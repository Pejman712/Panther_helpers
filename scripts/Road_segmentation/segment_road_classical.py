import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ColorEdgeRoadFollowerNode(Node):
    def __init__(self):
        super().__init__('color_edge_road_follower')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/zed_rear/zed_node_1/rgb/color/rect/image/compressed',
            self.image_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info(
            'Subscribed to /zed_rear/zed_node_1/rgb/color/rect/image/compressed'
        )
        self.get_logger().info('Publishing Twist commands to /cmd_vel')

        # ------------------------------------------------------------
        # Classical road detection settings
        # ------------------------------------------------------------
        self.roi_top_ratio = 0.40
        self.blur_kernel = 7

        # Bottom-center seed patch, assumed to be road
        self.seed_x1_ratio = 0.45
        self.seed_x2_ratio = 0.55
        self.seed_y1_ratio = 0.82
        self.seed_y2_ratio = 0.96

        # Lab color distance threshold
        self.color_dist_base = 18.0
        self.color_dist_gain = 2.0
        self.color_dist_min = 12.0
        self.color_dist_max = 42.0

        # Edge detection
        self.canny_low = 50
        self.canny_high = 150
        self.edge_dilate_size = 3

        # Row-wise corridor tracing
        self.max_row_gap_px = 12
        self.row_snap_radius_px = 80
        self.row_min_width_px = 30

        # ------------------------------------------------------------
        # Controller settings
        # ------------------------------------------------------------
        self.linear_speed = 0.20
        self.kp_angular = 0.90
        self.max_angular_speed = 0.70
        self.angular_smoothing = 0.30
        self.stop_when_road_lost = False
        self.lost_timeout_frames = 6

        # Scan-line road-center extraction
        self.lookahead_y_start_ratio = 0.62
        self.lookahead_y_end_ratio = 0.90
        self.num_scan_rows = 4
        self.min_road_width_px = 20

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------
        self.last_mask = None
        self.last_overlay = None
        self.last_frame = None
        self.last_edges = None
        self.last_color_mask = None

        self.latest_target_point = None
        self.latest_target_x = None
        self.latest_img_width = None
        self.latest_heading_deg = 0.0

        self.prev_angular_z = 0.0
        self.last_good_angular_z = 0.0
        self.frames_since_target = 999999

        self.last_center_points = []
        self.last_edge_points = []

    def keep_bottom_center_component(self, mask):
        if mask is None:
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return mask

        h, w = mask.shape
        seed_y1, seed_y2 = int(h * 0.80), h
        seed_x1, seed_x2 = int(w * 0.40), int(w * 0.60)

        best_label = 0
        best_overlap = 0

        for label in range(1, num_labels):
            overlap = np.count_nonzero(
                labels[seed_y1:seed_y2, seed_x1:seed_x2] == label
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        if best_label == 0:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                best_label = 1 + int(np.argmax(areas))

        clean = np.zeros_like(mask)
        if best_label > 0:
            clean[labels == best_label] = 255

        return clean

    def refine_mask(self, mask):
        if mask is None:
            return None

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = self.keep_bottom_center_component(mask)

        # Fill horizontal holes row-by-row
        h, w = mask.shape
        filled = np.zeros_like(mask)
        for y in range(h):
            xs = np.where(mask[y] > 0)[0]
            if len(xs) >= 2:
                filled[y, xs[0]:xs[-1] + 1] = 255
            elif len(xs) == 1:
                filled[y, xs[0]] = 255

        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel, iterations=1)
        return filled

    def compute_seed_color(self, lab_roi):
        roi_h, roi_w = lab_roi.shape[:2]

        sx1 = int(roi_w * self.seed_x1_ratio)
        sx2 = int(roi_w * self.seed_x2_ratio)
        sy1 = int(roi_h * self.seed_y1_ratio)
        sy2 = int(roi_h * self.seed_y2_ratio)

        seed_patch = lab_roi[sy1:sy2, sx1:sx2]
        if seed_patch.size == 0:
            return None, None

        seed_pixels = seed_patch.reshape(-1, 3)
        seed_color = np.median(seed_pixels, axis=0)
        seed_std = float(np.mean(np.std(seed_pixels, axis=0)))

        return seed_color, seed_std

    def build_color_mask(self, roi_bgr):
        blurred = cv2.GaussianBlur(
            roi_bgr,
            (self.blur_kernel, self.blur_kernel),
            0
        )
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

        seed_color, seed_std = self.compute_seed_color(lab)
        if seed_color is None:
            return np.zeros(roi_bgr.shape[:2], dtype=np.uint8)

        dist_thresh = np.clip(
            self.color_dist_base + self.color_dist_gain * seed_std,
            self.color_dist_min,
            self.color_dist_max
        )

        color_dist = np.linalg.norm(lab - seed_color.reshape(1, 1, 3), axis=2)
        color_mask = (color_dist <= dist_thresh).astype(np.uint8) * 255
        return color_mask

    def build_edge_mask(self, roi_bgr):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        kernel = np.ones((self.edge_dilate_size, self.edge_dilate_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def trace_row_bounds(self, color_row, edge_row, start_x):
        w = len(color_row)
        if w == 0:
            return None

        start_x = int(np.clip(start_x, 0, w - 1))

        # If start_x is not on road color, snap to nearest road-colored pixel
        if not color_row[start_x]:
            xs = np.where(color_row > 0)[0]
            if len(xs) == 0:
                return None

            nearest = int(xs[np.argmin(np.abs(xs - start_x))])
            if abs(nearest - start_x) > self.row_snap_radius_px:
                return None
            center_x = nearest
        else:
            center_x = start_x

        # Expand left
        last_good_left = center_x
        gap = 0
        for x in range(center_x, -1, -1):
            if x < center_x and edge_row[x]:
                break
            if color_row[x]:
                last_good_left = x
                gap = 0
            else:
                gap += 1
                if gap > self.max_row_gap_px:
                    break

        # Expand right
        last_good_right = center_x
        gap = 0
        for x in range(center_x, w):
            if x > center_x and edge_row[x]:
                break
            if color_row[x]:
                last_good_right = x
                gap = 0
            else:
                gap += 1
                if gap > self.max_row_gap_px:
                    break

        left_x = int(last_good_left)
        right_x = int(last_good_right)

        if right_x - left_x + 1 < self.row_min_width_px:
            return None

        center_x = (left_x + right_x) // 2
        return left_x, right_x, center_x

    def build_road_mask_rowwise(self, color_mask, edge_mask, start_x):
        h, w = color_mask.shape
        road_mask = np.zeros((h, w), dtype=np.uint8)

        current_x = int(np.clip(start_x, 0, w - 1))

        for y in range(h - 1, -1, -1):
            color_row = color_mask[y] > 0
            edge_row = edge_mask[y] > 0

            bounds = self.trace_row_bounds(color_row, edge_row, current_x)
            if bounds is None:
                continue

            left_x, right_x, center_x = bounds
            road_mask[y, left_x:right_x + 1] = 255
            current_x = center_x

        return road_mask

    def detect_road_mask(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        roi_top = int(h * self.roi_top_ratio)
        roi = frame_bgr[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]

        if roi_h <= 0 or roi_w <= 0:
            return np.zeros((h, w), dtype=np.uint8)

        color_mask = self.build_color_mask(roi)
        edges = self.build_edge_mask(roi)

        start_x = w // 2
        if self.latest_target_x is not None:
            start_x = int(np.clip(self.latest_target_x, 0, w - 1))

        road_mask_roi = self.build_road_mask_rowwise(color_mask, edges, start_x)
        road_mask_roi = self.refine_mask(road_mask_roi)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[roi_top:, :] = road_mask_roi

        self.last_color_mask = np.zeros((h, w), dtype=np.uint8)
        self.last_color_mask[roi_top:, :] = color_mask

        self.last_edges = np.zeros((h, w), dtype=np.uint8)
        self.last_edges[roi_top:, :] = edges

        return full_mask

    def compute_heading_angle_deg(self, origin_point, target_point):
        ox, oy = origin_point
        tx, ty = target_point

        dx = tx - ox
        dy = oy - ty  # invert image y-axis

        return math.degrees(math.atan2(dx, dy))

    def get_road_center_target(self, mask):
        if mask is None:
            return None, [], []

        h, w = mask.shape

        ys = np.linspace(
            int(h * self.lookahead_y_start_ratio),
            int(h * self.lookahead_y_end_ratio),
            self.num_scan_rows
        ).astype(int)

        center_points = []
        edge_points = []

        for y in ys:
            xs = np.where(mask[y] > 0)[0]
            if len(xs) < self.min_road_width_px:
                continue

            left_x = int(xs[0])
            right_x = int(xs[-1])
            center_x = (left_x + right_x) // 2

            center_points.append((center_x, int(y)))
            edge_points.append((left_x, int(y)))
            edge_points.append((right_x, int(y)))

        if not center_points:
            return None, [], []

        weights = np.linspace(1.0, 2.0, len(center_points))
        target_x = int(np.average([p[0] for p in center_points], weights=weights))
        target_y = int(np.average([p[1] for p in center_points], weights=weights))

        return (target_x, target_y), center_points, edge_points

    def compute_angular_velocity(self, image_width, target_x):
        image_center_x = image_width / 2.0
        error_px = target_x - image_center_x
        error_norm = error_px / image_center_x

        # target on right -> negative angular.z
        angular_z = -self.kp_angular * error_norm
        angular_z = float(
            np.clip(angular_z, -self.max_angular_speed, self.max_angular_speed)
        )

        angular_z = (
            (1.0 - self.angular_smoothing) * self.prev_angular_z
            + self.angular_smoothing * angular_z
        )
        self.prev_angular_z = angular_z

        return angular_z, error_px, error_norm

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode image')
            return

        h, w = frame.shape[:2]
        self.last_frame = frame.copy()
        self.latest_img_width = w

        try:
            self.last_mask = self.detect_road_mask(frame)
        except Exception as e:
            self.get_logger().error(f'Color+edge road detection failed: {e}')
            return

        road_target, center_points, edge_points = self.get_road_center_target(
            self.last_mask
        )

        self.last_center_points = center_points
        self.last_edge_points = edge_points
        self.latest_target_point = road_target

        if road_target is not None:
            self.latest_target_x = road_target[0]
            self.frames_since_target = 0

            control_origin = (w // 2, h - 30)
            self.latest_heading_deg = self.compute_heading_angle_deg(
                control_origin,
                road_target
            )
        else:
            self.frames_since_target += 1
            self.latest_target_x = None

        overlay = frame.copy()
        overlay[self.last_mask > 0] = (0, 255, 0)

        # Draw detected edges in red for debugging
        if self.last_edges is not None:
            overlay[self.last_edges > 0] = (0, 0, 255)

        self.last_overlay = cv2.addWeighted(frame, 0.70, overlay, 0.30, 0)

        self.update_visualization()

    def control_loop(self):
        cmd = Twist()

        linear_x = 0.0
        angular_z = 0.0

        if self.latest_img_width is not None and self.latest_target_x is not None:
            angular_z, _, _ = self.compute_angular_velocity(
                self.latest_img_width,
                self.latest_target_x
            )
            linear_x = self.linear_speed
            self.last_good_angular_z = angular_z
        else:
            if self.frames_since_target < self.lost_timeout_frames:
                linear_x = self.linear_speed * 0.7
                angular_z = self.last_good_angular_z
            else:
                if self.stop_when_road_lost:
                    linear_x = 0.0
                    angular_z = 0.0
                else:
                    linear_x = self.linear_speed * 0.5
                    angular_z = 0.0

        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def update_visualization(self):
        if self.last_frame is None or self.last_overlay is None:
            return

        frame = self.last_frame
        h, w = frame.shape[:2]
        result = self.last_overlay.copy()

        control_origin = (w // 2, h - 30)
        cv2.circle(result, control_origin, 6, (255, 0, 0), -1)

        for p in self.last_edge_points:
            cv2.circle(result, p, 3, (0, 255, 0), -1)

        for p in self.last_center_points:
            cv2.circle(result, p, 4, (0, 165, 255), -1)

        if self.latest_target_point is not None:
            cv2.circle(result, self.latest_target_point, 7, (0, 0, 255), -1)
            cv2.line(result, control_origin, self.latest_target_point, (0, 255, 255), 2)

            text_x = (control_origin[0] + self.latest_target_point[0]) // 2
            text_y = (control_origin[1] + self.latest_target_point[1]) // 2 - 10

            cv2.putText(
                result,
                f'{self.latest_heading_deg:+.1f} deg',
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                result,
                f'target=({self.latest_target_point[0]}, {self.latest_target_point[1]})',
                (self.latest_target_point[0] + 10, self.latest_target_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                result,
                'Road target not found',
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            result,
            f'linear={self.linear_speed:.2f} last_ang={self.prev_angular_z:+.2f}',
            (15, 58 if self.latest_target_point is not None else 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow('Image', frame)
        cv2.imshow(
            'Road Color Mask',
            self.last_color_mask
            if self.last_color_mask is not None
            else np.zeros((h, w), dtype=np.uint8)
        )
        cv2.imshow(
            'Road Edges',
            self.last_edges
            if self.last_edges is not None
            else np.zeros((h, w), dtype=np.uint8)
        )
        cv2.imshow(
            'Road Mask',
            self.last_mask
            if self.last_mask is not None
            else np.zeros((h, w), dtype=np.uint8)
        )
        cv2.imshow('Overlay + Control', result)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ColorEdgeRoadFollowerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_cmd = Twist()
            node.cmd_pub.publish(stop_cmd)
        except Exception:
            pass

        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()