#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# Put pp_liteseg.py from the PPLiteSeg.pytorch repo in the same folder
from pp_liteseg import PPLiteSeg


class RoadFollowerNode(Node):
    def __init__(self):
        super().__init__('ppliteseg_road_follower')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/zed_rear/zed_node_1/rgb/color/rect/image/compressed',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control timer: publish cmd_vel independently of image callback
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info(
            'Subscribed to /zed_rear/zed_node_1/rgb/color/rect/image/compressed'
        )
        self.get_logger().info('Publishing Twist commands to /cmd_vel')

        # ------------------------------------------------------------
        # Model settings
        # ------------------------------------------------------------
        self.weights_path = '/u/97/habibip1/data/Downloads/ppliteset_pp2torch_cityscape_pretrained.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Lower input size than before for better speed
        self.model_input_w = 512
        self.model_input_h = 256

        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Cityscapes trainId for road
        self.road_class_ids = {0}

        # ------------------------------------------------------------
        # Controller settings
        # ------------------------------------------------------------
        self.linear_speed = 0.20          # constant forward speed
        self.kp_angular = 0.90            # steering gain
        self.max_angular_speed = 0.70     # rad/s
        self.angular_smoothing = 0.30     # low-pass filter [0..1]

        self.stop_when_road_lost = False  # keep moving slowly if road is briefly lost
        self.lost_timeout_frames = 6      # use previous steering for a few frames

        # Scan-line road-center extraction
        self.lookahead_y_start_ratio = 0.62
        self.lookahead_y_end_ratio = 0.90
        self.num_scan_rows = 4
        self.min_road_width_px = 20

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------
        self.model = self.load_model()

        self.last_mask = None
        self.last_pred = None
        self.last_color_mask = None
        self.last_infer_ms = 0.0

        self.latest_target_point = None
        self.latest_target_x = None
        self.latest_img_width = None
        self.latest_heading_deg = 0.0

        self.prev_angular_z = 0.0
        self.last_good_angular_z = 0.0
        self.frames_since_target = 999999

        # Visualization helpers
        self.last_frame = None
        self.last_overlay = None
        self.last_center_points = []
        self.last_edge_points = []

        # Cityscapes trainId colors (BGR for OpenCV)
        self.class_colors_bgr = {
            0:  (128,  64, 128),  # road
            1:  (232,  35, 244),  # sidewalk
            2:  ( 70,  70,  70),  # building
            3:  (156, 102, 102),  # wall
            4:  (153, 153, 190),  # fence
            5:  (153, 153, 153),  # pole
            6:  ( 30, 170, 250),  # traffic light
            7:  (  0, 220, 220),  # traffic sign
            8:  ( 35, 142, 107),  # vegetation
            9:  (152, 251, 152),  # terrain
            10: (180, 130,  70),  # sky
            11: ( 60,  20, 220),  # person
            12: (  0,   0, 255),  # rider
            13: (142,   0,   0),  # car
            14: ( 70,   0,   0),  # truck
            15: (100,  60,   0),  # bus
            16: (100,  80,   0),  # train
            17: (230,   0,   0),  # motorcycle
            18: ( 32,  11, 119),  # bicycle
        }

        self.class_names = {
            0: 'road',
            1: 'sidewalk',
            2: 'building',
            3: 'wall',
            4: 'fence',
            5: 'pole',
            6: 'traffic light',
            7: 'traffic sign',
            8: 'vegetation',
            9: 'terrain',
            10: 'sky',
            11: 'person',
            12: 'rider',
            13: 'car',
            14: 'truck',
            15: 'bus',
            16: 'train',
            17: 'motorcycle',
            18: 'bicycle',
        }

    def load_model(self):
        model = PPLiteSeg()
        ckpt = torch.load(self.weights_path, map_location=self.device)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.get_logger().info(
            f'Loaded PPLiteSeg weights from {self.weights_path} on {self.device}'
        )
        return model

    def preprocess(self, frame_bgr):
        resized = cv2.resize(
            frame_bgr,
            (self.model_input_w, self.model_input_h),
            interpolation=cv2.INTER_LINEAR
        )

        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)

        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return tensor

    def refine_mask(self, mask):
        if mask is None:
            return None

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        if num_labels <= 1:
            return mask

        h, w = mask.shape

        seed_y1, seed_y2 = int(h * 0.80), h
        seed_x1, seed_x2 = int(w * 0.35), int(w * 0.65)

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

    def decode_color_mask(self, pred):
        color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        present_classes = np.unique(pred)
        for class_id in present_classes:
            class_id = int(class_id)
            color = self.class_colors_bgr.get(
                class_id,
                ((37 * class_id) % 255, (17 * class_id) % 255, (97 * class_id) % 255)
            )
            color_mask[pred == class_id] = color

        return color_mask

    def extract_road_mask(self, pred):
        road_mask = np.isin(pred, list(self.road_class_ids)).astype(np.uint8) * 255
        road_mask = self.refine_mask(road_mask)
        return road_mask

    def segment_classes(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        input_tensor = self.preprocess(frame_bgr)

        with torch.inference_mode():
            t0 = time.perf_counter()
            outputs = self.model(input_tensor)
            self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        pred_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)
        color_mask = self.decode_color_mask(pred)
        road_mask = self.extract_road_mask(pred)

        return pred, color_mask, road_mask

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

        # Weighted average: closer rows (near bottom) matter more
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
        angular_z = float(np.clip(angular_z, -self.max_angular_speed, self.max_angular_speed))

        # smoothing
        angular_z = (
            (1.0 - self.angular_smoothing) * self.prev_angular_z
            + self.angular_smoothing * angular_z
        )
        self.prev_angular_z = angular_z

        return angular_z, error_px, error_norm

    def draw_class_legend(self, image, pred):
        present_classes = [int(c) for c in np.unique(pred)]
        x0, y0 = 15, 90
        box_w, box_h = 18, 18
        line_h = 24

        for i, class_id in enumerate(present_classes[:10]):
            y = y0 + i * line_h
            color = self.class_colors_bgr.get(
                class_id,
                ((37 * class_id) % 255, (17 * class_id) % 255, (97 * class_id) % 255)
            )
            label = self.class_names.get(class_id, f'class {class_id}')

            cv2.rectangle(image, (x0, y), (x0 + box_w, y + box_h), color, -1)
            cv2.putText(
                image,
                label,
                (x0 + box_w + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

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
            pred, color_mask, road_mask = self.segment_classes(frame)
            self.last_pred = pred
            self.last_color_mask = color_mask
            self.last_mask = road_mask
        except Exception as e:
            self.get_logger().error(f'PPLiteSeg inference failed: {e}')
            return

        road_target, center_points, edge_points = self.get_road_center_target(self.last_mask)

        self.last_center_points = center_points
        self.last_edge_points = edge_points
        self.latest_target_point = road_target

        if road_target is not None:
            self.latest_target_x = road_target[0]
            self.frames_since_target = 0
            control_origin = (w // 2, h - 30)
            self.latest_heading_deg = self.compute_heading_angle_deg(control_origin, road_target)
        else:
            self.frames_since_target += 1
            self.latest_target_x = None

        self.last_overlay = cv2.addWeighted(frame, 0.45, self.last_color_mask, 0.55, 0)

        self.update_visualization()

    def control_loop(self):
        cmd = Twist()

        linear_x = 0.0
        angular_z = 0.0

        if self.latest_img_width is not None and self.latest_target_x is not None:
            angular_z, _, _ = self.compute_angular_velocity(self.latest_img_width, self.latest_target_x)
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
            f'PPLiteSeg {self.last_infer_ms:.1f} ms',
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            result,
            f'linear={self.linear_speed:.2f}  last_ang={self.prev_angular_z:+.2f}',
            (15, 58 if self.latest_target_point is not None else 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if self.last_pred is not None:
            self.draw_class_legend(result, self.last_pred)

        cv2.imshow('Image', frame)
        cv2.imshow('Road Mask', self.last_mask if self.last_mask is not None else np.zeros((h, w), dtype=np.uint8))
        cv2.imshow('All Masks Color', self.last_color_mask if self.last_color_mask is not None else np.zeros((h, w, 3), dtype=np.uint8))
        cv2.imshow('PPLiteSeg Overlay + Control', result)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RoadFollowerNode()

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