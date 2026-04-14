#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from ultralytics import FastSAM


class CompressedImageViewer(Node):
    def __init__(self):
        super().__init__('compressed_image_viewer')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/zed_rear/zed_node_1/rgb/color/rect/image/compressed',
            self.image_callback,
            10
        )

        self.get_logger().info(
            'Subscribed to /zed_rear/zed_node_1/rgb/color/rect/image/compressed'
        )

        # FastSAM model
        self.model = FastSAM('/u/97/habibip1/data/Downloads/FastSAM-s.pt')

        # Speed settings
        self.frame_count = 0
        self.process_every_n = 1
        self.input_width = 640
        self.roi_top_ratio = 0.1
        self.last_mask = None

    def select_road_mask(self, masks, h, w):
        """
        Pick the most road-like mask.
        Softer scoring, no hard rejection except extreme cases.
        """
        if masks is None or len(masks) == 0:
            return None

        best_idx = 0
        best_score = -1e18

        # Favor masks covering lower-middle region
        y1, y2 = int(h * 0.60), int(h * 0.98)
        x1, x2 = int(w * 0.20), int(w * 0.80)

        for i, mask in enumerate(masks):
            mask_u8 = mask.astype(np.uint8)
            area = float(mask_u8.sum())
            area_ratio = area / float(h * w)
            center_area = float(mask_u8[y1:y2, x1:x2].sum())

            penalty = 0.0
            if area_ratio < 0.005:
                penalty -= 10000.0
            if area_ratio > 0.95:
                penalty -= 10000.0

            score = (3.0 * center_area) + (0.2 * area) + penalty
            if score > best_score:
                best_score = score
                best_idx = i

        return (masks[best_idx].astype(np.uint8) * 255)

    def refine_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        if num_labels > 1:
            h, w = mask.shape
            seed_y1, seed_y2 = int(h * 0.85), h
            seed_x1, seed_x2 = int(w * 0.40), int(w * 0.60)

            best_label = 0
            best_overlap = -1

            for label in range(1, num_labels):
                overlap = np.sum(labels[seed_y1:seed_y2, seed_x1:seed_x2] == label)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label

            if best_label > 0:
                clean = np.zeros_like(mask)
                clean[labels == best_label] = 255
                mask = clean

        return mask

    def get_mask_center(self, mask):
        """
        Returns the centroid (cx, cy) of the binary mask in image coordinates.
        """
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)

    def compute_heading_angle_deg(self, img_center, target_point):
        """
        Heading angle from image center to target point.

        Convention:
        - 0 deg = straight up in the image
        - positive = target is to the right
        - negative = target is to the left
        """
        ix, iy = img_center
        tx, ty = target_point

        dx = tx - ix
        dy = iy - ty  # invert image y-axis so up is positive

        angle_deg = math.degrees(math.atan2(dx, dy))
        return angle_deg

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode image')
            return

        h, w = frame.shape[:2]
        self.frame_count += 1

        if self.frame_count % self.process_every_n == 0:
            roi_top = int(h * self.roi_top_ratio)
            roi = frame[roi_top:, :]
            roi_h, roi_w = roi.shape[:2]

            scale = self.input_width / float(roi_w)
            small_w = self.input_width
            small_h = max(64, int(roi_h * scale))
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

            mask_full = np.zeros((h, w), dtype=np.uint8)

            try:
                results = self.model(
                    small,
                    retina_masks=False,
                    imgsz=self.input_width,
                    conf=0.40,
                    iou=0.7,
                    verbose=False
                )

                if results and len(results) > 0:
                    r = results[0]
                    if r.masks is not None and r.masks.data is not None:
                        masks = r.masks.data.cpu().numpy()
                        self.get_logger().info(f'FastSAM returned {len(masks)} masks')

                        best_mask = self.select_road_mask(masks, small_h, small_w)

                        if best_mask is None and len(masks) > 0:
                            areas = [m.astype(np.uint8).sum() for m in masks]
                            best_mask = (masks[int(np.argmax(areas))].astype(np.uint8) * 255)

                        if best_mask is not None:
                            best_mask = self.refine_mask(best_mask)
                            best_mask = cv2.resize(
                                best_mask,
                                (roi_w, roi_h),
                                interpolation=cv2.INTER_NEAREST
                            )
                            mask_full[roi_top:, :] = best_mask
                        else:
                            self.get_logger().warn('FastSAM returned no usable masks')
                    else:
                        self.get_logger().warn('FastSAM returned no masks')
                else:
                    self.get_logger().warn('FastSAM returned no results')

            except Exception as e:
                self.get_logger().error(f'FastSAM inference failed: {e}')

            self.last_mask = mask_full

        if self.last_mask is None:
            self.last_mask = np.zeros((h, w), dtype=np.uint8)

        overlay = frame.copy()
        overlay[self.last_mask > 0] = (0, 255, 0)

        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Image center
        img_center = (w // 2, h // 2)

        # Draw image center
        cv2.circle(result, img_center, 5, (255, 0, 0), -1)  # blue center point

        # Mask center and heading angle
        mask_center = self.get_mask_center(self.last_mask)
        if mask_center is not None:
            # Red point at mask center
            cv2.circle(result, mask_center, 6, (0, 0, 255), -1)

            # Line from image center to mask center
            cv2.line(result, img_center, mask_center, (0, 255, 255), 2)

            # Compute heading angle
            heading_angle_deg = self.compute_heading_angle_deg(img_center, mask_center)

            # Put heading text near the middle of the line
            text_x = (img_center[0] + mask_center[0]) // 2
            text_y = (img_center[1] + mask_center[1]) // 2 - 10
            cv2.putText(
                result,
                f'{heading_angle_deg:+.1f} deg',
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Optional: label the red point
            cv2.putText(
                result,
                f'({mask_center[0]}, {mask_center[1]})',
                (mask_center[0] + 10, mask_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow('Image', frame)
        cv2.imshow('FastSAM Mask', self.last_mask)
        cv2.imshow('FastSAM Overlay', result)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
