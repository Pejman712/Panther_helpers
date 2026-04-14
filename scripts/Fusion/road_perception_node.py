#!/usr/bin/env python3

import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from road_nav_interfaces.msg import RoadObservation

# Put pp_liteseg.py from the PPLiteSeg.pytorch repo in the same folder
from pp_liteseg import PPLiteSeg


class RoadPerceptionNode(Node):
    def __init__(self):
        super().__init__("road_perception_node")

        self.declare_parameter(
            "image_topic",
            "/zed_rear/zed_node_1/rgb/color/rect/image/compressed"
        )
        self.declare_parameter("observation_topic", "/road_observation")
        self.declare_parameter(
            "weights_path",
            "/u/97/habibip1/data/Downloads/ppliteset_pp2torch_cityscape_pretrained.pth"
        )
        self.declare_parameter("show_debug_windows", True)

        self.image_topic = self.get_parameter("image_topic").value
        self.observation_topic = self.get_parameter("observation_topic").value
        self.weights_path = self.get_parameter("weights_path").value
        self.show_debug_windows = bool(self.get_parameter("show_debug_windows").value)

        self.subscription = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10,
        )
        self.obs_pub = self.create_publisher(RoadObservation, self.observation_topic, 10)

        self.get_logger().info(f"Subscribed to {self.image_topic}")
        self.get_logger().info(f"Publishing road observations to {self.observation_topic}")

        # ------------------------------------------------------------
        # Model settings
        # ------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_input_w = 512
        self.model_input_h = 256

        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Cityscapes trainId for road
        self.road_class_ids = {0}

        # Scan-line road-center extraction
        self.lookahead_y_start_ratio = 0.62
        self.lookahead_y_end_ratio = 0.90
        self.num_scan_rows = 4
        self.min_road_width_px = 20

        self.model = self.load_model()

        self.last_mask = None
        self.last_overlay = None
        self.last_infer_ms = 0.0
        self.last_center_points = []
        self.last_edge_points = []
        self.prev_target_x: Optional[int] = None

    def load_model(self):
        model = PPLiteSeg()
        ckpt = torch.load(self.weights_path, map_location=self.device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.get_logger().info(
            f"Loaded PPLiteSeg weights from {self.weights_path} on {self.device}"
        )
        return model

    def preprocess(self, frame_bgr):
        resized = cv2.resize(
            frame_bgr,
            (self.model_input_w, self.model_input_h),
            interpolation=cv2.INTER_LINEAR,
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
            overlap = np.count_nonzero(labels[seed_y1:seed_y2, seed_x1:seed_x2] == label)
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

    def extract_road_mask(self, pred):
        road_mask = np.isin(pred, list(self.road_class_ids)).astype(np.uint8) * 255
        road_mask = self.refine_mask(road_mask)
        return road_mask

    def segment_road(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        input_tensor = self.preprocess(frame_bgr)

        with torch.inference_mode():
            t0 = time.perf_counter()
            outputs = self.model(input_tensor)
            self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        pred_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

        road_mask = self.extract_road_mask(pred)
        return road_mask

    def compute_heading_angle_rad(self, origin_point, target_point):
        ox, oy = origin_point
        tx, ty = target_point

        dx = tx - ox
        dy = oy - ty  # invert image y-axis

        return math.atan2(dx, dy)

    def get_road_center_target(self, mask):
        if mask is None:
            return None, [], [], []

        h, w = mask.shape
        ys = np.linspace(
            int(h * self.lookahead_y_start_ratio),
            int(h * self.lookahead_y_end_ratio),
            self.num_scan_rows,
        ).astype(int)

        center_points = []
        edge_points = []
        widths = []

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
            widths.append(float(right_x - left_x))

        if not center_points:
            return None, [], [], []

        weights = np.linspace(1.0, 2.0, len(center_points))
        target_x = int(np.average([p[0] for p in center_points], weights=weights))
        target_y = int(np.average([p[1] for p in center_points], weights=weights))

        return (target_x, target_y), center_points, edge_points, widths

    def estimate_confidence(
        self,
        mask: np.ndarray,
        widths: List[float],
        center_points: List[Tuple[int, int]],
        target_point: Optional[Tuple[int, int]],
    ) -> float:
        if mask is None or target_point is None or not center_points:
            return 0.0

        h, w = mask.shape
        valid_rows_ratio = len(center_points) / float(self.num_scan_rows)

        avg_width_norm = 0.0
        if widths:
            avg_width_norm = min(1.0, (float(np.mean(widths)) / max(1.0, 0.7 * w)))

        y1 = int(h * self.lookahead_y_start_ratio)
        y2 = int(h * self.lookahead_y_end_ratio)
        roi = mask[y1:y2, :]
        roi_area_ratio = float(np.count_nonzero(roi)) / max(1.0, roi.size)

        area_score = min(1.0, roi_area_ratio / 0.25)

        stability = 1.0
        if self.prev_target_x is not None:
            dx = abs(target_point[0] - self.prev_target_x)
            stability = max(0.0, 1.0 - dx / max(1.0, 0.25 * w))

        confidence = (
            0.40 * valid_rows_ratio
            + 0.35 * avg_width_norm
            + 0.25 * area_score
        )
        confidence *= (0.5 + 0.5 * stability)
        return float(np.clip(confidence, 0.0, 1.0))

    def publish_observation(
        self,
        img_msg: CompressedImage,
        image_shape,
        target_point: Optional[Tuple[int, int]],
        heading_error_rad: float,
        confidence: float,
        road_width_px: float,
        valid_rows: int,
    ):
        h, w = image_shape[:2]
        obs = RoadObservation()
        obs.header = img_msg.header

        if target_point is None:
            obs.valid = False
            obs.lateral_error_norm = 0.0
            obs.heading_error_rad = 0.0
            obs.confidence = 0.0
            obs.road_width_px = 0.0
            obs.target_x = -1
            obs.target_y = -1
            obs.valid_scan_rows = 0
        else:
            image_center_x = w / 2.0
            lateral_error_norm = (target_point[0] - image_center_x) / max(1.0, image_center_x)

            obs.valid = True
            obs.lateral_error_norm = float(np.clip(lateral_error_norm, -1.0, 1.0))
            obs.heading_error_rad = float(heading_error_rad)
            obs.confidence = float(confidence)
            obs.road_width_px = float(road_width_px)
            obs.target_x = int(target_point[0])
            obs.target_y = int(target_point[1])
            obs.valid_scan_rows = int(valid_rows)

        self.obs_pub.publish(obs)

    def draw_debug(self, frame, target_point, center_points, edge_points):
        if not self.show_debug_windows:
            return

        result = cv2.cvtColor(self.last_mask, cv2.COLOR_GRAY2BGR) if self.last_mask is not None else frame.copy()
        result = cv2.addWeighted(frame, 0.7, result, 0.3, 0)

        h, w = frame.shape[:2]
        control_origin = (w // 2, h - 30)
        cv2.circle(result, control_origin, 6, (255, 0, 0), -1)

        for p in edge_points:
            cv2.circle(result, p, 3, (0, 255, 0), -1)

        for p in center_points:
            cv2.circle(result, p, 4, (0, 165, 255), -1)

        if target_point is not None:
            cv2.circle(result, target_point, 7, (0, 0, 255), -1)
            cv2.line(result, control_origin, target_point, (0, 255, 255), 2)

        cv2.putText(
            result,
            f"PPLiteSeg {self.last_infer_ms:.1f} ms",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Road Mask", self.last_mask if self.last_mask is not None else np.zeros((h, w), dtype=np.uint8))
        cv2.imshow("Road Perception Debug", result)
        cv2.waitKey(1)

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warning("Failed to decode image")
            return

        try:
            road_mask = self.segment_road(frame)
        except Exception as e:
            self.get_logger().error(f"PPLiteSeg inference failed: {e}")
            return

        self.last_mask = road_mask

        target_point, center_points, edge_points, widths = self.get_road_center_target(road_mask)

        heading_error_rad = 0.0
        confidence = 0.0
        road_width_px = float(np.mean(widths)) if widths else 0.0

        if target_point is not None:
            control_origin = (frame.shape[1] // 2, frame.shape[0] - 30)
            heading_error_rad = self.compute_heading_angle_rad(control_origin, target_point)
            confidence = self.estimate_confidence(road_mask, widths, center_points, target_point)
            self.prev_target_x = int(target_point[0])

        self.publish_observation(
            img_msg=msg,
            image_shape=frame.shape,
            target_point=target_point,
            heading_error_rad=heading_error_rad,
            confidence=confidence,
            road_width_px=road_width_px,
            valid_rows=len(center_points),
        )

        self.draw_debug(frame, target_point, center_points, edge_points)


def main(args=None):
    rclpy.init(args=args)
    node = RoadPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()