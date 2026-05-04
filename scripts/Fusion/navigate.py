#!/usr/bin/env python3

import argparse
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import torch

from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, Twist
from nav_msgs.msg import Path
from nav2_simple_commander.robot_navigator import BasicNavigator
from pyproj import CRS, Transformer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from robot_localization.srv import FromLL, ToLL
from sensor_msgs.msg import CompressedImage
from tf2_ros import Buffer, TransformListener

from road_nav_interfaces.msg import RoadObservation

# Put pp_liteseg.py from the PPLiteSeg.pytorch repo in the same folder
from pp_liteseg import PPLiteSeg


# ============================================================
# Shared geometry helpers
# ============================================================

def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = (32700 + zone) if lat < 0.0 else (32600 + zone)
    return CRS.from_epsg(epsg)


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


def yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def resample_polyline(
    points: List[Tuple[float, float]],
    spacing_m: float,
) -> List[Tuple[float, float]]:
    """
    Densify polyline so consecutive points are approximately spacing_m apart.
    points: [(x, y), ...] in meters.
    """
    if spacing_m <= 0 or len(points) < 2:
        return points

    out: List[Tuple[float, float]] = [points[0]]
    carry = 0.0

    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)

        if seg_len < 1e-9:
            continue

        ux = dx / seg_len
        uy = dy / seg_len

        dist = spacing_m - carry if carry > 1e-9 else spacing_m

        while dist < seg_len - 1e-9:
            out.append((x0 + ux * dist, y0 + uy * dist))
            dist += spacing_m

        out.append((x1, y1))

        last_mark = dist - spacing_m
        remain = seg_len - last_mark
        carry = 0.0 if remain < 1e-6 else spacing_m - remain

    filtered: List[Tuple[float, float]] = []
    last = None

    for p in out:
        if last is None or abs(p[0] - last[0]) > 1e-6 or abs(p[1] - last[1]) > 1e-6:
            filtered.append(p)
        last = p

    return filtered


# ============================================================
# OSMnx planner helper
# ============================================================

def plan_osmnx_route_lonlat(
    start_lat: float,
    start_lon: float,
    goal_lat: float,
    goal_lon: float,
    network_type: str = "all",
    dist_m: float = 1200.0,
) -> List[Tuple[float, float]]:
    """
    Returns route polyline as [(lon, lat), ...] using OSM edge geometries where possible.
    """
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import LineString

    ox.settings.use_cache = True
    ox.settings.log_console = True

    mid = ((start_lat + goal_lat) / 2.0, (start_lon + goal_lon) / 2.0)
    G = ox.graph_from_point(mid, dist=dist_m, network_type=network_type)

    orig = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
    dest = ox.distance.nearest_nodes(G, X=goal_lon, Y=goal_lat)

    route = nx.shortest_path(G, orig, dest, weight="length")

    coords: List[Tuple[float, float]] = []

    for u, v in zip(route[:-1], route[1:]):
        data_dict = G.get_edge_data(u, v)

        if not data_dict:
            continue

        best = min(data_dict.values(), key=lambda d: d.get("length", float("inf")))
        geom = best.get("geometry", None)

        if geom is None:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            geom = LineString([(x1, y1), (x2, y2)])

        seg = list(geom.coords)

        if coords and seg and tuple(seg[0]) == tuple(coords[-1]):
            seg = seg[1:]

        coords.extend([(float(x), float(y)) for x, y in seg])

    filtered: List[Tuple[float, float]] = []
    last = None

    for p in coords:
        if last is None or p != last:
            filtered.append(p)
        last = p

    return filtered


# ============================================================
# Road perception node
# ============================================================

class RoadPerceptionNode(Node):
    def __init__(self):
        super().__init__("road_perception_node")

        self.declare_parameter(
            "image_topic",
            "/zed_rear/zed_node_1/rgb/color/rect/image/compressed",
        )
        self.declare_parameter("observation_topic", "/road_observation")
        self.declare_parameter(
            "weights_path",
            "/u/97/habibip1/data/Downloads/ppliteset_pp2torch_cityscape_pretrained.pth",
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

        self.obs_pub = self.create_publisher(
            RoadObservation,
            self.observation_topic,
            10,
        )

        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")
        self.get_logger().info(f"Publishing road observations to: {self.observation_topic}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_input_w = 512
        self.model_input_h = 256

        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Cityscapes trainId for road
        self.road_class_ids = {0}

        self.lookahead_y_start_ratio = 0.62
        self.lookahead_y_end_ratio = 0.90
        self.num_scan_rows = 4
        self.min_road_width_px = 20

        self.model = self.load_model()

        self.last_mask = None
        self.last_infer_ms = 0.0
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

        pred = cv2.resize(
            pred_small,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        road_mask = self.extract_road_mask(pred)
        return road_mask

    def compute_heading_angle_rad(self, origin_point, target_point):
        ox, oy = origin_point
        tx, ty = target_point

        dx = tx - ox
        dy = oy - ty

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
            avg_width_norm = min(
                1.0,
                float(np.mean(widths)) / max(1.0, 0.7 * w),
            )

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

        confidence *= 0.5 + 0.5 * stability

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
            lateral_error_norm = (
                target_point[0] - image_center_x
            ) / max(1.0, image_center_x)

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

        h, w = frame.shape[:2]

        if self.last_mask is not None:
            result = cv2.cvtColor(self.last_mask, cv2.COLOR_GRAY2BGR)
        else:
            result = frame.copy()

        result = cv2.addWeighted(frame, 0.7, result, 0.3, 0)

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

        cv2.imshow(
            "Road Mask",
            self.last_mask if self.last_mask is not None else np.zeros((h, w), dtype=np.uint8),
        )
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

        target_point, center_points, edge_points, widths = self.get_road_center_target(
            road_mask
        )

        heading_error_rad = 0.0
        confidence = 0.0
        road_width_px = float(np.mean(widths)) if widths else 0.0

        if target_point is not None:
            control_origin = (frame.shape[1] // 2, frame.shape[0] - 30)
            heading_error_rad = self.compute_heading_angle_rad(
                control_origin,
                target_point,
            )
            confidence = self.estimate_confidence(
                road_mask,
                widths,
                center_points,
                target_point,
            )
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


# ============================================================
# Road/Nav fusion node
# ============================================================

class RoadNavFusionNode(Node):
    def __init__(self):
        super().__init__("road_nav_fusion_node")

        self.declare_parameter("nav_cmd_topic", "/cmd_vel_nav")
        self.declare_parameter("road_obs_topic", "/road_observation")
        self.declare_parameter("path_topic", "/gps_route_path")
        self.declare_parameter("output_cmd_topic", "/cmd_vel")
        self.declare_parameter("base_frame", "base_link")

        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("nav_cmd_timeout_sec", 0.50)
        self.declare_parameter("road_obs_timeout_sec", 0.50)

        self.declare_parameter("road_blend_gain", 0.60)
        self.declare_parameter("max_blend", 0.50)
        self.declare_parameter("min_confidence_for_blend", 0.25)

        self.declare_parameter("kp_road_lateral", 0.70)
        self.declare_parameter("kp_road_heading", 0.80)
        self.declare_parameter("max_road_correction", 0.35)
        self.declare_parameter("max_angular_speed", 1.00)

        self.declare_parameter("turn_lookahead_m", 3.0)
        self.declare_parameter("sharp_turn_angle_deg", 20.0)
        self.declare_parameter("sharp_turn_blend_scale", 0.25)

        self.nav_cmd_topic = self.get_parameter("nav_cmd_topic").value
        self.road_obs_topic = self.get_parameter("road_obs_topic").value
        self.path_topic = self.get_parameter("path_topic").value
        self.output_cmd_topic = self.get_parameter("output_cmd_topic").value
        self.base_frame = self.get_parameter("base_frame").value

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.nav_cmd_timeout_sec = float(self.get_parameter("nav_cmd_timeout_sec").value)
        self.road_obs_timeout_sec = float(self.get_parameter("road_obs_timeout_sec").value)

        self.road_blend_gain = float(self.get_parameter("road_blend_gain").value)
        self.max_blend = float(self.get_parameter("max_blend").value)
        self.min_confidence_for_blend = float(
            self.get_parameter("min_confidence_for_blend").value
        )

        self.kp_road_lateral = float(self.get_parameter("kp_road_lateral").value)
        self.kp_road_heading = float(self.get_parameter("kp_road_heading").value)
        self.max_road_correction = float(self.get_parameter("max_road_correction").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self.turn_lookahead_m = float(self.get_parameter("turn_lookahead_m").value)
        self.sharp_turn_angle_deg = float(self.get_parameter("sharp_turn_angle_deg").value)
        self.sharp_turn_blend_scale = float(
            self.get_parameter("sharp_turn_blend_scale").value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)

        self.create_subscription(Twist, self.nav_cmd_topic, self.nav_cmd_cb, 10)
        self.create_subscription(RoadObservation, self.road_obs_topic, self.road_obs_cb, 10)
        self.create_subscription(Path, self.path_topic, self.path_cb, 10)

        self.latest_nav_cmd: Optional[Twist] = None
        self.latest_road_obs: Optional[RoadObservation] = None
        self.latest_path: Optional[Path] = None

        self.last_nav_cmd_t = 0.0
        self.last_road_obs_t = 0.0
        self.last_debug_t = 0.0

        self.control_timer = self.create_timer(
            1.0 / self.control_hz,
            self.control_loop,
        )

        self.get_logger().info(f"Listening nav cmd: {self.nav_cmd_topic}")
        self.get_logger().info(f"Listening road obs: {self.road_obs_topic}")
        self.get_logger().info(f"Listening path: {self.path_topic}")
        self.get_logger().info(f"Publishing fused cmd: {self.output_cmd_topic}")

    def nav_cmd_cb(self, msg: Twist):
        self.latest_nav_cmd = msg
        self.last_nav_cmd_t = time.monotonic()

    def road_obs_cb(self, msg: RoadObservation):
        self.latest_road_obs = msg
        self.last_road_obs_t = time.monotonic()

    def path_cb(self, msg: Path):
        self.latest_path = msg

    def get_robot_pose_in_frame(self, frame_id: str) -> Tuple[float, float, float]:
        tf = self.tf_buffer.lookup_transform(
            frame_id,
            self.base_frame,
            rclpy.time.Time(),
        )

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        yaw = yaw_from_quaternion(tf.transform.rotation)

        return x, y, yaw

    def path_to_numpy(self, path: Path):
        if path is None or len(path.poses) < 2:
            return None

        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in path.poses],
            dtype=np.float32,
        )

        return pts

    def advance_index_by_distance(
        self,
        pts: np.ndarray,
        start_idx: int,
        dist_m: float,
    ) -> int:
        acc = 0.0
        i = start_idx

        while i < len(pts) - 1 and acc < dist_m:
            dx = float(pts[i + 1, 0] - pts[i, 0])
            dy = float(pts[i + 1, 1] - pts[i, 1])
            acc += math.hypot(dx, dy)
            i += 1

        return i

    def compute_upcoming_turn_angle(self) -> float:
        if self.latest_path is None:
            return 0.0

        pts = self.path_to_numpy(self.latest_path)

        if pts is None or len(pts) < 4:
            return 0.0

        try:
            rx, ry, _ = self.get_robot_pose_in_frame(self.latest_path.header.frame_id)
        except Exception:
            return 0.0

        d2 = np.sum((pts - np.array([rx, ry], dtype=np.float32)) ** 2, axis=1)
        idx = int(np.argmin(d2))

        if idx >= len(pts) - 2:
            return 0.0

        i0 = idx
        i1 = min(idx + 1, len(pts) - 1)
        i2 = self.advance_index_by_distance(pts, idx, self.turn_lookahead_m)
        i3 = min(i2 + 1, len(pts) - 1)

        if i1 <= i0 or i3 <= i2:
            return 0.0

        h0 = math.atan2(
            pts[i1, 1] - pts[i0, 1],
            pts[i1, 0] - pts[i0, 0],
        )
        h1 = math.atan2(
            pts[i3, 1] - pts[i2, 1],
            pts[i3, 0] - pts[i2, 0],
        )

        return abs(normalize_angle(h1 - h0))

    def compute_road_correction(self, obs: RoadObservation) -> float:
        # Road target to the right should make angular.z more negative.
        correction = -(
            self.kp_road_lateral * float(obs.lateral_error_norm)
            + self.kp_road_heading * float(obs.heading_error_rad)
        )

        correction = float(
            np.clip(
                correction,
                -self.max_road_correction,
                self.max_road_correction,
            )
        )

        return correction

    def control_loop(self):
        now = time.monotonic()

        if (
            self.latest_nav_cmd is None
            or (now - self.last_nav_cmd_t) > self.nav_cmd_timeout_sec
        ):
            stop = Twist()
            self.cmd_pub.publish(stop)
            return

        fused = Twist()
        fused.linear.x = self.latest_nav_cmd.linear.x
        fused.linear.y = self.latest_nav_cmd.linear.y
        fused.linear.z = self.latest_nav_cmd.linear.z
        fused.angular.x = self.latest_nav_cmd.angular.x
        fused.angular.y = self.latest_nav_cmd.angular.y
        fused.angular.z = self.latest_nav_cmd.angular.z

        alpha = 0.0
        road_corr = 0.0
        turn_angle = 0.0

        road_obs_is_fresh = (
            self.latest_road_obs is not None
            and (now - self.last_road_obs_t) <= self.road_obs_timeout_sec
        )

        if road_obs_is_fresh and self.latest_road_obs.valid:
            obs = self.latest_road_obs

            if float(obs.confidence) >= self.min_confidence_for_blend:
                road_corr = self.compute_road_correction(obs)
                alpha = min(
                    self.max_blend,
                    self.road_blend_gain * float(obs.confidence),
                )

                turn_angle = self.compute_upcoming_turn_angle()
                sharp_turn_rad = math.radians(self.sharp_turn_angle_deg)

                if turn_angle > sharp_turn_rad:
                    alpha *= self.sharp_turn_blend_scale

                if (
                    abs(self.latest_nav_cmd.angular.z) > 0.35
                    and self.latest_nav_cmd.angular.z * road_corr < 0.0
                ):
                    alpha *= 0.5

                fused.angular.z += alpha * road_corr

                speed_scale = 1.0
                speed_scale *= max(0.70, 0.70 + 0.30 * float(obs.confidence))

                if turn_angle > sharp_turn_rad:
                    speed_scale *= 0.80

                fused.linear.x *= speed_scale

        fused.angular.z = float(
            np.clip(
                fused.angular.z,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
        )

        self.cmd_pub.publish(fused)

        if now - self.last_debug_t > 1.0:
            self.last_debug_t = now

            self.get_logger().info(
                f"nav_w={self.latest_nav_cmd.angular.z:+.3f} "
                f"road_corr={road_corr:+.3f} "
                f"alpha={alpha:.2f} "
                f"turn_deg={math.degrees(turn_angle):.1f} "
                f"fused_w={fused.angular.z:+.3f}"
            )


# ============================================================
# GPS click planner node
# ============================================================

class ClickPlanner(Node):
    def __init__(
        self,
        enable_plot: bool,
        use_fromll: bool,
        fromll_service: str,
        toll_service: str,
        localized: bool,
        spacing_m: float,
        graph_dist_m: float,
        network_type: str,
        clicked_topic: str,
        require_frame: str,
        goal_frame: str,
        base_frame: str,
        continue_on_abort: bool,
        max_goals: int,
        skip_near_m: float,
        anchor_yaw_offset_deg: float,
    ):
        super().__init__("gps_click_planner")

        self.enable_plot = enable_plot
        self.use_fromll = use_fromll
        self.fromll_service = fromll_service
        self.toll_service = toll_service
        self.localized = localized
        self.spacing_m = spacing_m
        self.graph_dist_m = graph_dist_m
        self.network_type = network_type
        self.clicked_topic = clicked_topic
        self.require_frame = require_frame
        self.goal_frame = goal_frame
        self.base_frame = base_frame
        self.continue_on_abort = continue_on_abort
        self.max_goals = max_goals
        self.skip_near_m = skip_near_m
        self.anchor_yaw_offset_deg = anchor_yaw_offset_deg
        self.anchor_yaw_offset_rad = math.radians(anchor_yaw_offset_deg)

        self.navigator = BasicNavigator("basic_navigator")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.fromll_client = None
        self.toll_client = None

        self._pending_lonlats: List[Tuple[float, float]] = []
        self._fromll_results: List[Tuple[float, float]] = []
        self._current_fromll_future: Optional[Future] = None
        self._current_fromll_index = 0
        self._converting = False
        self._fromll_start_time: Optional[float] = None

        self._fromll_timeout_sec = 30.0
        self._toll_timeout_sec = 5.0

        if self.use_fromll:
            self.fromll_client = self.create_client(FromLL, self.fromll_service)

            while not self.fromll_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f"[fromLL] waiting for service '{self.fromll_service}' ..."
                )

            self.get_logger().info(f"[fromLL] service ready ({self.fromll_service})")

        if self.localized and self.use_fromll:
            self.toll_client = self.create_client(ToLL, self.toll_service)

            while not self.toll_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f"[toLL] waiting for service '{self.toll_service}' ..."
                )

            self.get_logger().info(f"[toLL] service ready ({self.toll_service})")

        self.create_subscription(
            PointStamped,
            self.clicked_topic,
            self._clicked_cb,
            10,
        )

        self.start_ll: Optional[Tuple[float, float]] = None
        self.goal_ll: Optional[Tuple[float, float]] = None

        self._route_map: Optional[List[Tuple[float, float]]] = None
        self._idx = 0
        self._nav_active = False

        if self.enable_plot:
            self._init_plot()
            self._plot_timer = self.create_timer(0.2, self._plot_update)

        self._worker_timer = self.create_timer(0.1, self._worker)

        mode_name = "fromLL" if self.use_fromll else "anchor"

        if self.localized:
            self.get_logger().info(
                "[ready] Localized mode ON: click GOAL in Mapviz "
                f"(expect frame '{self.require_frame}', x=lon,y=lat). "
                f"mode={mode_name} spacing_m={self.spacing_m} "
                f"anchor_yaw_offset_deg={self.anchor_yaw_offset_deg:.2f}"
            )
        else:
            self.get_logger().info(
                "[ready] Click START then GOAL in Mapviz "
                f"(expect frame '{self.require_frame}', x=lon,y=lat). "
                f"mode={mode_name} spacing_m={self.spacing_m} "
                f"anchor_yaw_offset_deg={self.anchor_yaw_offset_deg:.2f}"
            )

    def _init_plot(self):
        import matplotlib

        matplotlib.use("TkAgg")

        import matplotlib.pyplot as plt

        self._plt = plt
        self._fig, self._ax = plt.subplots()
        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.set_title("Route in map frame")
        self._ax.set_xlabel("x (m)")
        self._ax.set_ylabel("y (m)")

        (self._route_line,) = self._ax.plot(
            [],
            [],
            marker="o",
            linestyle="-",
            label="Route/Goals",
        )

        self._robot_scatter = self._ax.scatter([], [], marker="x", label="Robot")
        self._current_goal_scatter = self._ax.scatter(
            [],
            [],
            marker="o",
            label="Current goal",
        )

        self._ax.legend()
        self._plt.ion()
        self._plt.show()

    def _plot_update(self):
        if not self.enable_plot or self._route_map is None:
            return

        xs = [p[0] for p in self._route_map]
        ys = [p[1] for p in self._route_map]

        self._route_line.set_data(xs, ys)

        try:
            rx, ry, _ = self._get_robot_pose()
            self._robot_scatter.set_offsets([[rx, ry]])
        except Exception:
            pass

        if 0 <= self._idx < len(self._route_map):
            gx, gy = self._route_map[self._idx]
            self._current_goal_scatter.set_offsets([[gx, gy]])

        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def _get_robot_pose(self) -> Tuple[float, float, float]:
        tf = self.tf_buffer.lookup_transform(
            self.goal_frame,
            self.base_frame,
            rclpy.time.Time(),
        )

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        yaw = yaw_from_quaternion(tf.transform.rotation)

        return x, y, yaw

    def _get_robot_lonlat_from_toll(self) -> Tuple[float, float]:
        if self.toll_client is None:
            raise RuntimeError("ToLL client not available")

        rx, ry, _ = self._get_robot_pose()

        req = ToLL.Request()
        req.map_point = Point()
        req.map_point.x = float(rx)
        req.map_point.y = float(ry)
        req.map_point.z = 0.0

        fut = self.toll_client.call_async(req)
        t0 = time.time()

        while rclpy.ok() and not fut.done():
            if time.time() - t0 > self._toll_timeout_sec:
                raise RuntimeError(
                    f"/toLL timeout after {self._toll_timeout_sec:.1f}s. "
                    "navsat_transform_node is likely not ready."
                )

            rclpy.spin_once(self, timeout_sec=0.05)

        exc = fut.exception()

        if exc is not None:
            raise RuntimeError(f"/toLL call failed: {exc}")

        resp = fut.result()

        if resp is None:
            raise RuntimeError("/toLL returned no response")

        lat = float(resp.ll_point.latitude)
        lon = float(resp.ll_point.longitude)

        return lat, lon

    def _clicked_cb(self, msg: PointStamped):
        if self.require_frame and msg.header.frame_id != self.require_frame:
            self.get_logger().warning(
                f"[click] ignored frame_id='{msg.header.frame_id}', "
                f"expected '{self.require_frame}'"
            )
            return

        lat = float(msg.point.y)
        lon = float(msg.point.x)

        if self.localized:
            self.get_logger().info(
                f"[state] GOAL set lat={lat:.7f} lon={lon:.7f}. "
                "Planning from current robot pose..."
            )
            self._reset_all()
            self.goal_ll = (lat, lon)
            self._plan_route_and_prepare()
            return

        if self.start_ll is None:
            self._reset_all()
            self.start_ll = (lat, lon)
            self.get_logger().info(
                f"[state] START set lat={lat:.7f} lon={lon:.7f}. "
                "Now click GOAL."
            )
            return

        if self.goal_ll is None:
            self.goal_ll = (lat, lon)
            self.get_logger().info(
                f"[state] GOAL set lat={lat:.7f} lon={lon:.7f}. Planning..."
            )
            self._plan_route_and_prepare()
            return

        self.get_logger().info("[state] resetting due to new START")
        self._reset_all()
        self.start_ll = (lat, lon)
        self.get_logger().info(
            f"[state] START set lat={lat:.7f} lon={lon:.7f}. "
            "Now click GOAL."
        )

    def _reset_all(self):
        self.start_ll = None
        self.goal_ll = None
        self._route_map = None
        self._idx = 0
        self._nav_active = False

        self._pending_lonlats.clear()
        self._fromll_results.clear()

        self._current_fromll_future = None
        self._current_fromll_index = 0
        self._converting = False
        self._fromll_start_time = None

    def _plan_route_and_prepare(self):
        if self.goal_ll is None:
            self.get_logger().error("[plan] goal not set")
            return

        if self.localized:
            try:
                if self.use_fromll:
                    s_lat, s_lon = self._get_robot_lonlat_from_toll()
                    self.get_logger().info(
                        f"[localized] robot start from /toLL "
                        f"lat={s_lat:.7f} lon={s_lon:.7f}"
                    )
                else:
                    self.get_logger().warning(
                        "[localized] anchor mode without /toLL uses approximate start. "
                        "For best accuracy use --localized --use-fromll."
                    )
                    g_lat, g_lon = self.goal_ll
                    s_lat, s_lon = g_lat, g_lon
            except Exception as e:
                self.get_logger().error(
                    f"[localized] failed to determine robot start: {e}"
                )
                self._reset_all()
                return
        else:
            if self.start_ll is None:
                self.get_logger().error("[plan] start not set")
                return

            s_lat, s_lon = self.start_ll

        g_lat, g_lon = self.goal_ll

        t0 = time.time()

        try:
            lonlats = plan_osmnx_route_lonlat(
                s_lat,
                s_lon,
                g_lat,
                g_lon,
                network_type=self.network_type,
                dist_m=self.graph_dist_m,
            )
        except Exception as e:
            self.get_logger().error(f"[plan] OSMnx failed: {e}")
            self._reset_all()
            return

        self.get_logger().info(
            f"[plan] route computed points={len(lonlats)} "
            f"time={time.time() - t0:.2f}s"
        )

        if len(lonlats) < 2:
            self.get_logger().error("[plan] route too short")
            self._reset_all()
            return

        if self.use_fromll:
            if self.fromll_client is None:
                self.get_logger().error(
                    "[fromLL] use_fromll=True but client not created"
                )
                self._reset_all()
                return

            self._pending_lonlats = lonlats[:]
            self._fromll_results = []
            self._current_fromll_future = None
            self._current_fromll_index = 0
            self._converting = True
            self._fromll_start_time = None

            self.get_logger().info(
                f"[fromLL] starting sequential conversion "
                f"points={len(self._pending_lonlats)}"
            )

            self._send_next_fromll_request()
            return

        try:
            route_map = self._convert_anchor(lonlats)
        except Exception as e:
            self.get_logger().error(f"[anchor] conversion failed: {e}")
            self._reset_all()
            return

        self._finalize_route_and_start(route_map)

    def _send_next_fromll_request(self):
        if not self._pending_lonlats:
            self.get_logger().error("[fromLL] no pending lon/lat points")
            self._reset_all()
            return

        if self._current_fromll_index >= len(self._pending_lonlats):
            route_map = self._fromll_results[:]

            self._converting = False
            self._pending_lonlats.clear()
            self._fromll_results.clear()
            self._current_fromll_future = None
            self._current_fromll_index = 0
            self._fromll_start_time = None

            self.get_logger().info(
                f"[fromLL] conversion complete points={len(route_map)}"
            )

            self._finalize_route_and_start(route_map)
            return

        lon, lat = self._pending_lonlats[self._current_fromll_index]

        req = FromLL.Request()
        req.ll_point.longitude = float(lon)
        req.ll_point.latitude = float(lat)
        req.ll_point.altitude = 0.0

        self._current_fromll_future = self.fromll_client.call_async(req)
        self._fromll_start_time = time.time()

        i = self._current_fromll_index + 1
        n = len(self._pending_lonlats)

        self.get_logger().info(
            f"[fromLL] requested {i}/{n} lat={lat:.7f} lon={lon:.7f}"
        )

    def _convert_anchor(
        self,
        lonlats: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        lon0, lat0 = lonlats[0]

        utm = utm_crs_for_lonlat(lon0, lat0)
        to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)

        x0, y0 = to_utm.transform(lon0, lat0)

        local_xy: List[Tuple[float, float]] = []

        for lon, lat in lonlats:
            x, y = to_utm.transform(lon, lat)
            local_xy.append((x - x0, y - y0))

        rx, ry, _ = self._get_robot_pose()

        anchor_yaw = self.anchor_yaw_offset_rad
        c = math.cos(anchor_yaw)
        s = math.sin(anchor_yaw)

        self.get_logger().info(
            f"[anchor] robot pose anchor only: x={rx:.2f} y={ry:.2f} "
            f"fixed_offset={self.anchor_yaw_offset_deg:.2f} deg"
        )

        route_map: List[Tuple[float, float]] = []

        for dx, dy in local_xy:
            mx = rx + c * dx - s * dy
            my = ry + s * dx + c * dy
            route_map.append((mx, my))

        return route_map

    def _finalize_route_and_start(self, route_map: List[Tuple[float, float]]):
        before = len(route_map)
        route_map = resample_polyline(route_map, self.spacing_m)
        after = len(route_map)

        self.get_logger().info(
            f"[route] densified points={after} "
            f"(was {before}) spacing_m={self.spacing_m:.2f}"
        )

        if self.max_goals > 0 and len(route_map) > self.max_goals:
            original_last = route_map[-1]
            step = max(1, len(route_map) // self.max_goals)
            route_map = route_map[::step]

            if route_map[-1] != original_last:
                route_map.append(original_last)

            self.get_logger().warning(
                f"[route] capped goals to {len(route_map)} "
                f"(max_goals={self.max_goals})"
            )

        try:
            rx, ry, _ = self._get_robot_pose()

            while route_map:
                d = math.hypot(route_map[0][0] - rx, route_map[0][1] - ry)

                if d < self.skip_near_m:
                    self.get_logger().info(
                        f"[route] skipping near start goal "
                        f"d={d:.2f}m < {self.skip_near_m:.2f}m"
                    )
                    route_map.pop(0)
                else:
                    break

        except Exception as e:
            self.get_logger().warning(
                f"[route] could not evaluate skip_near_m: {e}"
            )

        if not route_map:
            self.get_logger().error("[route] no usable goals remain after filtering")
            self._reset_all()
            return

        self._route_map = route_map
        self._idx = 0
        self._nav_active = False

        self.get_logger().info(
            f"[nav2] route ready goals={len(self._route_map)} starting..."
        )

        self._send_next_goal()

    def _worker(self):
        if self._converting:
            if self._current_fromll_future is None:
                self._send_next_fromll_request()
                return

            if not self._current_fromll_future.done():
                if self._fromll_start_time is not None:
                    elapsed = time.time() - self._fromll_start_time

                    if elapsed > self._fromll_timeout_sec:
                        i = self._current_fromll_index + 1
                        n = len(self._pending_lonlats)
                        lon, lat = self._pending_lonlats[self._current_fromll_index]

                        self.get_logger().error(
                            f"[fromLL] timeout on point {i}/{n} "
                            f"after {elapsed:.1f}s "
                            f"lat={lat:.7f} lon={lon:.7f}. "
                            "Try calling /fromLL manually with this same lat/lon."
                        )

                        self._reset_all()

                return

            try:
                exc = self._current_fromll_future.exception()

                if exc is not None:
                    raise RuntimeError(exc)

                resp = self._current_fromll_future.result()

                if resp is None:
                    raise RuntimeError("/fromLL returned no response")

                x = float(resp.map_point.x)
                y = float(resp.map_point.y)

                self._fromll_results.append((x, y))

                i = self._current_fromll_index + 1
                n = len(self._pending_lonlats)

                self.get_logger().info(
                    f"[fromLL] converted {i}/{n}: x={x:.2f} y={y:.2f}"
                )

                self._current_fromll_index += 1
                self._current_fromll_future = None
                self._fromll_start_time = None

                self._send_next_fromll_request()

            except Exception as e:
                self.get_logger().error(f"[fromLL] conversion failed: {e}")
                self._reset_all()

            return

        if self._nav_active and self._route_map is not None:
            if not self.navigator.isTaskComplete():
                feedback = self.navigator.getFeedback()

                if feedback is not None:
                    self.get_logger().debug(f"[nav2] feedback: {feedback}")

                return

            result = self.navigator.getResult()

            self.get_logger().info(f"[nav2] result: {result}")

            self._nav_active = False
            result_str = str(result).lower()

            if "succeeded" in result_str:
                self._idx += 1
                self._send_next_goal()
                return

            if self.continue_on_abort:
                self.get_logger().warning(
                    "[nav2] goal failed, continuing because continue_on_abort=True"
                )
                self._idx += 1
                self._send_next_goal()
            else:
                self.get_logger().error(
                    "[nav2] goal failed, stopping because continue_on_abort=False"
                )

    def _send_next_goal(self):
        if self._route_map is None:
            return

        if self._nav_active:
            return

        if self._idx >= len(self._route_map):
            self.get_logger().info("[nav2] finished all goals")
            return

        x, y = self._route_map[self._idx]

        if self._idx < len(self._route_map) - 1:
            nx, ny = self._route_map[self._idx + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            try:
                rx, ry, _ = self._get_robot_pose()
                yaw = math.atan2(y - ry, x - rx)
            except Exception:
                yaw = 0.0

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.goal_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation = yaw_to_quat(yaw)

        self.get_logger().info(
            f"[nav2] goal {self._idx + 1}/{len(self._route_map)}: "
            f"x={x:.2f} y={y:.2f} yaw={yaw:.2f}"
        )

        self.navigator.goToPose(goal_pose)
        self._nav_active = True


# ============================================================
# Combined main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined GPS click planner, road perception, and road/nav fusion node."
    )

    # Original planner arguments
    parser.add_argument("--plot", action="store_true", help="Show live matplotlib plot")
    parser.add_argument(
        "--use-fromll",
        action="store_true",
        help="Use robot_localization /fromLL to convert lon/lat to map",
    )
    parser.add_argument(
        "--localized",
        action="store_true",
        help="Use current robot pose as START; only click GOAL",
    )
    parser.add_argument(
        "--fromll-service",
        default="/fromLL",
        help="Service name for FromLL",
    )
    parser.add_argument(
        "--toll-service",
        default="/toLL",
        help="Service name for ToLL",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=10.0,
        help="Densify spacing in meters",
    )
    parser.add_argument(
        "--graph-dist",
        type=float,
        default=1200.0,
        help="OSMnx graph distance in meters",
    )
    parser.add_argument(
        "--network-type",
        default="all",
        help="OSMnx network_type: all, walk, drive, bike",
    )
    parser.add_argument(
        "--clicked-topic",
        default="/clicked_point",
        help="Mapviz click topic",
    )
    parser.add_argument(
        "--require-frame",
        default="wgs84",
        help="Expected frame_id for clicked points",
    )
    parser.add_argument(
        "--goal-frame",
        default="map",
        help="Nav2 goal frame",
    )
    parser.add_argument(
        "--base-frame",
        default="base_link",
        help="Robot base frame",
    )
    parser.add_argument(
        "--continue-on-abort",
        action="store_true",
        help="If a goal fails, continue to next",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=400,
        help="Safety cap on number of goals. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-near",
        type=float,
        default=1.0,
        help="Skip initial goals closer than this distance to robot",
    )
    parser.add_argument(
        "--anchor-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Manual heading offset in degrees for anchor mode",
    )

    # New combined-node arguments
    parser.add_argument(
        "--enable-perception",
        action="store_true",
        help="Run RoadPerceptionNode in the same process",
    )
    parser.add_argument(
        "--enable-fusion",
        action="store_true",
        help="Run RoadNavFusionNode in the same process",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run planner, perception, and fusion together",
    )

    args = parser.parse_args()

    rclpy.init()

    nodes = []

    planner_node = ClickPlanner(
        enable_plot=args.plot,
        use_fromll=args.use_fromll,
        fromll_service=args.fromll_service,
        toll_service=args.toll_service,
        localized=args.localized,
        spacing_m=args.spacing,
        graph_dist_m=args.graph_dist,
        network_type=args.network_type,
        clicked_topic=args.clicked_topic,
        require_frame=args.require_frame,
        goal_frame=args.goal_frame,
        base_frame=args.base_frame,
        continue_on_abort=args.continue_on_abort,
        max_goals=args.max_goals,
        skip_near_m=args.skip_near,
        anchor_yaw_offset_deg=args.anchor_yaw_offset_deg,
    )
    nodes.append(planner_node)

    if args.enable_perception or args.run_all:
        perception_node = RoadPerceptionNode()
        nodes.append(perception_node)
    else:
        perception_node = None

    if args.enable_fusion or args.run_all:
        fusion_node = RoadNavFusionNode()
        nodes.append(fusion_node)
    else:
        fusion_node = None

    executor = MultiThreadedExecutor()

    for node in nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if fusion_node is not None:
            stop = Twist()
            fusion_node.cmd_pub.publish(stop)

        for node in nodes:
            executor.remove_node(node)
            node.destroy_node()

        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()