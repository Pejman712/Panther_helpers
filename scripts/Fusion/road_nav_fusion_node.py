#!/usr/bin/env python3

import math
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from road_nav_interfaces.msg import RoadObservation


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


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
        self.min_confidence_for_blend = float(self.get_parameter("min_confidence_for_blend").value)

        self.kp_road_lateral = float(self.get_parameter("kp_road_lateral").value)
        self.kp_road_heading = float(self.get_parameter("kp_road_heading").value)
        self.max_road_correction = float(self.get_parameter("max_road_correction").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self.turn_lookahead_m = float(self.get_parameter("turn_lookahead_m").value)
        self.sharp_turn_angle_deg = float(self.get_parameter("sharp_turn_angle_deg").value)
        self.sharp_turn_blend_scale = float(self.get_parameter("sharp_turn_blend_scale").value)

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

        self.control_timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.get_logger().info(f"Listening nav cmd: {self.nav_cmd_topic}")
        self.get_logger().info(f"Listening road obs: {self.road_obs_topic}")
        self.get_logger().info(f"Listening path:     {self.path_topic}")
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
        tf = self.tf_buffer.lookup_transform(frame_id, self.base_frame, rclpy.time.Time())
        x = tf.transform.translation.x
        y = tf.transform.translation.y

        q = tf.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return x, y, yaw

    def path_to_numpy(self, path: Path):
        if path is None or len(path.poses) < 2:
            return None
        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in path.poses],
            dtype=np.float32,
        )
        return pts

    def advance_index_by_distance(self, pts: np.ndarray, start_idx: int, dist_m: float) -> int:
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

        h0 = math.atan2(pts[i1, 1] - pts[i0, 1], pts[i1, 0] - pts[i0, 0])
        h1 = math.atan2(pts[i3, 1] - pts[i2, 1], pts[i3, 0] - pts[i2, 0])

        return abs(normalize_angle(h1 - h0))

    def compute_road_correction(self, obs: RoadObservation) -> float:
        # Road target to the right should make angular.z more negative.
        correction = -(
            self.kp_road_lateral * float(obs.lateral_error_norm)
            + self.kp_road_heading * float(obs.heading_error_rad)
        )
        correction = float(
            np.clip(correction, -self.max_road_correction, self.max_road_correction)
        )
        return correction

    def control_loop(self):
        now = time.monotonic()

        if self.latest_nav_cmd is None or (now - self.last_nav_cmd_t) > self.nav_cmd_timeout_sec:
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
                alpha = min(self.max_blend, self.road_blend_gain * float(obs.confidence))

                turn_angle = self.compute_upcoming_turn_angle()
                sharp_turn_rad = math.radians(self.sharp_turn_angle_deg)
                if turn_angle > sharp_turn_rad:
                    alpha *= self.sharp_turn_blend_scale

                # Reduce vision influence if it strongly fights a large nav turn command.
                if abs(self.latest_nav_cmd.angular.z) > 0.35 and (self.latest_nav_cmd.angular.z * road_corr) < 0.0:
                    alpha *= 0.5

                fused.angular.z += alpha * road_corr

                # Optional speed reduction when confidence is modest or turn ahead is sharp.
                speed_scale = 1.0
                speed_scale *= max(0.70, 0.70 + 0.30 * float(obs.confidence))
                if turn_angle > sharp_turn_rad:
                    speed_scale *= 0.80
                fused.linear.x *= speed_scale

        fused.angular.z = float(
            np.clip(fused.angular.z, -self.max_angular_speed, self.max_angular_speed)
        )

        self.cmd_pub.publish(fused)

        if (now - self.last_debug_t) > 1.0:
            self.last_debug_t = now
            self.get_logger().info(
                f"nav_w={self.latest_nav_cmd.angular.z:+.3f} "
                f"road_corr={road_corr:+.3f} alpha={alpha:.2f} "
                f"turn_deg={math.degrees(turn_angle):.1f} "
                f"fused_w={fused.angular.z:+.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RoadNavFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()