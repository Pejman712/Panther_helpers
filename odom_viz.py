#!/usr/bin/env python3
"""
odom_imu_plotter.py  (ROS 2 Humble)

- Subscribes to all nav_msgs/msg/Odometry topics and plots their X-Y paths.
- Subscribes to an IMU topic (default: /panther/imu/data) and plots yaw vs time.
"""

import math
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import matplotlib.pyplot as plt


class OdomImuPlotter(Node):
    def __init__(self):
        super().__init__('odom_imu_plotter')

        # Parameters
        self.declare_parameter('topic_scan_period', 2.0)
        self.declare_parameter('plot_update_period', 0.1)
        self.declare_parameter('max_points', 5000)
        self.declare_parameter('imu_topic', '/panther/imu/data')

        self.topic_scan_period = float(
            self.get_parameter('topic_scan_period').get_parameter_value().double_value
        )
        self.plot_update_period = float(
            self.get_parameter('plot_update_period').get_parameter_value().double_value
        )
        self.max_points = int(
            self.get_parameter('max_points').get_parameter_value().double_value
        )
        self.imu_topic = (
            self.get_parameter('imu_topic').get_parameter_value().string_value
        )

        # Data: odom topic_name -> {"x": [...], "y": [...]}
        self.tracks = defaultdict(lambda: {"x": [], "y": []})
        self.subscribers = {}

        # IMU data
        self.imu_yaw = []   # yaw in radians
        self.imu_time = []  # relative time (seconds)
        self.imu_sub = None
        self.start_time = time.time()

        # Setup matplotlib: 1 row, 2 columns (traj + yaw)
        plt.ion()
        self.fig, (self.ax_traj, self.ax_yaw) = plt.subplots(1, 2, figsize=(10, 5))
        self._setup_axes()

        # Timers
        self.topic_scan_timer = self.create_timer(
            self.topic_scan_period, self.scan_odom_topics
        )
        self.plot_update_timer = self.create_timer(
            self.plot_update_period, self.update_plot
        )

        # IMU subscription
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10
        )
        self.get_logger().info(
            f"Subscribing to IMU topic: {self.imu_topic}"
        )

        self.get_logger().info(
            f"odom_imu_plotter started (scan every {self.topic_scan_period:.1f} s)."
        )

    # ---------------- Helpers ---------------- #

    def _setup_axes(self):
        # Trajectory axes
        self.ax_traj.clear()
        self.ax_traj.set_xlabel("X [m]")
        self.ax_traj.set_ylabel("Y [m]")
        self.ax_traj.set_title("Odometry X-Y Tracks")
        self.ax_traj.grid(True)
        self.ax_traj.set_aspect("equal", adjustable="box")

        # IMU yaw axes
        self.ax_yaw.clear()
        self.ax_yaw.set_xlabel("Time [s]")
        self.ax_yaw.set_ylabel("Yaw [rad]")
        self.ax_yaw.set_title(f"IMU Yaw ({self.imu_topic})")
        self.ax_yaw.grid(True)

    # ---------------- Topic discovery ---------------- #

    def scan_odom_topics(self):
        """
        Discover all topics of type nav_msgs/msg/Odometry and subscribe to any new ones.
        """
        topics_and_types = self.get_topic_names_and_types()
        odom_topics = [
            (name, types)
            for name, types in topics_and_types
            if 'nav_msgs/msg/Odometry' in types
        ]

        for topic_name, _ in odom_topics:
            if topic_name not in self.subscribers:
                self.get_logger().info(f"Subscribing to odometry topic: {topic_name}")
                sub = self.create_subscription(
                    Odometry,
                    topic_name,
                    lambda msg, t=topic_name: self.odom_callback(msg, t),
                    10
                )
                self.subscribers[topic_name] = sub

        # (Optional) unsubscribe from vanished topics could be added here.

    # ---------------- Callbacks ---------------- #

    def odom_callback(self, msg: Odometry, topic_name: str):
        """
        Callback for each Odometry message.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        track = self.tracks[topic_name]
        track["x"].append(x)
        track["y"].append(y)

        # Limit history
        if len(track["x"]) > self.max_points:
            track["x"] = track["x"][-self.max_points:]
            track["y"] = track["y"][-self.max_points:]

    def imu_callback(self, msg: Imu):
        """
        Callback for IMU messages: compute yaw from quaternion and store it.
        """
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        # Quaternion -> yaw (Z axis) using standard formula
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        t = time.time() - self.start_time

        self.imu_yaw.append(yaw)
        self.imu_time.append(t)

        # Limit history
        if len(self.imu_yaw) > self.max_points:
            self.imu_yaw = self.imu_yaw[-self.max_points:]
            self.imu_time = self.imu_time[-self.max_points:]

    # ---------------- Plotting ---------------- #

    def update_plot(self):
        """
        Timer callback to refresh the Matplotlib plots.
        """
        self._setup_axes()

        # --- Trajectories ---
        for topic, track in self.tracks.items():
            if len(track["x"]) < 2:
                continue

            self.ax_traj.plot(track["x"], track["y"], label=topic)
            # mark current position
            self.ax_traj.scatter(track["x"][-1], track["y"][-1])

        if self.tracks:
            self.ax_traj.legend(loc="best")

        # --- IMU yaw ---
        if len(self.imu_time) > 1:
            self.ax_yaw.plot(self.imu_time, self.imu_yaw)

        # Redraw non-blocking
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = OdomImuPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down odom_imu_plotter.")
        node.destroy_node()
        rclpy.shutdown()

        try:
            plt.ioff()
            plt.show()
        except Exception:
            pass


if __name__ == '__main__':
    main()

