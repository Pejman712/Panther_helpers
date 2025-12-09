#!/usr/bin/env python3
"""

- Subscribes to all nav_msgs/msg/Odometry topics --->>> plots their X-Y paths.
- Subscribes to all sensor_msgs/msg/Imu topics and plots: --->>> plotts: 
    * Orientation: roll, pitch, yaw 
    * Angular velocity: wx, wy, wz 
    * Linear acceleration: ax, ay, az 
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
        #self.declare_parameter('imu_topic', '/panther/imu/data')

        self.topic_scan_period = float(
            self.get_parameter('topic_scan_period').get_parameter_value().double_value
        )
        self.plot_update_period = float(
            self.get_parameter('plot_update_period').get_parameter_value().double_value
        )
        self.max_points = int(
            self.get_parameter('max_points').get_parameter_value().double_value
        )
        #self.imu_topic = (
        #    self.get_parameter('imu_topic').get_parameter_value().string_value
        #)

        # Data: odom topic_name -> {"x": [...], "y": [...]}
        self.tracks = defaultdict(lambda: {"x": [], "y": []})
        self.odom_subscribers = {}

        # IMU data: imu topic_name -> dict with time, orientation, gyro, accel
        self.imu_data = defaultdict(lambda: {
            "time": [],
            "roll": [], "pitch": [], "yaw": [],
            "wx": [], "wy": [], "wz": [],
            "ax": [], "ay": [], "az": [],
        })
        self.imu_subscribers = {}

        self.start_time = time.time()
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        (self.ax_traj, self.ax_orientation), (self.ax_gyro, self.ax_accel) = self.axes
        self._setup_axes()

        # Timers
        self.topic_scan_timer = self.create_timer(
            self.topic_scan_period, self.scan_topics
        )
        self.plot_update_timer = self.create_timer(
            self.plot_update_period, self.update_plot
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

        # Orientation axes
        self.ax_orientation.clear()
        self.ax_orientation.set_xlabel("Time [s]")
        self.ax_orientation.set_ylabel("Angle [rad]")
        self.ax_orientation.set_title("IMU Orientation (roll/pitch/yaw) - all IMU topics")
        self.ax_orientation.grid(True)

        # Angular velocity axes
        self.ax_gyro.clear()
        self.ax_gyro.set_xlabel("Time [s]")
        self.ax_gyro.set_ylabel("Angular velocity [rad/s]")
        self.ax_gyro.set_title("IMU Angular Velocity (wx, wy, wz)")
        self.ax_gyro.grid(True)

        # Linear acceleration axes
        self.ax_accel.clear()
        self.ax_accel.set_xlabel("Time [s]")
        self.ax_accel.set_ylabel("Linear acceleration [m/s²]")
        self.ax_accel.set_title("IMU Linear Acceleration (ax, ay, az)")
        self.ax_accel.grid(True)

    # ---------------- Topic discovery ---------------- #

    def scan_topics(self):
        """
        Discover all topics of type nav_msgs/msg/Odometry and sensor_msgs/msg/Imu
        and subscribe to any new ones.
        """
        topics_and_types = self.get_topic_names_and_types()

        # Odometry topics
        odom_topics = [
            (name, types)
            for name, types in topics_and_types
            if 'nav_msgs/msg/Odometry' in types
        ]

        for topic_name, _ in odom_topics:
            if topic_name not in self.odom_subscribers:
                self.get_logger().info(f"Subscribing to odometry topic: {topic_name}")
                sub = self.create_subscription(
                    Odometry,
                    topic_name,
                    lambda msg, t=topic_name: self.odom_callback(msg, t),
                    10
                )
                self.odom_subscribers[topic_name] = sub

        # IMU topics
        imu_topics = [
            (name, types)
            for name, types in topics_and_types
            if 'sensor_msgs/msg/Imu' in types
        ]

        for topic_name, _ in imu_topics:
            if topic_name not in self.imu_subscribers:
                self.get_logger().info(f"Subscribing to IMU topic: {topic_name}")
                sub = self.create_subscription(
                    Imu,
                    topic_name,
                    lambda msg, t=topic_name: self.imu_callback(msg, t),
                    10
                )
                self.imu_subscribers[topic_name] = sub

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

    def imu_callback(self, msg: Imu, topic_name: str):
        """
        Callback for IMU messages: compute roll, pitch, yaw from quaternion
        and store angular velocity + linear acceleration for each IMU topic.
        """
        data = self.imu_data[topic_name]

        # Orientation quaternion
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        # Quaternion -> roll, pitch, yaw Euclearion/cartasion (Z-Y-X / yaw-pitch-roll convention)

        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2.0, sinp)  # use 90 deg if out of range
        else:
            pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Angular velocity
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z

        # Linear acceleration
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        # Time
        t = time.time() - self.start_time
        data["time"].append(t)

        # Store orientation
        data["roll"].append(roll)
        data["pitch"].append(pitch)
        data["yaw"].append(yaw)

        # Store angular velocity
        data["wx"].append(wx)
        data["wy"].append(wy)
        data["wz"].append(wz)

        # Store linear acceleration
        data["ax"].append(ax)
        data["ay"].append(ay)
        data["az"].append(az)

        # Limit history for this IMU topic
        if len(data["time"]) > self.max_points:
            for key in data.keys():
                data[key] = data[key][-self.max_points:]

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

        # --- IMU plots ---
        any_imu_data = any(len(d["time"]) > 1 for d in self.imu_data.values())

        if any_imu_data:
            # Orientation
            for topic, d in self.imu_data.items():
                if len(d["time"]) < 2:
                    continue
                self.ax_orientation.plot(d["time"], d["roll"], label=f"{topic} roll")
                self.ax_orientation.plot(d["time"], d["pitch"], label=f"{topic} pitch")
                self.ax_orientation.plot(d["time"], d["yaw"], label=f"{topic} yaw")
            self.ax_orientation.legend(loc="best")

            # Angular velocity
            for topic, d in self.imu_data.items():
                if len(d["time"]) < 2:
                    continue
                self.ax_gyro.plot(d["time"], d["wx"], label=f"{topic} wx")
                self.ax_gyro.plot(d["time"], d["wy"], label=f"{topic} wy")
                self.ax_gyro.plot(d["time"], d["wz"], label=f"{topic} wz")
            self.ax_gyro.legend(loc="best")

            # Linear acceleration
            for topic, d in self.imu_data.items():
                if len(d["time"]) < 2:
                    continue
                self.ax_accel.plot(d["time"], d["ax"], label=f"{topic} ax")
                self.ax_accel.plot(d["time"], d["ay"], label=f"{topic} ay")
                self.ax_accel.plot(d["time"], d["az"], label=f"{topic} az")
            self.ax_accel.legend(loc="best")

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

