#!/usr/bin/env python3
"""
ROS2 Navigation Debugger - terminal output only

- No Tkinter UI
- No tf2_ros.TransformListener
- Manual TF feed from /tf and /tf_static
- Fixed /tf_static QoS (TRANSIENT_LOCAL)
- Fixed TF buffer feed:
    * dynamic TF -> set_transform(...)
    * static TF  -> set_transform_static(...)
- Periodically prints a color-coded text report to the terminal

Legend:
- RED    = failure / missing
- YELLOW = warning / exists but not publishing
- GREEN  = OK
- GRAY   = optional and inactive
"""

import os
import sys
import time
import math
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

import tf2_ros

try:
    from tf2_msgs.msg import TFMessage
except Exception as e:
    raise RuntimeError(
        "tf2_msgs is not available. Install:\n"
        "  sudo apt install ros-humble-tf2-msgs"
    ) from e


# Try common message types for echo/subscription
MSG_TYPE_IMPORTS = [
    ("nav_msgs/msg/Odometry", "nav_msgs.msg", "Odometry"),
    ("geometry_msgs/msg/Twist", "geometry_msgs.msg", "Twist"),
    ("sensor_msgs/msg/Imu", "sensor_msgs.msg", "Imu"),
    ("sensor_msgs/msg/NavSatFix", "sensor_msgs.msg", "NavSatFix"),
    ("nav_msgs/msg/Path", "nav_msgs.msg", "Path"),
    ("rosgraph_msgs/msg/Clock", "rosgraph_msgs.msg", "Clock"),
]
IMPORTED_MSGS = {}
for pretty, mod, cls in MSG_TYPE_IMPORTS:
    try:
        m = __import__(mod, fromlist=[cls])
        IMPORTED_MSGS[pretty] = getattr(m, cls)
    except Exception:
        pass


@dataclass
class TopicStatus:
    name: str
    type_strs: List[str] = field(default_factory=list)
    exists: bool = False
    last_msg_time: Optional[float] = None
    msg_count: int = 0
    last_preview: str = ""


@dataclass
class TfCheck:
    parent: str
    child: str
    ok: bool
    note: str = ""


@dataclass
class ActionCheck:
    name: str
    ok: bool
    note: str = ""


@dataclass
class HealthReport:
    timestamp: float
    namespace_hint: str = ""
    use_sim_time: Optional[bool] = None
    required_topics: Dict[str, TopicStatus] = field(default_factory=dict)
    optional_topics: Dict[str, TopicStatus] = field(default_factory=dict)
    tf_checks: List[TfCheck] = field(default_factory=list)
    action_checks: List[ActionCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tf_connected: bool = False
    tf_note: str = ""


DEFAULT_EXPECTED = {
    "frames": {
        "map": "map",
        "odom": "odom",
        "base_link": "base_link",
        "base_footprint": "base_footprint",
    },
    "topics_required": {
        "/tf": None,
        "/tf_static": None,
        "/odom": "nav_msgs/msg/Odometry",
        "/cmd_vel": "geometry_msgs/msg/Twist",
    },
    "topics_optional": {
        "/clock": "rosgraph_msgs/msg/Clock",
        "/imu/data": "sensor_msgs/msg/Imu",
        "/gps/fix": "sensor_msgs/msg/NavSatFix",
        "/odometry/filtered": "nav_msgs/msg/Odometry",
        "/odometry/global": "nav_msgs/msg/Odometry",
        "/odometry/local": "nav_msgs/msg/Odometry",
        "/plan": "nav_msgs/msg/Path",
    },
    "actions": [
        "/navigate_to_pose",
        "/navigate_through_poses",
        "/follow_path",
    ],
}


def _now() -> float:
    return time.time()


def _fmt_age(last_time: Optional[float]) -> str:
    if last_time is None:
        return "never"
    age = _now() - last_time
    if age < 0:
        return "0.0s"
    if age < 10:
        return f"{age:.2f}s"
    return f"{age:.1f}s"


def _topic_basename(t: str) -> str:
    return t.strip().rstrip("/")


def _action_service_endpoints(action_name: str) -> List[str]:
    base = _topic_basename(action_name)
    return [
        f"{base}/_action/send_goal",
        f"{base}/_action/get_result",
        f"{base}/_action/cancel_goal",
    ]


# ANSI colors
RST = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GRN = "\033[32m"
YEL = "\033[33m"
CYN = "\033[36m"
GRY = "\033[90m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RST}"


def clear_terminal():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


class RosInspectorNode(Node):
    def __init__(self, expected: dict):
        super().__init__("ros_nav_debugger_terminal")
        self.expected = expected

        self._qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self._qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self._qos_tf_static = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self._topic_status: Dict[str, TopicStatus] = {}
        self._my_subscriptions = {}
        self._lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_sub = self.create_subscription(
            TFMessage, "/tf", self._on_tf, self._qos_best_effort
        )
        self.tf_static_sub = self.create_subscription(
            TFMessage, "/tf_static", self._on_tf_static, self._qos_tf_static
        )

        self._last_tf_time: Optional[float] = None
        self._last_tf_static_time: Optional[float] = None

        self._ensure_subscriptions()

    def _mark_topic_seen(self, topic_name: str, preview: str = ""):
        with self._lock:
            st = self._topic_status.get(topic_name)
            if st is None:
                st = TopicStatus(name=topic_name)
                self._topic_status[topic_name] = st
            st.msg_count += 1
            st.last_msg_time = _now()
            if preview:
                st.last_preview = preview

    def _on_tf(self, msg: TFMessage):
        try:
            for t in msg.transforms:
                self.tf_buffer.set_transform(t, "ros_nav_debugger")
            self._last_tf_time = _now()
            self._mark_topic_seen("/tf", f"{len(msg.transforms)} transform(s)")
        except Exception as e:
            self.get_logger().error(f"/tf callback error: {e}")

    def _on_tf_static(self, msg: TFMessage):
        try:
            for t in msg.transforms:
                self.tf_buffer.set_transform_static(t, "ros_nav_debugger")
            self._last_tf_static_time = _now()
            self._mark_topic_seen("/tf_static", f"{len(msg.transforms)} static transform(s)")
        except Exception as e:
            self.get_logger().error(f"/tf_static callback error: {e}")

    def _tf_connected(self, stale_after: float = 2.0) -> Tuple[bool, str]:
        now = _now()
        last_times = [
            t for t in (self._last_tf_time, self._last_tf_static_time) if t is not None
        ]
        if not last_times:
            return False, "No TF received yet"

        latest = max(last_times)
        age = now - latest
        if age <= stale_after:
            return True, f"Last TF {age:.2f}s ago"
        return False, f"TF stale ({age:.2f}s ago)"

    def _ensure_subscriptions(self):
        for group_key in ("topics_required", "topics_optional"):
            topics = self.expected.get(group_key, {})
            for tname, type_pretty in topics.items():
                if type_pretty is None:
                    continue
                if tname in self._my_subscriptions:
                    continue
                msg_cls = IMPORTED_MSGS.get(type_pretty)
                if msg_cls is None:
                    continue
                try:
                    sub = self.create_subscription(
                        msg_cls,
                        tname,
                        lambda msg, tn=tname: self._on_msg(tn, msg),
                        self._qos_best_effort,
                    )
                    self._my_subscriptions[tname] = sub
                except Exception as e:
                    self.get_logger().warning(f"Failed subscription for {tname}: {e}")

    def _on_msg(self, topic_name: str, msg):
        with self._lock:
            st = self._topic_status.get(topic_name)
            if st is None:
                st = TopicStatus(name=topic_name)
                self._topic_status[topic_name] = st
            st.msg_count += 1
            st.last_msg_time = _now()
            st.last_preview = self._preview_message(msg)

    @staticmethod
    def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw * 180.0 / math.pi

    def _preview_message(self, msg) -> str:
        try:
            cname = msg.__class__.__name__
            if cname == "Odometry":
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                v = msg.twist.twist.linear
                wz = msg.twist.twist.angular.z
                yaw = self._quat_to_yaw(o.x, o.y, o.z, o.w)
                return f"pos=({p.x:.2f},{p.y:.2f}) yaw={yaw:.1f}° v=({v.x:.2f},{v.y:.2f}) wz={wz:.2f}"
            if cname == "Twist":
                return f"lin=({msg.linear.x:.2f},{msg.linear.y:.2f}) ang_z={msg.angular.z:.2f}"
            if cname == "Imu":
                o = msg.orientation
                yaw = self._quat_to_yaw(o.x, o.y, o.z, o.w)
                return f"yaw={yaw:.1f}° ang_vel_z={msg.angular_velocity.z:.3f}"
            if cname == "NavSatFix":
                return f"lat={msg.latitude:.6f} lon={msg.longitude:.6f} alt={msg.altitude:.2f}"
            if cname == "Path":
                return f"frame={msg.header.frame_id} n={len(msg.poses)}"
            if cname == "Clock":
                return f"clock={msg.clock.sec}.{msg.clock.nanosec:09d}"
        except Exception:
            pass
        s = str(msg)
        return s[:160] + (" …" if len(s) > 160 else "")

    def _check_actions(self) -> List[ActionCheck]:
        services = self.get_service_names_and_types()
        service_names = {s[0] for s in services}
        out: List[ActionCheck] = []
        for act in self.expected.get("actions", []):
            endpoints = _action_service_endpoints(act)
            missing = [e for e in endpoints if e not in service_names]
            if missing:
                out.append(ActionCheck(act, False, "Missing: " + ", ".join(missing)))
            else:
                out.append(ActionCheck(act, True, "OK"))
        return out

    def _check_tf(self) -> List[TfCheck]:
        checks: List[TfCheck] = []
        if self._last_tf_time is None and self._last_tf_static_time is None:
            checks.append(TfCheck("(tf feed)", "", False, "No /tf or /tf_static received yet"))
            return checks

        frames = self.expected.get("frames", {})
        pairs = []
        if frames.get("map") and frames.get("odom"):
            pairs.append((frames["map"], frames["odom"]))
        if frames.get("odom") and frames.get("base_link"):
            pairs.append((frames["odom"], frames["base_link"]))
        if frames.get("map") and frames.get("base_link"):
            pairs.append((frames["map"], frames["base_link"]))
        if frames.get("odom") and frames.get("base_footprint"):
            pairs.append((frames["odom"], frames["base_footprint"]))

        for parent, child in pairs:
            try:
                ok = self.tf_buffer.can_transform(
                    parent,
                    child,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
                if not ok:
                    checks.append(TfCheck(parent, child, False, "No transform"))
                    continue
                t = self.tf_buffer.lookup_transform(
                    parent,
                    child,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
                tr = t.transform.translation
                rot = t.transform.rotation
                yaw_deg = self._quat_to_yaw(rot.x, rot.y, rot.z, rot.w)
                checks.append(
                    TfCheck(
                        parent,
                        child,
                        True,
                        f"xyz=({tr.x:.2f},{tr.y:.2f},{tr.z:.2f}) yaw={yaw_deg:.1f}°",
                    )
                )
            except Exception as e:
                checks.append(TfCheck(parent, child, False, f"Error: {e}"))
        return checks

    def _topic_presence(
        self,
        all_topics: List[Tuple[str, List[str]]],
        expected_map: Dict[str, Optional[str]],
    ) -> Dict[str, TopicStatus]:
        topic_dict = {name: types for name, types in all_topics}
        out: Dict[str, TopicStatus] = {}
        for tname, expected_type in expected_map.items():
            st = TopicStatus(name=tname)
            if tname in topic_dict:
                st.exists = True
                st.type_strs = topic_dict[tname]
            with self._lock:
                if tname in self._topic_status:
                    s = self._topic_status[tname]
                    st.last_msg_time = s.last_msg_time
                    st.msg_count = s.msg_count
                    st.last_preview = s.last_preview
            if expected_type and st.exists and expected_type not in st.type_strs:
                st.last_preview = (
                    (st.last_preview + " | ") if st.last_preview else ""
                ) + f"TYPE MISMATCH (expected {expected_type})"
            out[tname] = st
        return out

    def build_report(self) -> HealthReport:
        all_topics = self.get_topic_names_and_types()
        required_topics = self._topic_presence(
            all_topics, self.expected.get("topics_required", {})
        )
        optional_topics = self._topic_presence(
            all_topics, self.expected.get("topics_optional", {})
        )
        tf_checks = self._check_tf()
        action_checks = self._check_actions()
        tf_connected, tf_note = self._tf_connected()

        use_sim_time = None
        try:
            if not self.has_parameter("use_sim_time"):
                self.declare_parameter("use_sim_time", False)
            use_sim_time = bool(self.get_parameter("use_sim_time").value)
        except Exception:
            use_sim_time = None

        warnings = []
        if use_sim_time is True:
            clk = optional_topics.get("/clock")
            if clk and (not clk.exists or clk.msg_count == 0):
                warnings.append("use_sim_time=TRUE but /clock missing or not publishing.")
        if self._last_tf_time is None:
            warnings.append("No dynamic /tf received yet.")
        if self._last_tf_static_time is None:
            warnings.append("No /tf_static received yet (robot_state_publisher?).")
        if not tf_connected:
            warnings.append(f"TF connection issue: {tf_note}")

        return HealthReport(
            timestamp=_now(),
            use_sim_time=use_sim_time,
            required_topics=required_topics,
            optional_topics=optional_topics,
            tf_checks=tf_checks,
            action_checks=action_checks,
            warnings=warnings,
            tf_connected=tf_connected,
            tf_note=tf_note,
        )


def topic_state(name: str, st: TopicStatus, required: bool) -> Tuple[str, str]:
    if required:
        if not st.exists:
            return "BAD", RED
        if name not in ("/tf", "/tf_static") and st.msg_count == 0:
            return "WARN", YEL
        return "OK", GRN

    if st.exists and st.msg_count > 0:
        return "OK", GRN
    if st.exists and name not in ("/tf", "/tf_static") and st.msg_count == 0:
        return "WARN", YEL
    return "OPT", GRY


def render_report(r: HealthReport):
    clear_terminal()

    ok_req = all(st.exists for st in r.required_topics.values())
    ok_tf = all(c.ok for c in r.tf_checks if c.parent != "(tf feed)")
    ok_act = all(a.ok for a in r.action_checks)

    req_s = colorize("REQ OK", GRN) if ok_req else colorize("REQ FAIL", RED)
    tf_link_s = (
        colorize(f"TF LINK OK ({r.tf_note})", GRN)
        if r.tf_connected
        else colorize(f"TF LINK FAIL ({r.tf_note})", RED)
    )
    tf_s = colorize("TF OK", GRN) if ok_tf else colorize("TF FAIL", RED)
    act_s = colorize("ACT OK", GRN) if ok_act else colorize("ACT FAIL", RED)

    print(f"{BOLD}ROS2 Navigation Debugger (terminal){RST}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r.timestamp))}")
    print(f"Status: {req_s} | {tf_link_s} | {tf_s} | {act_s}")
    print(f"use_sim_time: {r.use_sim_time}")
    print()

    print(f"{BOLD}{CYN}Topics{RST}")
    print("-" * 120)
    print(f"{'STATE':<8} {'KIND':<5} {'TOPIC':<24} {'EXISTS':<6} {'COUNT':<8} {'AGE':<10} {'TYPES / PREVIEW'}")
    print("-" * 120)

    for required, topic_map in ((True, r.required_topics), (False, r.optional_topics)):
        for name, st in topic_map.items():
            state_txt, color = topic_state(name, st, required)
            kind = "REQ" if required else "OPT"
            exists_txt = "YES" if st.exists else "NO"
            types_txt = ", ".join(st.type_strs) if st.type_strs else "-"
            preview = st.last_preview if st.last_preview else "-"
            tail = f"{types_txt} | {preview}"
            print(
                f"{colorize(state_txt, color):<17} {kind:<5} {name:<24} {exists_txt:<6} "
                f"{st.msg_count:<8} {_fmt_age(st.last_msg_time):<10} {tail}"
            )

    print()
    print(f"{BOLD}{CYN}TF checks{RST}")
    print("-" * 120)
    for c in r.tf_checks:
        status = colorize("OK", GRN) if c.ok else colorize("FAIL", RED if c.parent != "(tf feed)" else YEL)
        if c.parent == "(tf feed)":
            label = "(tf feed)"
        else:
            label = f"{c.parent} -> {c.child}"
        print(f"{status:<12} {label:<30} {c.note}")

    print()
    print(f"{BOLD}{CYN}Action checks{RST}")
    print("-" * 120)
    for a in r.action_checks:
        status = colorize("OK", GRN) if a.ok else colorize("FAIL", RED)
        print(f"{status:<12} {a.name:<30} {a.note}")

    print()
    print(f"{BOLD}{CYN}Warnings / hints{RST}")
    print("-" * 120)
    if r.warnings:
        for w in r.warnings:
            print(colorize(f"- {w}", YEL))
    else:
        print(colorize("No warnings.", GRN))

    print()
    print(colorize("Press Ctrl+C to exit.", GRY))


def main():
    expected = dict(DEFAULT_EXPECTED)

    bl = os.environ.get("NAVDBG_BASE_LINK")
    if bl:
        expected["frames"] = dict(expected["frames"])
        expected["frames"]["base_link"] = bl

    try:
        rclpy.init(args=None)
        node = RosInspectorNode(expected)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        last_render = 0.0
        render_period = 0.5

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
            now = _now()
            if now - last_render >= render_period:
                try:
                    report = node.build_report()
                    render_report(report)
                except Exception:
                    clear_terminal()
                    print(colorize("ERROR while building/rendering report:", RED))
                    print(traceback.format_exc())
                last_render = now

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        print(colorize("Fatal error:", RED))
        print(traceback.format_exc())
    finally:
        try:
            if "executor" in locals() and "node" in locals():
                executor.remove_node(node)
                node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
