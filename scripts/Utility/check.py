#!/usr/bin/env python3
"""
ROS2 Navigation Debugger - terminal output only

Improvements in this version
- Topic aliases for real robot stacks
- QoS chosen from live publishers instead of hardcoded BEST_EFFORT everywhere
- TF checks adapted for panther/base_link + Fixposition stack
- Optional static TF check: vrtk_link -> panther/base_link
- Better action detection with optional namespace prefix
- Better type-mismatch / publisher diagnostics
- Safer subscriptions for /tf and /tf_static

Environment overrides
- NAVDBG_NAMESPACE=/robot_ns
- NAVDBG_MAP_FRAME=map
- NAVDBG_ODOM_FRAME=odom
- NAVDBG_BASE_LINK=panther/base_link
- NAVDBG_BASE_FOOTPRINT=panther/base_footprint
- NAVDBG_VRTK_FRAME=vrtk_link
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
from rclpy.topic_endpoint_info import TopicEndpointTypeEnum

import tf2_ros

try:
    from tf2_msgs.msg import TFMessage
except Exception as e:
    raise RuntimeError(
        "tf2_msgs is not available. Install:\n"
        "  sudo apt install ros-humble-tf2-msgs"
    ) from e


MSG_TYPE_IMPORTS = [
    ("nav_msgs/msg/Odometry", "nav_msgs.msg", "Odometry"),
    ("geometry_msgs/msg/Twist", "geometry_msgs.msg", "Twist"),
    ("sensor_msgs/msg/Imu", "sensor_msgs.msg", "Imu"),
    ("sensor_msgs/msg/NavSatFix", "sensor_msgs.msg", "NavSatFix"),
    ("sensor_msgs/msg/LaserScan", "sensor_msgs.msg", "LaserScan"),
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
    label: str
    actual_name: Optional[str] = None
    expected_types: List[str] = field(default_factory=list)
    type_strs: List[str] = field(default_factory=list)
    exists: bool = False
    matched_alias: Optional[str] = None
    last_msg_time: Optional[float] = None
    msg_count: int = 0
    last_preview: str = ""
    publisher_count: int = 0
    subscriber_count: int = 0
    qos_note: str = ""


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
    use_sim_time: Optional[bool] = None
    required_topics: Dict[str, TopicStatus] = field(default_factory=dict)
    optional_topics: Dict[str, TopicStatus] = field(default_factory=dict)
    tf_checks: List[TfCheck] = field(default_factory=list)
    action_checks: List[ActionCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tf_connected: bool = False
    tf_note: str = ""


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


def _join_ns(ns: str, name: str) -> str:
    if not ns:
        return name
    ns = ns.rstrip("/")
    if not name.startswith("/"):
        name = "/" + name
    return ns + name


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


DEFAULT_EXPECTED = {
    "frames": {
        "map": os.environ.get("NAVDBG_MAP_FRAME", "map"),
        "odom": os.environ.get("NAVDBG_ODOM_FRAME", "odom"),
        "base_link": os.environ.get("NAVDBG_BASE_LINK", "panther/base_link"),
        "base_footprint": os.environ.get("NAVDBG_BASE_FOOTPRINT", "panther/base_footprint"),
        "vrtk_link": os.environ.get("NAVDBG_VRTK_FRAME", "vrtk_link"),
    },
    # label -> {"aliases": [...], "types": [...]}
    "topics_required": {
        "tf": {"aliases": ["/tf"], "types": []},
        "tf_static": {"aliases": ["/tf_static"], "types": []},
        "odom": {
            "aliases": ["/fixposition/odometry_enu", "/odometry/filtered", "/odom"],
            "types": ["nav_msgs/msg/Odometry"],
        },
        "cmd_vel": {
            "aliases": ["/panther/cmd_vel", "/cmd_vel"],
            "types": ["geometry_msgs/msg/Twist"],
        },
    },
    "topics_optional": {
        "clock": {
            "aliases": ["/clock"],
            "types": ["rosgraph_msgs/msg/Clock"],
        },
        "imu": {
            "aliases": ["/fixposition/poiimu", "/imu/data"],
            "types": ["sensor_msgs/msg/Imu"],
        },
        "gps_fix": {
            "aliases": ["/fixposition/odometry_llh", "/gps/fix"],
            "types": ["sensor_msgs/msg/NavSatFix"],
        },
        "odom_gps": {
            "aliases": ["/odometry/gps"],
            "types": ["nav_msgs/msg/Odometry"],
        },
        "plan": {
            "aliases": ["/plan", "/planner_server/plan"],
            "types": ["nav_msgs/msg/Path"],
        },
        "scan": {
            "aliases": ["/panther/velodyne/scan", "/scan"],
            "types": ["sensor_msgs/msg/LaserScan"],
        },
        "cmd_vel_nav": {
            "aliases": ["/cmd_vel_nav"],
            "types": ["geometry_msgs/msg/Twist"],
        },
    },
    "actions": [
        "/navigate_to_pose",
        "/navigate_through_poses",
        "/follow_path",
    ],
}


class RosInspectorNode(Node):
    def __init__(self, expected: dict):
        super().__init__("ros_nav_debugger_terminal")
        self.expected = expected
        self.namespace_hint = os.environ.get("NAVDBG_NAMESPACE", "").strip()

        self._qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self._qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self._qos_tf_static = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
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
        self._last_topic_refresh = 0.0
        self._topic_refresh_period = 1.0

        self._ensure_subscriptions(force=True)

    def _mark_topic_seen(self, label: str, preview: str = ""):
        with self._lock:
            st = self._topic_status.get(label)
            if st is None:
                st = TopicStatus(label=label)
                self._topic_status[label] = st
            st.msg_count += 1
            st.last_msg_time = _now()
            if preview:
                st.last_preview = preview

    def _on_tf(self, msg: TFMessage):
        try:
            for t in msg.transforms:
                self.tf_buffer.set_transform(t, "ros_nav_debugger")
            self._last_tf_time = _now()
            self._mark_topic_seen("tf", f"{len(msg.transforms)} transform(s)")
        except Exception as e:
            self.get_logger().error(f"/tf callback error: {e}")

    def _on_tf_static(self, msg: TFMessage):
        try:
            for t in msg.transforms:
                self.tf_buffer.set_transform_static(t, "ros_nav_debugger")
            self._last_tf_static_time = _now()
            self._mark_topic_seen("tf_static", f"{len(msg.transforms)} static transform(s)")
        except Exception as e:
            self.get_logger().error(f"/tf_static callback error: {e}")

    def _tf_connected(self, stale_after: float = 2.0) -> Tuple[bool, str]:
        now = _now()
        last_times = [t for t in (self._last_tf_time, self._last_tf_static_time) if t is not None]
        if not last_times:
            return False, "No TF received yet"
        latest = max(last_times)
        age = now - latest
        if age <= stale_after:
            return True, f"Last TF {age:.2f}s ago"
        return False, f"TF stale ({age:.2f}s ago)"

    def _resolve_alias(self, aliases: List[str], all_topics: List[Tuple[str, List[str]]]) -> Tuple[Optional[str], List[str]]:
        topic_dict = {name: types for name, types in all_topics}

        expanded_aliases = []
        for a in aliases:
            expanded_aliases.append(a)
            if self.namespace_hint and a.startswith("/"):
                expanded_aliases.append(_join_ns(self.namespace_hint, a))

        for a in expanded_aliases:
            if a in topic_dict:
                return a, topic_dict[a]
        return None, []

    def _choose_qos_for_topic(self, topic_name: str, is_static_tf: bool = False) -> Tuple[QoSProfile, str]:
        if is_static_tf:
            return self._qos_tf_static, "RELIABLE + TRANSIENT_LOCAL"

        infos = []
        try:
            infos = self.get_publishers_info_by_topic(topic_name)
        except Exception:
            infos = []

        if not infos:
            return self._qos_best_effort, "BEST_EFFORT (default, no publishers yet)"

        reliable_seen = False
        best_effort_seen = False
        transient_local_seen = False
        depths = []

        for info in infos:
            try:
                q = info.qos_profile
                if q.reliability == QoSReliabilityPolicy.RELIABLE:
                    reliable_seen = True
                elif q.reliability == QoSReliabilityPolicy.BEST_EFFORT:
                    best_effort_seen = True

                if q.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL:
                    transient_local_seen = True

                if getattr(q, "depth", None) is not None and q.depth > 0:
                    depths.append(int(q.depth))
            except Exception:
                pass

        depth = max(depths) if depths else 20
        depth = max(10, min(depth, 200))

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE if reliable_seen else QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL if transient_local_seen else QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
        )

        note = []
        note.append("RELIABLE" if qos.reliability == QoSReliabilityPolicy.RELIABLE else "BEST_EFFORT")
        if qos.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL:
            note.append("TRANSIENT_LOCAL")
        else:
            note.append("VOLATILE")
        note.append(f"depth={depth}")

        if reliable_seen and best_effort_seen:
            note.append("mixed publisher reliability detected")

        return qos, " + ".join(note)

    def _ensure_subscriptions(self, force: bool = False):
        now = _now()
        if not force and (now - self._last_topic_refresh) < self._topic_refresh_period:
            return
        self._last_topic_refresh = now

        all_topics = self.get_topic_names_and_types()
        groups = [
            ("topics_required", self.expected.get("topics_required", {})),
            ("topics_optional", self.expected.get("topics_optional", {})),
        ]

        for _, topic_map in groups:
            for label, cfg in topic_map.items():
                aliases = list(cfg.get("aliases", []))
                type_pretty_list = list(cfg.get("types", []))
                if not type_pretty_list:
                    continue
                if label in self._my_subscriptions:
                    continue

                actual_name, actual_types = self._resolve_alias(aliases, all_topics)
                if actual_name is None:
                    continue

                msg_cls = None
                matched_type = None
                for t in type_pretty_list:
                    if t in actual_types and t in IMPORTED_MSGS:
                        msg_cls = IMPORTED_MSGS[t]
                        matched_type = t
                        break

                if msg_cls is None:
                    for t in type_pretty_list:
                        if t in IMPORTED_MSGS:
                            msg_cls = IMPORTED_MSGS[t]
                            matched_type = t
                            break

                if msg_cls is None:
                    continue

                try:
                    qos, qos_note = self._choose_qos_for_topic(actual_name)
                    sub = self.create_subscription(
                        msg_cls,
                        actual_name,
                        lambda msg, lbl=label: self._on_msg(lbl, msg),
                        qos,
                    )
                    self._my_subscriptions[label] = sub

                    with self._lock:
                        st = self._topic_status.get(label)
                        if st is None:
                            st = TopicStatus(label=label)
                            self._topic_status[label] = st
                        st.actual_name = actual_name
                        st.matched_alias = actual_name
                        st.expected_types = type_pretty_list
                        st.qos_note = qos_note

                    self.get_logger().info(
                        f"Subscribed: label={label} topic={actual_name} type={matched_type} qos={qos_note}"
                    )
                except Exception as e:
                    self.get_logger().warning(
                        f"Failed subscription for {label} ({actual_name}): {e}"
                    )

    def _on_msg(self, label: str, msg):
        with self._lock:
            st = self._topic_status.get(label)
            if st is None:
                st = TopicStatus(label=label)
                self._topic_status[label] = st
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
                return (
                    f"frame={msg.header.frame_id} child={msg.child_frame_id} "
                    f"pos=({p.x:.2f},{p.y:.2f}) yaw={yaw:.1f}° "
                    f"v=({v.x:.2f},{v.y:.2f}) wz={wz:.2f}"
                )
            if cname == "Twist":
                return f"lin=({msg.linear.x:.2f},{msg.linear.y:.2f}) ang_z={msg.angular.z:.2f}"
            if cname == "Imu":
                o = msg.orientation
                yaw = self._quat_to_yaw(o.x, o.y, o.z, o.w)
                return f"frame={msg.header.frame_id} yaw={yaw:.1f}° ang_vel_z={msg.angular_velocity.z:.3f}"
            if cname == "NavSatFix":
                return f"lat={msg.latitude:.6f} lon={msg.longitude:.6f} alt={msg.altitude:.2f}"
            if cname == "LaserScan":
                return f"frame={msg.header.frame_id} n={len(msg.ranges)} angle=[{msg.angle_min:.2f},{msg.angle_max:.2f}]"
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
            candidates = [act]
            if self.namespace_hint:
                candidates.append(_join_ns(self.namespace_hint, act))

            found = False
            found_name = None
            missing_note = ""
            for c in candidates:
                endpoints = _action_service_endpoints(c)
                missing = [e for e in endpoints if e not in service_names]
                if not missing:
                    found = True
                    found_name = c
                    break
                missing_note = "Missing: " + ", ".join(missing)

            if found:
                out.append(ActionCheck(found_name or act, True, "OK"))
            else:
                out.append(ActionCheck(act, False, missing_note or "No action endpoints found"))
        return out

    def _check_tf_pair(self, parent: str, child: str) -> TfCheck:
        try:
            ok = self.tf_buffer.can_transform(
                parent,
                child,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            if not ok:
                return TfCheck(parent, child, False, "No transform")

            t = self.tf_buffer.lookup_transform(
                parent,
                child,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            tr = t.transform.translation
            rot = t.transform.rotation
            yaw_deg = self._quat_to_yaw(rot.x, rot.y, rot.z, rot.w)
            return TfCheck(
                parent,
                child,
                True,
                f"xyz=({tr.x:.2f},{tr.y:.2f},{tr.z:.2f}) yaw={yaw_deg:.1f}°",
            )
        except Exception as e:
            return TfCheck(parent, child, False, f"Error: {e}")

    def _check_tf(self) -> List[TfCheck]:
        checks: List[TfCheck] = []
        if self._last_tf_time is None and self._last_tf_static_time is None:
            checks.append(TfCheck("(tf feed)", "", False, "No /tf or /tf_static received yet"))
            return checks

        frames = self.expected.get("frames", {})
        map_f = frames.get("map")
        odom_f = frames.get("odom")
        base_f = frames.get("base_link")
        footprint_f = frames.get("base_footprint")
        vrtk_f = frames.get("vrtk_link")

        pairs = []
        if map_f and odom_f:
            pairs.append((map_f, odom_f))
        if odom_f and base_f:
            pairs.append((odom_f, base_f))
        if map_f and base_f:
            pairs.append((map_f, base_f))
        if odom_f and footprint_f:
            pairs.append((odom_f, footprint_f))
        if vrtk_f and base_f:
            pairs.append((vrtk_f, base_f))

        for parent, child in pairs:
            checks.append(self._check_tf_pair(parent, child))
        return checks

    def _topic_presence(
        self,
        all_topics: List[Tuple[str, List[str]]],
        expected_map: Dict[str, dict],
    ) -> Dict[str, TopicStatus]:
        topic_dict = {name: types for name, types in all_topics}
        out: Dict[str, TopicStatus] = {}

        for label, cfg in expected_map.items():
            aliases = list(cfg.get("aliases", []))
            expected_types = list(cfg.get("types", []))
            st = TopicStatus(label=label, expected_types=expected_types)

            actual_name, actual_types = self._resolve_alias(aliases, all_topics)
            if actual_name is not None:
                st.exists = True
                st.actual_name = actual_name
                st.matched_alias = actual_name
                st.type_strs = actual_types

                try:
                    st.publisher_count = len(self.get_publishers_info_by_topic(actual_name))
                except Exception:
                    st.publisher_count = 0

                try:
                    st.subscriber_count = len(self.get_subscriptions_info_by_topic(actual_name))
                except Exception:
                    st.subscriber_count = 0

            with self._lock:
                if label in self._topic_status:
                    s = self._topic_status[label]
                    st.last_msg_time = s.last_msg_time
                    st.msg_count = s.msg_count
                    st.last_preview = s.last_preview
                    if s.qos_note:
                        st.qos_note = s.qos_note

            if expected_types and st.exists:
                if not any(t in st.type_strs for t in expected_types):
                    msg = f"TYPE MISMATCH (expected one of {expected_types})"
                    st.last_preview = ((st.last_preview + " | ") if st.last_preview else "") + msg

            out[label] = st
        return out

    def build_report(self) -> HealthReport:
        self._ensure_subscriptions()
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
            clk = optional_topics.get("clock")
            if clk and (not clk.exists or clk.msg_count == 0):
                warnings.append("use_sim_time=TRUE but /clock missing or not publishing.")

        if self._last_tf_time is None:
            warnings.append("No dynamic /tf received yet.")
        if self._last_tf_static_time is None:
            warnings.append("No /tf_static received yet (robot_state_publisher or static_transform_publisher?).")
        if not tf_connected:
            warnings.append(f"TF connection issue: {tf_note}")

        odom_st = required_topics.get("odom")
        if odom_st and odom_st.exists and odom_st.msg_count == 0:
            warnings.append(
                f"Odom topic exists but no messages received: {odom_st.actual_name} "
                f"(check QoS or publisher health)"
            )

        cmd_st = required_topics.get("cmd_vel")
        if cmd_st and not cmd_st.exists:
            warnings.append("No cmd_vel output topic found. Checked aliases: /panther/cmd_vel, /cmd_vel")

        imu_st = optional_topics.get("imu")
        if imu_st and imu_st.exists and imu_st.msg_count == 0:
            warnings.append(
                f"IMU topic exists but no messages received: {imu_st.actual_name} "
                f"(common sign of QoS mismatch)"
            )

        gps_st = optional_topics.get("gps_fix")
        if gps_st and gps_st.exists and gps_st.msg_count == 0:
            warnings.append(
                f"GPS/NavSatFix topic exists but no messages received: {gps_st.actual_name} "
                f"(common sign of QoS mismatch)"
            )

        scan_st = optional_topics.get("scan")
        if scan_st and scan_st.exists and scan_st.msg_count == 0:
            warnings.append(
                f"LaserScan topic exists but no messages received: {scan_st.actual_name} "
                f"(sensor data often uses BEST_EFFORT)"
            )

        vrtk_tf = [c for c in tf_checks if c.parent == self.expected["frames"]["vrtk_link"] and c.child == self.expected["frames"]["base_link"]]
        if vrtk_tf and not vrtk_tf[0].ok:
            warnings.append(
                f"Missing static TF {self.expected['frames']['vrtk_link']} -> {self.expected['frames']['base_link']} "
                f"(required for Fixposition nav setup if you use that frame chain)"
            )

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


def topic_state(st: TopicStatus, required: bool) -> Tuple[str, str]:
    type_mismatch = "TYPE MISMATCH" in (st.last_preview or "")

    if required:
        if not st.exists:
            return "BAD", RED
        if type_mismatch:
            return "BAD", RED
        if st.msg_count == 0 and st.publisher_count > 0:
            return "WARN", YEL
        return "OK", GRN

    if st.exists and type_mismatch:
        return "BAD", RED
    if st.exists and st.msg_count > 0:
        return "OK", GRN
    if st.exists and st.publisher_count > 0 and st.msg_count == 0:
        return "WARN", YEL
    return "OPT", GRY


def render_report(r: HealthReport):
    clear_terminal()

    ok_req = all(topic_state(st, True)[0] == "OK" for st in r.required_topics.values())
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
    print("-" * 160)
    print(
        f"{'STATE':<8} {'KIND':<5} {'LABEL':<14} {'ACTUAL TOPIC':<32} {'EXISTS':<6} "
        f"{'PUB':<4} {'SUB':<4} {'COUNT':<8} {'AGE':<10} {'TYPE / PREVIEW / QOS'}"
    )
    print("-" * 160)

    for required, topic_map in ((True, r.required_topics), (False, r.optional_topics)):
        for label, st in topic_map.items():
            state_txt, color = topic_state(st, required)
            kind = "REQ" if required else "OPT"
            exists_txt = "YES" if st.exists else "NO"
            actual = st.actual_name or "-"
            types_txt = ", ".join(st.type_strs) if st.type_strs else "-"
            preview = st.last_preview if st.last_preview else "-"
            qos = f" | qos={st.qos_note}" if st.qos_note else ""
            tail = f"{types_txt} | {preview}{qos}"

            print(
                f"{colorize(state_txt, color):<17} {kind:<5} {label:<14} {actual:<32} {exists_txt:<6} "
                f"{st.publisher_count:<4} {st.subscriber_count:<4} {st.msg_count:<8} {_fmt_age(st.last_msg_time):<10} {tail}"
            )

    print()
    print(f"{BOLD}{CYN}TF checks{RST}")
    print("-" * 160)
    for c in r.tf_checks:
        status = colorize("OK", GRN) if c.ok else colorize("FAIL", RED if c.parent != "(tf feed)" else YEL)
        label = "(tf feed)" if c.parent == "(tf feed)" else f"{c.parent} -> {c.child}"
        print(f"{status:<12} {label:<45} {c.note}")

    print()
    print(f"{BOLD}{CYN}Action checks{RST}")
    print("-" * 160)
    for a in r.action_checks:
        status = colorize("OK", GRN) if a.ok else colorize("FAIL", RED)
        print(f"{status:<12} {a.name:<40} {a.note}")

    print()
    print(f"{BOLD}{CYN}Warnings / hints{RST}")
    print("-" * 160)
    if r.warnings:
        for w in r.warnings:
            print(colorize(f"- {w}", YEL))
    else:
        print(colorize("No warnings.", GRN))

    print()
    print(colorize("Press Ctrl+C to exit.", GRY))


def main():
    expected = dict(DEFAULT_EXPECTED)
    expected["frames"] = dict(DEFAULT_EXPECTED["frames"])
    expected["topics_required"] = dict(DEFAULT_EXPECTED["topics_required"])
    expected["topics_optional"] = dict(DEFAULT_EXPECTED["topics_optional"])
    expected["actions"] = list(DEFAULT_EXPECTED["actions"])

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