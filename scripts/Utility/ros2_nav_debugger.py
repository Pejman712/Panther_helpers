#!/usr/bin/env python3
"""
ROS2 Navigation Debugger UI (Tkinter) - Humble-safe version with COLOR-CODED rows.

- No tf2_ros.TransformListener (avoids Humble init/destructor issues)
- Manual TF feed from /tf and /tf_static
- FIXED: do NOT override Node._subscriptions (we use self._my_subscriptions)
- Restores color-coded representation:
    * Required topic missing  -> red
    * Required topic exists but not publishing -> yellow
    * Topic OK -> green
    * TF check fail -> red, pass -> green
    * Action check fail -> red, pass -> green

Added:
- Simple TF connection status:
    * TF LINK OK   -> /tf or /tf_static received recently
    * TF LINK FAIL -> no TF received yet or TF feed is stale
"""

import os
import time
import math
import threading
import queue
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import tf2_ros

try:
    from tf2_msgs.msg import TFMessage
except Exception as e:
    raise RuntimeError(
        "tf2_msgs is not available. Install:\n"
        "  sudo apt install ros-humble-tf2-msgs"
    ) from e

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    raise RuntimeError("tkinter not available. Install: sudo apt install python3-tk") from e


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


class RosInspectorNode(Node):
    def __init__(self, ui_queue: queue.Queue, expected: dict):
        super().__init__("ros_nav_debugger_ui")
        self.ui_queue = ui_queue
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

        self._topic_status: Dict[str, TopicStatus] = {}
        self._my_subscriptions = {}  # IMPORTANT: don't overwrite Node._subscriptions
        self._lock = threading.Lock()

        # TF buffer fed manually
        self.tf_buffer = tf2_ros.Buffer()

        # TF topics
        self.tf_sub = self.create_subscription(
            TFMessage, "/tf", self._on_tf, self._qos_best_effort
        )
        self.tf_static_sub = self.create_subscription(
            TFMessage, "/tf_static", self._on_tf_static, self._qos_reliable
        )

        self._last_tf_time: Optional[float] = None
        self._last_tf_static_time: Optional[float] = None

        self._ensure_subscriptions()
        self._timer = self.create_timer(0.5, self._on_timer)

    def _on_tf(self, msg: TFMessage):
        try:
            for t in msg.transforms:
                self.tf_buffer.set_transform(t, "ros_nav_debugger")
            self._last_tf_time = _now()
        except Exception as e:
            self.ui_queue.put(("error", f"/tf callback error: {e}"))


    def _on_tf_static(self, msg: TFMessage):
        try:
            for t in msg.transforms:
                self.tf_buffer.set_transform_static(t, "ros_nav_debugger")
            self._last_tf_static_time = _now()
        except Exception as e:
            self.ui_queue.put(("error", f"/tf_static callback error: {e}"))

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
                except Exception:
                    pass

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
        return s[:800] + (" …" if len(s) > 800 else "")

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

    def _on_timer(self):
        try:
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

            report = HealthReport(
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
            self.ui_queue.put(("report", report))
        except Exception:
            self.ui_queue.put(("error", traceback.format_exc()))


class RosThread:
    def __init__(self, ui_queue: queue.Queue, expected: dict):
        self.ui_queue = ui_queue
        self.expected = expected
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._node: Optional[RosInspectorNode] = None
        self._executor: Optional[SingleThreadedExecutor] = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if self._executor and self._node:
                self._executor.remove_node(self._node)
                self._node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def _run(self):
        try:
            rclpy.init(args=None)
            self._node = RosInspectorNode(self.ui_queue, self.expected)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            while rclpy.ok() and not self._stop_event.is_set():
                self._executor.spin_once(timeout_sec=0.2)
        except Exception:
            self.ui_queue.put(("error", traceback.format_exc()))
        finally:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass


class DebuggerUI:
    def __init__(self, root: tk.Tk, expected: dict):
        self.root = root
        self.expected = expected
        self.ui_queue: queue.Queue = queue.Queue()
        self.ros_thread = RosThread(self.ui_queue, expected)
        self.last_report: Optional[HealthReport] = None

        self.root.title("ROS2 Nav Debugger")
        self.root.geometry("1100x720")

        self._build_ui()
        self.ros_thread.start()
        self._poll_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        self.status_label = ttk.Label(top, text="Status: starting…")
        self.status_label.pack(side=tk.LEFT)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=1)

        # --- Topics table ---
        topics_frame = ttk.LabelFrame(left, text="Topics (presence + publishing)", padding=8)
        topics_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("exists", "types", "count", "age", "preview")
        self.topics_tree = ttk.Treeview(topics_frame, columns=cols, show="headings", height=20)
        self.topics_tree.heading("exists", text="Exists")
        self.topics_tree.heading("types", text="Types")
        self.topics_tree.heading("count", text="Msg Count")
        self.topics_tree.heading("age", text="Last Msg Age")
        self.topics_tree.heading("preview", text="Last Preview")

        self.topics_tree.column("exists", width=60, anchor=tk.CENTER)
        self.topics_tree.column("types", width=220, anchor=tk.W)
        self.topics_tree.column("count", width=80, anchor=tk.E)
        self.topics_tree.column("age", width=100, anchor=tk.E)
        self.topics_tree.column("preview", width=520, anchor=tk.W)
        self.topics_tree.pack(fill=tk.BOTH, expand=True)

        # Color tags
        self.topics_tree.tag_configure("bad", background="#ffe5e5")   # red-ish
        self.topics_tree.tag_configure("warn", background="#fff2cc")  # yellow-ish
        self.topics_tree.tag_configure("ok", background="#e6ffe6")    # green-ish
        self.topics_tree.tag_configure("opt", background="#f5f5f5")   # light gray

        # --- TF table ---
        tf_frame = ttk.LabelFrame(right, text="TF checks", padding=8)
        tf_frame.pack(fill=tk.BOTH, expand=True)
        self.tf_tree = ttk.Treeview(tf_frame, columns=("ok", "note"), show="headings", height=10)
        self.tf_tree.heading("ok", text="OK")
        self.tf_tree.heading("note", text="Details")
        self.tf_tree.column("ok", width=50, anchor=tk.CENTER)
        self.tf_tree.column("note", width=520, anchor=tk.W)
        self.tf_tree.pack(fill=tk.BOTH, expand=True)
        self.tf_tree.tag_configure("bad", background="#ffe5e5")
        self.tf_tree.tag_configure("ok", background="#e6ffe6")
        self.tf_tree.tag_configure("warn", background="#fff2cc")

        # --- Action table ---
        act_frame = ttk.LabelFrame(right, text="Action checks", padding=8)
        act_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.act_tree = ttk.Treeview(act_frame, columns=("ok", "note"), show="headings", height=6)
        self.act_tree.heading("ok", text="OK")
        self.act_tree.heading("note", text="Details")
        self.act_tree.column("ok", width=50, anchor=tk.CENTER)
        self.act_tree.column("note", width=520, anchor=tk.W)
        self.act_tree.pack(fill=tk.BOTH, expand=True)
        self.act_tree.tag_configure("bad", background="#ffe5e5")
        self.act_tree.tag_configure("ok", background="#e6ffe6")

        # --- Warnings ---
        warn_frame = ttk.LabelFrame(self.root, text="Warnings / hints", padding=8)
        warn_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))
        self.warn_text = tk.Text(warn_frame, height=7, wrap=tk.WORD)
        self.warn_text.pack(fill=tk.BOTH, expand=True)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "report":
                    self.last_report = payload
                    self._render(payload)
                elif kind == "error":
                    self.warn_text.delete("1.0", tk.END)
                    self.warn_text.insert(tk.END, payload)
                    self.status_label.config(text="Status: ERROR")
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def _render(self, r: HealthReport):
        ok_req = all(st.exists for st in r.required_topics.values())
        ok_tf = all(c.ok for c in r.tf_checks if c.parent != "(tf feed)")
        ok_act = all(a.ok for a in r.action_checks)
        self.status_label.config(
            text=(
                f"Status: {'REQ OK' if ok_req else 'REQ FAIL'} | "
                f"{'TF LINK OK' if r.tf_connected else 'TF LINK FAIL'} ({r.tf_note}) | "
                f"{'TF OK' if ok_tf else 'TF FAIL'} | "
                f"{'ACT OK' if ok_act else 'ACT FAIL'}"
            )
        )

        # Topics
        for item in self.topics_tree.get_children():
            self.topics_tree.delete(item)

        def topic_tag(name: str, st: TopicStatus, required: bool) -> str:
            if required:
                if not st.exists:
                    return "bad"
                # For /tf + /tf_static, existence is usually enough
                if name not in ("/tf", "/tf_static") and st.msg_count == 0:
                    return "warn"
                return "ok"
            # optional topics
            if st.exists and st.msg_count > 0:
                return "ok"
            if st.exists and name not in ("/tf", "/tf_static") and st.msg_count == 0:
                return "warn"
            return "opt"

        def add_row(name: str, st: TopicStatus, required: bool):
            exists_txt = "YES" if st.exists else "NO"
            types_txt = ", ".join(st.type_strs) if st.type_strs else "-"
            count_txt = str(st.msg_count)
            age_txt = _fmt_age(st.last_msg_time)
            preview = f"[{'REQ' if required else 'OPT'}] {name} | {st.last_preview}"
            tag = topic_tag(name, st, required)
            self.topics_tree.insert(
                "", tk.END,
                values=(exists_txt, types_txt, count_txt, age_txt, preview),
                tags=(tag,)
            )

        for name, st in r.required_topics.items():
            add_row(name, st, True)
        for name, st in r.optional_topics.items():
            add_row(name, st, False)

        # TF
        for item in self.tf_tree.get_children():
            self.tf_tree.delete(item)
        for c in r.tf_checks:
            if c.parent == "(tf feed)":
                tag = "warn" if not c.ok else "ok"
            else:
                tag = "ok" if c.ok else "bad"
            self.tf_tree.insert(
                "",
                tk.END,
                values=("YES" if c.ok else "NO", f"{c.parent}->{c.child} | {c.note}"),
                tags=(tag,),
            )

        # Actions
        for item in self.act_tree.get_children():
            self.act_tree.delete(item)
        for a in r.action_checks:
            tag = "ok" if a.ok else "bad"
            self.act_tree.insert(
                "",
                tk.END,
                values=("YES" if a.ok else "NO", f"{a.name} | {a.note}"),
                tags=(tag,),
            )

        # Warnings
        self.warn_text.delete("1.0", tk.END)
        if r.warnings:
            self.warn_text.insert(tk.END, "\n".join(f"- {w}" for w in r.warnings))
        else:
            self.warn_text.insert(tk.END, "No warnings.")

    def _on_close(self):
        try:
            self.ros_thread.stop()
        finally:
            self.root.destroy()


def main():
    expected = dict(DEFAULT_EXPECTED)
    # optional override:
    #   NAVDBG_BASE_LINK=panther/base_link python3 ros2_nav_debugger.py
    bl = os.environ.get("NAVDBG_BASE_LINK")
    if bl:
        expected["frames"] = dict(expected["frames"])
        expected["frames"]["base_link"] = bl

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    DebuggerUI(root, expected)
    root.mainloop()


if __name__ == "__main__":
    main()