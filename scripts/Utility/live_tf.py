#!/usr/bin/env python3
"""
Live TF Tree Checker for ROS 2

Features:
- Subscribes to /tf and /tf_static
- Builds a live TF graph
- Verifies connectivity of expected frames
- Detects:
    * missing frames
    * disconnected subtrees
    * stale dynamic TF
    * multiple parents for same child
    * cycles
- Prints terminal status repeatedly

Usage:
  python3 live_tf_tree_checker.py

Optional environment overrides:
  export TFTREE_ROOT=map
  export TFTREE_EXPECT="map,odom,panther/base_link,vrtk_link"
  export TFTREE_STALE_SEC=2.0
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from tf2_msgs.msg import TFMessage


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


def now_s() -> float:
    return time.time()


def fmt_age(ts: Optional[float]) -> str:
    if ts is None:
        return "never"
    age = now_s() - ts
    if age < 10:
        return f"{age:.2f}s"
    return f"{age:.1f}s"


@dataclass
class EdgeInfo:
    parent: str
    child: str
    is_static: bool
    stamp_rx: float
    authority: str = ""
    last_msg_stamp_sec: Optional[float] = None


@dataclass
class FrameStatus:
    name: str
    exists: bool
    connected_to_root: bool
    parent: Optional[str] = None
    is_static_edge: Optional[bool] = None
    edge_age_sec: Optional[float] = None
    note: str = ""


@dataclass
class TreeReport:
    timestamp: float
    root: str
    frame_count: int
    edge_count: int
    static_edge_count: int
    dynamic_edge_count: int
    root_present: bool
    tf_ok: bool
    warnings: List[str] = field(default_factory=list)
    frame_statuses: List[FrameStatus] = field(default_factory=list)
    disconnected_frames: List[str] = field(default_factory=list)
    multi_parent_children: Dict[str, List[str]] = field(default_factory=dict)
    cycles: List[str] = field(default_factory=list)


class LiveTfTreeChecker(Node):
    def __init__(self):
        super().__init__("live_tf_tree_checker")

        self.root = os.environ.get("TFTREE_ROOT", "map").strip()
        expect_env = os.environ.get(
            "TFTREE_EXPECT",
            "map,odom,panther/base_link,vrtk_link",
        )
        self.expected_frames = [x.strip() for x in expect_env.split(",") if x.strip()]
        self.stale_sec = float(os.environ.get("TFTREE_STALE_SEC", "2.0"))

        qos_tf = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
        )
        qos_tf_static = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.edges_by_child: Dict[str, EdgeInfo] = {}
        self.all_frames: Set[str] = set()
        self.multi_parent_history: Dict[str, Set[str]] = {}
        self.last_tf_rx: Optional[float] = None
        self.last_tf_static_rx: Optional[float] = None

        self.create_subscription(TFMessage, "/tf", self._on_tf, qos_tf)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf_static, qos_tf_static)

        self.get_logger().info(
            f"Live TF checker started. root={self.root} expected={self.expected_frames}"
        )

    def _norm(self, frame: str) -> str:
        return frame.lstrip("/").strip()

    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _update_edge(self, parent: str, child: str, is_static: bool, stamp_sec: Optional[float]):
        rx = now_s()
        parent = self._norm(parent)
        child = self._norm(child)

        if not parent or not child:
            return

        self.all_frames.add(parent)
        self.all_frames.add(child)

        if child in self.edges_by_child:
            prev = self.edges_by_child[child]
            if prev.parent != parent:
                hist = self.multi_parent_history.setdefault(child, set())
                hist.add(prev.parent)
                hist.add(parent)

        self.edges_by_child[child] = EdgeInfo(
            parent=parent,
            child=child,
            is_static=is_static,
            stamp_rx=rx,
            last_msg_stamp_sec=stamp_sec,
        )

    def _on_tf(self, msg: TFMessage):
        self.last_tf_rx = now_s()
        for t in msg.transforms:
            stamp_sec = self._stamp_to_sec(t.header.stamp)
            self._update_edge(t.header.frame_id, t.child_frame_id, False, stamp_sec)

    def _on_tf_static(self, msg: TFMessage):
        self.last_tf_static_rx = now_s()
        for t in msg.transforms:
            stamp_sec = self._stamp_to_sec(t.header.stamp)
            self._update_edge(t.header.frame_id, t.child_frame_id, True, stamp_sec)

    def _children_map(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for child, edge in self.edges_by_child.items():
            out.setdefault(edge.parent, []).append(child)
        return out

    def _reachable_from_root(self) -> Set[str]:
        children = self._children_map()
        seen: Set[str] = set()
        stack = [self.root]

        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for ch in children.get(cur, []):
                if ch not in seen:
                    stack.append(ch)
        return seen

    def _find_cycles(self) -> List[str]:
        cycles = []
        visited: Set[str] = set()
        active: Set[str] = set()

        def dfs(frame: str, path: List[str]):
            if frame in active:
                if frame in path:
                    i = path.index(frame)
                    cyc = path[i:] + [frame]
                    cycles.append(" -> ".join(cyc))
                return
            if frame in visited:
                return

            visited.add(frame)
            active.add(frame)

            children = [c for c, e in self.edges_by_child.items() if e.parent == frame]
            for ch in children:
                dfs(ch, path + [ch])

            active.remove(frame)

        frames = set(self.all_frames)
        for f in frames:
            if f not in visited:
                dfs(f, [f])

        return sorted(set(cycles))

    def _path_to_root(self, frame: str) -> Tuple[bool, List[str], Optional[str]]:
        frame = self._norm(frame)
        path = [frame]
        seen = {frame}
        cur = frame

        while cur != self.root:
            edge = self.edges_by_child.get(cur)
            if edge is None:
                return False, path, "No parent"
            cur = edge.parent
            if cur in seen:
                path.append(cur)
                return False, path, "Cycle"
            path.append(cur)
            seen.add(cur)

        return True, path, None

    def build_report(self) -> TreeReport:
        warnings: List[str] = []
        frame_statuses: List[FrameStatus] = []

        edges = list(self.edges_by_child.values())
        reachable = self._reachable_from_root() if self.root else set()
        root_present = self.root in self.all_frames or self.root in [e.parent for e in edges]

        static_edge_count = sum(1 for e in edges if e.is_static)
        dynamic_edge_count = sum(1 for e in edges if not e.is_static)

        if self.last_tf_rx is None:
            warnings.append("No /tf received yet.")
        else:
            age = now_s() - self.last_tf_rx
            if age > self.stale_sec:
                warnings.append(f"/tf is stale: last message {age:.2f}s ago.")

        if self.last_tf_static_rx is None:
            warnings.append("No /tf_static received yet.")

        if not root_present:
            warnings.append(f"Root frame '{self.root}' not seen in TF graph.")

        multi_parent_children = {
            child: sorted(list(parents))
            for child, parents in self.multi_parent_history.items()
            if len(parents) > 1
        }
        for child, parents in multi_parent_children.items():
            warnings.append(f"Child '{child}' seen with multiple parents: {parents}")

        cycles = self._find_cycles()
        for c in cycles:
            warnings.append(f"Cycle detected: {c}")

        expected_set = list(dict.fromkeys(self.expected_frames))
        for frame in expected_set:
            exists = frame in self.all_frames or frame == self.root
            if not exists:
                frame_statuses.append(
                    FrameStatus(
                        name=frame,
                        exists=False,
                        connected_to_root=False,
                        note="Missing from TF graph",
                    )
                )
                continue

            ok, path, why = self._path_to_root(frame)
            parent = None
            edge_age = None
            is_static_edge = None
            if frame in self.edges_by_child:
                edge = self.edges_by_child[frame]
                parent = edge.parent
                edge_age = now_s() - edge.stamp_rx
                is_static_edge = edge.is_static

            note = "Connected" if ok else why or "Disconnected"
            if frame == self.root:
                ok = True
                note = "Root"

            frame_statuses.append(
                FrameStatus(
                    name=frame,
                    exists=True,
                    connected_to_root=ok,
                    parent=parent,
                    is_static_edge=is_static_edge,
                    edge_age_sec=edge_age,
                    note=note + f" | path: {' <- '.join(path)}",
                )
            )

        disconnected_frames = sorted(
            f for f in self.all_frames
            if f not in reachable and f != self.root
        ) if root_present else sorted(self.all_frames)

        if disconnected_frames:
            warnings.append(f"Disconnected frames from root '{self.root}': {disconnected_frames[:12]}{' ...' if len(disconnected_frames) > 12 else ''}")

        tf_ok = (
            root_present
            and not cycles
            and not multi_parent_children
            and all(fs.exists and fs.connected_to_root for fs in frame_statuses)
            and (self.last_tf_rx is not None or self.last_tf_static_rx is not None)
        )

        return TreeReport(
            timestamp=now_s(),
            root=self.root,
            frame_count=len(self.all_frames),
            edge_count=len(edges),
            static_edge_count=static_edge_count,
            dynamic_edge_count=dynamic_edge_count,
            root_present=root_present,
            tf_ok=tf_ok,
            warnings=warnings,
            frame_statuses=frame_statuses,
            disconnected_frames=disconnected_frames,
            multi_parent_children=multi_parent_children,
            cycles=cycles,
        )


def render(report: TreeReport):
    clear_terminal()

    overall = colorize("OK", GRN) if report.tf_ok else colorize("FAIL", RED)

    print(f"{BOLD}Live TF Tree Checker{RST}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
    print(f"Root: {report.root}")
    print(f"Overall TF status: {overall}")
    print(
        f"Frames: {report.frame_count} | Edges: {report.edge_count} | "
        f"Dynamic: {report.dynamic_edge_count} | Static: {report.static_edge_count}"
    )
    print()

    print(f"{BOLD}{CYN}Expected frames{RST}")
    print("-" * 140)
    print(f"{'STATE':<10} {'FRAME':<24} {'PARENT':<24} {'EDGE':<10} {'AGE':<10} NOTE")
    print("-" * 140)

    for fs in report.frame_statuses:
        if not fs.exists:
            state = colorize("MISSING", RED)
        elif fs.connected_to_root:
            state = colorize("OK", GRN)
        else:
            state = colorize("BAD", RED)

        parent = fs.parent or "-"
        edge_kind = "-" if fs.is_static_edge is None else ("static" if fs.is_static_edge else "dynamic")
        age = "-" if fs.edge_age_sec is None else f"{fs.edge_age_sec:.2f}s"

        print(f"{state:<19} {fs.name:<24} {parent:<24} {edge_kind:<10} {age:<10} {fs.note}")

    print()
    print(f"{BOLD}{CYN}Warnings{RST}")
    print("-" * 140)
    if report.warnings:
        for w in report.warnings:
            print(colorize(f"- {w}", YEL))
    else:
        print(colorize("No warnings.", GRN))

    print()
    print(f"{BOLD}{CYN}Disconnected frames{RST}")
    print("-" * 140)
    if report.disconnected_frames:
        for f in report.disconnected_frames[:30]:
            print(colorize(f"- {f}", YEL))
        if len(report.disconnected_frames) > 30:
            print(colorize(f"... and {len(report.disconnected_frames) - 30} more", YEL))
    else:
        print(colorize("None.", GRN))

    print()
    print(colorize("Press Ctrl+C to exit.", GRY))


def main():
    rclpy.init()
    node = None
    executor = None

    try:
        node = LiveTfTreeChecker()
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        last_render = 0.0
        render_period = 0.5

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
            t = now_s()
            if t - last_render >= render_period:
                report = node.build_report()
                render(report)
                last_render = t

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        clear_terminal()
        print(colorize("Fatal error:", RED))
        print(traceback.format_exc())
    finally:
        try:
            if executor and node:
                executor.remove_node(node)
                node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()