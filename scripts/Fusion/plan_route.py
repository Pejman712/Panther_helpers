#!/usr/bin/env python3

import argparse
import math
import time
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.task import Future

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Point
from tf2_ros import Buffer, TransformListener
from pyproj import CRS, Transformer
from robot_localization.srv import FromLL, ToLL
from nav2_simple_commander.robot_navigator import BasicNavigator


# ======================= geometry helpers =======================

def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = (32700 + zone) if (lat < 0.0) else (32600 + zone)
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


def resample_polyline(points: List[Tuple[float, float]], spacing_m: float) -> List[Tuple[float, float]]:
    """
    Densify polyline so consecutive points are ~spacing_m apart.
    points: [(x,y), ...] in meters.
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

        dist = (spacing_m - carry) if carry > 1e-9 else spacing_m
        while dist < seg_len - 1e-9:
            out.append((x0 + ux * dist, y0 + uy * dist))
            dist += spacing_m

        out.append((x1, y1))

        last_mark = dist - spacing_m
        remain = seg_len - last_mark
        carry = 0.0 if remain < 1e-6 else (spacing_m - remain)

    filtered: List[Tuple[float, float]] = []
    last = None
    for p in out:
        if last is None or (abs(p[0] - last[0]) > 1e-6 or abs(p[1] - last[1]) > 1e-6):
            filtered.append(p)
        last = p
    return filtered


# ======================= OSMnx planning =======================

def plan_osmnx_route_lonlat(
    start_lat: float,
    start_lon: float,
    goal_lat: float,
    goal_lon: float,
    network_type: str = "all",
    dist_m: float = 1200.0,
) -> List[Tuple[float, float]]:
    """
    Returns route polyline as [(lon, lat), ...] using edge geometries where possible.
    """
    import osmnx as ox
    import networkx as nx
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

        seg = list(geom.coords)  # (lon, lat)
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


# ======================= ROS2 Node =======================

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

        # Sequential /fromLL conversion state
        self._pending_lonlats: List[Tuple[float, float]] = []
        self._fromll_results: List[Tuple[float, float]] = []
        self._current_fromll_future: Optional[Future] = None
        self._current_fromll_index = 0
        self._converting = False
        self._fromll_start_time: Optional[float] = None

        # Increased because navsat_transform_node can be slow just after startup
        self._fromll_timeout_sec = 30.0
        self._toll_timeout_sec = 5.0

        if self.use_fromll:
            self.fromll_client = self.create_client(FromLL, self.fromll_service)
            while not self.fromll_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"[fromLL] waiting for service '{self.fromll_service}' ...")
            self.get_logger().info(f"[fromLL] service ready ({self.fromll_service})")

        if self.localized and self.use_fromll:
            self.toll_client = self.create_client(ToLL, self.toll_service)
            while not self.toll_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"[toLL] waiting for service '{self.toll_service}' ...")
            self.get_logger().info(f"[toLL] service ready ({self.toll_service})")

        self.create_subscription(PointStamped, self.clicked_topic, self._clicked_cb, 10)

        self.start_ll: Optional[Tuple[float, float]] = None  # (lat, lon)
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

    # ---------------- plotting ----------------

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
        (self._route_line,) = self._ax.plot([], [], marker="o", linestyle="-", label="Route/Goals")
        self._robot_scatter = self._ax.scatter([], [], marker="x", label="Robot")
        self._current_goal_scatter = self._ax.scatter([], [], marker="o", label="Current goal")
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

    # ---------------- TF helpers ----------------

    def _get_robot_pose(self) -> Tuple[float, float, float]:
        tf = self.tf_buffer.lookup_transform(self.goal_frame, self.base_frame, rclpy.time.Time())
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
            if (time.time() - t0) > self._toll_timeout_sec:
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

    # ---------------- clicks ----------------

    def _clicked_cb(self, msg: PointStamped):
        if self.require_frame and msg.header.frame_id != self.require_frame:
            self.get_logger().warning(
                f"[click] ignored frame_id='{msg.header.frame_id}', expected '{self.require_frame}'"
            )
            return

        lat = float(msg.point.y)
        lon = float(msg.point.x)

        if self.localized:
            self.get_logger().info(
                f"[state] GOAL set lat={lat:.7f} lon={lon:.7f}. Planning from current robot pose..."
            )
            self._reset_all()
            self.goal_ll = (lat, lon)
            self._plan_route_and_prepare()
            return

        if self.start_ll is None:
            self._reset_all()
            self.start_ll = (lat, lon)
            self.get_logger().info(f"[state] START set lat={lat:.7f} lon={lon:.7f}. Now click GOAL.")
            return

        if self.goal_ll is None:
            self.goal_ll = (lat, lon)
            self.get_logger().info(f"[state] GOAL set lat={lat:.7f} lon={lon:.7f}. Planning...")
            self._plan_route_and_prepare()
            return

        self.get_logger().info("[state] resetting (new START)")
        self._reset_all()
        self.start_ll = (lat, lon)
        self.get_logger().info(f"[state] START set lat={lat:.7f} lon={lon:.7f}. Now click GOAL.")

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

    # ---------------- planning + conversion ----------------

    def _plan_route_and_prepare(self):
        if self.goal_ll is None:
            self.get_logger().error("[plan] goal not set")
            return

        if self.localized:
            try:
                if self.use_fromll:
                    s_lat, s_lon = self._get_robot_lonlat_from_toll()
                    self.get_logger().info(f"[localized] robot start from /toLL lat={s_lat:.7f} lon={s_lon:.7f}")
                else:
                    self.get_logger().warning(
                        "[localized] anchor mode without /toLL uses an approximate start for OSMnx. "
                        "For best accuracy use --localized --use-fromll with /toLL available."
                    )
                    g_lat, g_lon = self.goal_ll
                    s_lat, s_lon = g_lat, g_lon
            except Exception as e:
                self.get_logger().error(f"[localized] failed to determine robot start: {e}")
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

        self.get_logger().info(f"[plan] route computed points={len(lonlats)} time={time.time() - t0:.2f}s")
        if len(lonlats) < 2:
            self.get_logger().error("[plan] route too short")
            self._reset_all()
            return

        if self.use_fromll:
            if self.fromll_client is None:
                self.get_logger().error("[fromLL] use_fromll=True but client not created")
                self._reset_all()
                return

            self._pending_lonlats = lonlats[:]
            self._fromll_results = []
            self._current_fromll_future = None
            self._current_fromll_index = 0
            self._converting = True
            self._fromll_start_time = None

            self.get_logger().info(f"[fromLL] starting sequential conversion points={len(self._pending_lonlats)}")
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

            self.get_logger().info(f"[fromLL] conversion complete points={len(route_map)}")
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
        self.get_logger().info(f"[fromLL] requested {i}/{n} lat={lat:.7f} lon={lon:.7f}")

    def _convert_anchor(self, lonlats: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Convert lon/lat route to local UTM coordinates, then anchor it into map frame
        using only translation plus an optional fixed manual yaw offset.

        No robot heading is used here.
        """
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
        self.get_logger().info(f"[route] densified points={after} (was {before}) spacing_m={self.spacing_m:.2f}")

        if self.max_goals > 0 and len(route_map) > self.max_goals:
            original_last = route_map[-1]
            step = max(1, len(route_map) // self.max_goals)
            route_map = route_map[::step]
            if route_map[-1] != original_last:
                route_map.append(original_last)
            self.get_logger().warning(f"[route] capped goals to {len(route_map)} (max_goals={self.max_goals})")

        try:
            rx, ry, _ = self._get_robot_pose()
            while route_map:
                d = math.hypot(route_map[0][0] - rx, route_map[0][1] - ry)
                if d < self.skip_near_m:
                    self.get_logger().info(f"[route] skipping near start goal d={d:.2f}m < {self.skip_near_m:.2f}m")
                    route_map.pop(0)
                else:
                    break
        except Exception as e:
            self.get_logger().warning(f"[route] could not evaluate skip_near_m: {e}")

        if not route_map:
            self.get_logger().error("[route] no usable goals remain after filtering")
            self._reset_all()
            return

        self._route_map = route_map
        self._idx = 0
        self._nav_active = False

        self.get_logger().info(f"[nav2] route ready goals={len(self._route_map)} starting...")
        self._send_next_goal()

    # ---------------- worker ----------------

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
                            f"[fromLL] timeout on point {i}/{n} after {elapsed:.1f}s "
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
                self.get_logger().info(f"[fromLL] converted {i}/{n}: x={x:.2f} y={y:.2f}")

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
                self.get_logger().warning("[nav2] goal failed, continuing (continue_on_abort=True)")
                self._idx += 1
                self._send_next_goal()
            else:
                self.get_logger().error("[nav2] goal failed, stopping (continue_on_abort=False)")

    # ---------------- Nav2 BasicNavigator sequential sending ----------------

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


# ======================= main =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Show live matplotlib plot")
    parser.add_argument("--use-fromll", action="store_true", help="Use robot_localization /fromLL to convert lon/lat to map")
    parser.add_argument("--localized", action="store_true", help="Use current robot pose as START; only click GOAL")
    parser.add_argument("--fromll-service", default="/fromLL", help="Service name for FromLL (default: /fromLL)")
    parser.add_argument("--toll-service", default="/toLL", help="Service name for ToLL (default: /toLL)")
    parser.add_argument("--spacing", type=float, default=10.0, help="Densify spacing in meters (smaller => more points)")
    parser.add_argument("--graph-dist", type=float, default=1200.0, help="OSMnx graph dist in meters")
    parser.add_argument("--network-type", default="all", help="OSMnx network_type (all, walk, drive, bike)")
    parser.add_argument("--clicked-topic", default="/clicked_point", help="Mapviz click topic")
    parser.add_argument("--require-frame", default="wgs84", help="Expected frame_id for clicked points")
    parser.add_argument("--goal-frame", default="map", help="Nav2 goal frame")
    parser.add_argument("--base-frame", default="base_link", help="Robot base frame")
    parser.add_argument("--continue-on-abort", action="store_true", help="If a goal fails, continue to next")
    parser.add_argument("--max-goals", type=int, default=400, help="Safety cap on number of goals (0 disables)")
    parser.add_argument("--skip-near", type=float, default=1.0, help="Skip initial goals closer than this distance to robot (m)")
    parser.add_argument(
        "--anchor-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Manual heading offset in degrees for anchor mode (e.g. -90 or 90)"
    )

    args = parser.parse_args()

    rclpy.init()
    node = ClickPlanner(
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

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()