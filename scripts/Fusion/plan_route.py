#!/usr/bin/env python3
"""
Mapviz click (wgs84) START+GOAL -> OSMnx route planning -> (choose conversion) -> densify -> Nav2 NavigateToPose

Conversion modes:
  A) Anchor (default): lon/lat -> UTM local XY -> anchor at current robot pose in "map" using TF
  B) FromLL: lon/lat -> robot_localization /fromLL -> "map" points

Features:
- Waits for EACH NavigateToPose result before sending next goal
- Densifies route with spacing_m (more points = smaller spacing)
- Publishes the route as nav_msgs/Path on /gps_route_path
- Optional live matplotlib plot (--plot)

Run examples:
  python3 gps_planner_mapviz.py
  python3 gps_planner_mapviz.py --plot
  python3 gps_planner_mapviz.py --use-fromll
  python3 gps_planner_mapviz.py --use-fromll --fromll-service /fromLL --spacing 0.5
"""

import argparse
import math
import time
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from tf2_ros import Buffer, TransformListener

from pyproj import CRS, Transformer
from robot_localization.srv import FromLL


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
    # planar yaw only
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

        # update carry: how far are we past the last perfect spacing mark?
        last_mark = dist - spacing_m
        remain = seg_len - last_mark
        carry = 0.0 if remain < 1e-6 else (spacing_m - remain)

    # remove consecutive duplicates
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
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]  # lon, lat
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            geom = LineString([(x1, y1), (x2, y2)])

        seg = list(geom.coords)  # (lon,lat)
        if coords and seg and tuple(seg[0]) == tuple(coords[-1]):
            seg = seg[1:]
        coords.extend([(float(x), float(y)) for x, y in seg])

    # remove consecutive duplicates
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
        spacing_m: float,
        graph_dist_m: float,
        network_type: str,
        clicked_topic: str,
        require_frame: str,
        goal_frame: str,
        base_frame: str,
        server_name: str,
        continue_on_abort: bool,
        max_goals: int,
    ):
        super().__init__("gps_click_planner")

        self.enable_plot = enable_plot
        self.use_fromll = use_fromll
        self.fromll_service = fromll_service
        self.spacing_m = spacing_m
        self.graph_dist_m = graph_dist_m
        self.network_type = network_type
        self.clicked_topic = clicked_topic
        self.require_frame = require_frame
        self.goal_frame = goal_frame
        self.base_frame = base_frame
        self.server_name = server_name
        self.continue_on_abort = continue_on_abort
        self.max_goals = max_goals

        # Nav2 action client
        self._action_client = ActionClient(self, NavigateToPose, self.server_name)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # FromLL client (optional)
        self.fromll_client = None
        self._fromll_futures: List[Future] = []
        self._pending_lonlats: List[Tuple[float, float]] = []
        self._converted_map_pts: List[Tuple[float, float]] = []
        self._converting = False

        if self.use_fromll:
            self.fromll_client = self.create_client(FromLL, self.fromll_service)
            while not self.fromll_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"[fromLL] waiting for service '{self.fromll_service}' ...")
            self.get_logger().info(f"[fromLL] service ready ({self.fromll_service})")

        # Click subscription
        self.create_subscription(PointStamped, self.clicked_topic, self._clicked_cb, 10)

        # Route path publisher
        self.path_pub = self.create_publisher(Path, "/gps_route_path", 1)

        # Route / goal state
        self.start_ll: Optional[Tuple[float, float]] = None  # (lat, lon)
        self.goal_ll: Optional[Tuple[float, float]] = None

        self._route_map: Optional[List[Tuple[float, float]]] = None  # [(x,y) in map]
        self._idx = 0
        self._in_flight = False

        # Plot
        if self.enable_plot:
            self._init_plot()
            self._plot_timer = self.create_timer(0.2, self._plot_update)

        # Main periodic worker
        self._worker_timer = self.create_timer(0.1, self._worker)

        self.get_logger().info(
            "[ready] Click START then GOAL in Mapviz "
            f"(expect frame '{self.require_frame}', x=lon,y=lat). "
            f"mode={'fromLL' if self.use_fromll else 'anchor'} spacing_m={self.spacing_m}"
        )

    # ---------------- plotting ----------------

    def _init_plot(self):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt
        self._fig, self._ax = plt.subplots()
        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.set_title("Route in map frame (anchored or fromLL)")
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

    # ---------------- clicks ----------------

    def _clicked_cb(self, msg: PointStamped):
        if self.require_frame and msg.header.frame_id != self.require_frame:
            self.get_logger().warning(
                f"[click] ignored frame_id='{msg.header.frame_id}', expected '{self.require_frame}'"
            )
            return

        lat = float(msg.point.y)
        lon = float(msg.point.x)

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

        # third click resets
        self.get_logger().info("[state] resetting (new START)")
        self._reset_all()
        self.start_ll = (lat, lon)
        self.get_logger().info(f"[state] START set lat={lat:.7f} lon={lon:.7f}. Now click GOAL.")

    def _reset_all(self):
        self.start_ll = None
        self.goal_ll = None
        self._route_map = None
        self._idx = 0
        self._in_flight = False

        self._fromll_futures.clear()
        self._pending_lonlats.clear()
        self._converted_map_pts.clear()
        self._converting = False

    # ---------------- planning + conversion ----------------

    def _plan_route_and_prepare(self):
        assert self.start_ll and self.goal_ll
        s_lat, s_lon = self.start_ll
        g_lat, g_lon = self.goal_ll

        # Plan
        t0 = time.time()
        try:
            lonlats = plan_osmnx_route_lonlat(
                s_lat, s_lon, g_lat, g_lon,
                network_type=self.network_type,
                dist_m=self.graph_dist_m,
            )
        except Exception as e:
            self.get_logger().error(f"[plan] OSMnx failed: {e}")
            self._reset_all()
            return

        self.get_logger().info(f"[plan] route computed points={len(lonlats)} time={time.time()-t0:.2f}s")
        if len(lonlats) < 2:
            self.get_logger().error("[plan] route too short")
            self._reset_all()
            return

        if self.use_fromll:
            if self.fromll_client is None:
                self.get_logger().error("[fromLL] use_fromll=True but client not created")
                self._reset_all()
                return

            self._pending_lonlats = lonlats[:]  # [(lon,lat)]
            self._converted_map_pts = []
            self._fromll_futures = []
            self._converting = True

            for i, (lon, lat) in enumerate(self._pending_lonlats, start=1):
                req = FromLL.Request()
                req.ll_point.longitude = float(lon)
                req.ll_point.latitude = float(lat)
                req.ll_point.altitude = 0.0
                self._fromll_futures.append(self.fromll_client.call_async(req))
                if i == 1 or i == len(self._pending_lonlats) or i % 50 == 0:
                    self.get_logger().info(f"[fromLL] queued {i}/{len(self._pending_lonlats)}")

            self.get_logger().info("[fromLL] all conversions queued")
        else:
            try:
                route_map = self._convert_anchor(lonlats)
            except Exception as e:
                self.get_logger().error(f"[anchor] conversion failed: {e}")
                self._reset_all()
                return

            self._finalize_route_and_start(route_map)

    def _convert_anchor(self, lonlats: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        lon0, lat0 = lonlats[0]
        utm = utm_crs_for_lonlat(lon0, lat0)
        to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        x0, y0 = to_utm.transform(lon0, lat0)

        local_xy: List[Tuple[float, float]] = []
        for lon, lat in lonlats:
            x, y = to_utm.transform(lon, lat)
            local_xy.append((x - x0, y - y0))

        # anchor at current robot pose in map frame
        rx, ry, ryaw = self._get_robot_pose()
        self.get_logger().info(f"[anchor] robot pose: x={rx:.2f} y={ry:.2f} yaw={ryaw:.2f}")

        return [(rx + dx, ry + dy) for (dx, dy) in local_xy]

    def _publish_route_path(self, route_map: List[Tuple[float, float]]):
        path = Path()
        path.header.frame_id = self.goal_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for i, (x, y) in enumerate(route_map):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            if i < len(route_map) - 1:
                nx, ny = route_map[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = 0.0

            pose.pose.orientation = yaw_to_quat(yaw)
            path.poses.append(pose)

        self.path_pub.publish(path)
        self.get_logger().info(f"[path] published /gps_route_path with {len(path.poses)} poses")

    def _finalize_route_and_start(self, route_map: List[Tuple[float, float]]):
        # Densify
        before = len(route_map)
        route_map = resample_polyline(route_map, self.spacing_m)
        after = len(route_map)
        self.get_logger().info(f"[route] densified points={after} (was {before}) spacing_m={self.spacing_m:.2f}")

        # Cap goals
        if self.max_goals > 0 and len(route_map) > self.max_goals:
            step = max(1, len(route_map) // self.max_goals)
            orig_last = route_map[-1]
            sampled = route_map[::step]
            if sampled[-1] != orig_last:
                sampled.append(orig_last)
            route_map = sampled
            self.get_logger().warning(f"[route] capped goals to {len(route_map)} (max_goals={self.max_goals})")

        self._route_map = route_map
        self._publish_route_path(self._route_map)

        self._idx = 0
        self._in_flight = False

        self.get_logger().info(f"[nav2] route ready goals={len(self._route_map)} starting...")
        self._send_next_goal()

    # ---------------- worker for fromLL async conversions ----------------

    def _worker(self):
        if not self._converting:
            return

        if not self._fromll_futures:
            return

        if any(not f.done() for f in self._fromll_futures):
            return

        route_map: List[Tuple[float, float]] = []
        try:
            for i, f in enumerate(self._fromll_futures, start=1):
                resp = f.result()
                route_map.append((float(resp.map_point.x), float(resp.map_point.y)))
                if i == 1 or i == len(self._fromll_futures) or i % 50 == 0:
                    self.get_logger().info(f"[fromLL] converted {i}/{len(self._fromll_futures)}")
        except Exception as e:
            self.get_logger().error(f"[fromLL] conversion failed: {e}")
            self._reset_all()
            return
        finally:
            self._converting = False
            self._fromll_futures.clear()
            self._pending_lonlats.clear()

        self.get_logger().info(f"[fromLL] conversion complete points={len(route_map)}")
        self._finalize_route_and_start(route_map)

    # ---------------- Nav2 goal sending (sequential) ----------------

    def _send_next_goal(self):
        if self._route_map is None:
            return
        if self._in_flight:
            return
        if self._idx >= len(self._route_map):
            self.get_logger().info("[nav2] finished all goals")
            return

        while rclpy.ok() and not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(f"[nav2] waiting for action server '{self.server_name}' ...")

        x, y = self._route_map[self._idx]
        if self._idx < len(self._route_map) - 1:
            nx, ny = self._route_map[self._idx + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = 0.0

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.goal_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation = yaw_to_quat(yaw)

        self._in_flight = True
        self.get_logger().info(f"[nav2] goal {self._idx+1}/{len(self._route_map)}: x={x:.2f} y={y:.2f}")

        fut = self._action_client.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("[nav2] goal rejected")
            self._in_flight = False
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        self.get_logger().info(f"[nav2] result status={status}")
        self._in_flight = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._idx += 1
            self._send_next_goal()
            return

        if self.continue_on_abort:
            self.get_logger().warning("[nav2] goal failed, continuing (continue_on_abort=True)")
            self._idx += 1
            self._send_next_goal()
        else:
            self.get_logger().error("[nav2] goal failed, stopping (continue_on_abort=False)")


# ======================= main =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Show live matplotlib plot")
    parser.add_argument("--use-fromll", action="store_true", help="Use robot_localization /fromLL to convert lon/lat to map")
    parser.add_argument("--fromll-service", default="/fromLL", help="Service name for FromLL (default: /fromLL)")
    parser.add_argument("--spacing", type=float, default=10.0, help="Densify spacing in meters (smaller => more points)")
    parser.add_argument("--graph-dist", type=float, default=1200.0, help="OSMnx graph dist in meters")
    parser.add_argument("--network-type", default="all", help="OSMnx network_type (all, walk, drive, bike)")
    parser.add_argument("--clicked-topic", default="/clicked_point", help="Mapviz click topic")
    parser.add_argument("--require-frame", default="wgs84", help="Expected frame_id for clicked points")
    parser.add_argument("--goal-frame", default="map", help="Nav2 goal frame")
    parser.add_argument("--base-frame", default="base_link", help="Robot base frame")
    parser.add_argument("--server-name", default="navigate_to_pose", help="Nav2 NavigateToPose action name")
    parser.add_argument("--continue-on-abort", action="store_true", help="If a goal fails, continue to next")
    parser.add_argument("--max-goals", type=int, default=400, help="Safety cap on number of goals (0 disables)")

    args = parser.parse_args()

    rclpy.init()
    node = ClickPlanner(
        enable_plot=args.plot,
        use_fromll=args.use_fromll,
        fromll_service=args.fromll_service,
        spacing_m=args.spacing,
        graph_dist_m=args.graph_dist,
        network_type=args.network_type,
        clicked_topic=args.clicked_topic,
        require_frame=args.require_frame,
        goal_frame=args.goal_frame,
        base_frame=args.base_frame,
        server_name=args.server_name,
        continue_on_abort=args.continue_on_abort,
        max_goals=args.max_goals,
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