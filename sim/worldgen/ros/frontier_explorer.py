#!/usr/bin/env python3
"""Frontier exploration over Nav2's global costmap, replacing explore_lite.

explore_lite stopped for good after one empty frontier search, even when the costmap had hundreds of reachable
frontier cells (measured, 2026-10-08). This node is the plain version: flood-fill from the robot through cells
cheaper than `traverse_cost`, collect unknown cells bordering that region, cluster them, and send Nav2 to the
reachable cell next to the best cluster (largest size / distance). Goals that abort or time out are blacklisted.
It stops only after `max_empty` consecutive searches find nothing.

    python3 /worldgen/frontier_explorer.py --ros-args -p use_sim_time:=true
"""

import math
from collections import deque

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        self.traverse_cost = self.declare_parameter("traverse_cost", 99).value   # cells below lethal (99) are passable, as for the Nav2 planner
        self.min_cluster = self.declare_parameter("min_cluster", 8).value       # frontier cells; 8 x 5 cm = 0.4 m
        self.blacklist_radius = self.declare_parameter("blacklist_radius", 0.75).value
        self.goal_timeout = self.declare_parameter("goal_timeout", 180.0).value  # sim seconds
        self.period = self.declare_parameter("period", 3.0).value
        self.max_empty = self.declare_parameter("max_empty", 40).value
        self.global_frame = self.declare_parameter("global_frame", "map").value
        self.robot_frame = self.declare_parameter("robot_frame", "base_link").value

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.costmap = None
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap", self._on_costmap, qos)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.blacklist = []
        self.goal_handle = None
        self.goal_xy = None
        self.goal_sent_at = None
        self.empty = 0
        self.reached = 0
        self.failed = 0
        self.timer = self.create_timer(self.period, self._tick)
        self.get_logger().info("frontier explorer up; waiting for costmap, TF and Nav2")

    def _on_costmap(self, msg):
        self.costmap = msg

    def _robot_xy(self):
        try:
            t = self.tf.lookup_transform(self.global_frame, self.robot_frame, Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def _tick(self):
        if self.goal_handle is not None:
            elapsed = (self.get_clock().now() - self.goal_sent_at).nanoseconds * 1e-9
            if elapsed > self.goal_timeout:
                self.get_logger().warn(f"goal {self._fmt(self.goal_xy)} timed out after {elapsed:.0f} s; blacklisting")
                self.blacklist.append(self.goal_xy)
                self.failed += 1
                if not isinstance(self.goal_handle, str):
                    self.goal_handle.cancel_goal_async()
                self.goal_handle = None
            return
        if self.costmap is None or not self.nav.server_is_ready():
            return
        robot = self._robot_xy()
        if robot is None:
            return
        goal = self._pick_goal(robot)
        if goal is None:
            self.empty += 1
            self.get_logger().info(f"no frontiers ({self.empty}/{self.max_empty}); reached={self.reached} failed={self.failed}")
            if self.empty >= self.max_empty:
                self.get_logger().info("exploration finished")
                self.timer.cancel()
            return
        self.empty = 0
        self._send(goal)

    def _pick_goal(self, robot):
        m = self.costmap
        w, h, res = m.info.width, m.info.height, m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        grid = np.asarray(m.data, dtype=np.int16).reshape(h, w)
        rx, ry = int((robot[0] - ox) / res), int((robot[1] - oy) / res)
        if not (0 <= rx < w and 0 <= ry < h):
            return None

        # Flood-fill the passable region from the robot (its own cell may be inflated, so always start there).
        reach = np.zeros((h, w), bool)
        reach[ry, rx] = True
        queue = deque([(rx, ry)])
        frontier = np.zeros((h, w), bool)
        while queue:
            x, y = queue.popleft()
            for dx, dy in NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or reach[ny, nx]:
                    continue
                v = grid[ny, nx]
                if v < 0:
                    frontier[ny, nx] = True
                elif v < self.traverse_cost:
                    reach[ny, nx] = True
                    queue.append((nx, ny))

        # Cluster frontier cells (8-connected) and score them.
        best, best_score = None, 0.0
        seen = np.zeros((h, w), bool)
        for fy, fx in zip(*np.nonzero(frontier)):
            if seen[fy, fx]:
                continue
            cluster, queue = [], deque([(fx, fy)])
            seen[fy, fx] = True
            while queue:
                x, y = queue.popleft()
                cluster.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and frontier[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((nx, ny))
            if len(cluster) < self.min_cluster:
                continue
            # Goal: the reachable cell bordering the cluster closest to its centroid (centroids can sit in unknown space).
            cx = sum(c[0] for c in cluster) / len(cluster)
            cy = sum(c[1] for c in cluster) / len(cluster)
            candidates = [(x + dx, y + dy) for x, y in cluster for dx, dy in NEIGHBOURS
                          if 0 <= x + dx < w and 0 <= y + dy < h and reach[y + dy, x + dx]]
            if not candidates:
                continue
            gx, gy = min(candidates, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
            goal = (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res)
            if any(math.dist(goal, b) < self.blacklist_radius for b in self.blacklist):
                continue
            dist = max(math.dist(goal, robot), 0.5)
            if dist < 0.3:
                continue
            score = len(cluster) / dist
            if score > best_score:
                best, best_score = goal, score
        return best

    def _send(self, goal):
        msg = NavigateToPose.Goal()
        msg.pose.header.frame_id = self.global_frame
        msg.pose.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = goal
        msg.pose.pose.orientation.w = 1.0
        self.goal_xy = goal
        self.goal_sent_at = self.get_clock().now()
        self.get_logger().info(f"new frontier goal {self._fmt(goal)}")
        future = self.nav.send_goal_async(msg)
        future.add_done_callback(self._on_accepted)
        self.goal_handle = "pending"

    def _on_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn(f"goal {self._fmt(self.goal_xy)} rejected; blacklisting")
            self.blacklist.append(self.goal_xy)
            self.failed += 1
            self.goal_handle = None
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.reached += 1
            self.get_logger().info(f"reached {self._fmt(self.goal_xy)} (total {self.reached})")
        elif status != GoalStatus.STATUS_CANCELED:
            self.failed += 1
            self.blacklist.append(self.goal_xy)
            self.get_logger().warn(f"goal {self._fmt(self.goal_xy)} ended with status {status}; blacklisting")
        self.goal_handle = None

    @staticmethod
    def _fmt(xy):
        return f"({xy[0]:.2f}, {xy[1]:.2f})" if xy else "?"


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
