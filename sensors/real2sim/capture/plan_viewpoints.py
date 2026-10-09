"""Camera viewpoints from the ROS 2 map: a grid over the free space the robot can reach, a few headings each.

    python3 plan_viewpoints.py --out /data/<run>/viewpoints.json [--spacing 1.5 --headings 4 --region X0 Y0 X1 Y1]

Reads /map (nav_msgs/OccupancyGrid) and the robot pose (TF map -> base_link), keeps grid points at least
--clearance from obstacles (scipy distance transform), and orders them as a nearest-neighbour tour from the robot.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.ndimage import distance_transform_edt

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--spacing", type=float, default=1.5, help="Grid spacing (m)")
ap.add_argument("--headings", type=int, default=4, help="Headings per position")
ap.add_argument("--clearance", type=float, default=0.8, help="Min distance to obstacles and unknown space (m)")
ap.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Map-frame box to cover (m)")
args = ap.parse_args()

rclpy.init()
nav = BasicNavigator()   # a node with sim-time support; used here for /map and the robot pose
maps = []
nav.create_subscription(OccupancyGrid, "/map", maps.append, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
while not maps:
    rclpy.spin_once(nav, timeout_sec=0.5)
m = maps[0].info
grid = np.asarray(maps[0].data).reshape(m.height, m.width)
clear = distance_transform_edt(grid == 0) * m.resolution >= args.clearance

xs = m.origin.position.x + (np.arange(m.width) + 0.5) * m.resolution
ys = m.origin.position.y + (np.arange(m.height) + 0.5) * m.resolution
step = max(1, round(args.spacing / m.resolution))
points = [(xs[c], ys[r]) for r in range(0, m.height, step) for c in range(0, m.width, step) if clear[r, c]]
if args.region:
    x0, y0, x1, y1 = args.region
    points = [p for p in points if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]

# Nearest-neighbour tour from the robot.
pos = (0.0, 0.0)
try:
    import tf2_ros
    tf = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf, nav)
    for _ in range(20):
        rclpy.spin_once(nav, timeout_sec=0.2)
        if tf.can_transform("map", "base_link", rclpy.time.Time()):
            t = tf.lookup_transform("map", "base_link", rclpy.time.Time()).transform.translation
            pos = (t.x, t.y)
            break
except Exception:
    pass
tour = []
while points:
    nxt = min(points, key=lambda p: math.dist(pos, p))
    points.remove(nxt)
    tour.append(nxt)
    pos = nxt

views = [{"x": round(float(x), 3), "y": round(float(y), 3), "yaw": round(2 * math.pi * h / args.headings, 4)}
         for x, y in tour for h in range(args.headings)]
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps({"frame_id": "map", "viewpoints": views}, indent=1))
print(f"{len(tour)} positions x {args.headings} headings = {len(views)} viewpoints -> {args.out}")
rclpy.shutdown()
