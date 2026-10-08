#!/usr/bin/env python3
"""Block until every given transform resolves on /tf (sim time), then exit 0. Used to gate Nav2 startup:
its costmaps abort the whole bringup if odom->base_link or map->odom is missing when they activate.

    python3 wait_for_tf.py odom:base_link map:odom [--timeout 600]
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    timeout = float(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 600.0
    pairs = [tuple(a.split(":")) for a in args if ":" in a]
    rclpy.init()
    node = Node("wait_for_tf", parameter_overrides=[Parameter("use_sim_time", value=True)])
    buf = Buffer()
    TransformListener(buf, node)
    start, pending = time.time(), list(pairs)
    while pending and time.time() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        pending = [p for p in pending if not buf.can_transform(p[0], p[1], Time())]
    ok = not pending
    node.get_logger().info(f"transforms {'ready' if ok else 'still missing: ' + str(pending)} after {time.time() - start:.0f} s")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
