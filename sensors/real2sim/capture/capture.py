"""Drive Nav2 through planned viewpoints, capture posed stereo images and lidar, write a COLMAP dataset.

    python3 capture.py --viewpoints /data/<run>/viewpoints.json --out /data/<run>/colmap

At each viewpoint: Nav2 NavigateToPose (nav2_simple_commander), wait --settle seconds (sim or robot clock), take the
next left/right images, their poses from TF (map -> camera optical frame at the image stamp) and one lidar cloud in
the map frame. Writes images/ and sparse/0/{cameras,images,points3D}.txt (COLMAP text model; world = map frame, so
the result is metric and gravity-aligned). Train with sim/splat/train.py.

Camera optical frames are parameters because Isaac Sim's Nova Carter stamps images with `<cam>_left_optical` while its
TF tree calls the optical frame `<cam>_left_rgb`.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--viewpoints", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cameras", nargs="+", default=["/front_stereo_camera/left", "/front_stereo_camera/right"],
                help="Camera namespaces with image_raw and camera_info")
ap.add_argument("--optical_frames", nargs="+", default=["front_stereo_camera_left_rgb", "front_stereo_camera_right_rgb"])
ap.add_argument("--lidar", default="/front_3d_lidar/lidar_points")
ap.add_argument("--settle", type=float, default=1.0, help="Seconds to wait after arriving (robot clock)")
ap.add_argument("--voxel", type=float, default=0.05, help="Lidar point voxel size for the initial point cloud (m)")
args = ap.parse_args()

out = Path(args.out)
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "sparse" / "0").mkdir(parents=True, exist_ok=True)

rclpy.init()
nav = BasicNavigator()
nav.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
bridge, tf = CvBridge(), tf2_ros.Buffer(Duration(seconds=60))
tf2_ros.TransformListener(tf, nav)
latest = {}
for ns in args.cameras:
    nav.create_subscription(Image, f"{ns}/image_raw", lambda m, k=ns: latest.__setitem__(k, m), 2)
    nav.create_subscription(CameraInfo, f"{ns}/camera_info", lambda m, k=ns + "#info": latest.__setitem__(k, m), 2)
nav.create_subscription(PointCloud2, args.lidar, lambda m: latest.__setitem__("lidar", m), 2)
nav.waitUntilNav2Active()


def wait_for(pred, timeout=30.0):
    end = time.time() + timeout
    while not pred() and time.time() < end:
        rclpy.spin_once(nav, timeout_sec=0.05)
    return pred()


views = json.loads(Path(args.viewpoints).read_text())["viewpoints"]
frames, clouds, intrinsics = [], [], {}
for i, v in enumerate(views):
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = nav.get_clock().now().to_msg()
    goal.pose.position.x, goal.pose.position.y = v["x"], v["y"]
    goal.pose.orientation.z, goal.pose.orientation.w = math.sin(v["yaw"] / 2), math.cos(v["yaw"] / 2)
    nav.goToPose(goal)
    while not nav.isTaskComplete():
        rclpy.spin_once(nav, timeout_sec=0.1)
    if nav.getResult() != TaskResult.SUCCEEDED:
        print(f"[{i + 1}/{len(views)}] {v}: navigation failed, skipped", flush=True)
        continue

    arrived = nav.get_clock().now()
    settled = lambda: (nav.get_clock().now() - arrived).nanoseconds * 1e-9 >= args.settle  # noqa: E731
    wait_for(settled, timeout=60)
    t0 = nav.get_clock().now().nanoseconds
    fresh = lambda k: k in latest and rclpy.time.Time.from_msg(latest[k].header.stamp).nanoseconds >= t0  # noqa: E731
    if not wait_for(lambda: all(fresh(ns) for ns in args.cameras) and fresh("lidar")):
        print(f"[{i + 1}/{len(views)}] no fresh images/lidar, skipped", flush=True)
        continue
    for ns, frame in zip(args.cameras, args.optical_frames):
        img, info = latest[ns], latest[ns + "#info"]
        try:
            T = tf.lookup_transform("map", frame, img.header.stamp, Duration(seconds=2)).transform
        except Exception as e:
            print(f"  {ns}: no TF map->{frame}: {e}", flush=True)
            continue
        name = f"{len(frames):05d}_{ns.strip('/').replace('/', '_')}.png"
        cv2.imwrite(str(out / "images" / name), bridge.imgmsg_to_cv2(img, "bgr8"))
        intrinsics[ns] = (info.width, info.height, info.k[0], info.k[4], info.k[2], info.k[5])
        frames.append((name, ns, T))
    cloud = do_transform_cloud(latest["lidar"], tf.lookup_transform("map", latest["lidar"].header.frame_id,
                                                                     latest["lidar"].header.stamp, Duration(seconds=2)))
    clouds.append(read_points_numpy(cloud, field_names=("x", "y", "z"), skip_nans=True))
    print(f"[{i + 1}/{len(views)}] captured at ({v['x']}, {v['y']}, yaw {v['yaw']})", flush=True)

# COLMAP text model. images.txt holds world-to-camera poses: R_cw = R_wc^T, t_cw = -R_cw t_wc; quaternion as qw qx qy qz.
cam_ids = {ns: i + 1 for i, ns in enumerate(intrinsics)}
with open(out / "sparse/0/cameras.txt", "w") as f:
    for ns, (w, h, fx, fy, cx, cy) in intrinsics.items():
        f.write(f"{cam_ids[ns]} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
with open(out / "sparse/0/images.txt", "w") as f:
    for i, (name, ns, T) in enumerate(frames, start=1):
        r_wc = Rotation.from_quat([T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w])
        r_cw = r_wc.inv()
        t_cw = -r_cw.apply([T.translation.x, T.translation.y, T.translation.z])
        qx, qy, qz, qw = r_cw.as_quat()
        f.write(f"{i} {qw} {qx} {qy} {qz} {t_cw[0]} {t_cw[1]} {t_cw[2]} {cam_ids[ns]} {name}\n\n")
pts = np.concatenate(clouds) if clouds else np.zeros((0, 3))
pts = np.unique(np.round(pts / args.voxel), axis=0) * args.voxel   # voxel downsample
with open(out / "sparse/0/points3D.txt", "w") as f:
    for i, (x, y, z) in enumerate(pts, start=1):
        f.write(f"{i} {x} {y} {z} 128 128 128 0\n")
print(f"wrote {len(frames)} images, {len(pts)} points -> {out}")
rclpy.shutdown()
