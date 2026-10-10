"""Drive Nav2 through planned viewpoints, capture posed stereo images and lidar, write a COLMAP dataset.

    python3 capture.py --viewpoints /data/<run>/viewpoints.json --out /data/<run>/colmap \
        [--height_range 0.35 1.2 --height_steps 3] [--pitches 0 30]

At each viewpoint: Nav2 NavigateToPose (nav2_simple_commander), wait --settle seconds (sim or robot clock), take the
next left/right images, their poses from TF (map -> camera optical frame at the image stamp) and one lidar cloud in
the map frame. Each camera's lens comes from its CameraInfo: no distortion -> PINHOLE, plumb_bob or
rational_polynomial -> OPENCV or FULL_OPENCV, equidistant -> OPENCV_FISHEYE. Writes images/ and sparse/0/{cameras,images,points3D}.txt (COLMAP text model; world = map frame, so
the result is metric and gravity-aligned). Train with sim/splat/train.py.

Camera heights: with --height_steps N > 1, each viewpoint is captured at N camera heights spread over --height_range
(the robot's achievable camera heights above the floor). Each height is published on --height_topic
(std_msgs/Float64, metres) and accepted once TF puts the first optical frame within --height_tolerance of it; the
camera returns to the lowest height before driving on. In sim, isaac_sim.py moves the stereo rig; on a real robot, an
adapter turns the topic into a lift or torso command.

Camera pitch: with --pitches, each height is captured at each pitch (degrees, positive looks down), published on
--pitch_topic (std_msgs/Float64) and accepted once TF tilts the first optical frame within --pitch_tolerance of it;
the camera returns to the first pitch before driving on. In sim, isaac_sim.py pitches the stereo rig; on a real robot,
an adapter turns the topic into a head tilt command.

Camera optical frames are parameters because Isaac Sim's Nova Carter stamps images with `<cam>_left_optical` while its
TF tree calls the optical frame `<cam>_left_rgb`.
"""

import argparse
import json
import math
import sys
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
from std_msgs.msg import Float64
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
ap.add_argument("--height_range", type=float, nargs=2, metavar=("MIN", "MAX"),
                help="Robot's achievable camera heights above the floor (m)")
ap.add_argument("--height_steps", type=int, default=1, help="Heights per viewpoint (1: leave the camera where it is)")
ap.add_argument("--height_topic", default="/camera_height")
ap.add_argument("--height_tolerance", type=float, default=0.02, help="(m)")
ap.add_argument("--pitches", type=float, nargs="+", help="Camera pitches per height (deg, positive looks down)")
ap.add_argument("--pitch_topic", default="/camera_pitch")
ap.add_argument("--pitch_tolerance", type=float, default=1.0, help="(deg)")
ap.add_argument("--nav_retries", type=int, default=2, help="Retries of a failed goal, each after clearing the costmaps")
args = ap.parse_args()
if args.height_steps > 1 and not args.height_range:
    ap.error("--height_steps > 1 needs --height_range")
heights = list(np.linspace(*args.height_range, args.height_steps)) if args.height_steps > 1 else [None]
pitches = args.pitches or [None]

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
height_pub = nav.create_publisher(Float64, args.height_topic, 1)
pitch_pub = nav.create_publisher(Float64, args.pitch_topic, 1)
nav.waitUntilNav2Active()


def wait_for(pred, timeout=30.0):
    end = time.time() + timeout
    while not pred() and time.time() < end:
        rclpy.spin_once(nav, timeout_sec=0.05)
    return pred()


def camera_z():
    try:
        return tf.lookup_transform("map", args.optical_frames[0], rclpy.time.Time()).transform.translation.z
    except Exception:
        return None


def go_to_height(h):
    """Command a camera height and wait until TF shows it; True when reached."""
    height_pub.publish(Float64(data=float(h)))
    return wait_for(lambda: (z := camera_z()) is not None and abs(z - h) <= args.height_tolerance, timeout=30)


def camera_pitch():
    """Downward tilt (deg) of the first optical frame's viewing axis (+z) in the map frame."""
    try:
        q = tf.lookup_transform("map", args.optical_frames[0], rclpy.time.Time()).transform.rotation
    except Exception:
        return None
    forward_z = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply([0, 0, 1])[2]
    return math.degrees(math.asin(-forward_z))


def go_to_pitch(p):
    """Command a camera pitch and wait until TF shows it; True when reached."""
    pitch_pub.publish(Float64(data=float(p)))
    return wait_for(lambda: (q := camera_pitch()) is not None and abs(q - p) <= args.pitch_tolerance, timeout=30)


def colmap_camera(info):
    """COLMAP camera model and parameters for a CameraInfo's lens (OpenCV conventions on both sides)."""
    k, d, m = info.k, [float(x) for x in info.d], info.distortion_model
    head = f"{info.width} {info.height} {k[0]} {k[4]} {k[2]} {k[5]}"
    if not any(d):
        return f"PINHOLE {head}"
    if m == "equidistant":                                     # k1 k2 k3 k4
        return f"OPENCV_FISHEYE {head} " + " ".join(map(str, (d + [0.0] * 4)[:4]))
    if m in ("plumb_bob", "rational_polynomial"):             # k1 k2 p1 p2 [k3 [k4 k5 k6]]
        d = (d + [0.0] * 8)[:8]
        if not any(d[4:]):
            return f"OPENCV {head} " + " ".join(map(str, d[:4]))
        return f"FULL_OPENCV {head} " + " ".join(map(str, d))
    sys.exit(f"unsupported CameraInfo distortion_model {m!r}")


def capture(i, h, p=None):
    """Settle, then save the next stereo pair (posed by TF) and one lidar cloud in the map frame."""
    v = views[i]
    arrived = nav.get_clock().now()
    settled = lambda: (nav.get_clock().now() - arrived).nanoseconds * 1e-9 >= args.settle  # noqa: E731
    wait_for(settled, timeout=60)
    t0 = nav.get_clock().now().nanoseconds
    fresh = lambda k: k in latest and rclpy.time.Time.from_msg(latest[k].header.stamp).nanoseconds >= t0  # noqa: E731
    if not wait_for(lambda: all(fresh(ns) for ns in args.cameras) and fresh("lidar")):
        print(f"[{i + 1}/{len(views)}] no fresh images/lidar, skipped", flush=True)
        skipped.append(i + 1)
        return
    for ns, frame in zip(args.cameras, args.optical_frames):
        img, info = latest[ns], latest[ns + "#info"]
        try:
            T = tf.lookup_transform("map", frame, img.header.stamp, Duration(seconds=2)).transform
        except Exception as e:
            print(f"  {ns}: no TF map->{frame}: {e}", flush=True)
            skipped.append(i + 1)
            return
        name = f"{len(frames):05d}_{ns.strip('/').replace('/', '_')}.png"
        cv2.imwrite(str(out / "images" / name), bridge.imgmsg_to_cv2(img, "bgr8"))
        intrinsics[ns] = colmap_camera(info)
        frames.append((name, ns, T))
    cloud = do_transform_cloud(latest["lidar"], tf.lookup_transform("map", latest["lidar"].header.frame_id,
                                                                     latest["lidar"].header.stamp, Duration(seconds=2)))
    clouds.append(read_points_numpy(cloud, field_names=("x", "y", "z"), skip_nans=True))
    at = (f", camera {h:.2f} m" if h is not None else "") + (f", pitch {p:g} deg" if p is not None else "")
    print(f"[{i + 1}/{len(views)}] captured at ({v['x']}, {v['y']}, yaw {v['yaw']}{at})", flush=True)


views = json.loads(Path(args.viewpoints).read_text())["viewpoints"]
frames, clouds, intrinsics, skipped = [], [], {}, []
for i, v in enumerate(views):
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = nav.get_clock().now().to_msg()
    goal.pose.position.x, goal.pose.position.y = v["x"], v["y"]
    goal.pose.orientation.z, goal.pose.orientation.w = math.sin(v["yaw"] / 2), math.cos(v["yaw"] / 2)
    for attempt in range(args.nav_retries + 1):
        if attempt:
            print(f"[{i + 1}/{len(views)}] navigation failed, clearing costmaps and retrying", flush=True)
            nav.clearAllCostmaps()
        nav.goToPose(goal)
        while not nav.isTaskComplete():
            rclpy.spin_once(nav, timeout_sec=0.1)
        if nav.getResult() == TaskResult.SUCCEEDED:
            break
    else:
        print(f"[{i + 1}/{len(views)}] {v}: navigation failed, skipped", flush=True)
        skipped.append(i + 1)
        continue

    for h in heights:
        if h is not None and not go_to_height(h):
            print(f"[{i + 1}/{len(views)}] camera height {h:.2f} m not reached, skipped", flush=True)
            skipped.append(i + 1)
            continue
        for p in pitches:
            if p is not None and not go_to_pitch(p):
                print(f"[{i + 1}/{len(views)}] camera pitch {p:g} deg not reached, skipped", flush=True)
                skipped.append(i + 1)
                continue
            capture(i, h, p)
        if pitches[0] is not None:
            go_to_pitch(pitches[0])
    if heights[0] is not None:
        go_to_height(heights[0])   # drive with the camera lowered

# COLMAP text model. images.txt holds world-to-camera poses: R_cw = R_wc^T, t_cw = -R_cw t_wc; quaternion as qw qx qy qz.
cam_ids = {ns: i + 1 for i, ns in enumerate(intrinsics)}
with open(out / "sparse/0/cameras.txt", "w") as f:
    for ns, camera in intrinsics.items():
        f.write(f"{cam_ids[ns]} {camera}\n")
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
if skipped:   # the model is still written; a nonzero exit lets scripts notice the gaps
    sys.exit(f"skipped captures at viewpoints {sorted(set(skipped))} of {len(views)}")
