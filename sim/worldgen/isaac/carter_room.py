"""Nova Carter driving inside a NuRec room, publishing ROS 2 (bundled Jazzy) for the worldgen container.

Stage: the NuRec scene (Gaussian splat + collision mesh, Z-up, floor at z=0) referenced under /World/room, an
invisible physics ground plane at z=0 (the scanned floor is bumpy at the cm level), and NVIDIA's Nova Carter ROS
asset, whose OmniGraphs publish /chassis/odom, /tf, the lidars and (once enabled) the Hawk stereo cameras, and
subscribe to /cmd_vel. A clock graph publishes /clock so the container runs on sim time.

    bash sim/worldgen/isaac.sh                       # headless, runs until Ctrl-C
    bash sim/worldgen/isaac.sh --gui --seconds 120   # with the Kit window, stop after 120 s of sim time
"""

import argparse
import math
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOM = ROOT / "datasets" / "nurec-nova_carter-wormhole" / "nova_carter-wormhole"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--room", default=str(DEFAULT_ROOM / "particle_sh_optimized.usdz"), help="NuRec scene USD(Z)")
parser.add_argument("--trajectory", default=str(DEFAULT_ROOM / "training_trajectory_poses.tum"),
                    help="TUM poses of the original capture; the robot spawns at one of them (free space)")
parser.add_argument("--spawn-index", type=float, default=0.5, help="Fraction along the trajectory to spawn at")
parser.add_argument("--carter-scene", default="/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd",
                    help="USD holding NVIDIA's ROS-wired Nova Carter (relative to the Isaac assets root)")
parser.add_argument("--carter-prim", default="/World/Nova_Carter_ROS", help="Prim to reference out of --carter-scene")
parser.add_argument("--camera-hz", type=float, default=5.0, help="Publish rate of the stereo pair (omni:sensor:tickRate)")
parser.add_argument("--camera-res", default="960x600", help="Render resolution of the stereo pair, WxH (Hawk native is 1920x1200)")
parser.add_argument("--seconds", type=float, default=0.0, help="Stop after this much sim time (0 = run until Ctrl-C)")
parser.add_argument("--gui", action="store_true", help="Show the Kit window")
parser.add_argument("--cameras", action="store_true", help="Enable the front Hawk stereo pair publishers")
parser.add_argument("--ros-check", action="store_true", help="Every 15 s, list the ROS 2 topics this process sees (uses the bundled rclpy)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({
    "headless": not args.gui,
    "extra_args": ["--/crashreporter/enabled=false", "--/renderer/multiGpu/enabled=false"],
})

import omni.client  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
app.update()


def spawn_pose(tum_path: str, frac: float) -> tuple[float, float, float]:
    """(x, y, yaw) of the capture rig at `frac` of the way along the trajectory."""
    rows = [list(map(float, line.split())) for line in open(tum_path) if line.strip() and not line.startswith("#")]
    _, x, y, _, qx, qy, qz, qw = rows[min(int(frac * len(rows)), len(rows) - 1)]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, yaw


create_new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)

add_reference_to_stage(args.room, "/World/room")
GroundPlane("/World/ground", z_position=0.0, visible=False)

# Isaac Sim 6.1 ships no standalone Carter ROS asset; the warehouse navigation sample holds the wired-up robot
# (odometry, /tf, RPLidar scans, XT-32 point cloud, four Hawk stereo pairs, cmd_vel), so reference that prim.
assets_root = get_assets_root_path()
carter = assets_root + args.carter_scene
if omni.client.stat(carter)[0] != omni.client.Result.OK:
    sys.exit(f"Carter scene not found: {carter}")
x, y, yaw = spawn_pose(args.trajectory, args.spawn_index)
robot = stage.DefinePrim("/World/Carter", "Xform")
robot.GetReferences().AddReference(carter, args.carter_prim)
xf = UsdGeom.Xformable(robot)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.05))
xf.AddRotateZOp().Set(math.degrees(yaw))
print(f"[worldgen] Carter spawned at x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f} deg from {args.trajectory}")

# /clock from sim time, so Nav2/SLAM in the container can use_sim_time.
og.Controller.edit(
    {"graph_path": "/World/ClockGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("tick.outputs:tick", "clock.inputs:execIn"),
            ("sim_time.outputs:simulationTime", "clock.inputs:timeStamp"),
        ],
    },
)

# The Hawk image publishers ship disabled for performance. Enable the front pair (+ its camera_info) when asked.
FRONT_HAWK = "/World/Carter/chassis_link/sensors/front_hawk"
camera_nodes = [
    f"{FRONT_HAWK}/left/ROS_Camera_Left/left_camera_publish_image",
    f"{FRONT_HAWK}/right/ROS_Camera_Right/right_camera_publish_image",
    f"{FRONT_HAWK}/ROS_Camera_Info/ros2_camera_info_helper",
]
for node_path in camera_nodes:
    node = stage.GetPrimAtPath(node_path)
    if not node.IsValid():
        print(f"[worldgen] WARNING: camera node missing: {node_path}")
        continue
    ns = node.GetAttribute("inputs:nodeNamespace").Get() if node.HasAttribute("inputs:nodeNamespace") else None
    topic = node.GetAttribute("inputs:topicName").Get() if node.HasAttribute("inputs:topicName") else None
    was = node.GetAttribute("inputs:enabled").Get() if node.HasAttribute("inputs:enabled") else None
    if args.cameras:
        node.GetAttribute("inputs:enabled").Set(True)
    print(f"[worldgen] camera node {node_path.split('sensors/')[-1]}: ns={ns!r} topic={topic!r} enabled {was} -> {args.cameras or was}")

# Rendering two Hawk cameras at native 1920x1200 over a 1M-Gaussian splat drops the sim to ~0.2x real time.
# Lower the render-product resolution and publish at a fixed tick rate (frameSkipCount is deprecated).
cam_w, cam_h = (int(v) for v in args.camera_res.lower().split("x"))
for cam_path in (f"{FRONT_HAWK}/left/camera_left", f"{FRONT_HAWK}/right/camera_right"):
    cam = stage.GetPrimAtPath(cam_path)
    if cam.IsValid():
        attr = cam.GetAttribute("omni:sensor:tickRate") or cam.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float)
        attr.Set(float(args.camera_hz))
for node_path in camera_nodes[:2]:
    node = stage.GetPrimAtPath(node_path)
    rp = node.GetAttribute("inputs:renderProductPath").Get() if node.IsValid() and node.HasAttribute("inputs:renderProductPath") else None
    rp_prim = stage.GetPrimAtPath(str(rp)) if rp else None
    if rp_prim and rp_prim.IsValid() and rp_prim.HasAttribute("resolution"):
        rp_prim.GetAttribute("resolution").Set(Gf.Vec2i(cam_w, cam_h))
        print(f"[worldgen] render product {rp}: resolution -> {cam_w}x{cam_h}")
    else:
        print(f"[worldgen] no render product found for {node_path.split('sensors/')[-1]} (rp={rp!r}); resolution unchanged")

# Report what the asset publishes, so the ROS side can be checked against it.
topics = sorted({
    str(p.GetAttribute("inputs:topicName").Get())
    for p in stage.Traverse()
    if p.GetTypeName() == "OmniGraphNode" and p.HasAttribute("inputs:topicName") and p.GetAttribute("inputs:topicName").Get()
})
print(f"[worldgen] ROS 2 topics authored in the stage: {topics}")

ros_node = None
if args.ros_check:
    import os
    import rclpy
    print(f"[worldgen] ROS env: ROS_DISTRO={os.environ.get('ROS_DISTRO')} RMW={os.environ.get('RMW_IMPLEMENTATION')} "
          f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')} profile={os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE')}")
    rclpy.init()
    ros_node = rclpy.create_node("worldgen_ros_check")

timeline = omni.timeline.get_timeline_interface()
timeline.play()
last_check = -1.0
stop = {"now": False}
signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
print("[worldgen] running; Ctrl-C to stop")
while app.is_running() and not stop["now"]:
    app.update()
    t = timeline.get_current_time()
    if ros_node is not None and t - last_check >= 15.0:
        last_check = t
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        names = sorted(n for n, _ in ros_node.get_topic_names_and_types())
        print(f"[worldgen] t={t:5.1f}s topics seen by this process: {names}", flush=True)
    if args.seconds and t >= args.seconds:
        break
timeline.stop()
app.close()
