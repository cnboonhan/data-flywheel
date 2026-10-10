"""Open an Isaac Sim scene and run it; its OmniGraphs publish ROS 2 (bundled Jazzy). Launched by run_sim.sh.

    run_sim.sh [--scene <usd>] [--gui] [--cameras]

Default scene: Isaac Sim's Nova Carter Nav2 sample (warehouse + ROS-wired Nova Carter: /cmd_vel in;
/chassis/odom, /tf, /front_3d_lidar/lidar_points, /clock out).

With --cameras the front stereo publishers are enabled, and the stereo rig follows /camera_height (std_msgs/Float64,
camera height above the floor in metres) and /camera_pitch (std_msgs/Float64, degrees, positive looks down): the sim
stand-in for a real robot's lift or torso and head tilt. The rig pitches about its first camera, so pitch leaves the
camera height unchanged. The camera TF follows.
"""

import argparse
import signal

SAMPLE = "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
parser = argparse.ArgumentParser()
parser.add_argument("--scene", default=SAMPLE, help="USD path or URL (paths starting with /Isaac are on the assets root)")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--cameras", action="store_true", help="Enable the front stereo camera publishers (off in the sample)")
parser.add_argument("--camera_prefix", default="/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk")
parser.add_argument("--camera_height_topic", default="/camera_height")
parser.add_argument("--camera_pitch_topic", default="/camera_pitch")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui,
                     "extra_args": ["--/crashreporter/enabled=false", "--/renderer/multiGpu/enabled=false"]})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
app.update()

scene = get_assets_root_path() + args.scene if args.scene.startswith("/Isaac/") else args.scene
omni.usd.get_context().open_stage(scene)
while omni.usd.get_context().get_stage_loading_status()[2] > 0:
    app.update()
if args.cameras:
    stage = omni.usd.get_context().get_stage()
    for node in ("left/ROS_Camera_Left/left_camera_publish_image", "right/ROS_Camera_Right/right_camera_publish_image",
                 "ROS_Camera_Info/ros2_camera_info_helper"):
        prim = stage.GetPrimAtPath(f"{args.camera_prefix}/{node}")
        if not prim.IsValid():
            raise SystemExit(f"--cameras: no camera node at {prim.GetPath()}")
        prim.GetAttribute("inputs:enabled").Set(True)
    print("[sim] front stereo camera publishers enabled", flush=True)
pending = {}
if args.cameras:
    import rclpy
    from pxr import Gf, Usd, UsdGeom
    from std_msgs.msg import Float64

    rig = stage.GetPrimAtPath(args.camera_prefix)
    cam = next(p for p in Usd.PrimRange(rig) if p.IsA(UsdGeom.Camera))
    rig_xf = UsdGeom.Xformable(rig)
    local0 = rig_xf.GetLocalTransformation()
    cam_world = lambda: UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # noqa: E731
    # The first camera's position in the rig's parent (chassis) frame: the pitch pivot.
    pivot = rig_xf.ComputeParentToWorldTransform(Usd.TimeCode.Default()).GetInverse().Transform(cam_world().ExtractTranslation())
    pose = {"dz": 0.0, "pitch": 0.0}

    def apply_pose() -> None:
        """Rig = its original local transform, pitched about the pivot (chassis y), raised by dz (row-vector order)."""
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), pose["pitch"]))
        local = (local0 * Gf.Matrix4d().SetTranslate(-pivot) * rot * Gf.Matrix4d().SetTranslate(pivot)
                 * Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, pose["dz"])))
        rig_xf.MakeMatrixXform().Set(local)

    def set_height(target: float) -> None:
        """Raise or lower the rig so the first camera under it sits at `target` above z = 0."""
        pose["dz"] += target - cam_world().ExtractTranslation()[2]   # world z = chassis z (the chassis is upright)
        apply_pose()
        print(f"[sim] camera height -> {target:.3f} m", flush=True)

    def set_pitch(degrees: float) -> None:
        pose["pitch"] = degrees
        apply_pose()
        print(f"[sim] camera pitch -> {degrees:.1f} deg", flush=True)

    rclpy.init()
    node = rclpy.create_node("camera_pose")
    node.create_subscription(Float64, args.camera_height_topic, lambda m: pending.__setitem__("height", m.data), 1)
    node.create_subscription(Float64, args.camera_pitch_topic, lambda m: pending.__setitem__("pitch", m.data), 1)
omni.timeline.get_timeline_interface().play()
print(f"[sim] running {scene}; Ctrl-C to stop", flush=True)

stop = {"now": False}
signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
while app.is_running() and not stop["now"]:
    app.update()
    if args.cameras:
        rclpy.spin_once(node, timeout_sec=0)
        if "pitch" in pending:
            set_pitch(pending.pop("pitch"))
        if "height" in pending:
            set_height(pending.pop("height"))
omni.timeline.get_timeline_interface().stop()
app.close()
