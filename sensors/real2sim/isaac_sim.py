"""Open an Isaac Sim scene and run it; its OmniGraphs publish ROS 2 (bundled Jazzy). Launched by run_sim.sh.

    run_sim.sh [--scene <usd>] [--gui]

Default scene: Isaac Sim's Nova Carter Nav2 sample (warehouse + ROS-wired Nova Carter: /cmd_vel in;
/chassis/odom, /tf, /front_3d_lidar/lidar_points, /clock out).
"""

import argparse
import signal

SAMPLE = "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
parser = argparse.ArgumentParser()
parser.add_argument("--scene", default=SAMPLE, help="USD path or URL (paths starting with /Isaac are on the assets root)")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--cameras", action="store_true", help="Enable the front stereo camera publishers (off in the sample)")
parser.add_argument("--camera_prefix", default="/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk")
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
omni.timeline.get_timeline_interface().play()
print(f"[sim] running {scene}; Ctrl-C to stop", flush=True)

stop = {"now": False}
signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
while app.is_running() and not stop["now"]:
    app.update()
omni.timeline.get_timeline_interface().stop()
app.close()
