# real2sim

Isaac Sim's Nova Carter warehouse plus Nav2 on its known map, behind the same ROS 2 interface as a real robot. Goals go to `/goal_pose` (map frame). `capture/` drives Nav2 to viewpoints planned from `/map` and writes a COLMAP dataset for [`sim/splat`](../../sim/splat/README.md).

Isaac Sim runs from the IsaacLab-Arena venv (`run_sim.sh`). Nav2 and NVIDIA's `carter_navigation` run in Docker (`compose.yml`) on `ROS_DOMAIN_ID=42`, away from the real robot's domain 0. `fastdds.xml` makes DDS use UDP only.

```bash
export UID GID=$(id -g); mkdir -p datasets/real2sim      # /data in the containers
C="docker compose -f sensors/real2sim/compose.yml"

bash sensors/real2sim/run_sim.sh [--gui] [--cameras]     # wait for "[sim] running"
$C up -d nav2                                            # RViz on $DISPLAY, else on Xvfb
$C run --rm ros2 ros2 topic pub --once -w 1 /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 0.0, y: -8.0}, orientation: {w: 1.0}}}"

# Splat capture (sim started with --cameras)
$C run --rm ros2 python3 /capture/plan_viewpoints.py --out /data/run1/viewpoints.json [--region X0 Y0 X1 Y1]
$C run --rm ros2 python3 /capture/capture.py --viewpoints /data/run1/viewpoints.json --out /data/run1/colmap \
  [--height_range 0.35 1.2 --height_steps 3] [--pitches 0 30]
uv run sim/splat/train.py datasets/real2sim/run1/colmap datasets/real2sim/run1/runs n_iterations=7000 --floor
```

- The COLMAP world frame is the `map` frame, so the splat is metric. Lidar points seed it. The run's `scene.usda` is the splat
  in the map frame with a collision floor, ready to open or reference in Isaac Sim.
- Camera height: with `--height_steps N`, each viewpoint is captured at N heights within `--height_range` (the robot's
  achievable camera heights above the floor), sent on `/camera_height` (`std_msgs/Float64`, m) and confirmed from TF. In sim
  the stereo rig follows the topic; on a real robot, an adapter turns it into a lift or torso command. `--pitches`
  likewise tilts the camera at each height (`/camera_pitch`, degrees, positive looks down; a head tilt on a real robot),
  so the floor is seen from above, as an arm camera sees it. Pitch is not yet verified in a full capture.
- A failed goal is retried after clearing the costmaps (`--nav_retries`, default 2). If a viewpoint is still skipped, the
  model is written but `capture.py` exits nonzero and lists the gaps: the splat's collision floor only covers the captured
  footprint (plus 2 m).
- Coverage: `--spacing 1.0 --headings 8` in `plan_viewpoints.py` captures views between the positions, not only at them.
  Over a 5 x 5 m region (1008 images), it scored 18.2 dB on held-out views across the area, where a 5-position,
  4-heading capture scored 15.5 dB.
- Train for at least 7,000 iterations; 3DGRUT's USD export fails on shorter runs.
- On a real robot, use its map and Nav2 params, and pass its topics and frames with `--cameras`, `--optical_frames` and `--lidar`.
- **Not on GB300 (aarch64 Slurm nodes) yet.** Isaac Sim 6.1 starts and the ROS 2 bridge loads, but every frame fails in the RTX renderer (`NGX CreateFeature failed`, `DLSS RenderOp failed`, `Rendering failed`) and the simulation stops advancing once playing, so nothing is published. Switching to `RaytracedLighting`, FXAA and `/rtx-transient/post/dlss/supported=false` didn't help. Run it on an RTX GPU.
