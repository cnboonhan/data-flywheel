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
$C run --rm ros2 python3 /capture/capture.py --viewpoints /data/run1/viewpoints.json --out /data/run1/colmap
uv run sim/splat/train.py datasets/real2sim/run1/colmap datasets/real2sim/run1/runs n_iterations=7000 --floor
```

- The COLMAP world frame is the `map` frame, so the splat is metric. Lidar points seed it. The run's `scene.usda` is the splat
  in the map frame with a collision floor, ready to open or reference in Isaac Sim.
- Train for at least 7,000 iterations; 3DGRUT's USD export fails on shorter runs.
- On a real robot, use its map and Nav2 params, and pass its topics and frames with `--cameras`, `--optical_frames` and `--lidar`.
