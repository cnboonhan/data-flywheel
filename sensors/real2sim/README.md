# real2sim

Isaac Sim's Nova Carter warehouse plus Nav2 on its known map, behind the same ROS 2 interface as a real robot; `capture/` drives Nav2 to viewpoints planned from `/map` and writes a COLMAP dataset for [`sim/splat`](../../sim/splat/README.md). Isaac Sim runs from the IsaacLab-Arena venv; Nav2 and NVIDIA's `carter_navigation` run in Docker on `ROS_DOMAIN_ID=42`, away from the real robot's domain 0 (`fastdds.xml` makes DDS use UDP only).

1. Set up the shell (`datasets/real2sim` is `/data` in the containers).
   ```bash
   export UID GID=$(id -g); mkdir -p datasets/real2sim
   C="docker compose -f sensors/real2sim/compose.yml"
   ```
2. Start the sim (add `--cameras` for capture, `--gui` for a window) and wait for `[sim] running`.
   ```bash
   bash sensors/real2sim/run_sim.sh --cameras
   ```
3. Start Nav2 (RViz on `$DISPLAY`, else on Xvfb) and send a goal on `/goal_pose` in the map frame.
   ```bash
   $C up -d nav2
   $C run --rm ros2 ros2 topic pub --once -w 1 /goal_pose geometry_msgs/msg/PoseStamped \
     "{header: {frame_id: map}, pose: {position: {x: 0.0, y: -8.0}, orientation: {w: 1.0}}}"
   ```
4. Plan viewpoints and capture (optional: `--region X0 Y0 X1 Y1`, `--height_range 0.35 1.2 --height_steps 3`, `--pitches 0 30`).
   ```bash
   $C run --rm ros2 python3 /capture/plan_viewpoints.py --out /data/run1/viewpoints.json --spacing 1.0 --headings 8
   $C run --rm ros2 python3 /capture/capture.py --viewpoints /data/run1/viewpoints.json --out /data/run1/colmap
   ```
5. Train the splat (at least 7,000 iterations; 3DGRUT's USD export fails on shorter runs), or upload the capture and train on the cluster: [sim/splat](../../sim/splat/README.md#with-the-s3-gateway).
   ```bash
   uv run sim/splat/train.py datasets/real2sim/run1/colmap datasets/real2sim/run1/runs n_iterations=7000 --floor
   ```

## Notes

- **Frame:** the COLMAP world is the `map` frame, so the splat is metric and gravity-aligned; lidar points seed it, and `scene.usda` is the splat in the map frame with a collision floor.
- **Height and pitch:** with `--height_steps N`, each viewpoint is captured at N heights in `--height_range`, sent on `/camera_height` (`std_msgs/Float64`, m) and confirmed from TF. `--pitches` tilts the camera at each height (`/camera_pitch`, degrees, positive looks down). In sim the stereo rig follows; on a real robot an adapter turns them into lift/torso and head commands. Check the images after a pitched capture: pitch hasn't been through a full capture yet.
- **Skipped viewpoints:** a failed goal is retried after clearing the costmaps (`--nav_retries`, default 2). If a viewpoint is still skipped, the model is written but `capture.py` exits nonzero and lists the gaps; the collision floor only covers the captured footprint plus 2 m.
- **Coverage:** use `--spacing 1.0 --headings 8` so there are views between positions, not only at them; sparse grids (a few positions x 4 headings) train noticeably worse. Expect ~1000 images for 5 x 5 m.
- **Known limitation, lens and poses:** `capture.py` writes a `PINHOLE` camera and drops `CameraInfo`'s distortion, but the Hawk stereo images are barrel-distorted, and poses come from AMCL localisation rather than the sim's ground truth. Splats from current captures are therefore blurred (~18 dB held out, against ~27 dB for well-posed photos). Until `capture.py` records the distortion (as `OPENCV`/`FULL_OPENCV`/`OPENCV_FISHEYE`) and ground-truth poses, treat these splats as previews.
- **Real robot:** use its map and Nav2 params, and pass its topics and frames with `--cameras`, `--optical_frames` and `--lidar`.
- **Run it on an RTX GPU, not the GB300 nodes.** On the GB300s every Isaac Sim 6.1 frame fails in the RTX renderer (`NGX CreateFeature failed`, `DLSS RenderOp failed`), so the sim never advances and publishes nothing. Train the splat on the cluster instead ([sim/splat](../../sim/splat/README.md#on-the-slurm-cluster)).
