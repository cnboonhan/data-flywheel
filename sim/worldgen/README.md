# sim/worldgen

Prototype of the world-generation loop from [`docs/worldgen.md`](../../docs/worldgen.md): a robot explores a space autonomously while recording a stereo bag, NuRec turns the bag into a sim-ready scene, and the scene comes back into IsaacLab-Arena. This is the **sim→sim** version: the robot is NVIDIA's Nova Carter driving inside the NuRec `nova_carter-wormhole` room (the sample scene from [`../nurec`](../nurec/README.md)), so the output can be compared against NVIDIA's own reconstruction of the same room from the same robot.

```
 Isaac Sim 6.1 (host, bundled ROS 2 Jazzy)          worldgen container (ROS 2 Jazzy, host network)
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────────┐
 │ NuRec room (splat + collision mesh)  │ /tf /clock │ slam_toolbox  → /map, map→odom           │
 │ Nova Carter ROS asset:               │ /chassis/odom          │                               │
 │   XT-32 RTX lidar → lidar_points  ───┼──────────▶ pointcloud_to_laserscan → /scan            │
 │   front Hawk stereo → image_raw ×2   │            │ Nav2 (navigation_launch, Carter params)   │
 │   odometry, TF, IMUs                 │ ◀──────────┼ explore_lite → goals → /cmd_vel           │
 │   /cmd_vel → differential drive      │  /cmd_vel  │ ros2 bag record (mcap) → runs/<name>/bag  │
 └──────────────────────────────────────┘            └──────────────────────────────────────────┘
```

Why Carter rather than the R1 Lite the proposal starts from: Isaac Sim ships Carter already wired for ROS 2 (sensors, odometry, TF, drive), NVIDIA's Nav2 parameters for it are in [`IsaacSim-ros_workspaces`](IsaacSim-ros_workspaces/) (submodule), and the NuRec sample room was captured by a Carter. The R1 becomes a second embodiment once the loop works.

| Path | Purpose |
|---|---|
| `isaac.sh`, `isaac/carter_room.py` | Isaac Sim side: builds the stage (room, hidden ground plane, Carter referenced from NVIDIA's sample, `/clock` graph), enables the front stereo publishers, runs until Ctrl-C |
| `nav.sh`, `compose.yml`, `docker/` | The ROS 2 Jazzy container: Nav2, slam_toolbox, `explore_lite` (built from source; no Jazzy apt package), rosbag2 mcap |
| `ros/explore.launch.py` | SLAM + Nav2 + frontier exploration, all on sim time; `explore:=false` for SLAM + Nav2 only |
| `ros/nav2_params.yaml` | NVIDIA's Carter Nav2 params minus map server/AMCL (SLAM provides the map), 2D sources on `/scan` |
| `ros/slam_params.yaml`, `ros/explore_params.yaml` | slam_toolbox (async, mapping) and explore_lite settings |
| `ros/record.sh` | Records the capture bag NuRec's stereo workflow needs |
| `ros/fastdds.xml` | UDP-only Fast DDS profile so host and container discover each other |
| `runs/` | Bags, maps, logs (gitignored) |

## Run

```bash
bash sim/worldgen/nav.sh build                    # once; ~5 GB image
bash sim/worldgen/isaac.sh                        # terminal 1: Isaac Sim, headless (add --gui to watch)
bash sim/worldgen/nav.sh up explore               # terminal 2: SLAM + Nav2 + exploration
bash sim/worldgen/nav.sh run --rm nav /worldgen/record.sh <name>   # terminal 3: bag → runs/<name>/bag
bash sim/worldgen/nav.sh run --rm nav ros2 run nav2_map_server map_saver_cli -f /runs/<name>/map --ros-args -p use_sim_time:=true
```

`nav.sh` is `docker compose -f sim/worldgen/compose.yml` run as your user (so `runs/` stays yours). `nav.sh run --rm nav` gives a shell with ROS sourced for `ros2 topic list`, `tf2_echo`, etc. Isaac Sim takes about 90 s to come up (longer on the first start of the day).

## How the pieces fit

- **No ROS on the host.** Isaac Sim 6.1 bundles ROS 2 Jazzy (`isaacsim.ros2.core/jazzy`); `isaac.sh` sets `ROS_DISTRO`, `RMW_IMPLEMENTATION` and `LD_LIBRARY_PATH` to it. Without the library path the bridge logs "ROS2 Bridge startup failed" and publishes nothing. The bundled tree includes `rclpy`, which `carter_room.py --ros-check` uses to list the topics the sim process itself sees.
- **DDS across the container boundary.** The container shares the host network and IPC namespace, and both sides load `ros/fastdds.xml` (UDP only, no shared memory). Verified in both directions with a ping/pong test.
- **`ROS_DOMAIN_ID=42`** on both sides. This LAN also carries a real Galaxea R1 on domain 0 (its `/hdas/*` topics show up in a domain-0 `ros2 topic list`), so the sim keeps off it.
- **Carter comes from the warehouse sample.** Isaac Sim 6.1 has no standalone Carter ROS asset; `carter_room.py` references `/World/Nova_Carter_ROS` out of `Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd`. It spawns at a pose from the room's capture trajectory (free space by construction; the room origin is under a table).
- **Lidar sees the room.** The XT-32 RTX lidar returns ~43k points per sweep against the NuRec collision mesh; `pointcloud_to_laserscan` turns it into a 720-beam `/scan`. The sample's 2D RPLidar graphs publish nothing, so every 2D consumer points at `/scan`.
- **Sim time everywhere.** The stage has a `/clock` graph; all ROS nodes run with `use_sim_time`.

## Gotchas

- **slam_toolbox is a lifecycle node in Jazzy.** Launched as a plain `Node` it stays unconfigured and silent: no `/map`, no `map→odom`, Nav2's costmaps wait forever. Include its `online_async_launch.py` (`autostart:=true`), which emits the configure/activate transitions.
- **Scan geometry must divide exactly.** `(angle_max - angle_min) / angle_increment` has to be an integer or Karto rejects every scan ("contains N range readings, expected M"). The launch uses 720 beams of 0.5°.
- **Speed.** Rendering two Hawk cameras over a 1M-Gaussian splat plus the RTX lidar runs the sim at roughly 0.4× real time on the RTX 5090 Laptop (odometry 24 Hz, lidar 3.7 Hz, images ~1 Hz with the 5 Hz tick rate). Everything is on sim time so the stack copes, but exploration takes correspondingly longer in wall time. Lowering the camera render resolution would help; the helper creates its render product at runtime, so it has to be resized after the first frame.
- **Don't `pkill -f carter_room.py` from a script**: the pattern matches the calling shell. Use `pgrep -f "python.*[c]arter_room.py"`.
- **Headless CUDA installers** (3DGRUT) and Kit both misbehave with a stale `DISPLAY`; `isaac.sh` runs fine with it set, the 3DGRUT install does not (see `../nurec`).

## Status (2026-10-08)

**Autonomous exploration works.** From a fresh spawn, 30 minutes of frontier exploration (`runs/explore8`) reached 3 frontier goals, drove ~13 m and produced a closed SLAM map of the room: outer corridor (~27 × 17 m), the long room along the top and the row of alcoves, at 5 cm. Nav2 came up cleanly; 9 of 12 goals failed and were blacklisted, mostly when recoveries stopped on "Collision Ahead", and the run ended after 40 empty frontier searches with the unexplored right half beyond a doorway the explorer could not reach.

What it took, in order of discovery:
- slam_toolbox via its lifecycle-aware launch; 720-beam `/scan`.
- **Our own `ros/frontier_explorer.py` instead of explore_lite**, which quit after one empty search while the costmap had hundreds of reachable frontier cells. It flood-fills from the robot through non-lethal cells, clusters frontier cells, sends Nav2 to the best cluster and blacklists failures.
- Global costmap tracks unknown space (otherwise no frontiers exist).
- Inflation tightened (padding 0.05, radius 0.55): NVIDIA's 0.25 m padding turned every stray scan point into a lethal disc.
- 3D-lidar obstacles start 0.10 m above the floor and the cloud→scan min height skips floor noise; scanned-floor bumps otherwise box the robot in.
- Nav2 starts only after `ros/wait_for_tf.py` sees odom→base_link and map→odom, with `initial_transform_timeout` effectively unbounded: it is measured on sim time, and a node that starts its timer before its first `/clock` sees time jump from 0 to the current sim time and aborts the bringup.

Next: record a bag during exploration (`record.sh`) and run NuRec's stereo workflow on it; compare this SLAM map with NVIDIA's `occupancy_map.png` for the same room; reduce recovery failures (collision_monitor/behavior settings) so exploration reaches the right half.

## NuRec stereo reconstruction (in progress)

`nurec_stereo/` builds NVIDIA's stereo workflow toolchain into one image (`flywheel-nurec-stereo`, on the Isaac ROS 5.0 / ROS 2 Lyrical dev image, pulled anonymously from NGC): `isaac_mapping_ros` (`rosbag_to_mapping_data`, FoundationStereo offline), pyCuSFM (`cusfm_cli`) and nvblox (`fuse_cusfm`, built for `sm_120`). Build it with `docker build -t flywheel-nurec-stereo sim/worldgen/nurec_stereo`. The FoundationStereo engine is built at run time with `--gpus all` (the EULA was accepted on 2026-10-08) and cached in `runs/nurec_assets`. CUDA 13.2 in the container runs on the host's 580 driver.

Status: a 10-minute capture (`runs/capture1`, 3,091 stereo pairs) is recorded. Getting it into `rosbag_to_mapping_data` took two fixes, and the conversion itself was stopped before it finished, so cuSFM, depth, nvblox and 3DGRUT haven't run yet.
- **Compression:** record with mcap chunk compression (`--storage-preset-profile zstd_fast`, now in `record.sh`). With rosbag2 per-message compression, the converter fails to deserialize `/tf`.
- **Camera frames:** the Carter asset stamps camera data with `<cam>_left_optical`, but its TF only has `<cam>_left_rgb`, which is the optical frame. `nurec_stereo/fix_bag.py` copies the bag with identity `/tf_static` aliases between the two.

Next: `rosbag_to_mapping_data --sensor_data_bag_file=<bag_nurec> --pose_bag_file=<bag_nurec> --pose_topic_name=/chassis/odom --camera_topic_config=/cfg/topic_config_carter.yaml ...`, then `cusfm_cli`, `run_foundationstereo_trt_offline.py`, `fuse_cusfm`, and `sim/nurec/train.sh` with `apps/cusfm_3dgut.yaml`.

