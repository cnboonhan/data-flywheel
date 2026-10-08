# World generation: the physical-scene ↔ sim-scene loop

A second flywheel next to the data one, for the diagram's "Sync Specs" edge between Physical Scenes and Sim Scenes: a robot maps and explores a real space, the capture becomes a sim-ready scene, and the sim scene is where navigation and manipulation get trained and evaluated before going back to the robot. Status: **steps 1–2 prototyped in simulation** with a Nova Carter (see [`sim/worldgen`](../sim/worldgen/README.md), 2026-10-08); the reconstruction half (step 3–4) already exists in [`sim/nurec`](../sim/nurec/README.md). The R1 is not wired up yet.

```
 Physical scene                                   Sim scene
 ──────────────                                   ─────────
 1 nav stack on the R1 (Nav2 + slam_toolbox) ──▶ occupancy map
 2 autonomous exploration over that map (frontier / NavDP),
   recording a stereo bag: head L/R RGB + camera_info + /tf + lidar
                                  │
                         3 NuRec stereo workflow (Isaac ROS 4.0 container)
                           → Gaussian splat USDZ + collision mesh + occupancy map
                                  │
                         4 IsaacLab-Arena background (sim/nurec/arena_nurec pattern)
                           → system-2 navigation on the same map, system-1 manipulation in the room
                                  │
        feedback: weak reconstruction → where to drive next; sim eval → what to collect ◀───────┘
```

## Prototype in simulation first

Run steps 1–2 in a simulator standing in for the robot, push the synthetic bag through 3–4. That validates the whole chain (does a stereo bag reconstruct and align, does the map round-trip) before touching hardware, and leaves a sim→sim regression test behind.

### Decision: Nova Carter first, R1 second

The prototype uses NVIDIA's **Nova Carter** instead of an R1 Lite, for three reasons that each remove a step from the plan below: Isaac Sim ships Carter already wired for ROS 2 (odometry, `/tf`, RTX lidar, four Hawk stereo pairs, `/cmd_vel`), NVIDIA publishes Nav2 parameters for it ([`IsaacSim-ros_workspaces`](https://github.com/isaac-sim/IsaacSim-ros_workspaces), a submodule under `sim/worldgen/`), and the NuRec sample room we already load in Arena (`nova_carter-wormhole`) was captured by a Carter, so NVIDIA's own reconstruction of that room is the ground truth for the sim→sim test. The R1 becomes a second embodiment once the loop closes; the nav stack and topic remaps are the only R1-specific parts.

**ROS 2 runs in Docker, not on the host.** Isaac Sim 6.1 bundles its own ROS 2 Jazzy (no system install), and Nav2 + slam_toolbox + explore_lite live in a container on the host network (`sim/worldgen/compose.yml`). The two talk over a UDP-only Fast DDS profile on `ROS_DOMAIN_ID=42`, because a real R1 shares this LAN on domain 0.

### What is built (`sim/worldgen`)

| Step | State |
|---|---|
| 1 Robot in sim | Carter referenced out of Isaac Sim's warehouse sample into the NuRec room, spawned on the room's capture trajectory; `/clock`, odometry, TF, XT-32 point cloud (~43k pts/sweep against the room's collision mesh) and the front stereo pair verified from the container |
| 2 Nav stack | Working: slam_toolbox + Nav2 (NVIDIA's Carter params, retuned) + our own frontier explorer (explore_lite gave up after one empty search). A 30-minute autonomous run reached 3 frontier goals and mapped the room's corridor, top room and alcoves; recovery failures still leave part of the room unexplored |
| Bag | `record.sh` records `/tf`, odometry, lidar and the front stereo pair (+`camera_info`) as mcap, the inputs NuRec's stereo workflow lists |
| 3–4 | Not run on a sim bag yet; the NuRec stereo workflow needs the Isaac ROS 4.0 container |

Sim speed is the practical limit: two Hawk cameras over a 1M-Gaussian splat plus the RTX lidar give ~0.4× real time on the RTX 5090 laptop. Everything is on sim time, so correctness holds, but a 10-minute exploration is 25 minutes of wall clock.

### What exists for an R1 that navigates (surveyed 2026-10-08)

| Source | Gives | Navigates? | Sim? |
|---|---|---|---|
| [Galaxea R1 software guide](https://userguide-galaxea.github.io/Product_User_Guide/Guide/R1/Software_Guide/) | Official ROS 2 driver: chassis velocity `/motion_target/target_speed_chassis` (Twist), odometry `/hdas/feedback_chassis`, lidar `/hdas/lidar_chassis_left` (PointCloud2), IMUs, 5 chassis cameras, head + wrist RGB-D | No Nav2/SLAM shipped, but every input a standard stack needs is published | Mentions an "Isaac Lab usage tutorial" for R1 (locate in the docs sidebar) |
| [OpenGalaxea/GalaxeaManipSim](https://github.com/OpenGalaxea/GalaxeaManipSim) | SAPIEN 3 tabletop manipulation; R1 / R1 Lite / R1 Pro URDFs, mplib demo generation, LeRobot export | **No**: fixed-base URDF (torso + arms + grippers only), no lidar, no ROS | Manipulation only; x86, CUDA 11.8 |
| [dimos R1 Lite integration](https://github.com/dimensionalOS/dimos/pull/3014) | Hardware-validated ROS 2 module: 16-DOF upper body, 3-DOF holonomic chassis via Twist, 6 camera streams. Lesson: the chassis latches its last Twist, so commands must stream with a dead-man | No nav stack | No sim |
| BEHAVIOR-1K / [OmniGibson](https://github.com/StanfordVL/OmniGibson) | Isaac Sim-based household sim whose challenge robot is the **R1 Pro** on its mobile base; 50 scenes; base-motion primitives on traversability maps | Yes, internal planners (not ROS Nav2) | Isaac Sim 4.x, x86 only; own ecosystem |
| [VR-Robo](https://github.com/zst1406217/VR-Robo) (Tsinghua + Galaxea AI, MIT) | Real-to-sim-to-real for navigation: multi-view photos → 3DGS + mesh hybrid in Isaac Lab → RL nav policy → zero-shot real (Unitree Go2) | The sim side of this loop; capture by hand | Isaac Lab |
| [NavDP / InternVLA-N1](https://github.com/wzcai99/NavDP) | Mapless RGB-D navigation policy + Isaac Sim benchmark (explore, point-goal, image-goal) | Learned, generic wheeled | Isaac Sim 4.2 |
| [MoMaGen](https://github.com/ChengshuLi/MoMaGen) | Demo generation from one R1 demo, cross-embodiment | Base placement, not mapping | — |

Conclusion: no off-the-shelf "R1 + nav stack + sim". The real robot needs no vendor stack (Nav2 + slam_toolbox run on the published topics), and the sim prototype is best built in Isaac Lab/Arena so that the same stack and topic names run in sim and on the robot, and the result plugs into the existing NuRec → Arena path.

## Plan

### Step 1: a navigating robot in Isaac Sim with lidar, stereo and ROS 2
Done with Nova Carter (`sim/worldgen/isaac/carter_room.py`). For the R1 Lite later:
- Import the R1 Lite URDF from GalaxeaManipSim (`galaxea_sim/assets/r1_lite/robot.urdf`) into Isaac Sim; put it on a holonomic base (a kinematic/velocity-controlled base is enough for the prototype; Galaxea's chassis is omnidirectional).
- Add an RTX lidar at the chassis position and the head stereo pair matching the real `/hdas/camera_head/{left,right}` cameras (h2rc bags give the real intrinsics and placement).
- ROS 2 bridge publishing the real driver's topic names: `/hdas/lidar_chassis_left`, `/hdas/feedback_chassis`, `/hdas/camera_head/*/image_raw_color/compressed` + `camera_info`, `/tf`; subscribing `/motion_target/target_speed_chassis`.
- Register it as an Arena embodiment/environment the way `sim/nurec/arena_nurec` registers a background (external class path; submodule untouched).
- Scene for the prototype: an existing Arena background (the NuRec `nova_carter-wormhole` room is ideal: it already has a mesh and occupancy map to compare against).

### Step 2: navigation stack, identical for sim and robot
- `slam_toolbox` (online async) → map; Nav2 (DWB, NVIDIA's Carter tuning) for motion; `explore_lite` for frontier exploration. Built: `sim/worldgen/ros/explore.launch.py` + params, in the `flywheel-worldgen-nav` image.
- Still to add for the R1: Nav2 params for a holonomic base and a dead-man republisher for the chassis Twist (the chassis latches its last command).
- Deliverable: a bag + the SLAM map from a fully autonomous exploration run in sim.

### Step 3: reconstruction
- NuRec stereo workflow on the bag (Isaac ROS 4.0 container, needs a GPU; see `sim/nurec/README.md`), 3DGRUT training via `sim/nurec/train.sh`, export ParticleField USD.
- Compare the NuRec occupancy map with the SLAM map (alignment, scale).
- First real-data test: an h2rc bag from the R1 (stereo head topics are there; confirm `/tf` is recorded).

### Step 4: back into Arena
- Register the reconstructed scene as a background (copy `arena_nurec/nurec_wormhole_env.py`), spawn the R1 Lite from step 1 in it, run the same Nav2 stack on the reconstructed map, and GalaxeaManipSim-style manipulation tasks on the room's surfaces as Arena environments.
- Metrics into MLflow: reconstruction PSNR on held-out views (train.sh already computes it), map IoU vs SLAM map, nav success on point-goals, manipulation success.

### Step 5: the robot
- Same launch files on the R1 (real driver topics), capture a room, run steps 3–4 on the real bag.

## Where it runs
- Steps 1–2 run on the RTX 5090 laptop today (Isaac Sim from Arena's venv + the nav container; Docker with the NVIDIA toolkit is installed there). Isaac Sim headless on the GB300 node is being established by the RoboDojo evaluate work; if it holds, steps 1–4 become Slurm jobs like the rest of the stack.
- NuRec stereo needs the Isaac ROS 4.0 container with a GPU runtime; the node's Docker has none, so that step runs via Slurm with a rootless/apptainer-style container or on the laptop.

## Data layout (proposed)
```
raw/scenes/<scene>/<capture>.mcap                  robot bags (sim or real)
processed/scenes/<scene>/map/                      SLAM map (pgm/yaml)
processed/scenes/<scene>/nurec/                    splat USDZ, collision mesh, occupancy map
processed/scenes/<scene>/arena/                    registered background + environments
```
The mcap episode converter (`services/fiftyone/episodes_from_mcap.py`) already handles R1 bags, so captures become browsable in FiftyOne/Rerun once the validate-stage workflows are rebuilt.

## Open questions
- Does Galaxea's own R1 Isaac Lab tutorial provide a mobile-base USD? That would replace most of step 1.
- h2rc bags: is `/tf` present, and are the head cameras calibrated as a stereo pair (baseline in `camera_info`)? Partly answered: the live R1 on the lab LAN publishes `/hdas/camera_head/{left,right}_raw/image_raw_color/compressed` and `/calib/head_{left,right}/camera_info`; `/tf` was not checked.
- Lidar: NuRec recommends it for metric scale; the R1 publishes a 3D PointCloud2, which the stereo workflow accepts optionally.
