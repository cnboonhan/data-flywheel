# World generation: the physical-scene ↔ sim-scene loop

A second flywheel next to the data one, for the diagram's "Sync Specs" edge between Physical Scenes and Sim Scenes: a robot maps and explores a real space, the capture becomes a sim-ready scene, and the sim scene is where navigation and manipulation get trained and evaluated before going back to the robot. Status: **proposal + plan**, nothing implemented yet. The reconstruction half already exists in [`sim/nurec`](../sim/nurec/README.md).

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

Run steps 1–2 in a simulator standing in for the robot, push the synthetic bag through 3–4. That validates the whole chain (does an R1 stereo bag reconstruct and align, does the map round-trip) before touching hardware, and leaves a sim→sim regression test behind.

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

### Step 1: R1 Lite in Isaac Lab/Arena with a mobile base and sensors
- Import the R1 Lite URDF from GalaxeaManipSim (`galaxea_sim/assets/r1_lite/robot.urdf`) into Isaac Sim; put it on a holonomic base (a kinematic/velocity-controlled base is enough for the prototype; Galaxea's chassis is omnidirectional).
- Add an RTX lidar at the chassis position and the head stereo pair matching the real `/hdas/camera_head/{left,right}` cameras (h2rc bags give the real intrinsics and placement).
- ROS 2 bridge publishing the real driver's topic names: `/hdas/lidar_chassis_left`, `/hdas/feedback_chassis`, `/hdas/camera_head/*/image_raw_color/compressed` + `camera_info`, `/tf`; subscribing `/motion_target/target_speed_chassis`.
- Register it as an Arena embodiment/environment the way `sim/nurec/arena_nurec` registers a background (external class path; submodule untouched).
- Scene for the prototype: an existing Arena background (the NuRec `nova_carter-wormhole` room is ideal: it already has a mesh and occupancy map to compare against).

### Step 2: navigation stack, identical for sim and robot
- `slam_toolbox` (online async) → map; Nav2 (MPPI or DWB controller, smac planner) for motion; `explore_lite` for frontier exploration.
- A `worldgen` ROS 2 workspace in the repo: launch files, Nav2 params tuned for a holonomic base, a dead-man republisher for the chassis Twist, and a `ros2 bag record` profile for the capture topics.
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
- Isaac Sim headless on the GB300 node is being established by the RoboDojo evaluate work; if it holds, steps 1–4 are Slurm jobs like the rest of the stack. Otherwise the RTX 5090 laptop, where `sim/nurec` was verified, hosts steps 3–4.
- NuRec stereo needs the Isaac ROS 4.0 container with a GPU runtime; the node's Docker has none, so that step runs via Slurm with a rootless/apptainer-style container or on the laptop.

## Data layout (proposed)
```
raw/scenes/<scene>/<capture>.mcap                  robot bags (sim or real)
processed/scenes/<scene>/map/                      SLAM map (pgm/yaml)
processed/scenes/<scene>/nurec/                    splat USDZ, collision mesh, occupancy map
processed/scenes/<scene>/arena/                    registered background + environments
```
`episodes-mcap` already converts R1 bags for FiftyOne/Rerun, so captures are browsable like any other data.

## Open questions
- Does Galaxea's own R1 Isaac Lab tutorial provide a mobile-base USD? That would replace most of step 1.
- h2rc bags: is `/tf` present, and are the head cameras calibrated as a stereo pair (baseline in `camera_info`)?
- Lidar: NuRec recommends it for metric scale; the R1 publishes a 3D PointCloud2, which the stereo workflow accepts optionally.
