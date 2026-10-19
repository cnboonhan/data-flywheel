# Troubleshooting

What went wrong while bringing ASPIRE up on Genie Sim and BEHAVIOR-1K, what the cause turned out to be, and the check
that finds it again. The [README](README.md) says how to run things; this file says what to do when they don't work.
The probes live in `adapter/probes/` (ASPIRE programs run through `run_geniesim.py`, scripts, and `isaac/` scripts run
inside the container with Isaac's python while the benchmark is stopped).

## Method

A program that reports success at every step and still misses the object means one link in the chain "pixel → 3D
point → gripper pose → joints" is wrong. Test each link against an independent source of the same number; a check
that only reads the system's own report back proves nothing. Rerun the table when the robot model, URDF or camera
changes.

| Check | Probe | Sources compared | Expect |
|---|---|---|---|
| Kinematic model | `probes/isaac/usd_fk_check.py` | the URDF FK the adapter plans on vs the simulator's link poses at saved joint configurations | ≤ 5 mm. The benchmark's `end_pose` is its own FK from the same joints, so agreeing with it proves nothing |
| IK and planner closed loop | `programs/wave.py` | a commanded 8 cm lift vs the FK-measured rise | `rise_m` ≥ 0.075; less means the waist moved (rest cost or seed lost) |
| Head camera model | any run with `--debug` (`depth_at_centre` in each snapshot's json) | the head depth at the projected gripper *base* (solid, above any block) vs its FK depth; a block's top from the head vs the height a descent stalls at | equal within 2 cm; the table where the base should be means the camera pose is off |
| Joint order and direction | `probes/joint_probe.py` | each arm joint moved alone: the reported joints vs the command, and the head depth at the FK gripper vs FK's depth | echo error ≈ 0; depth matches for every joint |
| Depth validity | `probes/probe_cloud.py` | the head depth image | no zeros or NaNs on the table |
| Perception to action | `programs/pick_block.py` (`block_head`, `block_wrist`, `held`) | the head and wrist estimates of the block vs each other and vs the stall height | within 2 cm; a held cube reads 0.29–0.45 of gripper travel |
| Gripper geometry | `probes/isaac/finger_probe.py`, `grasp_test.py` | the robot USD driven headless: finger meshes in the centre-link frame; a 5 cm cube closed on at offsets, then pulled along the approach | fingertips meet 1 cm past the centre link; a face-aligned cube holds with its top 2.5–4.5 cm above the centre link, lost beyond 5 cm or at 30° yaw |
| Camera lag | `probes/lag_probe.py`, `far_calib.py`, `calib_probe.py` | the rendered gripper across snapshots after one move vs the reported pose; the depth around the projected gripper at far poses | the image is static after a move; the nearest surface is at the projection |

Projecting a point and reading the depth back is blind to a camera rotation, because a rotated camera still returns
a plausible depth at the shifted pixel. The decisive checks are a solid surface at a known pose and the stall height.

## Genie Sim: the pick missed the cube

Each of these was believed at some point: images lag the robot, the FK is wrong, the joint order is wrong, the drives
are slow, the block is offset laterally. The probes above disproved them all. The real causes:

| Symptom | Root cause | Fix |
|---|---|---|
| Head estimates 4–7 cm too high and too far | the head's three joints are never reported by the benchmark, so the adapter's camera pose assumed zero while the simulator had them tilted | the adapter commands the head to zero in every chunk (`reply["head"]`), so model and robot agree |
| Wrist estimates disagree with the head along the image's vertical axis | the wrist prims' vertical aperture in the USD does not match the image's aspect; Isaac fits the horizontal aperture and renders square pixels, so `fy` computed from the file was 45 % too large | `fy = fx` in `G2Model.camera()` |
| The fingers close on air at the right pixel | the planning frame and the pinch point differ: the omnipicker's centre link is the pinch point, 14.3 cm beyond the gripper base, and the fingertips reach 1 cm past it; the earlier approach aimed too deep | `PAD_OFFSET` in `pick_block.py`: centre link 3.5 cm below a 5 cm cube's top, descending 1 cm past the hold window so a low estimate stalls on the table |
| The gripper reading after a close is mid-travel and the same every time | `close_gripper` returned before the fingers stopped (30–60 ticks) | hold the command, then poll until the reading settles (`GRIPPER_STEPS`, `GRIPPER_POLL`, `GRIPPER_SETTLED`) |
| Held the cube but no score | the lift did not finish within the episode's ~330 steps | shorter rise and approach; no second lift |
| Grasped a look-alike block | the arm hid the target from the head camera and the re-localization took the nearest candidate | a new estimate counts only within 5 cm of the old one |
| "Purple" and "green" blocks never found | SAM3's colour words: the scene's green is teal and its purple is navy | the hue table in `pick_block.py` (`HUES`), candidates within 40° |

Facts measured from the robot USD along the way, used by the programs: the fingers swing in a near-horizontal arc from
13.6 cm apart (joint 45°, reported as −0.785) to meeting 1 cm past the centre link; a held 5 cm cube stops the gripper
at 0.29–0.45 of travel, an empty close goes below 0.1; in the pick scene the fingers pass through the table, so a
descent never stalls on it. The cameras are 1.93 mm focal; head 3.896 mm horizontal aperture (fx ≈ 317 px at 640),
wrist 3.76 mm (fx ≈ 657 px at 1280). The waist at rest reads about `[-0.83, 1.22, 0.1, 0, 0]`, outside the limits of
the benchmark's own `robot_cfg/G2_omnipicker/G2_omnipicker.urdf`, which is how that URDF was found not to match the
simulated robot; the data-collection planner's `G2_omnipicker_fixed_dual.urdf` matches to 5 mm.

## Genie Sim: what the learned placement program found

Findings a coding agent made in five attempts on `g2op_spatial_place_object_relative_position`, each with the fix now
in the adapter or the program:

| Finding | Where it went |
|---|---|
| The episode's first frame is taken while the head is still swinging to its hold pose: head estimates 6.5 cm too far and 3–5 cm too high | the adapter holds `SETTLE_TICKS` at reset before returning the first observation |
| The runner's `success` never came true on a scored episode: the benchmark's episode-ending frame carries no `task_progress`, and its two `Follow` checks are a left/right either-or | the adapter keeps the last scored progress; `success` is the final check; the benchmark's own verdict is `genie_sim/output/benchmark/<task>/<sub_task>/evaluate_ret_NN.json` (`E2E`) |
| `--debug` head snapshots were overwritten each episode: named by the benchmark's episode index, which restarts | named by the runner's count |
| The checker's `left` is the robot's +y and `top` is +x; B's top-face centre + 0.17 m along the direction passes `aligned` at 0.1 | `place_relative.py` |
| `Follow` scores only during the lift, never at the grasp; if its 300-step limit fires the episode ends unscored | lift promptly |
| Gripper reading 0.12–0.75 means held (bottle 0.47, apple 0.52, sponge 0.41, a 3 cm block 0.18; empty < 0.002) | `place_relative.py`'s held test |
| Closing on a ≤ 3 cm object at the table is a coin flip | regrasp across the other axis when the close reads empty |
| SAM3 scores ≥ 0.5 are reliable and ≤ 0.1 are guesses; it returns the grippers for any prompt | candidates with tops above 16 cm are dropped |
| An arm chosen from an unsettled estimate cannot carry across the body | choose the arm from the settled estimate by the midpoint of A and the target |
| The sandbox's builtins are only `abs all any bool dict enumerate float int len list max min print range reversed round set sorted str sum tuple zip` plus `Exception ValueError TypeError` and `math` | no `isinstance`, no `repr` in programs |

## Genie Sim: adding a task

| Symptom | Root cause | Fix |
|---|---|---|
| `'PiEnv' object has no attribute 'init_arm'` on a new sub-task | the benchmark keys robot init states by sub-task name in `robot_init_states.py` (`TASK_INFO_DICT`), and scoring steps in `eval_utils.py` | `task_info.json` with `{"like": "<shipped task>"}` in the task folder; `hooks/sitecustomize.py` fills both tables at import |
| `geniesim_generator`'s `app.py` does not run as its docs say | the container has `python3` with `pxr` but no `python`; `mitsuba`, `pygraphviz`, `jaxtyping` and `transforms3d` are installed nowhere; `app.py` imports `LLM_RESULT.py` from its own directory; it writes asset paths relative to a directory that only exists in the deployed layout, snapshots its template instead of the program it ran, and writes under the generator's own `src/benchmark/`, not the benchmark package | `adapter/tasks/place_object_relative_position_lab/gen/generate.sh`: shims for the missing packages, `runpy` from a staging directory, a post-process that makes payload paths absolute and the table kinematic, a copy into the benchmark package |
| A random layout passes `spec.py`'s own strip check and fails `verify.py` | `verify.py` measures the target strip on object extents, the layout on centres | the layout's strip is widened by a 6 cm half-size (`STRIP` in `spec.py`) |
| A panorama from `geniesim_world` never loads | this Genie Sim version loads only backgrounds shipped as USD + `.nurec`; PanoRecon's Gaussian PLYs have no loader yet | build layouts on the pack's backgrounds |

## BEHAVIOR-1K: the PyRoki runner

| Symptom | Root cause | Fix |
|---|---|---|
| `Joint 'wheel_motor_joint1' of type 'continuous' has no velocity limits` | PyRoki refuses a continuous joint without one; the R1 Pro's wheels have none | `ik.py` gives them one (no cost uses it) |
| IK residual of 7 m | the adapter's base transform maps odom from the articulation root, which OmniGibson leaves at the world origin with the base pose in virtual joints; the URDF's root is `base_link` | the runner builds the context from `base_link`'s live pose |
| IK residual of a few centimetres with no rotation error | OmniGibson's end-effector frame is a meta link the URDF lacks; the gripper link stands in at a fixed offset | the offset is measured once from the live robot (`link_offsets`) |
| An 8 cm lift from the tucked arm stops at 7 cm | the arm hangs straight down with the elbow at a limit it can only leave one way; a local solver from that seed walks into the limit | `RESTARTS` in `ik.py`: seeds pulled toward mid-range and jittered |
| The straight joint interpolation sweeps the object | no collision checking between PyRoki's IK and the interpolation planner | a waypoint over the table edge (in the campaign's skill library) |
| The base drives off the scene | the traversability map marks an opening with no floor as free | a seen-floor guard on candidate poses (in the skill library) |
| A campaign's per-seed policy navigates on new instances but never holds (1 of 20) | it was tuned on one radio pose; the seed-27 library entry had the finger axis wrong (the R1 Pro closes along hand +y, so it closed along the radio's long side onto the handle), the leaned arm sags 4–8 cm below its targets, `move_to_pose` refuses descents its current-seeded IK can't solve, and a radio whose long axis faces the base is unreachable at any lean | `pick_radio_r1pro.py`: fingers across the short axis, a base pose chosen by the radio's rotation, approaching high and descending in 3 cm Cartesian steps, refusing IK solutions far from the current joints |
| The general policy held on 8 of 10 development instances but 5 of 12 collection instances | the radio's body (8.6 cm) is as wide as the open jaw, so body grasps had under 1 cm of slack, and most misses were the robot's own joint-space moves knocking the upright radio over | grasp the 3 cm wire handle first; predict `move_to_pose`'s IK and refuse solutions far from the current joints; cross to the radio at its highest reachable point, then descend and lift in 3 cm steps (9 of 12 fresh instances) |
| Still missed: a radio at the edge of reach, a lying radio, a radio at the table corner | the torso lean chosen without margin at the handle height; no handle on a lying radio; the map-routed drive pushes a corner radio off the table | open; routing around the corner was tried and caused a fall off a floor edge, so it was removed |
| Exit status 139 after `outcome.json` exists | Isaac Sim segfaults on interpreter teardown | a complete run; judge by `outcome.json` |
| `collect_behavior.sh` stops at once with "another Isaac Sim process is running" | its `pgrep -f` matched a shell whose command line mentioned the process names (the shell that launched it) | the check matches python processes only; launch the collector with `setsid nohup` so it outlives the shell |

## ASPIRE itself

- **LIBERO:** `--init-mode saved` (the default) crashes on reset in ASPIRE v1.0 (`adapter.py` tests a NumPy array with `not`); pass `--init-mode seeded`. The same bug fails `cap-harness doctor --reset` and `validate --plan libero-smoke`.
- **BEHAVIOR bootstrap:** `pyarrow` is unpinned and the import check fails on NumPy 1.x; `setup.sh behavior` pins `pyarrow<23` and reruns it.
- **PyRoki service:** Panda-only (`panda_description`, an 8-value seed) and the embodiment labels are a closed set; hence `adapter/ik.py` and the `robosuite` label for the G2.
- **cuRobo on RTX 50-series:** OmniGibson's cuRobo drops the R1 Pro embodiment on sm_120; hence the PyRoki runner.
- **The benchmark's own recorder** (`--benchmark.record=true`) writes review videos and an instruction json only; the AgiBot-format episodes its converter expects come from the separate data-collection stack (its own image, gRPC server and planners, ~1.5 GB an episode). The adapter records its own stream instead (`--record`).
- **The benchmark keeps serving** across runner restarts (30 socket retries with backoff), so a crashed runner can be relaunched without restarting Isaac Sim; after that budget it exits and `setup.sh benchmark` starts it again.
- **Isaac Sim probes** in `probes/isaac/` need `from isaacsim import SimulationApp` before any `pxr` import; the `isaacsim.sensors.camera.Camera` annotators fail on these prims and a replicator render hung, so the probes read poses and physics only.
