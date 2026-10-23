# Troubleshooting

What went wrong while bringing ASPIRE up on BEHAVIOR-1K, what the cause turned out to be, and the fix. The
[README](README.md) says how to run things; this file says what to do when they don't work.

When a program reports success at every step and still misses the object, one link in the chain "pixel → 3D point →
gripper pose → joints" is wrong. Test each link against an independent source of the same number (`programs/wave.py`:
a commanded 8 cm lift against the measured rise); a check that only reads the system's own report back proves nothing.

## BEHAVIOR-1K: the PyRoki runner

| Symptom | Root cause | Fix |
|---|---|---|
| `Joint 'wheel_motor_joint1' of type 'continuous' has no velocity limits` | PyRoki refuses a continuous joint without one; the R1 Pro's wheels have none | `ik.py` gives them one (no cost uses it) |
| IK residual of 7 m | the adapter's base transform maps odom from the articulation root, which OmniGibson leaves at the world origin with the base pose in virtual joints; the URDF's root is `base_link` | the runner builds the context from `base_link`'s live pose |
| IK residual of a few centimetres with no rotation error | OmniGibson's end-effector frame is a meta link the URDF lacks; the gripper link stands in at a fixed offset | the offset is measured once from the live robot (`link_offsets`) |
| An 8 cm lift from the tucked arm stops at 7 cm | the arm hangs straight down with the elbow at a limit it can only leave one way; a local solver from that seed walks into the limit | `RESTARTS` in `ik.py`: seeds pulled toward mid-range and jittered |
| The straight joint interpolation sweeps the object | no collision checking between PyRoki's IK and the interpolation planner | a waypoint over the table edge (in the campaign's skill library) |
| The base drives off the scene | the traversability map marks an opening with no floor as free | a seen-floor guard on candidate poses (in the skill library) |
| Demonstrations end with the object barely off the table | ASPIRE's BEHAVIOR adapter terminates the episode on the tick the pickup witness latches (5 mm clearance) | `run_behavior.py` delays the termination by `CAP_HARNESS_SUCCESS_GRACE_TICKS` (the witness still records the first success); `collect_behavior.sh --grace` sets it; the policy lifts in 3 cm steps to 15 cm and keeps the hold |
| A BEHAVIOR dataset has the arms but not the driving or the lean | ASPIRE's per-tick record carries the arms and grippers only; the base and torso move through `behavior.*` calls | `run_behavior.py` writes `episode/whole_body.jsonl` per tick (base pose in odom, base and torso joints and targets); `behavior_to_lerobot.py` turns it into base pose, base velocity and torso features |
| Exit status 139 after `outcome.json` exists | Isaac Sim segfaults on interpreter teardown | a complete run; judge by `outcome.json` |
| `collect_behavior.sh` stops at once with "another Isaac Sim process is running" | its `pgrep -f` matched a shell whose command line mentioned the process names (the shell that launched it) | the check matches python processes only; launch the collector with `setsid nohup` so it outlives the shell |

## ASPIRE itself

- **LIBERO:** `--init-mode saved` (the default) crashes on reset in ASPIRE v1.0 (`adapter.py` tests a NumPy array with `not`); pass `--init-mode seeded`. The same bug fails `cap-harness doctor --reset` and `validate --plan libero-smoke`.
- **BEHAVIOR bootstrap:** `pyarrow` is unpinned and the import check fails on NumPy 1.x; `setup.sh behavior` pins `pyarrow<23` and reruns it.
- **PyRoki service:** Panda-only (`panda_description`, an 8-value seed); hence `adapter/ik.py`.
- **cuRobo on RTX 50-series:** OmniGibson's cuRobo drops the R1 Pro embodiment on sm_120; hence the PyRoki runner.
