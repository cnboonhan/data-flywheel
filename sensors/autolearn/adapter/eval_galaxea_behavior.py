"""Evaluate a policy server (galaxea/serve_galaxea.py, or fastwam/serve_fastwam.py with --space joint) on BEHAVIOR-1K's
picking_up_cup, the grasp at the counter.

    venv-behavior/bin/python eval_galaxea_behavior.py --seed 3 --start <held-out run's program-result.json> \\
        --server http://127.0.0.1:18200 --out results.jsonl [--video ep.mp4]

One episode: reset the task instance (run_behavior.py's registry, the pickup witness ASPIRE's collection used), drive to
the base pose the scripted policy parked at on this instance (its report's base_after_hop, by the same map route and
servo), then the policy alone: every policy step (10 Hz, the dataset's rate) sends the three 512x512 cameras, both
end-effector poses in the base frame, grippers 0..100 and the torso joints, and executes the returned chunk's first
--replan actions, each held for 3 control ticks (30 Hz). End-effector poses become arm joints through run_behavior.py's
PyRoki IK; torso targets go into the adapter's joint targets as move_torso puts them. Success is the witness: the cup
held and 5 mm up. Writes one JSON line per episode.
"""
import argparse
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_behavior as rb  # noqa: E402  registers picking_up_cup, PyRoki, no cuRobo base plan, arm64 material fix
from cap_harness.behavior.adapter import BehaviorAdapter  # noqa: E402
from cap_harness.behavior.registry import BehaviorTaskRegistry  # noqa: E402
from cap_harness.contracts import ArmCommand, Pose, RobotAction  # noqa: E402
from cap_harness.validation.evaluators.behavior_pickup import BehaviorPickupWitness  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--start", required=True, help="the scripted policy's program-result.json on this instance (base_after_hop)")
p.add_argument("--server", default="http://127.0.0.1:18200")
p.add_argument("--out", required=True)
p.add_argument("--label", default="", help="what is evaluated, written into each result line")
p.add_argument("--policy-steps", type=int, default=200)   # 20 s at 10 Hz: the scripted grasp and lift take ~8-12 s
p.add_argument("--replan", type=int, default=8)           # of the 32-step chunk; untuned
p.add_argument("--video", default=None)
p.add_argument("--space", choices=("ee", "joint"), default="ee",
               help="ee: GalaxeaVLA (serve_galaxea.py), end-effector poses through IK; joint: FastWAM (fastwam/serve_fastwam.py), "
                    "a 20-value joint vector (arms, grippers, torso; lerobot_to_fastwam.py's layout) sent as is")
args = p.parse_args()

TICKS_PER_STEP = 3          # the dataset's 10 fps over the 30 Hz control period (behavior_to_lerobot.py VIDEO_EVERY)
ARMS = {"left": "primary", "right": "secondary"}
TASK = "pick up the cup"


def quat_mul(a, b):   # xyzw
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def in_base(pos, wxyz, base):
    """An odom pose in the base frame as [x y z qx qy qz qw] (behavior_to_lerobot.py's in_base)."""
    x, y, yaw = base
    dx, dy = pos[0] - x, pos[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    q = quat_mul([0.0, 0.0, -math.sin(yaw / 2), math.cos(yaw / 2)], [wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return np.asarray([c * dx + s * dy, -s * dx + c * dy, pos[2]] + q, dtype=np.float32)


def to_odom(p7, base):
    """in_base's inverse: a base-frame [x y z qx qy qz qw] as an odom position and wxyz quaternion."""
    x, y, yaw = base
    c, s = math.cos(yaw), math.sin(yaw)
    q = quat_mul([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], list(p7[3:7]))
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return [x + c * p7[0] - s * p7[1], y + s * p7[0] + c * p7[1], float(p7[2])], [q[3] / n, q[0] / n, q[1] / n, q[2] / n]


start = json.load(open(args.start))
start = start.get("result", start)
bx0, by0, byaw0 = start["base_after_hop"]

reg = BehaviorTaskRegistry()
meta = reg.resolve("picking_up_cup")
adapter = BehaviorAdapter(registry=reg, camera_width=512, camera_height=512, horizon=100000)
t0 = time.time()
adapter.reset(meta, args.seed)
witness = BehaviorPickupWitness(meta.task_name, adapter.native_env, target_scope=meta.target_scope)
adapter.bind_protocol_evaluator(witness)
r = adapter.navigate_to_pose(bx0, by0, byaw0, planner="curobo", max_steps=1500)
bx, by, byaw = adapter.base_pose_odom()
if math.hypot(bx - bx0, by - by0) > 0.05 or abs(math.atan2(math.sin(byaw - byaw0), math.cos(byaw - byaw0))) > 0.1:
    adapter.navigate_to_pose(bx0, by0, byaw0, planner="servo", max_steps=600)
bx, by, byaw = adapter.base_pose_odom()
start_err = [round(math.hypot(bx - bx0, by - by0), 3), round(math.degrees(abs(byaw - byaw0)), 1)]
print(f"seed {args.seed}: at the counter, off by {start_err[0]} m / {start_err[1]} deg ({time.time() - t0:.0f} s)", flush=True)

ik = rb.pyroki_for(adapter)
codec = adapter._require_codec()
layout = adapter.layout
emb = adapter.get_robot_state().embodiment
# self-check of the frames: each hand's current pose, to the base frame and back, must solve IK where it is
check = {}
st0, base0 = adapter.get_robot_state(), adapter.base_pose_odom()
for side, arm in ARMS.items():
    pz = st0.end_effector_poses[arm]
    pos, wxyz = to_odom(in_base(pz.position, pz.quaternion_wxyz, base0), base0)
    sol = ik.solve_ik(Pose(position=pos, quaternion_wxyz=wxyz, frame=st0.base_frame), st0, arm=arm)
    check[side] = {"roundtrip_m": round(float(np.linalg.norm(np.asarray(pos) - np.asarray(pz.position))), 6), "ik_ok": bool(sol.ok),
                   "ik_residual_m": round(float(dict(sol.diagnostics or {}).get("position_error_m", -1)), 4)}
print("frame self-check:", check, flush=True)
frames, ik_fail, steps, success, ended = [], 0, 0, False, ""
arm_target = {a: np.asarray(adapter.get_robot_state().joint_positions[a], dtype=float) for a in ARMS.values()}
grip_target = {a: float(adapter.get_robot_state().gripper_positions[a]) for a in ARMS.values()}

while steps < args.policy_steps and not ended:
    obs = adapter.get_observation()
    st = obs.robot_state
    base = adapter.base_pose_odom()
    q = np.asarray(adapter.robot.get_joint_positions(), dtype=float)
    msg = {"task": np.asarray(TASK), "torso": q[layout.trunk].astype(np.float32)}
    for name in ("head", "left_wrist", "right_wrist"):
        msg[f"{name}_rgb"] = np.asarray(obs.cameras[name].rgb, dtype=np.uint8)
    if args.space == "joint":
        msg["state"] = np.concatenate([np.concatenate([np.asarray(st.joint_positions[arm], dtype=np.float32),
                                                       [float(st.gripper_positions[arm])]]) for arm in ARMS.values()]
                                      + [q[layout.trunk].astype(np.float32)])
    for side, arm in ARMS.items():
        pose = st.end_effector_poses[arm]
        msg[f"{side}_ee_pose"] = in_base(pose.position, pose.quaternion_wxyz, base)
        msg[f"{side}_gripper"] = np.asarray([float(st.gripper_positions[arm]) * 100.0], dtype=np.float32)
    buf = io.BytesIO()
    np.savez(buf, **msg)
    resp = urllib.request.urlopen(urllib.request.Request(args.server + "/act", buf.getvalue()), timeout=120).read()
    chunk = dict(np.load(io.BytesIO(resp)))
    n_chunk = len(chunk["action"]) if args.space == "joint" else len(chunk["torso"])
    for k in range(min(args.replan, n_chunk)):
        if args.space == "joint":       # [left arm 7, left gripper, right arm 7, right gripper, torso 4]
            a = np.asarray(chunk["action"][k], dtype=float)
            arm_target = {"primary": a[0:7], "secondary": a[8:15]}
            grip_target = {"primary": float(np.clip(a[7], 0.0, 1.0)), "secondary": float(np.clip(a[15], 0.0, 1.0))}
            chunk_torso = a[16:20]
        else:
            chunk_torso = None
        base = adapter.base_pose_odom()
        state = adapter.get_robot_state()
        for side, arm in (ARMS.items() if args.space == "ee" else ()):
            pos, wxyz = to_odom(chunk[f"{side}_ee_pose"][k], base)
            sol = ik.solve_ik(Pose(position=pos, quaternion_wxyz=wxyz, frame=state.base_frame), state, arm=arm)
            if sol.ok:
                arm_target[arm] = np.asarray(sol.joint_positions, dtype=float)
            else:
                ik_fail += 1          # keep the arm's last target
            grip_target[arm] = float(np.clip(chunk[f"{side}_gripper"][k][0] / 100.0, 0.0, 1.0))
        adapter._q_target = codec.apply_trunk(adapter._q_target, np.asarray(chunk_torso if chunk_torso is not None else chunk["torso"][k], dtype=float))
        action = RobotAction(arms={arm: ArmCommand(mode="joint_position", target=arm_target[arm],
                                                   gripper_position=grip_target[arm], embodiment=emb) for arm in ARMS.values()})
        for _ in range(TICKS_PER_STEP):
            res = adapter.step(action)
            if res.terminated or res.truncated or not res.ok:
                ended = "success" if (res.terminated and witness.success) else (
                    "terminated" if res.terminated else "truncated" if res.truncated else "step_error")
                break
        steps += 1
        if args.video:      # one frame per policy step (10 fps, as the dataset), left wrist | head | right wrist
            cams = adapter.get_observation().cameras
            frames.append(np.concatenate([np.asarray(cams[n].rgb, dtype=np.uint8) for n in ("left_wrist", "head", "right_wrist")], axis=1))
        if ended:
            break

ev = dict(witness.evidence())
success = bool(witness.success)
row = {"label": args.label, "seed": args.seed, "success": success, "ended": ended or "step_limit", "policy_steps": steps,
       "ik_failures": ik_fail, "ik_solves": 2 * steps, "self_check": check, "start_err": start_err, "max_clearance_m": (ev or {}).get("max_clearance_m"),
       "wall_s": round(time.time() - t0)}
print(json.dumps(row), flush=True)
with open(args.out, "a") as f:
    f.write(json.dumps(row) + "\n")
if args.video and frames:
    import imageio.v2 as imageio
    imageio.mimwrite(args.video, frames, fps=10, codec="libx264", quality=6)
