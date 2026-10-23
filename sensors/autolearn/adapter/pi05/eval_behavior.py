"""Evaluate a π0.5 server (serve_pi05.py) on BEHAVIOR-1K's picking_up_cup, the grasp at the counter.

    venv-behavior/bin/python eval_behavior.py --seed 3 --start <held-out run's program-result.json> \\
        --server http://127.0.0.1:18400 --out results.jsonl [--video ep.mp4]

One episode: reset the task instance (run_behavior.py's registry, the pickup witness ASPIRE's collection used), drive to
the base pose the scripted policy parked at on this instance (its report's base_after_hop), then the policy alone: every
policy step (10 Hz, the dataset's rate) sends the three 512x512 cameras and the 20-value joint state (left arm, left
gripper, right arm, right gripper, torso: to_lerobot.py's layout), and executes the returned chunk's first --replan
actions, each held for 3 control ticks (30 Hz). Success is the witness: the cup held and 5 mm up. Writes one JSON line
per episode.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_behavior  # noqa: E402,F401  registers picking_up_cup, PyRoki, no cuRobo base plan, arm64 material fix
from cap_harness.behavior.adapter import BehaviorAdapter  # noqa: E402
from cap_harness.behavior.registry import BehaviorTaskRegistry  # noqa: E402
from cap_harness.contracts import ArmCommand, RobotAction  # noqa: E402
from cap_harness.validation.evaluators.behavior_pickup import BehaviorPickupWitness  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--start", required=True, help="the scripted policy's program-result.json on this instance (base_after_hop)")
p.add_argument("--server", default="http://127.0.0.1:18400")
p.add_argument("--out", required=True)
p.add_argument("--label", default="", help="what is evaluated, written into each result line")
p.add_argument("--policy-steps", type=int, default=200)   # 20 s at 10 Hz: the scripted grasp and lift take ~8-12 s
p.add_argument("--replan", type=int, default=8)           # of the chunk; untuned
p.add_argument("--video", default=None)
args = p.parse_args()

TICKS_PER_STEP = 3          # the dataset's 10 fps over the 30 Hz control period (behavior_to_lerobot.py VIDEO_EVERY)
ARMS = ("primary", "secondary")      # left, right
TASK = "pick up the cup"

start = json.load(open(args.start))
bx0, by0, byaw0 = start.get("result", start)["base_after_hop"]

reg = BehaviorTaskRegistry()
meta = reg.resolve("picking_up_cup")
adapter = BehaviorAdapter(registry=reg, camera_width=512, camera_height=512, horizon=100000)
t0 = time.time()
adapter.reset(meta, args.seed)
witness = BehaviorPickupWitness(meta.task_name, adapter.native_env, target_scope=meta.target_scope)
adapter.bind_protocol_evaluator(witness)
adapter.navigate_to_pose(bx0, by0, byaw0, planner="curobo", max_steps=1500)
bx, by, byaw = adapter.base_pose_odom()
if math.hypot(bx - bx0, by - by0) > 0.05 or abs(math.atan2(math.sin(byaw - byaw0), math.cos(byaw - byaw0))) > 0.1:
    adapter.navigate_to_pose(bx0, by0, byaw0, planner="servo", max_steps=600)
bx, by, byaw = adapter.base_pose_odom()
start_err = [round(math.hypot(bx - bx0, by - by0), 3), round(math.degrees(abs(byaw - byaw0)), 1)]
print(f"seed {args.seed}: at the counter, off by {start_err[0]} m / {start_err[1]} deg ({time.time() - t0:.0f} s)", flush=True)

codec = adapter._require_codec()
trunk = adapter.layout.trunk
emb = adapter.get_robot_state().embodiment
frames, steps, ended = [], 0, ""
while steps < args.policy_steps and not ended:
    obs = adapter.get_observation()
    st = obs.robot_state
    q = np.asarray(adapter.robot.get_joint_positions(), dtype=np.float32)
    msg = {"task": np.asarray(TASK),
           "state": np.concatenate([np.append(np.asarray(st.joint_positions[a], dtype=np.float32), st.gripper_positions[a])
                                    for a in ARMS] + [q[trunk]]).astype(np.float32)}
    for name in ("head", "left_wrist", "right_wrist"):
        msg[f"{name}_rgb"] = np.asarray(obs.cameras[name].rgb, dtype=np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **msg)
    chunk = np.load(io.BytesIO(urllib.request.urlopen(urllib.request.Request(args.server + "/act", buf.getvalue()), timeout=120).read()))["action"]
    for a in np.asarray(chunk[:args.replan], dtype=float):
        # [left arm 7, left gripper, right arm 7, right gripper, torso 4]
        adapter._q_target = codec.apply_trunk(adapter._q_target, a[16:20])
        action = RobotAction(arms={arm: ArmCommand(mode="joint_position", target=a[8 * i:8 * i + 7],
                                                   gripper_position=float(np.clip(a[8 * i + 7], 0.0, 1.0)), embodiment=emb)
                                   for i, arm in enumerate(ARMS)})
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

ev = dict(witness.evidence() or {})
row = {"label": args.label, "seed": args.seed, "success": bool(witness.success), "ended": ended or "step_limit",
       "policy_steps": steps, "start_err": start_err, "max_clearance_m": ev.get("max_clearance_m"), "wall_s": round(time.time() - t0)}
print(json.dumps(row), flush=True)
with open(args.out, "a") as f:
    f.write(json.dumps(row) + "\n")
if args.video and frames:
    import imageio.v2 as imageio
    imageio.mimwrite(args.video, frames, fps=10, codec="libx264", quality=6)
