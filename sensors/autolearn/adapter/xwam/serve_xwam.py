"""An X-WAM (XPolicyLab's policy/X_WAM) checkpoint served over HTTP for eval_galaxea_behavior.py --space ee.

    venv-xwam/bin/python serve_xwam.py --exp <experiment dir> [--steps last] [--torso-json <dataset>/torso.json] [--port 18400]

The same protocol as galaxea/serve_galaxea.py. POST /act takes an .npz of one observation (head_rgb, left_wrist_rgb,
right_wrist_rgb: 512x512x3 uint8; left_ee_pose, right_ee_pose: x y z qx qy qz qw in the base frame; left_gripper,
right_gripper: 0 closed .. 100 open; torso: 4 joints; task: the instruction) and returns an .npz of the chunk:
{left,right}_ee_pose [T, 7] (base frame, x y z qx qy qz qw), {left,right}_gripper [T, 1] (0..100), torso [T, 4].

The model is X-WAM's own evaluation/deploy_policy.py XWAMPolicy (the experiment's config.yaml carries the statistics).
It predicts per-step deltas (position, rotation vector, gripper) for each hand; they are integrated from the current
pose as XPolicyLab's model.py does, without its axis redefinition (lerobot_to_xwam.py applies none). X-WAM has no torso
output: the torso is held at the lean the demonstrations end in (lerobot_to_xwam.py's torso.json).
"""
import argparse
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
XWAM = HERE.parents[3] / "eval/system1/RoboDojo/XPolicyLab/policy/X_WAM/X-WAM"
sys.path[:0] = [str(HERE / "shim"), str(XWAM), str(XWAM / "evaluation")]

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True, help="experiment dir: config.yaml + checkpoints/<steps>.ckpt/checkpoint/...")
p.add_argument("--steps", default="last")
p.add_argument("--wan", default="/tier1/mistral/flywheel/autolearn/models/diffsynth/Wan-AI/Wan2.2-TI2V-5B")
p.add_argument("--torso-json", default=None, help="default: torso.json next to the experiment's training data")
p.add_argument("--action-denoise-steps", type=int, default=10)    # X-WAM's deploy default (actions decode early)
p.add_argument("--denoise-steps", type=int, default=50)           # X-WAM's deploy default (the video's schedule)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--port", type=int, default=18400)
args = p.parse_args()

from omegaconf import OmegaConf  # noqa: E402
from evaluation.deploy_policy import XWAMPolicy  # noqa: E402

POLICY = XWAMPolicy(exp_path=args.exp, steps=args.steps, wan_checkpoint_dir=args.wan, denoise_steps=args.denoise_steps,
                    action_denoise_steps=args.action_denoise_steps, device="cuda", compile_model=False)
cfg = OmegaConf.load(os.path.join(args.exp, "config.yaml"))
TORSO = np.asarray(json.load(open(args.torso_json or os.path.join(cfg.dataset.dataset_path, "torso.json")))["torso"], dtype=np.float32)
SIDES = ("left", "right")
CALLS = [0]
print(f"[serve_xwam] {args.exp} {args.steps} ready: torso held at {TORSO.round(3).tolist()}", flush=True)


def act(obs):
    video = np.stack([obs["head_rgb"], obs["left_wrist_rgb"], obs["right_wrist_rgb"]])[None]    # X-WAM's sorted views
    prop = []
    for side in SIDES:
        pose = np.asarray(obs[f"{side}_ee_pose"], dtype=np.float64)
        prop += list(pose[:3]) + list(pose[[6, 3, 4, 5]]) + [float(obs[f"{side}_gripper"][0]) / 100.0]   # wxyz, 0..1
    prop = np.asarray(prop, dtype=np.float32)[None]
    CALLS[0] += 1
    deltas, _ = POLICY.infer_batch(video=video, proprios=prop, prompts=[str(obs["task"])], seeds=[args.seed + CALLS[0]])
    d = deltas[0]                                     # [T, 14]: per hand dxyz, drotvec, dgripper
    out = {}
    for i, side in enumerate(SIDES):
        pos = prop[0, 8 * i:8 * i + 3].astype(np.float64)
        R = Rotation.from_quat(np.asarray(obs[f"{side}_ee_pose"], dtype=np.float64)[3:7])
        g = float(prop[0, 8 * i + 7])
        poses, grips = [], []
        for k in range(len(d)):
            a = d[k, 7 * i:7 * i + 7].astype(np.float64)
            pos = pos + a[:3]
            R = Rotation.from_rotvec(a[3:6]) * R      # left-multiplied, as the training deltas (a_R @ p_R^T)
            g = float(np.clip(g + a[6], 0.0, 1.0))
            poses.append(np.concatenate([pos, R.as_quat()]))
            grips.append([g * 100.0])
        out[f"{side}_ee_pose"] = np.asarray(poses, dtype=np.float32)
        out[f"{side}_gripper"] = np.asarray(grips, dtype=np.float32)
    out["torso"] = np.tile(TORSO, (len(d), 1))
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply(200, b"ok", "text/plain")

    def do_POST(self):
        obs = dict(np.load(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"]))), allow_pickle=False))
        try:
            buf = io.BytesIO()
            np.savez(buf, **act(obs))
            self.reply(200, buf.getvalue(), "application/octet-stream")
        except Exception as e:   # noqa: BLE001  report, keep serving
            self.log_error("act failed: %r", e)
            self.reply(500, repr(e).encode(), "text/plain")

    def reply(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
