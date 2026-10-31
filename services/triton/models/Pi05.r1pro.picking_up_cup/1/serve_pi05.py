"""A π0.5 (openpi, XPolicyLab's policy/Pi_05) checkpoint served over HTTP.

    venv-pi05/bin/python serve_pi05.py --ckpt <checkpoint step dir, or pi05_base's> [--config pi05_r1pro_picking_up_cup] [--port 18400]

POST /act takes an .npz of one observation (head_rgb, left_wrist_rgb, right_wrist_rgb: HxWx3 uint8; state: the 20-value
joint vector, left arm, left gripper, right arm, right gripper, torso; task: the instruction) and returns an .npz with
`action`: the predicted chunk, horizon x 20, absolute joint targets.

The policy is openpi's own (policy_config.create_trained_policy) with r1pro.py's transforms. A fine-tuned checkpoint
carries its norm stats; the base checkpoint has none for the R1 Pro, so the config's (norm_stats.py's) are used, which
makes the base model's "before" run see the same normalized inputs and outputs as the fine-tuned one.
"""
import argparse
import io
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--config", default="pi05_r1pro_picking_up_cup")
p.add_argument("--port", type=int, default=18400)
args = p.parse_args()

import openpi.shared.normalize as normalize  # noqa: E402
import openpi.training.config as _config  # noqa: E402
from openpi.policies import policy_config  # noqa: E402

import r1pro  # noqa: E402

r1pro.register()
cfg = _config.get_config(args.config)
ckpt = Path(args.ckpt)
stats = None
if not (ckpt / "assets").is_dir():      # the base checkpoint: params only
    stats = normalize.load(cfg.assets_dirs / cfg.data.repo_id)
    ckpt = ckpt.parent if ckpt.name == "params" else ckpt
POLICY = policy_config.create_trained_policy(cfg, ckpt, norm_stats=stats)
print(f"[serve_pi05] {args.ckpt} ready ({'config' if stats else 'checkpoint'} norm stats)", flush=True)


def act(obs):
    out = POLICY.infer({"images": {"cam_high": obs["head_rgb"], "cam_left_wrist": obs["left_wrist_rgb"],
                                   "cam_right_wrist": obs["right_wrist_rgb"]},
                        "state": obs["state"].astype(np.float32), "prompt": str(obs["task"])})
    return {"action": np.asarray(out["actions"], dtype=np.float32)}


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
