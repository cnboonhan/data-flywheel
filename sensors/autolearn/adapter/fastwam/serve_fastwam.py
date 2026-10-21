"""A FastWAM (XPolicyLab's policy/FastWAM) checkpoint served over HTTP for eval_policy_behavior.py --joint.

    DIFFSYNTH_MODEL_BASE_PATH=$STATE/models/diffsynth DIFFSYNTH_SKIP_DOWNLOAD=true venv-fastwam/bin/python serve_fastwam.py \
        --ckpt <checkpoint .pt> --stats <dataset_stats.json> --action-dit <ActionDiT .pt> [--dim 20] [--port 18300]

POST /act takes an .npz of one observation (head_rgb, left_wrist_rgb, right_wrist_rgb: HxWx3 uint8; state: the joint
vector lerobot_to_fastwam.py writes, left arm, left gripper, right arm, right gripper, torso; task: the instruction) and
returns an .npz with `action`: the predicted chunk, horizon x dim, absolute joint targets.

FastWAM's own deploy code (experiments/robotwin/fastwam_policy/deploy_policy.py, which XPolicyLab's adapter wraps)
takes the action size from XPolicyLab's robot registry, which has no R1 Pro: the config here is composed with the
same dimension overrides as XPolicyLab's train.sh, then its policy class is built the way its get_model builds it.
"""
import argparse
import io
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FASTWAM = HERE.parents[3] / "eval/system1/RoboDojo/XPolicyLab/policy/FastWAM/FastWAM"
sys.path[:0] = [str(FASTWAM), str(FASTWAM / "src")]

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--stats", required=True)
p.add_argument("--action-dit", required=True)              # the backbone prep_fastwam.sbatch built (model.action_dit_pretrained_path)
p.add_argument("--dim", type=int, default=20)
p.add_argument("--task-config", default="robotwin_uncond_3cam_384_1e-4")
p.add_argument("--replan", type=int, default=24)              # FastWAM's deploy default; the runner may execute fewer
p.add_argument("--inference-steps", type=int, default=10)     # FastWAM's deploy default (flow-matching steps)
p.add_argument("--port", type=int, default=18300)
args = p.parse_args()


def dim_overrides(d):
    """XPolicyLab's FastWAM train.sh overrides for an action/state size d."""
    out = []
    for split in ("train", "val"):
        for group in ("action", "state"):
            out += [f"data.{split}.shape_meta.{group}.0.raw_shape={d}", f"data.{split}.shape_meta.{group}.0.shape={d}"]
        out += [f"data.{split}.processor.action_output_dim={d}", f"data.{split}.processor.proprio_output_dim={d}"]
    return out


def build():
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from experiments.robotwin.fastwam_policy import deploy_policy as dp

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(FASTWAM / "configs")):
        cfg = compose(config_name="sim_robotwin", overrides=[f"task={args.task_config}", "model.redirect_common_files=false",
                                                         f"model.action_dit_pretrained_path={args.action_dit}"] + dim_overrides(args.dim))
    horizon = int(cfg.data.train.num_frames) - 1
    policy = dp.WorldActionRobotWinPolicy(
        model_cfg=cfg.model, processor_cfg=cfg.data.train.processor, checkpoint_path=args.ckpt,
        dataset_stats_path=Path(args.stats), device="cuda", model_dtype=torch.bfloat16, action_horizon=horizon,
        replan_steps=args.replan, num_inference_steps=args.inference_steps, sigma_shift=None, seed=0, text_cfg_scale=1.0,
        negative_prompt="", rand_device="cpu", tiled=False, timing_enabled=False,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1)
    return policy, horizon


POLICY, HORIZON = build()
print(f"[serve_fastwam] {args.ckpt} ready: dim {args.dim} horizon {HORIZON}", flush=True)


def act(obs):
    o = {"observation": {"head_camera": {"rgb": obs["head_rgb"]}, "left_camera": {"rgb": obs["left_wrist_rgb"]},
                         "right_camera": {"rgb": obs["right_wrist_rgb"]}},
         "joint_action": {"vector": obs["state"].astype(np.float32)}}
    return {"action": np.asarray(POLICY._infer_action_chunk(o, str(obs["task"])), dtype=np.float32)}


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
