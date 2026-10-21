"""A GalaxeaVLA (XPolicyLab's policy/GalaxeaVLA, Galaxea's G0) checkpoint served over HTTP for eval_galaxea_behavior.py.

    venv-galaxea/bin/python serve_galaxea.py --ckpt <checkpoint dir or .pt> [--stats dataset_stats.json] [--port 18200]

POST /act takes an .npz of one observation (head_rgb, left_wrist_rgb, right_wrist_rgb: 512x512x3 uint8; left_ee_pose,
right_ee_pose: x y z qx qy qz qw in the base frame; left_gripper, right_gripper: 0 closed .. 100 open; torso: 4 joints;
task: the instruction) and returns an .npz of the predicted chunk (each action key: action_size x raw dim, absolute).
XPolicyLab's own GalaxeaVLA adapter (model.py) packs joint states; this repo's BEHAVIOR task config predicts
end-effector poses, so the model is built here from the same galaxea_fm calls with that config.

--stats overrides the checkpoint's dataset_stats.json: the zero-shot baseline runs pretrained G0Tiny with the fine-tune
data's normalization, so both checkpoints read and write the same units.
"""
import argparse
import io
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GALAXEA = HERE.parents[3] / "eval/system1/RoboDojo/XPolicyLab/policy/GalaxeaVLA/GalaxeaVLA"

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="a checkpoint directory (model weights + dataset_stats.json) or .pt")
p.add_argument("--stats", default=None, help="dataset_stats.json to normalize with instead of the checkpoint's")
p.add_argument("--task-config", default="r1pro_sim/g0tiny_behavior_grasp")
p.add_argument("--smolvlm", required=True, help="the SmolVLM2-500M-Video-Instruct directory (G0Tiny's backbone)")
p.add_argument("--port", type=int, default=18200)
args = p.parse_args()


def build():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from galaxea_fm.utils.config_resolvers import register_default_resolvers
    from galaxea_fm.utils.load_pretrained_resumed import _load_weight_state_dict, load_dataset_stats_from_json, load_checkpoint_for_eval

    from accelerate import PartialState
    PartialState()      # galaxea_fm logs through accelerate, which needs its state first (XPolicyLab's model.py does the same)
    register_default_resolvers()
    os.environ.setdefault("GALAXEA_FM_OUTPUT_DIR", "/tmp/galaxea_fm_output")
    with initialize_config_dir(config_dir=str(GALAXEA / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            f"task={args.task_config}", f"hydra.searchpath=[file://{HERE}]",
            f"model.model_arch.pretrained_model_path={args.smolvlm}",
            f"model.tokenizer.tokenizer_params.pretrained_model_name_or_path={args.smolvlm}"])
    model = instantiate(cfg.model.model_arch)
    if args.stats:
        model.load_state_dict(_load_weight_state_dict(Path(args.ckpt), device="cpu"), strict=True)
        stats = load_dataset_stats_from_json(Path(args.stats))
    else:
        model, stats = load_checkpoint_for_eval(args.ckpt, model, device="cpu")
    dtype = torch.bfloat16 if bool(getattr(cfg.model, "enable_bf16_training", False)) else torch.float32
    model = model.to(dtype=dtype).cuda().eval()
    proc = instantiate(cfg.data.processor)
    proc.set_normalizer_from_stats(stats)
    proc.eval()
    if hasattr(model, "set_tokenizer") and hasattr(proc, "tokenizer"):
        model.set_tokenizer(proc.tokenizer)
    return cfg, model, proc


cfg, MODEL, PROC = build()
IMAGES = [m["key"] for m in cfg.data.dataset.shape_meta.images]
STATE = [m["key"] for m in cfg.data.dataset.shape_meta.state]
print(f"[serve_galaxea] {args.ckpt} ready: images {IMAGES} state {STATE} horizon {cfg.data.dataset.action_size}", flush=True)


def act(obs):
    from galaxea_fm.utils.pytorch_utils import dict_apply
    sample = {"images": {k: torch.from_numpy(np.ascontiguousarray(obs[k].transpose(2, 0, 1))).unsqueeze(0) for k in IMAGES},
              "state": {k: torch.from_numpy(obs[k].astype(np.float32)).unsqueeze(0) for k in STATE},
              "task": str(obs["task"]), "state_is_pad": torch.tensor([False]), "image_is_pad": torch.tensor([False]),
              "idx": torch.tensor(0)}
    s = PROC.preprocess(sample)
    batch = {k: (v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else [v]) for k, v in s.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=next(MODEL.parameters()).dtype == torch.bfloat16):
        out = MODEL.predict_action(batch)
    out = dict_apply(out, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
    out = PROC.postprocess(out)
    return {k: np.asarray(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)[0] for k, v in out["action"].items()}


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
