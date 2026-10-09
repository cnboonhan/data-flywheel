#!/usr/bin/env python
"""Template training script for a model that reads the xspark hdf5 mix itself; metrics and checkpoints go to MLflow.

Copy models/mlp/ for a new model and replace the three marked parts (DATA, MODEL, LOOP); keep Mix.resolve and the
TrainRun calls so the run follows lib/flywheel_mlflow.py's naming. As shipped it trains a small MLP behaviour-cloning
baseline on proprio state -> action, which is enough to check the plumbing end to end. Run through train.sbatch:

    sbatch --export=ALL train.sbatch mlp arx_x5 RoboDojo/stack_bowls -- --epochs 20
    sbatch --export=ALL train.sbatch mlp arx_x5 'RoboDojo/*,galaxeaOpenWorldDataset/*' --mix arx_x5-all --action ee

xspark episode layout (one hdf5 per episode, T frames):
    state/{left,right}_{arm_joint_states,ee_joint_states,ee_poses}     (T, 6|1|7)
    action/... same keys as state                                      (T, 6|1|7)
    vision/<cam>/colors (T,) JPEG bytes, vision/<cam>/shape (3,)       cam_head, cam_left_wrist, cam_right_wrist
    instructions, subtasks, additional_info/frequency
"""

import argparse
import os
import sys
import tempfile

import h5py
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from embodiment import ACTION_KEYS, load  # noqa: E402
from flywheel_mlflow import Mix, TrainRun  # noqa: E402


# ---- DATA: episodes -> arrays. Replace for images, chunks, language. ----
def load_episode(path, keys):
    with h5py.File(path, "r") as f:
        cat = lambda group: np.concatenate([f[f"{group}/{k}"][()] for k in keys], axis=1)
        return cat("state").astype(np.float32), cat("action").astype(np.float32)


def load_mix(mix, keys, val_frac):
    eps = [load_episode(p, keys) for p in mix.episodes]
    n_val = max(1, int(len(eps) * val_frac)) if len(eps) > 1 else 0
    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    split = lambda idx: tuple(np.concatenate([eps[i][j] for i in idx]) for j in (0, 1)) if len(idx) else None
    return split(order[n_val:]), split(order[:n_val])


# ---- MODEL ----
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="MLP", help="model name used in run and model names")
    ap.add_argument("--embodiment", required=True, help="embodiments/<name>.yaml")
    ap.add_argument("--data", required=True, help="comma-separated <bench>/<task> globs under processed/xpolicylab")
    ap.add_argument("--mix", help="mix name (required when --data matches several datasets)")
    ap.add_argument("--action", default="joint", choices=sorted(ACTION_KEYS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=0, help="also keep a checkpoint every N epochs (0 = final only)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = load(args.embodiment)
    mix = Mix.resolve(args.data.split(","), emb, name=args.mix)
    (xs, ys), val = load_mix(mix, emb.keys(args.action), args.val_frac)
    assert ys.shape[1] == emb.action_dim(args.action), f"episodes have {ys.shape[1]}-D actions, {emb.name} expects {emb.action_dim(args.action)}"
    mean, std = xs.mean(0), xs.std(0) + 1e-6
    print(f"{mix.name}: {len(mix.datasets)} datasets, {len(mix.episodes)} episodes, {len(xs)} train frames, device {device}", flush=True)

    model = MLP(xs.shape[1], ys.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    norm = lambda a: torch.from_numpy((a - mean) / std).to(device)
    X, Y = norm(xs), torch.from_numpy(ys).to(device)
    VX, VY = (norm(val[0]), torch.from_numpy(val[1]).to(device)) if val else (None, None)

    with TrainRun(model=args.model, embodiment=emb, mix=mix, action=args.action, seed=args.seed, params=vars(args)) as run, \
            tempfile.TemporaryDirectory() as ckpt_dir:
        save = lambda name, epoch: torch.save({"model": model.state_dict(), "epoch": epoch, "state_mean": mean, "state_std": std,
                                               "action": args.action, "embodiment": emb.name, "args": vars(args)}, os.path.join(ckpt_dir, name))
        # ---- LOOP: one epoch = one pass over the frames. Log with run.log(..., step=epoch). ----
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(X), device=device)
            total = 0.0
            for i in range(0, len(X), args.batch_size):
                b = perm[i:i + args.batch_size]
                loss = nn.functional.mse_loss(model(X[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            run.log({"loss": total / len(X), "lr": opt.param_groups[0]["lr"]}, step=epoch)
            if VX is not None:
                model.eval()
                with torch.no_grad():
                    run.log({"loss": nn.functional.mse_loss(model(VX), VY).item()}, step=epoch, phase="val")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                save(f"epoch_{epoch + 1}.pt", epoch + 1)
        save("final.pt", args.epochs)
        run.save_checkpoints(ckpt_dir)


if __name__ == "__main__":
    main()
