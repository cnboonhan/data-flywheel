#!/usr/bin/env python
"""Run an XPolicyLab training script with its metrics, checkpoints and model version in MLflow (flywheel naming).

XPolicyLab policies don't ship a dashboard: ACT computes per-epoch loss summaries but never prints or saves them,
DP writes logs.json.txt. This wrapper patches the policy's own summary hook so every epoch/step lands in the run.
Called by models/<act|dp>/train.sh from the policy directory:

    python xpolicylab_train.py --model ACT --embodiment arx_x5 --data RoboDojo/stack_bowls --action joint --seed 0 \
        --ckpt-dir <dir> --ckpt-name RoboDojo-stack_bowls-arx_x5-joint-0 -- imitate_episodes.py <args...>
"""

import argparse
import os
import runpy
import sys

import mlflow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embodiment import load  # noqa: E402
from flywheel_mlflow import Mix, TrainRun  # noqa: E402


def patch_act(run):
    """ACT: compute_dict_mean() is called once for validation, then once for training, each epoch."""
    import utils  # policy/ACT/utils.py, run from that directory

    state = {"epoch": 0, "val": True}
    orig = utils.compute_dict_mean

    def logged(epoch_dicts):
        summary = orig(epoch_dicts)
        run.log(summary, step=state["epoch"], phase="val" if state["val"] else "train")
        if not state["val"]:
            state["epoch"] += 1
        state["val"] = not state["val"]
        return summary

    utils.compute_dict_mean = logged
    import imitate_episodes
    imitate_episodes.compute_dict_mean = logged


def patch_dp(run):
    """DP: JsonLogger.log(step_log) receives the per-step/epoch dict."""
    from diffusion_policy.common import json_logger

    orig = json_logger.JsonLogger.log

    def logged(self, data):
        nums = {k: float(v) for k, v in data.items() if isinstance(v, (int, float)) and k not in ("global_step", "epoch")}
        if nums:
            mlflow.log_metrics(nums, step=int(data.get("global_step", 0)))   # DP's keys already carry train_/val_
        return orig(self, data)

    json_logger.JsonLogger.log = logged


PATCHES = {"ACT": (patch_act, "epoch"), "DP": (patch_dp, "step")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PATCHES))
    ap.add_argument("--embodiment", required=True)
    ap.add_argument("--data", required=True, help="the one <bench>/<task> XPolicyLab trains on")
    ap.add_argument("--action", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-dir", required=True, help="where the policy writes checkpoints; uploaded and registered after training")
    ap.add_argument("--ckpt-name", required=True, help="XPolicyLab checkpoint name, which evaluate.sbatch uses to find this run")
    ap.add_argument("script", help="training script to run, from the policy directory")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="its arguments (after --)")
    a = ap.parse_args()
    args = a.args[1:] if a.args[:1] == ["--"] else a.args

    emb = load(a.embodiment)
    mix = Mix.resolve([a.data], emb)
    patch, step_unit = PATCHES[a.model]
    with TrainRun(model=a.model, embodiment=emb, mix=mix, action=a.action, seed=a.seed, step_unit=step_unit,
                  params={"command": " ".join([a.script] + args)}, tags={"xpolicylab_ckpt": a.ckpt_name}) as run:
        sys.path.insert(0, os.getcwd())
        patch(run)
        sys.argv = [a.script] + args
        runpy.run_path(a.script, run_name="__main__")
        if os.path.isdir(a.ckpt_dir):
            run.save_checkpoints(a.ckpt_dir)


if __name__ == "__main__":
    main()
