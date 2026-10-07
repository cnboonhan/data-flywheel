#!/usr/bin/env python
"""Run an XPolicyLab training script with its metrics mirrored to MLflow.

XPolicyLab policies don't ship a dashboard: ACT computes per-epoch loss
summaries but never prints or saves them, DP writes logs.json.txt, SmolVLA
logs to the console. This wrapper patches the policy's own summary hook so
every epoch/step lands in an MLflow run, where the Store stack's /mlflow UI
draws the curves.

    python train_mlflow.py --policy ACT --run <name> [--param k=v ...] -- imitate_episodes.py <args...>
    python train_mlflow.py --policy DP  --run <name> -- train.py <hydra overrides...>

MLflow connection comes from the usual env vars: MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME/PASSWORD, MLFLOW_TRACKING_SERVER_CERT_PATH.
"""

import argparse
import os
import runpy
import sys

import mlflow


def patch_act():
    """ACT: compute_dict_mean() is called once for validation, then once for
    training, each epoch. Log both and advance the step on the training call."""
    import utils  # policy/ACT/utils.py, run from that directory

    state = {"epoch": 0, "val": True}
    orig = utils.compute_dict_mean

    def logged(epoch_dicts):
        summary = orig(epoch_dicts)
        phase = "val" if state["val"] else "train"
        mlflow.log_metrics({f"{phase}/{k}": float(v) for k, v in summary.items()}, step=state["epoch"])
        if not state["val"]:
            state["epoch"] += 1
        state["val"] = not state["val"]
        return summary

    utils.compute_dict_mean = logged
    import imitate_episodes
    imitate_episodes.compute_dict_mean = logged


def patch_dp():
    """DP: JsonLogger.log(step_log) receives the per-step/epoch dict."""
    from diffusion_policy.common import json_logger

    orig = json_logger.JsonLogger.log

    def logged(self, data):
        nums = {k: float(v) for k, v in data.items() if isinstance(v, (int, float)) and k not in ("global_step", "epoch")}
        if nums:
            mlflow.log_metrics(nums, step=int(data.get("global_step", 0)))
        return orig(self, data)

    json_logger.JsonLogger.log = logged


PATCHES = {"ACT": patch_act, "DP": patch_dp}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, choices=sorted(PATCHES))
    ap.add_argument("--experiment", default="xpolicylab")
    ap.add_argument("--run", required=True, help="MLflow run name, e.g. ACT-Galaxea-Make_The_Bed-arx_x5-joint-0")
    ap.add_argument("--param", action="append", default=[], help="k=v to record as a run parameter")
    ap.add_argument("--ckpt-dir", help="where the policy writes checkpoints; uploaded as run artifacts and registered as a model version")
    ap.add_argument("--model-name", help="registered model name (default: <policy>-<bench>-<task>)")
    ap.add_argument("script", help="training script to run, from the policy directory")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="its arguments (after --)")
    a = ap.parse_args()
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    params = dict(p.split("=", 1) for p in a.param)

    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run) as run:
        mlflow.log_params(params)
        mlflow.log_params({"policy": a.policy, "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "command": " ".join([a.script] + args)})
        sys.path.insert(0, os.getcwd())
        PATCHES[a.policy]()
        sys.argv = [a.script] + args
        runpy.run_path(a.script, run_name="__main__")

        if a.ckpt_dir and os.path.isdir(a.ckpt_dir):
            # Checkpoints go to the run's artifact store (the Store's S3 bucket),
            # and the run becomes a version of the registered model.
            mlflow.log_artifacts(a.ckpt_dir, artifact_path="checkpoints")
            name = a.model_name or "-".join(filter(None, [a.policy, params.get("bench"), params.get("task")]))
            # Register the artifact directory as a model version. In MLflow 3 a
            # runs:/ URI must point at a LoggedModel (raw checkpoints aren't one),
            # and a run-artifact source must be accompanied by its run_id, so go
            # through the client API rather than mlflow.register_model.
            client = mlflow.MlflowClient()
            try:
                client.create_registered_model(name, tags={"policy": a.policy}, description=f"XPolicyLab {a.policy} checkpoints")
            except mlflow.exceptions.RestException as e:
                if e.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
            mv = client.create_model_version(name, f"{run.info.artifact_uri}/checkpoints", run_id=run.info.run_id,
                                             tags={"env_cfg": params.get("env_cfg", ""), "action": params.get("action", ""), "seed": params.get("seed", "")})
            print(f"registered {name} version {mv.version} from {a.ckpt_dir}")


if __name__ == "__main__":
    main()
