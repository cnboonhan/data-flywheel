#!/usr/bin/env python
"""Record a RoboDojo evaluation in MLflow.

Reads the newest `_result.json` RoboDojo wrote under eval_result/ for the given model/task/ckpt and logs
success_rate, score and the episode count, plus the result folder (videos), on the training run that produced the
checkpoint. That run is found by its tag xpolicylab_ckpt (lib/xpolicylab_train.py sets it) in any experiment, or,
for runs from before the flywheel naming, by run name in the experiment xpolicylab. The run's registered model
version gets eval_<task>_success_rate and eval_<task>_score tags. Without a training run (e.g. a hub checkpoint),
the result goes to a new run in the experiment eval/<model>.

    python eval_mlflow.py --model ACT --bench RoboDojo --task stack_bowls --ckpt RoboDojo-stack_bowls-arx_x5-joint-0 \
        --env-cfg arx_x5_gpu --action joint --seed 0 --result-root <RoboDojo>/eval_result --since <epoch>
"""

import argparse
import glob
import json
import os

import mlflow


def find_training_run(client, a):
    exps = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(exps, filter_string=f"tags.xpolicylab_ckpt = '{a.ckpt}' and params.model = '{a.model}'",
                              max_results=1, order_by=["start_time DESC"])
    if runs:
        return runs[0]
    legacy = client.get_experiment_by_name("xpolicylab")   # experiment, run and model names before services/slurm/lib
    if legacy:
        runs = client.search_runs([legacy.experiment_id], filter_string=f"tags.mlflow.runName = '{a.model}-{a.ckpt}'",
                                  max_results=1, order_by=["start_time DESC"])
        if runs:
            return runs[0]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--model", "--bench", "--task", "--ckpt", "--env-cfg", "--action"):
        ap.add_argument(a, required=True)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--since", type=float, default=0.0, help="ignore result files older than this epoch time (the job's start), so a failed run can't report an earlier result")
    a = ap.parse_args()

    pattern = f"{a.result_root}/{a.bench}/{a.task}/{a.model}/{a.env_cfg}/{a.seed}_ckpt_name={a.ckpt},action_type={a.action}/*/_result.json"
    results = sorted((p for p in glob.glob(pattern) if os.path.getmtime(p) >= a.since), key=os.path.getmtime)
    if not results:
        raise SystemExit(f"no _result.json newer than the job start under {pattern}")
    path = results[-1]
    r = json.load(open(path))
    metrics = {"eval/success_rate": float(r.get("success_rate", 0)), "eval/score": float(r.get("score", 0)), "eval/episodes": float(r.get("eval_time", 0))}
    print(f"{path}: {metrics}")

    client = mlflow.MlflowClient()
    train = find_training_run(client, a)
    if train is None:
        mlflow.set_experiment(f"eval/{a.model}")
    with mlflow.start_run(run_id=train.info.run_id if train else None, run_name=None if train else f"{a.model}-{a.ckpt}-{a.env_cfg}") as run:
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"eval_task": a.task, "eval_result": path, "eval_env": "RoboDojo", "eval_env_cfg": a.env_cfg})
        mlflow.log_artifacts(os.path.dirname(path), artifact_path=f"eval/{a.task}")   # _result.json and the episode videos
        print("logged to run", run.info.run_id, "(training run)" if train else "(new eval run)")

    if train is None:
        return
    name = train.data.tags.get("registered_model") or f"{a.model}-{a.bench}-{a.task}"   # legacy: <policy>-<bench>-<task>
    for mv in client.search_model_versions(f"name = '{name}'"):
        if mv.run_id == train.info.run_id:
            client.set_model_version_tag(name, mv.version, f"eval_{a.task}_success_rate", f"{metrics['eval/success_rate']:.3f}")
            client.set_model_version_tag(name, mv.version, f"eval_{a.task}_score", f"{metrics['eval/score']:.1f}")
            print(f"tagged model {name} v{mv.version}")


if __name__ == "__main__":
    main()
