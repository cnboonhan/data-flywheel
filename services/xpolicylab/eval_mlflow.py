#!/usr/bin/env python
"""Record a RoboDojo evaluation in MLflow.

Reads the newest `_result.json` RoboDojo wrote under eval_result/ for the
given policy/task/ckpt, logs success_rate, score and eval_time as metrics on
the training run that produced the checkpoint (found by run name) and tags
the registered model version with them. Without a matching run, a new run
named eval-<...> is created in the same experiment.

    python eval_mlflow.py --policy ACT --bench RoboDojo --task stack_bowls --ckpt stack_bowls --env-cfg arx_x5 --action joint --seed 0 --result-root <RoboDojo>/eval_result
"""

import argparse
import glob
import json
import os

import mlflow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--policy", "--bench", "--task", "--ckpt", "--env-cfg", "--action"):
        ap.add_argument(a, required=True)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--experiment", default="xpolicylab")
    args = ap.parse_args()

    pattern = f"{args.result_root}/{args.bench}/{args.task}/{args.policy}/{args.env_cfg}/{args.seed}_ckpt_name={args.ckpt},action_type={args.action}/*/_result.json"
    results = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not results:
        raise SystemExit(f"no _result.json under {pattern}")
    path = results[-1]
    r = json.load(open(path))
    metrics = {"eval/success_rate": float(r.get("success_rate", 0)), "eval/score": float(r.get("score", 0)), "eval/episodes": float(r.get("eval_time", 0))}
    print(f"{path}: {metrics}")

    mlflow.set_experiment(args.experiment)
    run_name = f"{args.policy}-{args.bench}-{args.ckpt}-{args.env_cfg}-{args.action}-{args.seed}"
    runs = mlflow.search_runs(experiment_names=[args.experiment], filter_string=f"tags.mlflow.runName = '{run_name}'", max_results=1, order_by=["start_time DESC"])
    run_id = runs.iloc[0]["run_id"] if len(runs) else None
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else f"eval-{run_name}") as run:
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"eval_task": args.task, "eval_result": path, "eval_env": "RoboDojo"})
        print("logged to run", run.info.run_id, "(training run)" if run_id else "(new eval run)")

    client = mlflow.MlflowClient()
    name = f"{args.policy}-{args.bench}-{args.ckpt}"
    for mv in client.search_model_versions(f"name = '{name}'"):
        if run_id and mv.run_id == run_id:
            client.set_model_version_tag(name, mv.version, f"eval_{args.task}_success_rate", f"{metrics['eval/success_rate']:.3f}")
            client.set_model_version_tag(name, mv.version, f"eval_{args.task}_score", f"{metrics['eval/score']:.1f}")
            print(f"tagged model {name} v{mv.version}")


if __name__ == "__main__":
    main()
