"""MLflow conventions for flywheel training runs. Every training entry point goes through TrainRun.

    from embodiment import load
    from flywheel_mlflow import Mix, TrainRun

    emb = load("arx_x5")
    mix = Mix.resolve(["RoboDojo/stack_bowls", "galaxeaOpenWorldDataset/*"], emb, name="bowls-fruits")
    with TrainRun(model="ACT", embodiment=emb, mix=mix, action="joint", seed=0, params=vars(args)) as run:
        run.log({"loss": loss}, step=epoch)              # -> train/loss
        run.log({"loss": vloss}, step=epoch, phase="val")
        run.save_checkpoints(ckpt_dir)                   # -> artifacts checkpoints/ + a registered model version

Naming:

    dataset      <bench>/<task>; its episodes are processed/xpolicylab/<bench>/<task>/<embodiment data_env_cfg>/data/
    mix          a name for a set of datasets; default: the single dataset, '/' -> '.'
    experiment   train/<embodiment>/<mix>                every model on the same robot and data, side by side
    run          <model>-<action>-s<seed>-<YYmmdd-HHMMSS>
    model        <model>.<embodiment>.<mix>               one registered model per model, robot and mix
    version      one per run, source = the run's artifacts checkpoints/; tagged like the run
    metrics      train/<name>, val/<name>; step = epoch or optimizer step (the param step_unit says which)
    params       model, embodiment, mix, datasets, episodes, data_fingerprint, action, seed, git_commit, slurm_job, + the script's
    tags         registered_model, plus whatever the caller passes (e.g. xpolicylab_ckpt for RoboDojo evaluation)

data_fingerprint hashes every episode's path, size and mtime, so two runs with the same fingerprint saw identical data.
MLflow connection: MLFLOW_TRACKING_URI/USERNAME/PASSWORD and MLFLOW_TRACKING_SERVER_CERT_PATH (slurm.env).
"""

import dataclasses
import glob
import hashlib
import os
import subprocess
import time

import mlflow

BUCKETS_DIR = os.environ.get("BUCKETS_DIR", "/tier1/htx_boonhan/services/versitygw/buckets")
XSPARK_ROOT = os.path.join(BUCKETS_DIR, "processed", "xpolicylab")


@dataclasses.dataclass
class Mix:
    name: str
    datasets: list  # <bench>/<task>
    episodes: list  # absolute paths to episode_*.hdf5

    @classmethod
    def resolve(cls, patterns, embodiment, name=None, root=XSPARK_ROOT):
        """Expand <bench>/<task> globs into the datasets that have episodes for this embodiment."""
        sub = embodiment.data_env_cfg
        datasets = sorted({os.path.relpath(os.path.dirname(d), root) for p in patterns
                           for d in glob.glob(os.path.join(root, p, sub)) if os.path.isdir(os.path.join(d, "data"))})
        if not datasets:
            raise SystemExit(f"no datasets match {patterns} with {sub}/data under {root}")
        episodes = [e for d in datasets for e in sorted(glob.glob(os.path.join(root, d, sub, "data", "episode_*.hdf5")))]
        if name is None:
            if len(datasets) > 1:
                raise SystemExit(f"{len(datasets)} datasets match ({', '.join(datasets[:5])}...); name the mix (--mix)")
            name = datasets[0].replace("/", ".")
        return cls(name, datasets, episodes)

    def fingerprint(self):
        h = hashlib.sha1()
        for e in self.episodes:
            st = os.stat(e)
            h.update(f"{os.path.relpath(e, XSPARK_ROOT)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        return h.hexdigest()[:12]


def _git_commit():
    root = os.environ.get("FLYWHEEL_ROOT", os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(["git", "-C", root, "describe", "--always", "--dirty", "--abbrev=7"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


class TrainRun:
    def __init__(self, model, embodiment, mix, action, seed, params=None, tags=None, step_unit="epoch"):
        self.model, self.embodiment, self.mix, self.action, self.seed = model, embodiment, mix, action, seed
        self.experiment = f"train/{embodiment.name}/{mix.name}"
        self.run_name = f"{model}-{action}-s{seed}-{time.strftime('%y%m%d-%H%M%S')}"
        self.model_name = f"{model}.{embodiment.name}.{mix.name}"
        self.params = dict(params or {})
        self.tags = dict(tags or {})
        self.step_unit = step_unit
        self.run = None

    def __enter__(self):
        # Multipart uploads hand out presigned URLs on the compose-internal S3 host, which nodes can't reach.
        os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
        mlflow.set_experiment(self.experiment)
        self.run = mlflow.start_run(run_name=self.run_name)
        self.git_commit = _git_commit()
        self.fingerprint = self.mix.fingerprint()
        mlflow.log_params({
            "model": self.model, "embodiment": self.embodiment.name, "mix": self.mix.name,
            "datasets": ",".join(self.mix.datasets)[:6000], "episodes": len(self.mix.episodes), "data_fingerprint": self.fingerprint,
            "action": self.action, "action_dim": self.embodiment.action_dim(self.action), "seed": self.seed, "step_unit": self.step_unit,
            "git_commit": self.git_commit, "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "host": os.uname().nodename,
        })
        own = {k: v for k, v in self.params.items() if k not in ("model", "embodiment", "mix", "action", "seed", "data")}
        if own:
            mlflow.log_params({k: str(v)[:6000] for k, v in own.items()})
        mlflow.set_tags({"registered_model": self.model_name, **self.tags})
        mlflow.log_dict({"mix": self.mix.name, "embodiment": dataclasses.asdict(self.embodiment), "datasets": self.mix.datasets,
                         "episodes": [os.path.relpath(e, XSPARK_ROOT) for e in self.mix.episodes]}, "config/mix.json")
        print(f"mlflow: experiment {self.experiment}, run {self.run_name} ({self.run.info.run_id})", flush=True)
        return self

    def log(self, metrics, step, phase="train"):
        mlflow.log_metrics({f"{phase}/{k}": float(v) for k, v in metrics.items()}, step=step)

    def save_checkpoints(self, ckpt_dir, description=None):
        """Upload ckpt_dir as artifacts checkpoints/ and register it as a new version of <model>.<embodiment>.<mix>."""
        mlflow.log_artifacts(ckpt_dir, artifact_path="checkpoints")
        client = mlflow.MlflowClient()
        try:
            client.create_registered_model(self.model_name, tags={"model": self.model, "embodiment": self.embodiment.name, "mix": self.mix.name},
                                           description=description or f"{self.model} on {self.embodiment.name}, mix {self.mix.name}")
        except mlflow.exceptions.RestException as e:
            if e.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        # MLflow 3: a runs:/ URI must name a LoggedModel, so register the artifact directory with its run_id.
        mv = client.create_model_version(self.model_name, f"{self.run.info.artifact_uri}/checkpoints", run_id=self.run.info.run_id, tags={
            "mix": self.mix.name, "datasets": ",".join(self.mix.datasets)[:5000], "action": self.action, "seed": str(self.seed),
            "run": self.run_name, "data_fingerprint": self.fingerprint, "git_commit": self.git_commit, **self.tags})
        print(f"mlflow: registered {self.model_name} version {mv.version}", flush=True)
        return mv

    def __exit__(self, exc_type, exc, tb):
        mlflow.end_run("FAILED" if exc_type else "FINISHED")
        return False
