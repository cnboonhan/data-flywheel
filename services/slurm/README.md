# slurm

GPU work (training, evaluation) runs as Slurm jobs, submitted by hand from a login node. Gitea doesn't submit Slurm jobs. The jobs name no partition and the cluster has no default, so pick one when submitting: `export SBATCH_PARTITION=<partition>` (as below) or `sbatch -p <partition>`.

```
slurm/
  train.sbatch, evaluate.sbatch   generic launchers: <model> <embodiment> ...
  models/<model>/                 what differs per model: train.sh, eval.sh (+ its own code, e.g. mlp/train.py)
  embodiments/<robot>.yaml        what differs per robot: arms, joint and gripper dims, cameras, env_cfg names
  lib/                            shared: paths and env setup, MLflow naming, XPolicyLab hooks, RoboDojo shim
```

A new model is a new `models/<name>/` folder (plus its env, built by Gitea `setup-envs`); a new robot is a new `embodiments/<name>.yaml` (plus an adapter writing its data). Neither touches the other.

```bash
set -a; . /tier1/htx_boonhan/services/slurm.env; set +a; export SBATCH_PARTITION=<partition>
cd /tier1/htx_boonhan/services/slurm-logs; S=$FLYWHEEL_ROOT/services/slurm
sbatch --export=ALL $S/train.sbatch act arx_x5 RoboDojo/stack_bowls -- --num_epochs 30 --save_freq 30
sbatch --export=ALL $S/train.sbatch mlp arx_x5 'RoboDojo/*,galaxeaOpenWorldDataset/*' --mix arx_x5-all -- --epochs 200
sbatch --export=ALL $S/evaluate.sbatch act arx_x5 stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 joint 0 5
squeue -u $USER; tail -f <job>-<id>.log; scancel <id>
```

| Launcher | Arguments |
|---|---|
| `train.sbatch` | `<model> <embodiment> <data>[,<data>...] [--mix NAME] [--action joint\|ee] [--seed N] [-- model args]`. `<data>` is a `<bench>/<task>` glob under `processed/xpolicylab`; the episodes are read from `<bench>/<task>/<data_env_cfg>/data/`. |
| `evaluate.sbatch` | `<model> <embodiment> <task> <ckpt> [action] [seed] [eval_num]`. For ACT/DP, `<ckpt>` is XPolicyLab's `<bench>-<task>-<env_cfg>-<action>-<seed>` (the training run's tag `xpolicylab_ckpt`) or a RoboDojo hub name. |

## Models

| Model | Env | Train | Evaluate |
|---|---|---|---|
| `act` | `act` | XPolicyLab ACT: `process_data.sh`, `imitate_episodes.py` through `lib/xpolicylab_train.py`; one dataset | RoboDojo via XPolicyLab `eval.sh` |
| `dp` | `dp` | XPolicyLab Diffusion Policy: zarr, `train.py` (Hydra overrides after `--`); one dataset | RoboDojo via XPolicyLab `eval.sh` |
| `mlp` | `act` | `models/mlp/train.py`: MLP on proprio state → action, reads any mix of hdf5 datasets | none |

**Recipe contract.** `models/<m>/train.sh` sets `ENV` (an env under `ENVS_DIR` with mlflow and pyyaml) and defines `train()`; `eval.sh` sets `ENV` and defines `evaluate()`. The launcher has already set `MODEL`, `MODEL_DIR`, `DATA` (array), `MIX`, `ACTION`, `SEED`, `EXTRA` (array, the args after `--`) for training, or `TASK`, `CKPT`, `ACTION`, `SEED`, `EVAL_NUM` for evaluation, plus `EMB_NAME`, `EMB_DATA_ENV_CFG`, `EMB_XPL_ENV_CFG`, `EMB_EVAL_ENV_CFG` from the embodiment and the paths in `lib/common.sh`. Training code goes through `TrainRun` (`lib/flywheel_mlflow.py`) so the names below hold. For a model that reads hdf5 itself, copy `models/mlp/` and replace the DATA, MODEL and LOOP parts of `train.py`.

## Embodiments

`embodiments/<robot>.yaml`: `arms` (xspark key prefixes, e.g. `[left, right]`), `arm_dim`, `ee_dim` (per arm), `cameras`, `data_env_cfg` (the folder the adapter writes) and, for XPolicyLab and RoboDojo, `xpolicylab.env_cfg` (a key of XPolicyLab's `utils/robot/_robot_info.json`) and `xpolicylab.eval_env_cfg` (a RoboDojo env_cfg, empty when RoboDojo can't simulate the robot). `_example.yaml` is the template. Today: `arx_x5`, which Galaxea r1lite data also uses (same 14-D layout). A robot that XPolicyLab or RoboDojo doesn't know can still train with models that read hdf5 directly; a `_robot_info.json` entry or a RoboDojo robot needs a fork of the submodule.

## Naming in MLflow

| | Convention | Example |
|---|---|---|
| experiment | `train/<embodiment>/<mix>` | `train/arx_x5/RoboDojo.stack_bowls` |
| run | `<model>-<action>-s<seed>-<YYmmdd-HHMMSS>` | `ACT-joint-s0-261009-121010` |
| registered model | `<model>.<embodiment>.<mix>`, one version per run (artifacts `checkpoints/`) | `ACT.arx_x5.RoboDojo.stack_bowls` |
| mix | `--mix`, or the single dataset with `/` → `.` | `RoboDojo.stack_bowls` |
| metrics | `train/<name>`, `val/<name>` (DP: its own `train_*`/`val_*` keys); `eval/success_rate`, `eval/score`, `eval/episodes` | |

Every run records `model`, `embodiment`, `datasets`, `episodes`, `data_fingerprint` (hash of the episodes' paths, sizes and mtimes: equal fingerprints mean identical data), `action`, `action_dim`, `git_commit` (`-dirty` with local changes to tracked files), `slurm_job`, the model's arguments, and `config/mix.json` with the episode list. Model versions carry the same tags; evaluation adds `eval_<task>_success_rate` and `eval_<task>_score`. Runs from before this layout stay in the experiment `xpolicylab` under the models `<policy>-<bench>-<task>`; evaluation still finds them.

## Environment

**Credentials and paths.** `ctl.sh up` writes `$STATE_DIR/slurm.env`, readable only by you: MLflow and S3 credentials plus `FLYWHEEL_ROOT`, `PROJECT_ROOT`, `ENVS_DIR`, `ROBODOJO_DIR`, `BUCKETS_DIR`, `DATA_ROOT`. The jobs run the code in this checkout (`FLYWHEEL_ROOT`).

**Envs** (RoboDojo eval env, ACT/DP policy envs) are built by the Gitea workflow `setup-envs` in a job container ([gitea/setup/](../gitea/setup/README.md)). `ctl.sh up` dispatches it when one is missing or was built for another checkout, and `ctl.sh setup` forces it.

**Evaluation on the GB300 nodes.** `arx_x5`'s `eval_env_cfg` is `arx_x5_gpu`, derived from the base config with `device: cuda:0`; RoboDojo's CPU device gets no camera frames here. `lib/robodojo-shim/sitecustomize.py` is injected via `PYTHONPATH` so CUDA tensors convert to numpy and a missing viewport camera is tolerated. The job excludes C01 and C02 (viewport never renders there; cause unknown). The policy server gets its own GPU with `--gres=gpu:2`.

**Data layout.** The checkout holds code only; `lib/xpolicylab.sh` links `XPolicyLab/policy/<P>/{processed_data,checkpoints}`, `policy/DP/data` → `$DATA_ROOT/<P>/`, RoboDojo `eval_result` → `$ROBODOJO_DIR/eval_result`, `data` → `$BUCKETS_DIR/processed/xpolicylab`, `Assets` → `$ROBODOJO_DIR/Assets`.
