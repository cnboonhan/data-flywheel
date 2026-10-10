# slurm

GPU work (training, evaluation) as Slurm jobs, submitted by hand from a login node; Gitea doesn't submit them.

```
slurm/
  train.sbatch, evaluate.sbatch   generic launchers: <model> <embodiment> ...
  models/<model>/                 what differs per model: train.sh, eval.sh (+ its own code, e.g. mlp/train.py)
  embodiments/<robot>.yaml        what differs per robot: arms, joint and gripper dims, cameras, env_cfg names
  lib/                            shared: paths and env setup, MLflow naming, XPolicyLab hooks, RoboDojo shim
```

## Submit jobs

1. Load the credentials and pick a partition (the jobs name none and the cluster has no default).
   ```bash
   set -a; . /tier1/htx_boonhan/services/slurm.env; set +a; export SBATCH_PARTITION=<partition>
   cd /tier1/htx_boonhan/services/slurm-logs; S=$FLYWHEEL_ROOT/services/slurm
   ```
2. Train.
   ```bash
   sbatch --export=ALL $S/train.sbatch act arx_x5 RoboDojo/stack_bowls -- --num_epochs 30 --save_freq 30
   sbatch --export=ALL $S/train.sbatch mlp arx_x5 'RoboDojo/*,galaxeaOpenWorldDataset/*' --mix arx_x5-all -- --epochs 200
   ```
3. Evaluate.
   ```bash
   sbatch --export=ALL $S/evaluate.sbatch act arx_x5 stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 joint 0 5
   ```
4. Watch or cancel.
   ```bash
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

**Add a model:** create `models/<name>/` with `train.sh` and `eval.sh`, and its env (built by Gitea `setup-envs`). For a model that reads hdf5 itself, copy `models/mlp/` and replace the DATA, MODEL and LOOP parts of `train.py`.

**Recipe contract.** `train.sh` sets `ENV` (an env under `ENVS_DIR` with mlflow and pyyaml) and defines `train()`; `eval.sh` sets `ENV` and defines `evaluate()`. The launcher has already set `MODEL`, `MODEL_DIR`, `DATA` (array), `MIX`, `ACTION`, `SEED`, `EXTRA` (array, the args after `--`) for training, or `TASK`, `CKPT`, `ACTION`, `SEED`, `EVAL_NUM` for evaluation, plus `EMB_NAME`, `EMB_DATA_ENV_CFG`, `EMB_XPL_ENV_CFG`, `EMB_EVAL_ENV_CFG` from the embodiment and the paths in `lib/common.sh`. Training code goes through `TrainRun` (`lib/flywheel_mlflow.py`) so the names below hold.

## Embodiments

**Add a robot:** copy `embodiments/_example.yaml` to `embodiments/<robot>.yaml`, and add an adapter that writes its data ([adapter/](../gitea/adapter/README.md)). Models and robots don't touch each other.

| Key | Meaning |
|---|---|
| `arms` | xspark key prefixes, e.g. `[left, right]` |
| `arm_dim`, `ee_dim` | joint and gripper/hand dimensions per arm |
| `cameras` | camera names |
| `data_env_cfg` | the folder under `processed/xpolicylab/<bench>/<task>/` the adapter writes |
| `xpolicylab.env_cfg` | a key of XPolicyLab's `utils/robot/_robot_info.json` (ACT/DP action dimension) |
| `xpolicylab.eval_env_cfg` | a RoboDojo env_cfg; empty when RoboDojo can't simulate the robot |

Today: `arx_x5`, which Galaxea r1lite data also uses (same 14-D layout). A robot that XPolicyLab or RoboDojo doesn't know can still train with models that read hdf5 directly; a `_robot_info.json` entry or a RoboDojo robot needs a fork of the submodule.

## Naming in MLflow

| | Convention | Example |
|---|---|---|
| experiment | `train/<embodiment>/<mix>` | `train/arx_x5/RoboDojo.stack_bowls` |
| run | `<model>-<action>-s<seed>-<YYmmdd-HHMMSS>` | `ACT-joint-s0-261009-121010` |
| registered model | `<model>.<embodiment>.<mix>`, one version per run (artifacts `checkpoints/`) | `ACT.arx_x5.RoboDojo.stack_bowls` |
| mix | `--mix`, or the single dataset with `/` → `.` | `RoboDojo.stack_bowls` |
| metrics | `train/<name>`, `val/<name>` (DP: its own `train_*`/`val_*` keys); `eval/success_rate`, `eval/score`, `eval/episodes` | |

Every run records `model`, `embodiment`, `datasets`, `episodes`, `data_fingerprint` (hash of the episodes' paths, sizes and mtimes: equal fingerprints mean identical data), `action`, `action_dim`, `git_commit` (`-dirty` with local changes to tracked files), `slurm_job`, the model's arguments, and `config/mix.json` with the episode list. Model versions carry the same tags; evaluation adds `eval_<task>_success_rate` and `eval_<task>_score`. Older runs live in the experiment `xpolicylab` under the models `<policy>-<bench>-<task>`; evaluation still finds them.

## Environment

- **Credentials and paths:** `ctl.sh up` writes `$STATE_DIR/slurm.env` (readable only by you): MLflow and S3 credentials, `TRITON_URL`/`TRITON_TOKEN`, `SERVICE_NODE`, and `FLYWHEEL_ROOT`, `PROJECT_ROOT`, `ENVS_DIR`, `ROBODOJO_DIR`, `BUCKETS_DIR`, `DATA_ROOT`. The jobs run the code in this checkout (`FLYWHEEL_ROOT`).
- **Envs** (RoboDojo eval env, ACT/DP policy envs): built by the Gitea workflow `setup-envs` ([gitea/setup/](../gitea/setup/README.md)).
- **Data layout:** the checkout holds code only; `lib/xpolicylab.sh` links `XPolicyLab/policy/<P>/{processed_data,checkpoints}`, `policy/DP/data` → `$DATA_ROOT/<P>/`, RoboDojo `eval_result` → `$ROBODOJO_DIR/eval_result`, `data` → `$BUCKETS_DIR/processed/xpolicylab`, `Assets` → `$ROBODOJO_DIR/Assets`.

**Evaluation on the GB300 nodes.** `evaluate.sbatch` already handles the following (except the GPU count, which is your choice); keep it that way when you change it.

- **Run the simulation on the GPU device.** RoboDojo defaults Isaac Lab's simulation device to CPU, and on these nodes Isaac Sim 5.1 then delivers no camera frames, so RoboDojo's capture loop spins forever. Use an `eval_env_cfg` ending in `_gpu` (`arx_x5_gpu` for `arx_x5`): the job derives it from the base config with `device: cuda:0` and aliases the base's evaluation layouts and checkpoints. Expect scores to differ slightly from the official CPU-physics numbers.
- **Keep C01 and C02 excluded.** Their Kit viewport doesn't render, and Isaac Lab's viewport camera then fails with "Accessed invalid null prim".
- **To give the policy server its own GPU,** submit with `sbatch --gres=gpu:2 ...`; with the default single GPU, the simulator and the policy server share it.
- **Keep the RoboDojo shim on `PYTHONPATH`.** RoboDojo assumes CPU tensors in places (`np.asarray` on a tensor, `.numpy()`); `lib/robodojo-shim/sitecustomize.py` converts CUDA tensors and makes the viewport camera placement best-effort. Drop it once a RoboDojo fork has a `device` setting.
- **`ffmpeg` comes from the env.** The nodes have none; `install-robodojo.sh` puts `imageio-ffmpeg`'s static build in `$ROBODOJO_DIR/bin`, which the job puts on `PATH`. The same installer handles the aarch64 Isaac Sim 5.1 and torch cu128 wheels, the `libgomp` and NVRTC 12.9 preloads (torch's 12.8 doesn't know sm_103), user-space GL and curobo from source. The policy envs use the cu128 index because the pinned `torch==2.4.1` has no CUDA build for aarch64 Blackwell.

## What to expect

- **Training ACT on a new dataset** spends most of its first run in `process_data.sh` decoding JPEGs: about 15 min for one Galaxea task (330k frames), against about a minute for 30 epochs. Later runs reuse the decoded frames.
- **The MLP** trains in seconds, even on a mix of every dataset; use it to check the plumbing.
- **An evaluation** of 2 RoboDojo episodes of 800 steps takes about 10 min on one GB300. The first Isaac Sim start on a node also compiles RTX pipelines for a few minutes; the shared caches in `$ROBODOJO_DIR/cache` make later starts take about 15 s.
- **A short training run won't succeed** in evaluation (a 30-epoch ACT scores 0 on `stack_bowls`); use one to check the plumbing, and train for XPolicyLab's default 6000 epochs for real results.
