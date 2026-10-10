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

Every run records `model`, `embodiment`, `datasets`, `episodes`, `data_fingerprint` (hash of the episodes' paths, sizes and mtimes: equal fingerprints mean identical data), `action`, `action_dim`, `git_commit` (`-dirty` with local changes to tracked files), `slurm_job`, the model's arguments, and `config/mix.json` with the episode list. Model versions carry the same tags; evaluation adds `eval_<task>_success_rate` and `eval_<task>_score`. Runs from before this layout stay in the experiment `xpolicylab` under the models `<policy>-<bench>-<task>`; evaluation still finds them.

## Environment

- **Credentials and paths:** `ctl.sh up` writes `$STATE_DIR/slurm.env` (readable only by you): MLflow and S3 credentials, `TRITON_URL`/`TRITON_TOKEN`, and `FLYWHEEL_ROOT`, `PROJECT_ROOT`, `ENVS_DIR`, `ROBODOJO_DIR`, `BUCKETS_DIR`, `DATA_ROOT`. The jobs run the code in this checkout (`FLYWHEEL_ROOT`).
- **Envs** (RoboDojo eval env, ACT/DP policy envs): built by the Gitea workflow `setup-envs` ([gitea/setup/](../gitea/setup/README.md)).
- **Data layout:** the checkout holds code only; `lib/xpolicylab.sh` links `XPolicyLab/policy/<P>/{processed_data,checkpoints}`, `policy/DP/data` → `$DATA_ROOT/<P>/`, RoboDojo `eval_result` → `$ROBODOJO_DIR/eval_result`, `data` → `$BUCKETS_DIR/processed/xpolicylab`, `Assets` → `$ROBODOJO_DIR/Assets`.

**Evaluation on the GB300 nodes.** The policy server gets its own GPU (`--gres=gpu:2`). What it took on this hardware:

- **Simulation on the GPU device (`env_cfg = *_gpu`).** RoboDojo defaults Isaac Lab's simulation device to CPU, where Isaac Sim 5.1 on these nodes never delivers Replicator camera frames (verified with Isaac Lab's tiled camera: empty buffers on `cpu`, frames on `cuda:0`) and RoboDojo's capture kernel spins forever. `arx_x5`'s `eval_env_cfg` is `arx_x5_gpu`, derived by the job with `device: cuda:0` and the base's evaluation layouts and checkpoints aliased. GPU physics may score slightly differently from the official CPU numbers.
- **C01 and C02 are excluded.** Their Kit viewport stopped rendering on 2026-10-08 ("Accessed invalid null prim" in the viewport camera controller); software, GPU mode, scratch and extension order match C03/C04; cause not found.
- **RoboDojo assumes CPU tensors** in places (`np.asarray` on a tensor, `.numpy()`): `lib/robodojo-shim/sitecustomize.py`, injected via `PYTHONPATH`, converts CUDA tensors and makes the viewport camera placement best-effort. The proper fix is a RoboDojo fork with a `device` setting.
- **No `ffmpeg` on the nodes:** `install-robodojo.sh` puts `imageio-ffmpeg`'s static build in `$ROBODOJO_DIR/bin`, on the job's `PATH`. It also covers aarch64 Isaac Sim 5.1 and torch cu128 wheels, `libgomp` and NVRTC 12.9 preloads (torch's 12.8 doesn't know sm_103), user-space GL, curobo from source. The first Isaac Sim start on a node compiles RTX pipelines (minutes); shared caches in `$ROBODOJO_DIR/cache` make later starts 15 s.
- The policies pin `torch==2.4.1`, which has no CUDA build for aarch64 Blackwell; the envs use the cu128 index.

## Results

- **Train.** ACT, 30 epochs on `galaxeaOpenWorldDataset/Arrange_Fruits_20250819_011`: 14 min 28 s, mostly `process_data` decoding 330k JPEGs (the epochs take a minute); `val/loss` 86.1 → 0.754, 336 MB checkpoints. ACT, 3 epochs on `RoboDojo/stack_bowls` (frames already decoded): 32 s, `ACT.arx_x5.RoboDojo.stack_bowls` v1. MLP, 20 epochs on every RoboDojo and Galaxea dataset: 14 s, `MLP.arx_x5.arx_x5-test` v1.
- **Evaluate.** The 30-epoch ACT on `stack_bowls`, 2 episodes of 800 steps: 10 min 32 s on one GB300, `eval/success_rate 0.0` (not expected to succeed; the point is the plumbing), videos and `_result.json` on the run. The 3-epoch checkpoint, 1 episode: 7 min 10 s on C03, logged to its training run and tagged on `ACT.arx_x5.RoboDojo.stack_bowls` v1.
