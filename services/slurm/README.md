# slurm

GPU work (conversion, training, evaluation) runs as Slurm jobs on the `raus_manual` partition; Gitea Actions submits them over SSH and streams the log (`follow.sh`).

**Bridge.** `ctl.sh up` creates a key at `$STATE_DIR/act_runner/ssh/`, adds it to `~/.ssh/authorized_keys` with the forced command `slurm-submit` (`restrict`: sbatch from the pipelines checkout at the run's commit, status, log, cancel; nothing else), stores it as the repo secrets `SLURM_SSH_KEY` / `SLURM_SSH_HOST` (`SLURM_LOGIN_HOST` in `.env`), and writes `$STATE_DIR/slurm.env`: MLflow and S3 credentials plus `FLYWHEEL_ROOT`, `PROJECT_ROOT`, `ENVS_DIR`, `ROBODOJO_DIR`, `BUCKETS_DIR`, `DATA_ROOT`. Jobs read `$PIPELINES_ROOT` = `$STATE_DIR/pipelines` (a checkout of `admin/pipelines`).

**By hand** (same scripts, same env):

```bash
set -a; . $STATE_DIR/slurm.env; set +a; cd $STATE_DIR/slurm-logs; P=$STATE_DIR/pipelines/slurm
sbatch --export=ALL $P/install-robodojo.sbatch                       # Isaac Sim 5.1 + Isaac Lab + curobo + XPolicyLab + ffmpeg env
sbatch --export=ALL $P/install-policy-env.sbatch ACT                 # also DP, demo_policy
sbatch --export=ALL $P/download-robodojo.sbatch assets               # also: ckpt ACT, data stack_bowls
sbatch --export=ALL $P/convert-galaxea-xspark.sbatch 'galaxea-open-world-r1lite/*' 0 arx_x5
sbatch --export=ALL $P/train-xpolicylab.sbatch ACT Galaxea Arrange_Fruits_20250819_011 arx_x5 joint 0 --num_epochs 30 --save_freq 30
sbatch --export=ALL $P/evaluate-xpolicylab.sbatch ACT stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 arx_x5_gpu joint 0 5
squeue -u $USER; tail -f $STATE_DIR/slurm-logs/<job>-<id>.log
```

| Script | Notes |
|---|---|
| `install-robodojo.sbatch` | uv env `$ENVS_DIR/robodojo` (aarch64 wheels, cu128 torch, NVRTC 12.9 preload, user-space GL libs, conda shim); re-run after moving the checkout, editable installs follow `PROJECT_ROOT` |
| `install-policy-env.sbatch <policy>` | uv env `$ENVS_DIR/<policy>` with the versions that work on this node |
| `download-robodojo.sbatch assets\|ckpt <P>\|data <task>` | RoboDojo hub → `$ROBODOJO_DIR/Assets`, `$ROBODOJO_DIR/ckpt`, `s3://raw/open_datasets/robodojo/<task>` |
| `convert-galaxea-xspark.sbatch` | LeRobot v2.1 → xspark in `processed/xpolicylab/` |
| `train-xpolicylab.sbatch <policy> <bench> <task> <env_cfg> <action> <seed> [args]` | policy `process_data.sh` + training through `xpolicylab/train_mlflow.py` |
| `evaluate-xpolicylab.sbatch <policy> <task> <ckpt> <env_cfg> <action> <seed> [eval_num]` | XPolicyLab `eval.sh` in RoboDojo (Isaac Sim headless) + `xpolicylab/eval_mlflow.py` |

**Evaluation on the GB300 nodes.** Use an `env_cfg` ending in `_gpu` (derived from the base config with `device: cuda:0`; RoboDojo's CPU device gets no camera frames here). `robodojo-shim/sitecustomize.py` is injected via `PYTHONPATH` so CUDA tensors convert to numpy and a missing viewport camera is tolerated. `ckpt` is the checkpoint directory name: trained `<bench>-<task>-<env_cfg>-<action>-<seed>`, hub checkpoints the bare task name. The job excludes C01 and C02 (viewport never renders there; cause unknown). The policy server gets its own GPU with `--gres=gpu:2`.

**Data layout.** The checkout holds code only; the scripts link `XPolicyLab/policy/<P>/{processed_data,checkpoints}`, `policy/DP/data` → `$DATA_ROOT/<P>/`, RoboDojo `eval_result` → `$ROBODOJO_DIR/eval_result`, `data` → `$BUCKETS_DIR/processed/xpolicylab`, `Assets` → `$ROBODOJO_DIR/Assets`.
