# data-flywheel

Monorepo for training and evaluating action models (system 1, VLA policies) and agentic models (system 2). See `README.md` for the human-facing overview.

## Layout

- `datasets/`, `checkpoints/`: Hugging Face downloads. Gitignored except `.gitkeep`.
- `scripts/download.sh`: the repo lists (`DATASETS=(...)`, `CHECKPOINTS=(...)`). `scripts/hf_download.py` is the downloader.
- `eval/system1/{RoboDojo,RoboTwin}`: VLA benchmarks. `eval/system2/IsaacLab-Arena`: agentic eval environments.
- `sensors/{yubi-hw,yubi-sw}`: data-collection hardware.
- `sim/nurec/`: NuRec real-to-sim pipeline (scripts, Arena hook, README). `sim/nurec/3dgrut` is the 3DGRUT trainer.
- Everything under `eval/` and `sensors/`, plus `sim/nurec/3dgrut`, is an **upstream git submodule**.

## Rules

- **Never commit large files** (data, weights, assets, videos). To add a dataset or checkpoint, add an entry to `scripts/download.sh`. Don't download data unless asked; use `--dry-run` to check gating and size.
- **Treat submodules as upstream code.** Don't commit inside them, because their remotes are third-party. Changes belong on a fork; ask first. data-flywheel only records each submodule's commit.
- **Keep nested submodules pinned.** The RoboDojo and RoboTwin installers move XPolicyLab to its latest commit. After running them, run `git submodule update` inside that benchmark.
- **IsaacLab-Arena `uv.lock`:** upstream's lock is stale, and it's regenerated locally with `git update-index --skip-worktree uv.lock`. Use `uv sync --frozen --extra dev` so uv doesn't rewrite it. See the README before pulling Arena.
- **Run one Isaac Sim instance at a time.** The GPU is an RTX 5090 Laptop with 24 GB.
- **NVIDIA driver 580-open is held on purpose**, because Isaac Sim 5.1 crashes on 595. Don't change drivers or system packages without asking.
- **Off-limits:** don't modify, run or copy from `~/workspaces/htx-robotics-release` or `~/workspaces/snippets`.
- **Isaac Sim crash reports:** Isaac Sim uploads them to NVIDIA by default. Pass `--/crashreporter/enabled=false` to ad-hoc Isaac Sim runs.

## Environments

| Benchmark | Env | Activate |
|---|---|---|
| RoboDojo | conda `RoboDojo` (Py 3.11, Isaac Sim 5.1, torch 2.7 cu128) | `eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate RoboDojo` |
| RoboDojo demo policy | conda `demo-policy` | passed to XPolicyLab `eval.sh` by name |
| RoboTwin | conda `RoboTwin` (Py 3.10, SAPIEN, torch 2.7 cu128, CUDA 12.8 nvcc in env) | `conda activate RoboTwin` |
| IsaacLab-Arena | uv `.venv` (Py 3.12, Isaac Sim 6.1) | `.venv/bin/python` |

The host has no system `nvcc`, and `python3` on PATH is Homebrew's (no `yaml`). Always run benchmark scripts inside their env.

## Verification

```bash
# RoboDojo: install checker (inside the RoboDojo env), then policy wiring without the simulator
cd eval/system1/RoboDojo && bash scripts/internal/verify_install.sh --policy-dir XPolicyLab/policy/demo_policy --policy-env demo-policy
cd XPolicyLab/policy/demo_policy && EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 demo-policy RoboDojo

# RoboTwin: collect episodes (set episode_num in a copy of env_cfg/task_config/demo_clean.yml for a quick test, then delete it)
cd eval/system1/RoboTwin && bash collect_data.sh beat_block_hammer demo_clean 0

# IsaacLab-Arena
cd eval/system2/IsaacLab-Arena && OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  .venv/bin/python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_steps 20 cube_goal_pose
```

XPolicyLab's `eval.sh` arguments: `<bench> <task> <ckpt> <env_cfg_type> <action_type> <seed> <policy_gpu> <env_gpu> <policy_env> <sim_env>`.

## Git

- Push goes over SSH (`remote.origin.pushurl`); fetch uses HTTPS.
- Commit only when asked.
