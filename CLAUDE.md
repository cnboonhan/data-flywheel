# data-flywheel

Monorepo for training and evaluating action models (system 1, VLA policies) and agentic models (system 2). See `README.md` for the human-facing overview.

## Layout

- Data and weights live in the S3 gateway (`services/`), not in the checkout. Gitea workflows fetch from Hugging Face: `services/gitea/ingest/download-datasets-hf.yml` into `raw/open_datasets/`, `download-models-hf.yml` into the MLflow model registry (self-contained workflows; each header lists the known repos).
- `eval/system1/{RoboDojo,RoboTwin}`: VLA benchmarks. `eval/system2/IsaacLab-Arena`: agentic eval environments.
- `sensors/{yubi-hw,yubi-sw}`: data-collection hardware.
- `sensors/real2sim/`: Isaac Sim robot (Arena venv, bundled ROS 2 Jazzy, `ROS_DOMAIN_ID=42`) + Nav2 in Docker on a known map; goals via `/goal_pose`. A real robot shares the LAN on domain 0: never publish to `/hdas/*` or `/motion_target/*` from sim.
- `sim/isaaclab_arena/`: Arena environment generation via a local LLM proxy and scripted demo recording (`envgen.sh`, `record.py`, `hooks/` import hooks: `cliproxy` endpoint, `ridgeback_franka_ik` mobile manipulator, `splat_scene` background from `ARENA_SPLAT_SCENE`).
- `sim/splat/`: Gaussian-splat training of real scenes; `sim/splat/3dgrut` is the 3DGRUT trainer (uv venv inside it, CUDA 12.8 bundled). Data in `datasets/splat/` (gitignored).
- Everything under `eval/`, `sensors/yubi-*` and `sim/splat/3dgrut` is an **upstream git submodule**.

## Rules

- **Never commit large files** (data, weights, assets, videos). To add a dataset or model, add it to the list in `services/gitea/ingest/download-datasets-hf.yml` or `download-models-hf.yml` and run that workflow. Don't download data unless asked.
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

## Docs

- **Reference, don't duplicate.** Each fact lives in one README, the one closest to the code it describes (`services/<service>/README.md`, `services/gitea/<stage>/README.md`, `sim/splat/README.md`, ...). Overviews (`docs/flywheel.md`, `services/README.md`, the top-level README) summarise in a line and link to it. When a fact changes, update its home; when adding to an overview, link instead of copying.

## Git

- Push goes over SSH on port 443, because port 22 is blocked from the cluster: `remote.origin.pushurl` is `ssh://git@ssh.github.com:443/cnboonhan/data-flywheel.git`. Fetch uses HTTPS. On a fresh clone, set the pushurl with `git config remote.origin.pushurl ssh://git@ssh.github.com:443/cnboonhan/data-flywheel.git`.
- Commit only when asked.
