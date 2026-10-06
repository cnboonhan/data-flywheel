# data-flywheel

Monorepo for training and evaluating **action models** (system 1, VLA policies) and **agentic models** (system 2).

```
datasets/      downloaded datasets (gitignored)
checkpoints/   downloaded checkpoints (gitignored)
eval/system1/  VLA benchmarks: RoboDojo, RoboTwin (git submodules)
eval/system2/  agentic eval environments: IsaacLab-Arena (git submodule)
sensors/       data-collection hardware: yubi-hw (CAD), yubi-sw (ROS software) (git submodules)
scripts/       download.sh (repo lists), hf_download.py (downloader)
```

## Download data and checkpoints

Add repos to `DATASETS=(...)` or `CHECKPOINTS=(...)` in `scripts/download.sh`, then run:

```bash
bash scripts/download.sh   # pick repos, see size and time, confirm
```

Entries in `DATASETS` download to `datasets/<name>/`, and entries in `CHECKPOINTS` to `checkpoints/<name>/`.

- Gated repos you can't access are shown but can't be selected. Run `hf auth login`, then request access on the repo's page.
- Finished downloads are skipped, and interrupted ones resume.
- Use `--dry-run` to only show the plan, or `--all --yes` for non-interactive runs.

## Evaluation

```bash
git submodule update --init --recursive
```

| Benchmark | Path | Simulator | Env |
|---|---|---|---|
| [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) | `eval/system1/RoboDojo` | Isaac Sim 5.1 | conda `RoboDojo` |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) | `eval/system1/RoboTwin` | SAPIEN | conda `RoboTwin` |
| [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) | `eval/system2/IsaacLab-Arena` | Isaac Sim 6.1 | uv `.venv` |

System 1 policies plug in through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) at `XPolicyLab/policy/<POLICY>/` inside each benchmark.

**Gotchas**
- **Driver:** NVIDIA 580-open is required, because Isaac Sim 5.1 crashes on 595. It's held with `apt-mark hold`.
- **Submodule versions:** the RoboDojo and RoboTwin installers move XPolicyLab to its latest commit. Run `git submodule update` inside the benchmark afterwards to go back to the pinned version.
- **Arena `uv.lock`:** upstream's lockfile is stale. Regenerate it locally and keep it out of commits:
  ```bash
  cd eval/system2/IsaacLab-Arena && uv lock && uv sync --extra dev
  git update-index --skip-worktree uv.lock   # run once per clone; undo with --no-skip-worktree before pulling
  ```

## Plan

**Action models (system 1)**
- [ ] Reproduce published RoboDojo / RoboTwin baselines
- [ ] Define the data mix (sim, teleop, synthetic) and fine-tune
- [ ] Use failure analysis to choose the next round of data

**Agentic models (system 2)**
- [ ] Define the agent ↔ action-model interface
- [ ] Build agentic eval environments in IsaacLab-Arena

## Status

| Date | Item | Status |
|---|---|---|
| 2026-10-06 | RoboDojo installed and verified (18/18 checks, sim episode runs) | ✅ |
| 2026-10-06 | IsaacLab-Arena verified on driver 580 (rollout + camera test) | ✅ |
| 2026-10-06 | RoboTwin installed (torch 2.7 cu128 for RTX 5090); assets in place; 1-episode collection works | ✅ |
