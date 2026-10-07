# scripts

| Script | Purpose |
|---|---|
| `download.sh` | Lists of Hugging Face repos (`DATASETS=(...)`, `CHECKPOINTS=(...)`); runs the downloader |
| `hf_download.py` | Downloader: checks access and size, lets you pick repos, resumes interrupted downloads |
| `arena_envgen.sh` | Runs IsaacLab-Arena agentic environment generation through a local LLM proxy (see [`eval/system2`](../eval/system2/README.md)) |
| `arena_cliproxy/` | `sitecustomize.py` that registers the `cliproxy` inference endpoint in Arena; loaded by `arena_envgen.sh` |
| `arena_nurec.sh` | Smoke-tests a NuRec (Gaussian splat) scene inside Arena and records robot-camera videos (see [`sim/`](../sim/README.md)) |
| `arena_nurec/` | External Arena environment: registers the downloaded NuRec room as a background and drops `cube_goal_pose`'s Franka + cube into it |

## Download data and checkpoints

Add repos to `DATASETS=(...)` or `CHECKPOINTS=(...)` in `download.sh`, then run:

```bash
bash scripts/download.sh   # pick repos, see size and time, confirm
```

Entries in `DATASETS` download to [`datasets/<name>/`](../datasets/README.md), and entries in `CHECKPOINTS` to [`checkpoints/<name>/`](../checkpoints/README.md).

Entry format: `"<model|dataset> <repo_id> <include globs, comma-separated or *> <local folder name>"`. The type is the Hugging Face repo type; some checkpoints live in dataset repos.

- Gated repos you can't access are shown but can't be selected. Run `hf auth login`, then request access on the repo's page.
- Finished downloads are skipped, and interrupted ones resume.
- Use `--dry-run` to only show the plan, or `--all --yes` for non-interactive runs.
