# scripts

| Script | Purpose |
|---|---|
| `download.sh` | Lists of Hugging Face repos (`DATASETS=(...)`, `CHECKPOINTS=(...)`); runs the downloader |
| `hf_download.py` | Downloader: checks access and size, lets you pick repos, resumes interrupted downloads |
| `arena_envgen.sh` | Runs IsaacLab-Arena agentic environment generation through a local LLM proxy (see [`eval/system2`](../eval/system2/README.md)) |
| `arena_cliproxy/` | `sitecustomize.py` that registers the `cliproxy` inference endpoint in Arena; loaded by `arena_envgen.sh` |

NuRec reconstruction and Arena smoke-test scripts live in [`sim/nurec/`](../sim/nurec/README.md).

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
