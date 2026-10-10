# The flywheel, end to end

The path one dataset takes through the [architecture diagram](../architecture.html): collect → look → adapt → train → evaluate → serve. Each step links to the README that holds its details; this page only connects them.

```
 Collect                 Look                       Adapt                      Train                  Evaluate             Serve
 ───────                 ────                       ─────                      ─────                  ────────             ─────
 public datasets,   ──▶  s3://raw mirrored     ──▶  s3://processed/       ──▶  Slurm GPU job     ──▶  Slurm GPU job   ──▶  Triton, from the
 UMI / teleop /          into FiftyOne,             xpolicylab/<bench>/        (sbatch by hand),      RoboDojo sim,        MLflow registry
 robot bags              bags open in Rerun         (adapter workflows)        MLflow runs, registry  scores on the run    (alias `triton`)
 (s3://raw)
      └────────────────────────── feedback: what to collect or reprocess next ◀──────────────────────────────┘
```

| Stage | Runs as | Started by |
|---|---|---|
| Collect, look, adapt, serve | Gitea Actions jobs on the service node ([gitea/](../services/gitea/README.md)) | dispatch, or a schedule |
| Environments | Gitea workflow `setup-envs` ([setup/](../services/gitea/setup/README.md)) | `ctl.sh up` when stale; `ctl.sh setup` |
| Train, evaluate | Slurm GPU jobs ([slurm/](../services/slurm/README.md)) | you, `sbatch` from a login node |

All state is on `/tier1`, so containers, Slurm jobs and your shell see the same bytes.

## 0. Before you start

- **Install or update the stack:** [services/README.md](../services/README.md#run) (fresh install included).
- **Reach it and log in:** [Access](../services/README.md#access) and [Users](../services/README.md#users); one Keycloak login covers every web UI.
- **S3 from your shell:** [versitygw/](../services/versitygw/README.md#client-setup). Gateway users read and write `raw` and `processed`.
- **Run a workflow:** Gitea `admin/pipelines` → Actions, or the API call in [gitea/](../services/gitea/README.md). The repo's copies are what runs: push changes to `$STATE_DIR/pipelines`.
- **Slurm jobs:** source `$STATE_DIR/slurm.env` and set `SBATCH_PARTITION` ([slurm/](../services/slurm/README.md)).

## 1. Collect: data lands in `raw`

A dataset is a folder under `s3://raw/open_datasets/<dataset>/` (public) or `s3://raw/internal_datasets/<dataset>/` (ours), in whatever format it was collected in. Upload with `aws s3 sync` ([versitygw/](../services/versitygw/README.md)), `mv` on the node for data already on `/tier1`, or the `download-datasets-hf` / `download-models-hf` workflows ([ingest/](../services/gitea/ingest/README.md)).

## 2. Look: `raw` in FiftyOne and Rerun

`sync-fiftyone-raw` mirrors `raw` into FiftyOne every 30 minutes without writing data; bags open in Rerun. Layouts, checks and results: [fiftyone/](../services/fiftyone/README.md), [rerun/](../services/rerun/README.md).

## 3. Adapt: `raw` → `processed/xpolicylab/`

One adapter workflow per source writes XPolicyLab's xspark HDF5, incrementally: [adapter/](../services/gitea/adapter/README.md).

## 4. Train under Slurm, tracked in MLflow

`train.sbatch <model> <embodiment> <bench>/<task>[,...]`. Per-model and per-embodiment layout, MLflow naming (experiment `train/<embodiment>/<mix>`, model `<model>.<embodiment>.<mix>`) and results: [slurm/](../services/slurm/README.md). Runs and models are browsed in MLflow ([mlflow/](../services/mlflow/README.md)).

## 5. Evaluate and feed back

`evaluate.sbatch` runs XPolicyLab's RoboDojo evaluation and logs `eval/success_rate` and `eval/score` on the training run and its model version, so the registry answers "how good is version N". GB300 specifics and results: [slurm/](../services/slurm/README.md#environment). Data without a simulator (Galaxea r1lite) is evaluated physically.

## 6. Serve: registry → Triton

Set the MLflow alias `triton` on a model version; `sync-triton` loads it within 10 minutes and tags the version `triton.status`. Remove the alias to unload: [triton/](../services/triton/README.md).

## The same loop for other data

| Raw format | Collect | Look | Adapt and train |
|---|---|---|---|
| LeRobot v2 archives (Galaxea) | archives into `raw` | catalog entry per archive | `galaxeaOpenWorldDataset_to_xpolicylab` → `train.sbatch` |
| LeRobot v3 (HiFi-UMI-2K) | `download-datasets-hf` | episodes with videos and state/action plots | EE-space data; XPolicyLab's joint-space layout doesn't fit yet |
| xspark HDF5 (RoboDojo) | `setup-envs` | episode groups with preview videos | `robodojo_to_xpolicylab` → `train.sbatch` |
| ROS 2 mcap (h2rc) | bag directories into `raw` | bags in FiftyOne and Rerun | no recorded actions in the bags; would need derived targets |
| COLMAP captures (real2sim) | `aws s3 sync` into `raw/internal_datasets/real2sim/` | photos with poses | splats: [sim/splat/](../sim/splat/README.md) |
| anything else | into `raw` | add a layout to `sync-fiftyone-raw` | one `<repo>_to_xpolicylab` adapter |

## Where things are

| Need | Where |
|---|---|
| Stack operations, users, state on disk | [services/README.md](../services/README.md) |
| Pipelines code | Gitea `admin/pipelines`, seeded from `services/gitea/`; editing checkout `$STATE_DIR/pipelines` |
| Slurm jobs, envs, logs | [services/slurm/](../services/slurm/README.md) |
| What Triton serves | [services/triton/](../services/triton/README.md#tracking) |
| Secrets | `services/.env` (gitignored); jobs read `$STATE_DIR/slurm.env` |
