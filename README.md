# data-flywheel

Monorepo for data collection, training and evaluating **action models** (system 1, VLA policies) and **agentic models** (system 2).

## Architecture

[![Data Flywheel architecture](architecture.svg)](https://htmlpreview.github.io/?https://github.com/cnboonhan/data-flywheel/blob/main/architecture.html)

Click the diagram for the interactive version, with links to each component ([source](architecture.html); `architecture.svg` is generated from it by `tools/architecture_svg.py`).

## Folder structure

| Folder | Contents |
|---|---|
| [`eval/`](eval/README.md) | Benchmarks for action models (system 1) and agentic models (system 2) (git submodules) |
| [`sensors/`](sensors/README.md) | Data-collection hardware (git submodules) and the real2sim capture robot |
| [`services/`](services/README.md) | Store stack as Docker Compose behind Caddy: Keycloak SSO, Gitea + Actions, MLflow, S3 gateway, FiftyOne, Rerun, Grafana/Loki, Triton, Slurm job scripts |
| [`sim/`](sim/README.md) | Simulation tooling: IsaacLab-Arena environment generation, Gaussian splats of real scenes |
| `tools/` | Repo maintenance scripts |

## Setup

1. **Install [Git LFS](https://git-lfs.com/)** and run `git lfs install`. Several submodules store assets in LFS and otherwise check out as pointer files.
2. **Clone with submodules.** IsaacLab-Arena pins two of its submodules with SSH URLs; the `-c` option fetches them over HTTPS, so no GitHub SSH key is needed. The full clone is several GB; to fetch one benchmark only, clone without `--recurse-submodules` and run `git submodule update --init --recursive <path>` (same `-c` option for `eval/system2/IsaacLab-Arena`).
   ```bash
   git -c url."https://github.com/".insteadOf=git@github.com: \
       clone --recurse-submodules --jobs 8 https://github.com/cnboonhan/data-flywheel.git
   ```
3. **Set the push URL** if you will push from the cluster, where port 22 is blocked: `git config remote.origin.pushurl ssh://git@ssh.github.com:443/cnboonhan/data-flywheel.git`.
4. **Bring up the services.** Copy `services/.env.example` to `services/.env`, fill in the secrets, and run `services/ctl.sh up`. It provisions every service and dispatches `setup-envs`, which builds the Slurm-side environments and downloads the RoboDojo data: [services/](services/README.md#run) (fresh install included).
5. **Get access.** Trust the stack's CA and, through a login node, forward its port: [Access](services/README.md#access). Create your account (Keycloak login, MLflow token, S3 keys) with `services/ctl.sh user add <name> <email>` ([Users](services/README.md#users)); set up the S3 client: [versitygw/](services/versitygw/README.md#client-setup).
6. **Install the simulation tooling you need**, each in its own venv: splat training `bash sim/splat/install.sh` ([sim/splat/](sim/splat/README.md#install-once)), IsaacLab-Arena `bash eval/system2/setup-arena.sh` ([eval/system2/](eval/system2/README.md#install)), the system 1 benchmarks per [eval/system1/](eval/system1/README.md).
7. **For Slurm jobs,** source the credentials and pick a partition (the cluster has no default): `set -a; . $STATE_DIR/slurm.env; set +a; export SBATCH_PARTITION=<partition>` ([slurm/](services/slurm/README.md)).

## The flywheel, end to end

The diagram reads left to right: data is built, stored raw, processed into training sets, trained on and evaluated, and evaluation feeds back into what to build next. Below, each box of the diagram and what implements it here. Gitea Actions jobs on the service node do the data work; training and evaluation are Slurm GPU jobs; all state is on `/tier1`, so containers, jobs and your shell see the same bytes.

### Build

| Box | Here |
|---|---|
| Egocentric UMI | YUBI glove and gripper rigs: [sensors/](sensors/README.md) (`yubi-hw`, `yubi-sw`) |
| Robot Teleoperation | Galaxea R1 data (public so far) |
| Public Datasets | `download-datasets-hf` into `s3://raw/open_datasets/` (HiFi-UMI-2K, Galaxea, RoboDojo, EgoPro): [ingest/](services/gitea/ingest/README.md) |
| Sim Scenes | Arena environment generation ([sim/isaaclab_arena/](sim/isaaclab_arena/README.md)), splats of real scenes ([sim/splat/](sim/splat/README.md)) from captures by [sensors/real2sim/](sensors/real2sim/README.md) |
| Physical Scenes | not built yet |
| Inference Compute | Triton on the service node's GPUs: [triton/](services/triton/README.md) |

**Upload and Sync:** collections go to `s3://raw/<open|internal>_datasets/<dataset>/` with `aws s3 sync` ([versitygw/](services/versitygw/README.md)) or the download workflows.

### Store (Raw)

| Box | Here |
|---|---|
| Connectivity | access through Caddy on one port, SSH-tunnelled from a login node ([Access](services/README.md#access)) |
| Data Storage | Versity S3 gateway, buckets `raw`, `processed`, `mlflow`, `triton` ([versitygw/](services/versitygw/README.md)); `raw` browsable in FiftyOne ([fiftyone/](services/fiftyone/README.md)) and Rerun ([rerun/](services/rerun/README.md)) |
| Logging | Loki and Grafana ([loki/](services/loki/README.md), [grafana/](services/grafana/README.md)); Prometheus not yet |
| Model Registry | MLflow: runs, registered models, checkpoints in `s3://mlflow` ([mlflow/](services/mlflow/README.md)) |

### Processed

| Box | Here |
|---|---|
| Data Cleaning, Data Validation | stage folders [clean/](services/gitea/clean/README.md), [validate/](services/gitea/validate/README.md); no workflows yet |
| Data Mixing | named mixes at training time (`train.sbatch --mix`, recorded with a data fingerprint); [mix/](services/gitea/mix/README.md) has no workflows yet |
| (conversion) | adapter workflows write `s3://processed/xpolicylab/` in XPolicyLab's format: [adapter/](services/gitea/adapter/README.md) |

**Architecture Experiments:** every model trained on one robot and mix lands in one MLflow experiment, `train/<embodiment>/<mix>`, for side-by-side comparison.

### Train

| Box | Here |
|---|---|
| Model Training and Pre-Evals | `train.sbatch <model> <embodiment> <data>`, per-model recipes and per-robot configs, tracked in MLflow: [slurm/](services/slurm/README.md) |
| Inference / Model Optimizations | not built yet |
| Distillation | not built yet |

**Deploy:** each training run registers a model version (`<model>.<embodiment>.<mix>`). Setting its MLflow alias `triton` serves it on Inference Compute: [triton/](services/triton/README.md).

### Evaluate

| Box | Here |
|---|---|
| Sim Evals | `evaluate.sbatch` runs RoboDojo through XPolicyLab ([slurm/](services/slurm/README.md)); RoboTwin and IsaacLab-Arena in [eval/](eval/README.md) |
| Physical Evals | not built yet |

**Eval Logs:** scores go onto the training run and its model version in MLflow, so the registry answers "how good is version N".

**Feedback Loop:** by hand for now: the scores and FiftyOne views decide what to collect or reprocess next.

### Data formats through the loop

| Raw format | Collect | Look | Adapt and train |
|---|---|---|---|
| LeRobot v2 archives (Galaxea) | archives into `raw` | catalog entry per archive | `galaxeaOpenWorldDataset_to_xpolicylab` → `train.sbatch` |
| LeRobot v3 (HiFi-UMI-2K) | `download-datasets-hf` | episodes with videos and state/action plots | EE-space data; XPolicyLab's joint-space layout doesn't fit yet |
| xspark HDF5 (RoboDojo) | `setup-envs` | episode groups with preview videos | `robodojo_to_xpolicylab` → `train.sbatch` |
| ROS 2 mcap (h2rc) | bag directories into `raw` | bags in FiftyOne and Rerun | no recorded actions in the bags; would need derived targets |
| COLMAP captures (real2sim) | `aws s3 sync` into `raw/internal_datasets/real2sim/` | photos with poses | splats: [sim/splat/](sim/splat/README.md) |
| anything else | into `raw` | add a layout to `sync-fiftyone-raw` | one `<repo>_to_xpolicylab` adapter |
