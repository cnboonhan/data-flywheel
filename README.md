# data-flywheel

Monorepo for data collection, training and evaluating **action models** (system 1, VLA policies) and **agentic models** (system 2).

Connecting to the stack? Start with the [quickstart](quickstart.md).

## Architecture

![Data Flywheel architecture](assets/architecture.svg)

Edit the [source](assets/architecture.html); `assets/architecture.svg` and its role variants are generated from it by `tools/architecture_svg.py`, run by the git hooks in `.githooks/`.

## Folder structure

| Folder | Contents |
|---|---|
| [`eval/`](eval/README.md) | Benchmarks for action models (system 1) and agentic models (system 2) (git submodules) |
| [`sensors/`](sensors/README.md) | Data-collection hardware (git submodules) |
| [`services/`](services/README.md) | Infra stack as Docker Compose behind Caddy: Keycloak SSO, Gitea + Actions, MLflow, S3 gateway, FiftyOne, Rerun, Grafana/Loki, Triton, Slurm job scripts |
| `assets/` | The architecture diagram and its role variants, the developer roles that push the flywheel along ([Scientist](assets/devroles/scientist.md), [Engineering](assets/devroles/engineering.md)), the quickstart's landing-page screenshot, and the `bufo` image submodule |
| `tools/` | Repo maintenance scripts; [`bin/`](tools/bin/README.md): the commands SSH keys may run on the service account (e.g. `dynamo-up` for `sync-dynamo`) |

## Setup

1. Install Git LFS.
   ```bash
   git lfs install
   ```
2. Clone with submodules.
   ```bash
   git -c url."https://github.com/".insteadOf=git@github.com: \
       clone --recurse-submodules --jobs 8 https://github.com/cnboonhan/data-flywheel.git
   ```
3. Set the push URL (port 22 is blocked on the cluster) and enable the repo's git hooks (they keep `architecture.svg` in step with `architecture.html`).
   ```bash
   git config remote.origin.pushurl ssh://git@ssh.github.com:443/cnboonhan/data-flywheel.git
   git config core.hooksPath .githooks
   ```
4. Bring up the services: [services/](services/README.md#run).
5. Get access and an account: [Access](services/README.md#access), [Users](services/README.md#users), [S3 client](services/versitygw/README.md#client-setup).
6. Install the evaluation tooling you need: [eval/system2/](eval/system2/README.md#install), [eval/system1/](eval/system1/README.md).
7. Set up Slurm jobs: [slurm/](services/slurm/README.md).

## The flywheel, end to end

The diagram reads left to right: tooling gathers data, the infra stores it raw, it is processed into training sets, trained on and evaluated, and evaluation feeds back into the tooling and the processing. Below, each box of the diagram and what implements it here. Gitea Actions jobs on the service node do the data work; training and evaluation are Slurm GPU jobs; all state is on `/tier1`, so containers, jobs and your shell see the same bytes.

### Tooling

| Box | Here |
|---|---|
| Off-Robot Data Collection | YUBI glove and gripper rigs: [sensors/](sensors/README.md) (`yubi-hw`, `yubi-sw`) |
| Robot Teleoperation | Galaxea R1 data (public so far) |
| Public Datasets | `download-datasets-hf` into `s3://raw/open_datasets/` (HiFi-UMI-2K, Galaxea, RoboDojo, EgoPro): [ingest/](services/gitea/ingest/README.md) |
| Virtual Scenes | not built yet |
| Physical Scenes | not built yet |
| Inference Compute | Triton on the service node's GPUs: [triton/](services/triton/README.md) |

**Upload and Sync:** collections go to `s3://raw/<open|internal>_datasets/<dataset>/` with `aws s3 sync` ([versitygw/](services/versitygw/README.md)) or the download workflows.

### Infra

| Box | Here |
|---|---|
| Connectivity | access through Caddy on one port, SSH-tunnelled from a login node ([Access](services/README.md#access)) |
| Data Storage | Versity S3 gateway, buckets `raw`, `processed`, `mlflow`, `triton` ([versitygw/](services/versitygw/README.md)); `raw` and `processed` browsable as folders with LeRobot dataset pages ([browse/](services/browse/README.md)), `raw` indexed in FiftyOne ([fiftyone/](services/fiftyone/README.md)), episodes viewed in Rerun ([rerun/](services/rerun/README.md)) |
| Logging | Loki and Grafana ([loki/](services/loki/README.md), [grafana/](services/grafana/README.md)) |
| Model Registry | MLflow: runs, registered models, checkpoints in `s3://mlflow` ([mlflow/](services/mlflow/README.md)) |

### Processed

| Box | Here |
|---|---|
| Data Cleaning | stage folder [clean/](services/gitea/clean/README.md); no workflows yet |
| Data Conversion | adapter workflows write `s3://processed/xpolicylab/` in XPolicyLab's format: [xpolicylab_adapter/](services/gitea/ingest/xpolicylab_adapter/README.md) |
| Data Mixing | named mixes at training time (`train.sbatch --mix`, recorded with a data fingerprint); [mix/](services/gitea/mix/README.md) has no workflows yet |

**Architecture Experiments:** every model trained on one robot and mix lands in one MLflow experiment, `train/<embodiment>/<mix>`, for side-by-side comparison.

### Train

| Box | Here |
|---|---|
| Model Training and Pre-Evals | `train.sbatch <model> <embodiment> <data>`, per-model recipes and per-robot configs, tracked in MLflow: [slurm/](services/slurm/README.md) |
| Inference / Model Optimizations | not built yet |
| Distillation | not built yet |

**Deploy:** each training run registers a model version (`<model>.<embodiment>.<mix>`). Listing it in the Gitea repo `triton`'s `models.yaml` serves it on Inference Compute, for the kinds Triton has a packager for (π0.5, MLP): [triton/](services/triton/README.md).

### Evaluate

| Box | Here |
|---|---|
| Virtual Evals | `evaluate.sbatch` runs RoboDojo through XPolicyLab ([slurm/](services/slurm/README.md)); RoboTwin and IsaacLab-Arena in [eval/](eval/README.md) |
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
| anything else | into `raw` | add a layout to `sync-fiftyone-raw` | one `<repo>_to_xpolicylab` adapter |
