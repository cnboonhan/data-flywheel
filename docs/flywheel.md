# The flywheel, end to end

How a dataset travels from collection to a registered checkpoint on this stack, following the columns of the [architecture diagram](../architecture.html). Each stage names the service that owns it, where the data sits, and the command that moves it on. [`services/README.md`](../services/README.md) has the per-service details; this page is the path across them.

```
 Build            Store (raw)         Processed                 Train                      Evaluate
 ─────            ───────────         ─────────                 ─────                      ────────
 UMI / teleop /   s3://raw            s3://processed            Slurm GPU job              RoboDojo / RoboTwin sim,
 public datasets  (versitygw)   ──▶   unpack / convert   ──▶    XPolicyLab policy    ──▶   physical evals
      │                │              (Gitea Actions)           train_mlflow.py                 │
      │                │                   │                         │                          │
      │                └── FiftyOne ◀──────┘                   MLflow: curves,              Eval logs ──▶ Store
      │                    (browse)                            artifacts, registry               │
      └──────────────────────────── feedback: what to collect / reprocess next ◀─────────────────┘
```

Everything except training runs as Docker Compose on the service node (`services/ctl.sh up`); training and heavy conversions run as Slurm jobs on the GB300 partition, because the node's Docker has no GPU runtime. All state is on `/tier1`, so the same bytes are visible to containers, Slurm jobs and your shell.

## 1. Build: data arrives

Sources today: public datasets downloaded from Hugging Face (HiFi-UMI-2K, Galaxea Open-World, h2rc), later teleoperation and UMI recordings.

Three ways into the raw bucket, all ending as a directory under `/tier1/htx_boonhan/services/versitygw/buckets/raw/<dataset>/`:

| Way | Command | Notes |
|---|---|---|
| S3 upload | `aws --endpoint-url https://<host>:8444 s3 sync ./data s3://raw/<dataset>/` | Objects get ETags; bucket events fire |
| Move on `/tier1` | `mv /tier1/.../download /tier1/htx_boonhan/services/versitygw/buckets/raw/<dataset>` | Instant rename, no copy; no ETag or event |
| Download job | (planned) a workflow running `hf download --local-dir /buckets/raw/<dataset>` | Would need `/buckets/raw` writable in job containers |

The S3 gateway (Versity, `services/docker-compose.yml`) maps `s3://raw/x` to that directory and back, so S3 clients and filesystem readers see the same data.

## 2. Store: raw, logged, registered

The Store column is the Compose stack behind Caddy (`https://<host>:8443`, one admin login for everything):

| Service | Role in the loop | Where |
|---|---|---|
| Versity S3 gateway | `raw`, `processed`, `mlflow` buckets | `:8444` API, `/s3/` UI |
| Gitea + act_runner | pipelines repo (`admin/pipelines`) and the Actions runner that executes them | `/gitea/` |
| MLflow | experiment tracking, run artifacts (checkpoints in `s3://mlflow`), model registry | `/mlflow/` |
| FiftyOne + Mongo | browsing datasets as samples | `fiftyone.<host>` |
| Prometheus, Loki, Grafana | metrics and logs of the stack | `/prometheus/`, `/loki/`, `/grafana/` |

Datasets become *browsable* only once an ingest pipeline has turned them into FiftyOne samples; see stage 3.

## 3. Processed: pipelines in Gitea Actions

The `admin/pipelines` repo holds the workflows (`.gitea/workflows/`) and the scripts they run (`fiftyone/`, `xpolicylab/`, `slurm/`). `services/ctl.sh up` seeds it from `services/gitea/examples`, `services/fiftyone`, `services/xpolicylab` and `services/slurm` once; after that the repo's copies are what runs, edited and versioned there like any code. Jobs run as containers on the compose network with the buckets mounted read-only at `/buckets` and `S3_ENDPOINT_URL=http://versitygw:7070`, so they read from the mount (fast) and write through S3 (ETags, events).

| Workflow | In → out | What it does |
|---|---|---|
| `process-raw` | `raw` → `processed/manifests/raw.json` | Lists the raw bucket; the smoke test, runs on every push |
| `unpack-archives` | `raw/<ds>/*.tar.gz` → `processed/<ds>/<archive>/` | Streams tar members to S3 (Galaxea ships one LeRobot dataset per task archive) |
| `convert-mcap` | `raw/h2rc/**/*.mcap` → `processed/h2rc/**/<camera>.mp4` + `episode.json`, then FiftyOne | Decodes ROS 2 `CompressedImage` topics through ffmpeg |
| `ingest-lerobot` | LeRobot v2/v3 roots in any bucket → FiftyOne dataset | v3: one sample per camera video with episodes as temporal detections and an **episodes** clips view; v2: one sample per episode video |
| `convert-xpolicylab` | `processed/<galaxea task>/` → `processed/xpolicylab/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5` | Galaxea LeRobot v2.1 → XPolicyLab xspark v1.0 (Slurm CPU job) |
| `train-xpolicylab` | xspark → MLflow run + registered model | Slurm GPU job, stage 4 |
| `episodes-lerobot`, `episodes-mcap` | raw (LeRobot v2/v3, ROS 2 mcap) → `processed/episodes/<dataset>/<episode>/` | The **canonical episode layout**: per-camera mp4s, `episode.json`, `signals.parquet`. One converter per raw format; everything downstream reads only this |
| `ingest-episodes` | canonical episodes → `processed/rerun/*.rrd` + FiftyOne `episodes/<dataset>` | A Rerun recording per episode (cameras and signals on one timeline) and a grouped FiftyOne dataset: one group per episode, one slice per camera, `rerun_url` field |

**Viewing an episode:** FiftyOne (`fiftyone.<host>`) for the catalogue — filter by task, robot, duration, gripper range, play any camera; its `rerun_url` field opens the same episode in **Rerun** (`rerun.<host>`, same SSO session) with all cameras and every joint/IMU/wrench signal scrubbing together. Both read the canonical layout, so a new raw format needs one converter and nothing else.

Triggers: `workflow_dispatch` (Actions tab or API) and `push` today. Reacting to uploads automatically is the one missing piece: the gateway can post bucket events to a webhook, and a small bridge turning those into `workflow_dispatch` calls would close it. Files `mv`'d into a bucket directory never raise events; a scheduled scan would catch those.

### Where each raw dataset stands

| Dataset | Format in raw | Processed | Browsable | Trainable |
|---|---|---|---|---|
| HiFi-UMI-2K | LeRobot v3, 399 chunks | not needed | yes, `ingest-lerobot` (3 chunks ingested) | EE-space only; XPolicyLab's joint-space layout doesn't fit yet |
| galaxea-open-world-r1lite | 227 tar.gz, LeRobot v2.1 inside | `unpack-archives` (1 of 227 done) | yes, `ingest-lerobot` from `processed` | **yes**: `convert-xpolicylab` → xspark; ACT and DP trained |
| h2rc | 4 124 ROS 2 mcap bags | `convert-mcap` (2 done) | yes, per-camera mp4s | no recorded actions; would need derived targets |

## 4. Train: Slurm jobs, tracked in MLflow

XPolicyLab (`eval/system1/RoboDojo/XPolicyLab`) trains any of its ~35 trainable policies from one input format, its xspark HDF5 (`PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`). `eval/system1/RoboDojo/data` is a symlink to `processed/xpolicylab`, so once a dataset is converted every policy's `process_data.sh`/`train.sh` sees it.

`slurm/train-xpolicylab.sbatch <policy> <bench> <task> <env_cfg> <action> <seed> [extra]`:

1. activates the policy's uv env (`/tier1/htx_boonhan/services/envs/<policy>`; torch from the cu128 index because the policies' pinned torch has no CUDA build for the aarch64 Blackwell node);
2. runs the policy's own `process_data.sh` (ACT → per-episode HDF5 at 480x640; DP → zarr at 240x320);
3. runs the policy's training command exactly as its `train.sh` would, but through `xpolicylab/train_mlflow.py`, which
   - mirrors every epoch's losses to an MLflow run (experiment `xpolicylab`, run `<policy>-<bench>-<task>-<env_cfg>-<action>-<seed>`) by hooking the policy's own summary function, since XPolicyLab has no dashboard of its own;
   - uploads the checkpoint directory as run artifacts (stored in `s3://mlflow` through the MLflow proxy);
   - registers it as a new version of the model `<policy>-<bench>-<task>`.

Submitting from the login node:

```bash
cd /tier1/htx_boonhan/services/slurm-logs
set -a; . /tier1/htx_boonhan/services/slurm.env; set +a        # MLflow + S3 credentials (written by ctl.sh)
sbatch --export=ALL /tier1/htx_boonhan/services/pipelines/slurm/train-xpolicylab.sbatch ACT Galaxea Make_The_Bed_20250730_012 arx_x5 joint 0
```

Or from Gitea: the `train-xpolicylab` workflow does the same over SSH and streams the Slurm log into the Actions log (the bridge is enabled by `SLURM_LOGIN_HOST` in `.env`; `ctl.sh up` sets up the key and secrets).

Results: **https://<SERVICE_HOST>:8443/mlflow/** → experiment `xpolicylab` for curves and parameters, **Models** for the registry. First runs on Galaxea `Make_The_Bed` (51 episodes): ACT 30 epochs in 71 s, val loss 87.8 → 1.7; DP 3 epochs, val loss 0.047; both registered as version 1.

`env_cfg=arx_x5` is a stand-in label: Galaxea's r1lite has the same 14-D layout (6 joints + 1 gripper per arm). A proper `r1lite` entry means editing `utils/robot/_robot_info.json` and adding `env_cfg/r1lite.yml` inside the submodules, i.e. a fork.

## 5. Evaluate and feed back

Not wired yet. The pieces that exist: RoboDojo and RoboTwin evaluators (`eval/system1/`), IsaacLab-Arena environments (`eval/system2/`), and XPolicyLab's `eval.sh`, which serves a checkpoint as a policy server for the simulator. Galaxea's r1lite has no simulator here, so its evaluation is physical. The intended loop: an `evaluate` workflow pulls a registered model version from MLflow, runs the benchmark as a Slurm job, writes scores back to the run and eval logs to the Store, and the results decide what to collect or reprocess next.

## Worked example: Galaxea, start to finish

```bash
# 1. raw: the HF download was moved into the bucket directory (18 TB total, no copy)
mv /tier1/htx_boonhan/datasets/galaxea-open-world-r1lite /tier1/htx_boonhan/services/versitygw/buckets/raw/

# 2. processed: unpack one task archive through the S3 gateway, then make it browsable
#    (Gitea → Actions → unpack-archives: archives=galaxea-open-world-r1lite/lerobot/Make_The_Bed_*.tar.gz)
#    (Gitea → Actions → ingest-lerobot: name=galaxea-open-world-r1lite bucket=processed path=galaxea-open-world-r1lite/*)
#    → FiftyOne https://fiftyone.<host>:8443, dataset galaxea-open-world-r1lite, 204 episode videos

# 3. training format: xspark HDF5 (Slurm CPU job, ~0.4 s/episode)
#    (Gitea → Actions → convert-xpolicylab: subsets=galaxea-open-world-r1lite/Make_The_Bed_*)
#    → processed/xpolicylab/Galaxea/Make_The_Bed_20250730_012/arx_x5/data/episode_0000000.hdf5 …

# 4. train two policies (Slurm GPU jobs)
#    (Gitea → Actions → train-xpolicylab: policy=ACT …; policy=DP …)
#    → MLflow runs with curves, checkpoints in s3://mlflow, models ACT-Galaxea-… and DP-Galaxea-… v1
```

## Operations cheat-sheet

| Need | Where |
|---|---|
| Start / stop / update the stack | `services/ctl.sh up` / `down` (runs compose on the service node over ssh) |
| Logins | Keycloak single sign-on (`/auth`); `services/ctl.sh user add <name> <email>` creates a person in Keycloak, MLflow and the S3 gateway; the admin login is `ADMIN_USER` / `ADMIN_PASSWORD` in `services/.env` |
| Reach it from a laptop | ssh tunnel on 8443 plus `/etc/hosts` entries for `flywheel.<ip>.sslip.io` and its `fiftyone.`, `rerun.`, `s3.` subdomains → `https://flywheel.<ip>.sslip.io:8443/`; trust `/ca.crt` once |
| State on disk | `/tier1/htx_boonhan/services/<service>/`; buckets under `versitygw/buckets/` |
| Pipelines code | Gitea `admin/pipelines`; checkout for Slurm at `/tier1/htx_boonhan/services/pipelines` |
| Slurm logs | `/tier1/htx_boonhan/services/slurm-logs/<job>-<id>.log` |
| Policy envs | `/tier1/htx_boonhan/services/envs/{act,dp}` (uv venvs) |
