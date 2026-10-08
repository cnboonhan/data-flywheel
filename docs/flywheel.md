# The flywheel, end to end

A runbook for carrying one dataset through the whole loop of the [architecture diagram](../architecture.html): collect → look → adapt → train → evaluate → feed back. [`services/README.md`](../services/README.md) has the per-service reference; this page is the path across them.

```
 Collect                 Look                       Adapt                      Train                  Evaluate
 ───────                 ────                       ─────                      ─────                  ────────
 public datasets,   ──▶  s3://raw mirrored     ──▶  s3://processed/       ──▶  Slurm GPU job     ──▶  Slurm GPU job
 UMI / teleop /          into FiftyOne,             xpolicylab/<bench>/        (sbatch by hand),      RoboDojo sim (Isaac Sim),
 robot bags              bags open in Rerun         (adapter workflows,        XPolicyLab + MLflow    scores back on the
 (s3://raw)              (sync-fiftyone-raw)        incremental)               runs, registry         run and model version
      └────────────────────────── feedback: what to collect or reprocess next ◀──────────────────────────────┘
```

Where things run:

| Stage | Runs as | Started by |
|---|---|---|
| Collect, look, adapt | Gitea Actions jobs: containers on the service node, buckets mounted read-only at `/buckets` | Gitea (dispatch, or the hourly schedule of `sync-fiftyone-raw`) |
| Environments | Gitea workflow `setup-envs` | `ctl.sh up` when an env is missing or stale; `ctl.sh setup` forces it |
| Train, evaluate | Slurm GPU jobs on the GB300 partition (the node's Docker has no GPU runtime) | you, with `sbatch` from a login node |

All state is on `/tier1`, so containers, Slurm jobs and your shell see the same bytes. The `clean/`, `validate/` and `mix/` stages in `services/gitea/` are empty so far.

## 0. Before you start

### Fresh install

Done from zero on 2026-10-08 (containers and service state wiped; raw bucket, policy envs, RoboDojo assets and caches kept), with the checkout in `$HOME` and the state on `/tier1`:

1. **Clone and configure.** `git clone --recurse-submodules https://github.com/cnboonhan/data-flywheel.git ~/workspaces/data-flywheel`; set `remote.origin.pushurl` to the port-443 SSH URL (CLAUDE.md). IsaacLab-Arena's nested submodules use SSH URLs the cluster can't reach; fetch them with `git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive eval/system2/IsaacLab-Arena` (not needed for this loop). Copy `services/.env` (or fill in `services/.env.example`).
2. **Bring the stack up.** `services/ctl.sh up`. On a clean state it starts Caddy alone first and waits for its CA, then everything else, then provisions: Gitea admin, runner token, Keycloak OpenID source, the `pipelines` repo with its workflows, secrets and variables; Grafana's admin email; Keycloak's `mlflow` client; the MLflow job token; `$STATE_DIR/slurm.env` (credentials and the paths of this checkout); the `$STATE_DIR/pipelines` checkout. About 5 minutes; run it a second time if MLflow was still migrating when the token was minted (it says so).
3. **Trust the CA** on the machines that will talk to the stack: `https://<host>:8443/ca.crt`.
4. **Environments and assets.** `ctl.sh up` dispatches `setup-envs`, which builds the RoboDojo evaluation env (Isaac Sim 5.1, Isaac Lab, curobo, XPolicyLab, ffmpeg, conda shim; ~1 h on a cold cache) and the ACT and DP policy envs under `$STATE_DIR/envs/`, and downloads the sim assets and the RoboDojo data (35 sim tasks, 1.85 TB, into `raw/open_datasets/robodojo/`; real-robot episodes, 273 GB, into `raw/open_datasets/robodojo_real/`). The RoboDojo job resumes where it stopped and can take a day for the data.

What the fresh run caught, all fixed in `ctl.sh` or the scripts: dockerd creating a directory where Caddy's CA file would be (Caddy now starts first), the pipelines checkout not being recreated, the `mlflow` bucket not being created (artifact uploads then fail with 500), the MLflow token minted before the plugin had migrated its database (rerun `up`), and the installers skipping editable installs that pointed at the old checkout (now path-aware). **Pushing to the `pipelines` repo can cancel in-flight runs** of push-triggered workflows; queue the pushes.

### Access

- **Reach the stack.** `https://$SERVICE_HOST:$CADDY_PORT/` (values from `services/.env`) plus the `fiftyone.`, `rerun.` and `s3.` subdomains on the same port; through a login node, forward the port and map the four names in `/etc/hosts`; trust `/ca.crt` once. Commands: [services/README.md](../services/README.md#access).
- **Log in once.** Keycloak (`/auth`) signs you into Gitea, Grafana, MLflow, FiftyOne and Rerun. MLflow and the S3 API use the account and keys that `services/ctl.sh user add <name> <email>` gives you.
- **S3 credentials in your shell** for the `aws` commands below:
  ```bash
  export AWS_ACCESS_KEY_ID=<name> AWS_SECRET_ACCESS_KEY=<secret> AWS_DEFAULT_REGION=us-east-1
  export AWS_CA_BUNDLE=flywheel-ca.crt
  S3=https://s3.$SERVICE_HOST:$CADDY_PORT
  ```
- **Workflows** live in the Gitea repo **`admin/pipelines`** → Actions → pick a workflow → Run workflow. The same from a shell:
  ```bash
  curl -u <name>:<password> -H 'Content-Type: application/json' -d '{"ref":"main","inputs":{...}}' \
    https://$SERVICE_HOST:$CADDY_PORT/gitea/api/v1/repos/admin/pipelines/actions/workflows/<workflow>.yml/dispatches
  ```
  `ctl.sh up` seeds the repo from `services/gitea/` once; after that the repo's copies are what runs, so copy changed files into `$STATE_DIR/pipelines` and push.
- **Slurm jobs** read the credentials and paths from `slurm.env`:
  ```bash
  set -a; . /tier1/htx_boonhan/services/slurm.env; set +a
  cd /tier1/htx_boonhan/services/slurm-logs; S=$FLYWHEEL_ROOT/services/slurm
  ```

## 1. Collect: data lands in `raw`

A dataset is a directory under `s3://raw/open_datasets/<dataset>/` (public: Galaxea, HiFi-UMI-2K, RoboDojo) or `s3://raw/internal_datasets/<dataset>/` (our own collections: h2rc), in whatever format it was collected in.

| Way | Command | Notes |
|---|---|---|
| S3 upload (laptop, robot) | `aws --endpoint-url $S3 s3 sync ./capture s3://raw/internal_datasets/<dataset>/` | |
| Already on `/tier1` | `mv <dir> $STATE_DIR/versitygw/buckets/raw/<open\|internal>_datasets/<dataset>` | instant rename, no copy |
| Hugging Face dataset | workflow **`download-datasets-hf`**, inputs `repo`, `include`, `dest`, `limit` | file by file through the S3 API, resumable; gated repos need `HF_TOKEN` in `.env` |
| Hugging Face model | workflow **`download-models-hf`** | into the MLflow model registry (experiment `hf-models`, artifacts in `s3://mlflow`) |

Known repos are listed in each workflow's header ([ingest/](../services/gitea/ingest/README.md)).

## 2. Look: `raw` in FiftyOne and Rerun

Workflow **`sync-fiftyone-raw`** ([ingest/](../services/gitea/ingest/README.md)) runs hourly and mirrors `raw` into FiftyOne without writing any data: each `raw/<group>/<name>/` becomes the FiftyOne dataset `raw/<group>/<name>`, with samples pointing at the raw files. Re-runs touch only files that are new, changed or gone; a dataset whose folder disappears is deleted. Dispatch it by hand (input `datasets`, a glob such as `internal_datasets/*`) to see new data sooner.

| Raw dataset | In FiftyOne |
|---|---|
| HiFi-UMI-2K (LeRobot v3) | one sample per episode, playing its window of the source videos; state/action plotted from the parquet |
| robodojo (xspark HDF5) | one group per episode, a slice per preview video; `instruction`, `task`, `hdf5` path |
| h2rc (ROS 2 mcap) | one sample per bag in FiftyOne's MCAP viewer (plot joints from the Message tile's **plot** buttons); `rerun_url` opens the bag in Rerun |
| galaxea-open-world-r1lite (tar.gz) | one catalog entry per archive (not playable until unpacked) |

Open **FiftyOne** at `https://fiftyone.<host>:8443/` and pick a `raw/...` dataset; filter by `task`, `duration_s` or `topics`. Use Chrome: Safari fails to load MCAP streams. Rerun links carry a path token (`RERUN_RAW_TOKEN` in `.env`) because the viewer's fetch sends no login cookie; anyone with a link can read bags, and changing the token revokes them ([rerun/](../services/rerun/README.md)). FiftyOne is one shared App: people browsing at the same time can move each other's view.

What to check, because training inherits it: episode count matches the collection log, every episode has all cameras, durations are plausible, task strings are right.

Result (2026-10-08, first full sync): HiFi-UMI-2K 482 060 episodes, robodojo 6 257 camera slices (2 099 episodes, download still running), h2rc 4 121 bags, galaxea 227 archives.

## 3. Adapt: `raw` → `processed/xpolicylab/`

XPolicyLab (`eval/system1/RoboDojo/XPolicyLab`) trains ~35 policies from one input format, its xspark HDF5 (`PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`). That tree is `processed/xpolicylab/`; the Slurm jobs symlink `eval/system1/RoboDojo/data` to it. One adapter workflow per source writes it ([adapter/](../services/gitea/adapter/README.md)); a per-task `manifest.json` maps source to output episodes, so re-runs only add what's new. Inputs: `tasks` (glob, default `*`), `limit` (episodes per task, 0 = all), `env_cfg` (default `arx_x5`).

| Workflow | Source → bench |
|---|---|
| **`galaxeaOpenWorldDataset_to_xpolicylab`** | `raw/open_datasets/galaxea-open-world-r1lite/lerobot/<task>.tar.gz` → `galaxeaOpenWorldDataset` |
| **`robodojo_to_xpolicylab`** | `raw/open_datasets/robodojo/<task>/<env_cfg>/data/` (server-side copy) → `RoboDojo` |

Galaxea LeRobot v2.1 → xspark: 6 arm joints + 1 gripper per arm (14-D, the same layout as XPolicyLab's `arx_x5`), EE poses reordered to `[x y z qw qx qy qz]`, the three cameras at 480×640 as JPEGs stamped with XPolicyLab's `XPL-RGB1` marker. Verified against `XPolicyLab.utils.data_loader.load`.

Result: `Arrange_Fruits_20250819_011` in full is 114 episodes of ~130 MB (110 064 frames); the earlier Slurm converter took 16 min for it. Incremental runs: `limit = 2` then `limit = 3` added 2 then 1 episode; an unchanged archive is skipped without extracting.

## 4. Train: XPolicyLab under Slurm, tracked in MLflow

```bash
sbatch --export=ALL $S/train-xpolicylab.sbatch ACT galaxeaOpenWorldDataset Arrange_Fruits_20250819_011 arx_x5 joint 0 --num_epochs 30 --save_freq 30
squeue -u $USER; tail -f train-xpolicylab-<id>.log
```

Arguments: `<policy> <bench> <task> <env_cfg> <action> <seed> [extra training args]`. Drop the extra arguments for the real 6000-epoch run; `DP` with `training.num_epochs=…` for diffusion policy. Policies with an env: ACT and DP. A new policy needs a recipe in `services/gitea/setup/install-policy-env.sh` and a case in the training job ([slurm/](../services/slurm/README.md)).

The job runs the policy's own `process_data.sh`, then its training command exactly as `policy/<P>/train.sh` would, through `services/xpolicylab/train_mlflow.py`, which hooks the policy's epoch summary (XPolicyLab prints no metrics itself), mirrors every epoch to MLflow, uploads the checkpoint directory as run artifacts (stored in `s3://mlflow`) and registers it as a model version. Decoded-frame caches and checkpoints go to `$STATE_DIR/xpolicylab/<policy>/`, linked from the checkout.

Result (ACT, 30 epochs, on the bench then named `Galaxea`): 14 min 28 s, most of it `process_data` decoding 330k JPEG frames; the 30 epochs take about a minute. MLflow run `ACT-Galaxea-Arrange_Fruits_20250819_011-arx_x5-joint-0`: `train/loss`, `train/l1`, `train/kl`, `val/loss` (86.1 → 0.754); artifacts `policy_epoch_30_seed_0.ckpt` and `policy_last.ckpt` (336 MB each) plus `dataset_stats.pkl`; registered model **`ACT-Galaxea-Arrange_Fruits_20250819_011` version 1**.

**Look at the results.** `https://<host>:8443/mlflow/` → experiment **xpolicylab**: curves, parameters (bench/task/env_cfg/seed/action_dim), the exact command, the Slurm job id. **Models** → the registered model, each version linked to its run and checkpoints. Compare runs across policies or seeds by selecting them.

Caveats: `env_cfg=arx_x5` is a stand-in label for Galaxea (r1lite has the same 14-D layout; a proper `r1lite` entry needs edits inside the XPolicyLab/RoboDojo submodules, i.e. a fork); the policies' pinned `torch==2.4.1` has no CUDA build for this aarch64 Blackwell node, so the envs use the cu128 index.

## 5. Evaluate and feed back

```bash
sbatch --export=ALL $S/evaluate-xpolicylab.sbatch ACT stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 arx_x5_gpu joint 0 5
```

Arguments: `<policy> <task> <ckpt> <env_cfg> <action> <seed> [eval_num]`; leave `eval_num` empty for the task's default 25 to 50 episodes.

The job runs XPolicyLab's `eval.sh` exactly as documented: a policy server in the policy's env and RoboDojo's eval client in the `robodojo` env (Isaac Sim 5.1 headless, both through the conda shim). RoboDojo plays `eval_num` episodes against its evaluation layouts, scores them and writes `_result.json` plus one mp4 per camera under `$ROBODOJO_DIR/eval_result/`. `services/xpolicylab/eval_mlflow.py` then logs `eval/success_rate`, `eval/score` and `eval/episodes` **on the training run that produced the checkpoint** (found by name; a new `eval-…` run if there is none), uploads the result and the videos under `eval/<task>/`, and tags the registered model version with the scores. So the registry answers "how good is version N" directly.

This closes the loop on data that has a simulator: RoboDojo's `stack_bowls` episodes (`raw/open_datasets/robodojo/`, from `setup-envs`) → `robodojo_to_xpolicylab` → `train-xpolicylab.sbatch` (bench `RoboDojo`) → evaluated here → scores on the run and the model version. Galaxea's r1lite has no simulator, so its evaluation stays physical.

Result: the 30-epoch ACT checkpoint (`RoboDojo-stack_bowls-arx_x5-joint-0`, trained in 5 min 54 s), 2 episodes of 800 steps, `COMPLETED` in 10 min 32 s on one GB300; `eval/success_rate 0.0` on the training run, six episode videos and `_result.json` as artifacts, model `ACT-RoboDojo-stack_bowls` v1 tagged `eval_stack_bowls_success_rate = 0.000`. A 30-epoch model is not expected to succeed; the point is the plumbing.

What it took on this hardware, and why the job script does what it does:

- **The simulation runs on the GPU device (`env_cfg = *_gpu`).** RoboDojo defaults the Isaac Lab simulation device to CPU. On the GB300 nodes Isaac Sim 5.1 then never delivers Replicator camera frames (verified with Isaac Lab's own tiled camera: empty buffers on `cpu`, frames on `cuda:0`), and RoboDojo's capture kernel spins forever on the empty buffer. An `env_cfg` ending in `_gpu` is derived from its base config by the job (`device: cuda:0`), with the base's evaluation layouts and checkpoints aliased. Physics on the GPU may score slightly differently from the official CPU-physics numbers.
- **Two nodes can't run it.** On C01 and C02 the eval client's Kit viewport stopped rendering on 2026-10-08 while C03 and C04 kept working, so Isaac Lab's viewport camera controller fails with "Accessed invalid null prim". Software, GPU mode, scratch and extension order are identical on all four nodes; the cause was not found. The job excludes those two nodes.
- **RoboDojo assumes CPU tensors** in a few places (`np.asarray` on a tensor, `.numpy()`); `services/slurm/robodojo-shim/sitecustomize.py`, injected through `PYTHONPATH`, makes CUDA tensors convert transparently and the viewport camera placement best-effort. The proper fix is a RoboDojo fork with a `device` setting and device-safe conversions.
- **No `ffmpeg` on the nodes.** RoboDojo streams camera frames through one; `install-robodojo.sh` installs `imageio-ffmpeg`'s static build into `$ROBODOJO_DIR/bin`, which the job puts on `PATH`.
- Already covered by `install-robodojo.sh`: aarch64 wheels for Isaac Sim 5.1 and torch cu128, `libgomp` preloaded, NVRTC 12.9 preloaded (torch's 12.8 doesn't know sm_103), user-space GL libraries, curobo built from source, robot configs rendered with absolute asset paths. The first Isaac Sim start on a node compiles the RTX pipelines (minutes); the caches under `$ROBODOJO_DIR/cache` are shared, so later starts take 15 s.

## The same loop for other data

| Raw format | Collect | Look | Adapt and train |
|---|---|---|---|
| LeRobot v2 archives (Galaxea) | archives into `raw` | catalog entry per archive | `galaxeaOpenWorldDataset_to_xpolicylab` → `train-xpolicylab.sbatch` |
| LeRobot v3 (HiFi-UMI-2K) | `download-datasets-hf` | episodes with videos and state/action plots | EE-space data; XPolicyLab's joint-space layout doesn't fit yet |
| xspark HDF5 (RoboDojo) | `setup-envs` | episode groups with preview videos | `robodojo_to_xpolicylab` → `train-xpolicylab.sbatch` |
| ROS 2 mcap (h2rc) | bag directories into `raw` | bags in FiftyOne and Rerun | no recorded actions in the bags; would need derived targets |
| anything else | into `raw` | add a layout to `sync-fiftyone-raw` | one `<repo>_to_xpolicylab` adapter |

## Operations cheat-sheet

| Need | Where |
|---|---|
| Start / stop / update the stack | `services/ctl.sh up` / `down` (runs compose on the service node over ssh) |
| Rebuild the envs | `services/ctl.sh setup` (Gitea workflow `setup-envs`) |
| Add a person | `services/ctl.sh user add <name> <email>` → Keycloak (SSO), MLflow account, S3 key |
| Reach it from a laptop | forward `$CADDY_PORT` + `/etc/hosts` for the four names; trust `/ca.crt` once ([services/README.md](../services/README.md#access)) |
| State on disk | `/tier1/htx_boonhan/services/<service>/`; buckets under `versitygw/buckets/` |
| Pipelines code | Gitea `admin/pipelines`, seeded from `services/gitea/`; editing checkout at `/tier1/htx_boonhan/services/pipelines` |
| Slurm jobs and logs | `services/slurm/*.sbatch`; logs in `/tier1/htx_boonhan/services/slurm-logs/<job>-<id>.log` |
| Environments | `/tier1/htx_boonhan/services/envs/{robodojo,act,dp}` (uv venvs) |
| Secrets | `services/.env` (gitignored); Slurm jobs read `/tier1/htx_boonhan/services/slurm.env` |
