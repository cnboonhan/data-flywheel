# The flywheel, end to end

A runbook for carrying one dataset through the whole loop of the [architecture diagram](../architecture.html): collect → store → process → verify → train → register. Every step below was executed exactly as written, from Gitea, on the Galaxea task `Arrange_Fruits_20250819_011`; the "Result" lines are from that run. [`services/README.md`](../services/README.md) has the per-service reference; this page is the path across them.

```
 Build            Store (raw)        Processed                    Verify                 Train                 Evaluate
 ─────            ───────────        ─────────                    ──────                 ─────                 ────────
 UMI / teleop /   s3://raw      ──▶  s3://processed          ──▶  FiftyOne + Rerun  ──▶  Slurm GPU job   ──▶   RoboDojo / RoboTwin,
 public datasets  (versitygw)        unpack / episodes / xspark   (one SSO login)        XPolicyLab + MLflow    physical (not wired)
                                     (Gitea Actions)                                     curves, artifacts,
                                                                                         model registry
      └──────────────────────── feedback: what to collect or reprocess next ◀────────────────────────────────────────┘
```

Everything except training runs as Docker Compose on the service node (`services/ctl.sh up`); training and heavy conversions are Slurm jobs on the GB300 partition, submitted by Gitea Actions over SSH. All state is on `/tier1`, so containers, Slurm jobs and your shell see the same bytes.

> **Clean setup in progress (2026-10-08).** The setup, ingest (`download-datasets-hf`, `download-models-hf`, `sync-fiftyone-raw`) and adapter workflows exist; the train and evaluate workflows named below were removed and will be rebuilt under `services/gitea/`. The scripts they call (`services/xpolicylab/`, `services/slurm/`) are still here.

## 0. Before you start

### Fresh install (what a clean redeploy takes)

Done from zero on 2026-10-08 (containers and service state wiped, raw bucket, policy envs, RoboDojo assets and caches kept), with the checkout in `$HOME` and the state on `/tier1`:

1. **Clone and configure.** `git clone --recurse-submodules https://github.com/cnboonhan/data-flywheel.git ~/workspaces/data-flywheel`; set `remote.origin.pushurl` to the port-443 SSH URL (CLAUDE.md). IsaacLab-Arena's nested submodules use SSH URLs that the cluster can't reach; fetch them with `git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive eval/system2/IsaacLab-Arena` (not needed for this loop). Copy `services/.env` (or fill in `services/.env.example`).
2. **Bring the stack up.** `services/ctl.sh up`. On a clean state it starts Caddy alone first and waits for its CA, then everything else, then provisions: Gitea admin, runner token, Keycloak OpenID source, the `pipelines` repo with workflows, scripts and shims, secrets and variables; Grafana's admin email; Keycloak's `mlflow` client; the MLflow job token; the Slurm bridge (runner key, one forced-command line in `~/.ssh/authorized_keys`, `$STATE_DIR/slurm.env` with the credentials and the paths of this checkout, `$STATE_DIR/pipelines` checkout). About 5 minutes; run it a second time if MLflow was still migrating when the token was minted (it says so).
3. **Trust the CA** on the machines that will talk to the stack: `https://<host>:8443/ca.crt`.
4. **Environments and assets.** `ctl.sh up` dispatches the Gitea workflow `setup-envs`, which builds the RoboDojo evaluation env (Isaac Sim 5.1, Isaac Lab, curobo, XPolicyLab, ffmpeg, conda shim; ~1 h on a cold cache), downloads the sim assets and the RoboDojo data (35 sim tasks, 1.85 TB, into `raw/open_datasets/robodojo/`; real-robot episodes, 273 GB, into `raw/open_datasets/robodojo_real/`) and the ACT and DP policy envs in a job container on the service node (`services/gitea/setup/*.sh`). It runs again whenever an env is missing or was built for another checkout; `ctl.sh setup` forces it. The RoboDojo job resumes where it stopped and can take a day for the data.
5. **Run the loop below** from Gitea. Every step was re-run after the wipe; the "Result" lines were reproduced.

What the fresh run caught, all fixed in `ctl.sh` or the scripts: dockerd creating a directory where Caddy's CA file would be (Caddy now starts first), the pipelines checkout not being recreated, the `mlflow` bucket not being created (artifact uploads then fail with 500), the MLflow token minted before the plugin had migrated its database (rerun `up`), and the installers skipping editable installs that pointed at the old checkout (now path-aware). One rule it taught: **don't push to the `pipelines` repo while runs are in flight**. Gitea cancels in-progress runs on the ref when it moves, whatever the workflow (two runs were lost that way), so queue the pushes.


- **Reach the stack.** `https://$SERVICE_HOST:$CADDY_PORT/` (values from `services/.env`) plus the `fiftyone.`, `rerun.` and `s3.` subdomains on the same port; through a login node, forward the port and map the four names in `/etc/hosts`; trust `/ca.crt` once. Commands: [services/README.md](../services/README.md#access).
- **Log in once.** Keycloak (`/auth`) signs you into Gitea, Grafana, MLflow, FiftyOne, Rerun and Loki. MLflow and the S3 API use the account and keys that `services/ctl.sh user add <name> <email>` gives you.
- **S3 credentials in your shell** for the `aws` commands below:
  ```bash
  export AWS_ACCESS_KEY_ID=<name> AWS_SECRET_ACCESS_KEY=<secret> AWS_DEFAULT_REGION=us-east-1
  export AWS_CA_BUNDLE=flywheel-ca.crt
  S3=https://s3.flywheel.<ip>.sslip.io:8443
  ```
- Workflows live in the Gitea repo **`admin/pipelines`** → Actions → pick a workflow → Run workflow, and fill in the inputs listed below. The same from a shell:
  ```bash
  curl -u <name>:<password> -H 'Content-Type: application/json' \
    -d '{"ref":"main","inputs":{...}}' https://flywheel.<ip>.sslip.io:8443/gitea/api/v1/repos/admin/pipelines/actions/workflows/<workflow>.yml/dispatches
  ```

## 1. Collect: data lands in `raw`

A dataset is a directory under `s3://raw/open_datasets/<dataset>/` (public: Galaxea, HiFi-UMI-2K, RoboDojo, Hugging Face downloads) or `s3://raw/internal_datasets/<dataset>/` (our own collections: h2rc), in whatever format it was collected in. Three ways in:

| Way | Command | Notes |
|---|---|---|
| S3 upload (laptop, robot, HF mirror) | `aws --endpoint-url $S3 s3 sync ./capture s3://raw/internal_datasets/<dataset>/` | Objects get ETags and bucket events |
| Already on `/tier1` | `mv <dir> $STATE_DIR/versitygw/buckets/raw/<open|internal>_datasets/<dataset>` | Instant rename, no copy (how the 18 TB of public data went in) |
| Download job | workflow **`download-datasets-hf`**, inputs `repo`, `repo_type`, `include`, `dest` | file by file from the Hugging Face hub through the S3 API; gated repos need the `HF_TOKEN` secret |

**Example.** The Galaxea Open-World set was moved in as 227 per-task archives, `raw/open_datasets/galaxea-open-world-r1lite/lerobot/<task>.tar.gz`. A newly collected episode is uploaded instead; to exercise that path, push a marker and read the prefix back:

```bash
echo "collected $(date -u +%FT%TZ) on r1lite" > collection.log
aws --endpoint-url $S3 s3 cp collection.log s3://raw/open_datasets/galaxea-open-world-r1lite/logs/collection.log
aws --endpoint-url $S3 s3 ls s3://raw/open_datasets/galaxea-open-world-r1lite/
```

Result: `PRE lerobot/`, `PRE logs/`, `lerobot_info.json`, `README.md`, `demo.mp4`. The file is also visible in the S3 web UI (`https://s3.<host>:8443/ui/`, log in with the access key) and as a plain file under `buckets/raw/`.

## 2. Process: `raw` → `processed` with Gitea Actions

Pipelines are code in `admin/pipelines` (`.gitea/workflows/*.yml` plus `setup/`, `adapter/`, `xpolicylab/*.py`, `slurm/*`). Jobs run as containers on the compose network with the buckets mounted read-only at `/buckets` and `S3_ENDPOINT_URL=http://versitygw:7070`, so they read from the mount and write through S3. `services/ctl.sh up` seeds the repo from `services/` once; after that the repo's copies are what runs.

Workflow **`galaxeaOpenWorldDataset_to_xpolicylab`** ([adapter/](../services/gitea/adapter/README.md)), inputs `tasks = Arrange_Fruits_*`, `limit = 0`, `env_cfg = arx_x5`, converts the task archives into XPolicyLab xspark at `processed/xpolicylab/galaxeaOpenWorldDataset/<task>/arx_x5/data/`. Incremental: a per-task `manifest.json` records which source episodes are done.

## 3. Verify: look at what was collected

Workflow **`sync-fiftyone-raw`** ([ingest/](../services/gitea/ingest/README.md)) mirrors `raw` into FiftyOne hourly without writing any data: each `raw/<group>/<name>/` becomes the FiftyOne dataset `raw/<group>/<name>`, with samples pointing at the raw files.

| Raw dataset | In FiftyOne |
|---|---|
| HiFi-UMI-2K (LeRobot v3) | one sample per episode, playing its window of the source videos with the parquet signals |
| robodojo (xspark HDF5) | one group per episode, a slice per preview video; `instruction`, `task`, `hdf5` path |
| h2rc (ROS 2 mcap) | one sample per bag in FiftyOne's MCAP viewer; `duration_s`, `topics` |
| galaxea-open-world-r1lite (tar.gz) | one catalog entry per archive (not playable until unpacked) |

Then, in the browser, open **FiftyOne** `https://fiftyone.<host>:8443/` and pick a `raw/...` dataset. Filter by `task`, `duration_s` or `topics`; play any episode. **Rerun** opens raw mcap or LeRobot directories directly (`rerun --save out.rrd <path>`).

What to check here, because training inherits it: episode count matches the collection log, every episode has all cameras, durations are plausible, task strings are right.

## 4. Train: XPolicyLab under Slurm, tracked in MLflow

XPolicyLab (`eval/system1/RoboDojo/XPolicyLab`) trains ~35 policies from one input format, its xspark HDF5 (`PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`). That tree is `processed/xpolicylab/`, and `eval/system1/RoboDojo/data` is a symlink to it. Training runs as Slurm GPU jobs (the node's Docker has no GPU runtime); Gitea submits them over SSH with a key locked to `services/slurm/slurm-submit` and streams the Slurm log into the Actions log.

### 4a. Training format

Workflow **`convert-xpolicylab`** (a Slurm CPU job), inputs `subsets = galaxea-open-world-r1lite/Arrange_Fruits_*`, `limit = 0`, `env_cfg = arx_x5`.

Galaxea LeRobot v2.1 → xspark: 6 arm joints + 1 gripper per arm (14-D, the same layout as XPolicyLab's `arx_x5`), EE poses reordered to `[x y z qw qx qy qz]`, the three cameras at 480×640 as JPEGs stamped with XPolicyLab's `XPL-RGB1` marker. Verified against `XPolicyLab.utils.data_loader.load`.

Result: `processed/xpolicylab/galaxeaOpenWorldDataset/Arrange_Fruits_20250819_011/arx_x5/data/episode_0000000.hdf5 … 0000113`, 114 files of ~130 MB (110 064 frames), Slurm job `COMPLETED` in 16 min, log streamed into the Actions run.

### 4b. Train

Workflow **`train-xpolicylab`**, inputs `policy = ACT`, `bench = galaxeaOpenWorldDataset`, `task = Arrange_Fruits_20250819_011`, `env_cfg = arx_x5`, `action = joint`, `seed = 0`, `extra = --num_epochs 30 --save_freq 30` (drop `extra` for the real 6000-epoch run; `policy = DP` with `training.num_epochs=…` for diffusion policy).

The Slurm script runs the policy's own `process_data.sh`, then its training command exactly as `policy/<P>/train.sh` would, through `xpolicylab/train_mlflow.py`, which hooks the policy's epoch summary (XPolicyLab prints no metrics itself), mirrors every epoch to MLflow, uploads the checkpoint directory as run artifacts (stored in `s3://mlflow`) and registers it as a model version.

Result: Slurm job `COMPLETED` in 10 min 47 s (most of it the policy's `process_data` decoding 330k JPEG frames; the 30 epochs take about a minute). Reproduced after the clean reinstall with the checkout on the NFS home: 14 min 28 s, same `val/loss 0.754`; the extra minutes were `process_data` writing 399 GB of decoded frames into the policy directory on NFS; those caches, the checkpoints and the evaluation results now live under `$STATE_DIR/xpolicylab` and `$STATE_DIR/robodojo` with symlinks in the checkout. MLflow run `ACT-Galaxea-Arrange_Fruits_20250819_011-arx_x5-joint-0`: 30 epochs of `train/loss`, `train/l1`, `train/kl`, `val/loss` (86.1 → 0.754); artifacts `policy_epoch_30_seed_0.ckpt` and `policy_last.ckpt` (336 MB each) plus `dataset_stats.pkl`; registered model **`ACT-Galaxea-Arrange_Fruits_20250819_011` version 1**.

### 4c. Look at the results

`https://<host>:8443/mlflow/` → experiment **xpolicylab**: curves, parameters (bench/task/env_cfg/seed/action_dim), the exact command, the Slurm job id. **Models** → the registered model, each version linked to its run and checkpoints. Compare runs across policies or seeds by selecting them.

Caveats: `env_cfg=arx_x5` is a stand-in label (Galaxea's r1lite has the same 14-D layout; a proper `r1lite` entry needs edits inside the XPolicyLab/RoboDojo submodules, i.e. a fork); the policies' pinned `torch==2.4.1` has no CUDA build for this aarch64 Blackwell node, so the envs use the cu128 index.

## 5. Evaluate and feed back

Workflow **`evaluate-xpolicylab`**, inputs `policy = ACT`, `task = stack_bowls`, `ckpt = RoboDojo-stack_bowls-arx_x5-joint-0`, `env_cfg = arx_x5_gpu`, `action = joint`, `seed = 0`, `eval_num = 5`.

The Slurm GPU job (`slurm/evaluate-xpolicylab.sbatch`) runs XPolicyLab's `eval.sh` exactly as documented: a policy server in the policy's env and RoboDojo's eval client in the `robodojo` env (Isaac Sim 5.1 headless, both through the conda shim), on the GPU Slurm allocated. RoboDojo plays `eval_num` episodes of the task against its evaluation layouts, scores them and writes `eval_result/<bench>/<task>/<policy>/<env_cfg>/<seed>_ckpt_name=…/<timestamp>/_result.json` plus one mp4 per camera. `xpolicylab/eval_mlflow.py` then logs `eval/success_rate`, `eval/score` and `eval/episodes` **on the training run that produced the checkpoint** (found by name; a new `eval-…` run if there is none), uploads the result file and the videos as artifacts under `eval/<task>/`, and tags the registered model version with the scores. So the registry answers "how good is version N" directly.

This closes the loop on data that has a simulator: RoboDojo's own `stack_bowls` episodes (`raw/open_datasets/robodojo/stack_bowls`, pulled from the hub by `setup-envs`) → copied into `processed/xpolicylab/RoboDojo/stack_bowls/arx_x5` (workflow `robodojo_to_xpolicylab`) → trained (`train-xpolicylab`, `bench = RoboDojo`) → evaluated here → scores on the run and the model version. For Galaxea's r1lite there is no simulator, so its evaluation stays physical.

Result (reproduced after the clean reinstall, Gitea run 43): the 30-epoch ACT checkpoint (`RoboDojo-stack_bowls-arx_x5-joint-0`, trained in 5 min 54 s), 2 episodes of 800 steps, Slurm job `COMPLETED` in 10 min 32 s on one GB300; `eval/success_rate 0.0`, `eval/score 0.0` on the training run `ACT-RoboDojo-stack_bowls-arx_x5-joint-0`, six episode videos and `_result.json` under its `eval/stack_bowls/` artifacts, and model `ACT-RoboDojo-stack_bowls` v1 tagged `eval_stack_bowls_success_rate = 0.000`. A 30-epoch model is not expected to succeed; the point is the plumbing. The zero-action smoke test (`demo_policy`, 1 episode, 2 min 40 s) lands as a new `eval-demo_policy-…` run. A full evaluation uses the task's default episode count (25 to 50; leave `eval_num` empty).

What it took on this hardware, and why the job script does what it does:

- **The simulation runs on the GPU device (`env_cfg = *_gpu`).** RoboDojo defaults the Isaac Lab simulation device to CPU. On the GB300 nodes Isaac Sim 5.1 then never delivers Replicator camera frames (verified with Isaac Lab's own tiled camera: empty buffers on `cpu`, frames on `cuda:0`, regardless of annotator device, Fabric or the zero-delay render settings), and RoboDojo's capture kernel spins forever on the empty buffer. An `env_cfg` ending in `_gpu` is derived from its base config by the job (`device: cuda:0`), with the base's evaluation layouts and checkpoints aliased. Physics on the GPU may score slightly differently from the official CPU-physics numbers.
- **Two nodes can't run it.** On C01 and C02 the eval client's Kit viewport stopped rendering a frame during the afternoon of 2026-10-08 (first beside the ACT policy server, later with the zero-action demo policy as well, while C03 and C04 kept working), so Isaac Lab's viewport camera controller fails with "Accessed invalid null prim" while creating the environment. Software, GPU mode, scratch and extension order are identical on all four nodes; the cause was not found. The job excludes those two nodes (`#SBATCH --exclude`), and the shim makes the viewport camera placement best-effort anyway, since RoboDojo's cameras are separate render products.
- **RoboDojo assumes CPU tensors** in a few places (`np.asarray` on a tensor, `.numpy()`); `slurm/robodojo-shim/sitecustomize.py`, injected through `PYTHONPATH`, makes CUDA tensors convert transparently. Together with the `_gpu` config this avoids editing the submodule; the proper fix is a RoboDojo fork with a `device` setting and device-safe conversions.
- **No `ffmpeg` on the nodes.** RoboDojo streams camera frames through one; `install-robodojo.sh` installs `imageio-ffmpeg`'s static build into `$ROBODOJO_DIR/bin`, which the job puts on `PATH`.
- Already covered by `install-robodojo.sh`: aarch64 wheels for Isaac Sim 5.1 and torch cu128, `libgomp` preloaded, NVRTC 12.9 preloaded (torch's 12.8 doesn't know sm_103), user-space GL libraries, curobo built from source, robot configs rendered with absolute asset paths. The first Isaac Sim start on a node compiles the RTX pipelines (minutes); the caches under `$ROBODOJO_DIR/cache` are shared, so later starts take 15 s.

## The same loop for other data

| Raw format | Collect | Look at it | Train |
|---|---|---|---|
| LeRobot v2 archives (Galaxea) | archive into `raw` | catalog entry in FiftyOne | `galaxeaOpenWorldDataset_to_xpolicylab` → `train-xpolicylab` |
| LeRobot v3 (HiFi-UMI-2K) | directory into `raw` | episodes in FiftyOne | EE-space data; XPolicyLab's joint-space layout doesn't fit yet |
| xspark HDF5 (RoboDojo) | directory into `raw` | episode groups in FiftyOne | `robodojo_to_xpolicylab` → `train-xpolicylab` |
| ROS 2 mcap (h2rc) | bag directories into `raw` | bags in FiftyOne | no recorded actions in the bags; would need derived targets |
| anything else | into `raw` | add a layout to `sync-fiftyone-raw` | one `<repo>_to_xpolicylab` adapter |

## Operations cheat-sheet

| Need | Where |
|---|---|
| Start / stop / update the stack | `services/ctl.sh up` / `down` (runs compose on the service node over ssh) |
| Add a person | `services/ctl.sh user add <name> <email>` → Keycloak (SSO), MLflow account, S3 key |
| Reach it from a laptop | forward `$CADDY_PORT` + `/etc/hosts` for the four names; trust `/ca.crt` once ([services/README.md](../services/README.md#access)) |
| State on disk | `/tier1/htx_boonhan/services/<service>/`; buckets under `versitygw/buckets/` |
| Pipelines code | Gitea `admin/pipelines`; Slurm's checkout at `/tier1/htx_boonhan/services/pipelines` |
| Slurm logs | `/tier1/htx_boonhan/services/slurm-logs/<job>-<id>.log` (also streamed into the Actions log) |
| Policy envs | `/tier1/htx_boonhan/services/envs/{act,dp}` (uv venvs) |
| Secrets | `services/.env` (gitignored); Slurm jobs read `/tier1/htx_boonhan/services/slurm.env` |
