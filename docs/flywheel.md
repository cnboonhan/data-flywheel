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

## 0. Before you start

### Fresh install (what a clean redeploy takes)

Done from zero on 2026-10-08 (containers and service state wiped, raw bucket, policy envs, RoboDojo assets and caches kept), with the checkout in `$HOME` and the state on `/tier1`:

1. **Clone and configure.** `git clone --recurse-submodules https://github.com/cnboonhan/data-flywheel.git ~/workspaces/data-flywheel`; set `remote.origin.pushurl` to the port-443 SSH URL (CLAUDE.md). IsaacLab-Arena's nested submodules use SSH URLs that the cluster can't reach; fetch them with `git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive eval/system2/IsaacLab-Arena` (not needed for this loop). Copy `services/.env` (or fill in `services/.env.example`).
2. **Bring the stack up.** `services/ctl.sh up`. On a clean state it starts Caddy alone first and waits for its CA, then everything else, then provisions: Gitea admin, runner token, Keycloak OpenID source, the `pipelines` repo with workflows, scripts and shims, secrets and variables; Grafana's admin email; Keycloak's `mlflow` client; the MLflow job token; the Slurm bridge (runner key, one forced-command line in `~/.ssh/authorized_keys`, `$STATE_DIR/slurm.env` with the credentials and the paths of this checkout, `$STATE_DIR/pipelines` checkout). About 5 minutes; run it a second time if MLflow was still migrating when the token was minted (it says so).
3. **Trust the CA** on the machines that will talk to the stack: `https://<host>:8443/ca.crt`.
4. **Environments and assets** (Slurm jobs, inputs from `$STATE_DIR/slurm.env`; `set -a; . /tier1/htx_boonhan/services/slurm.env; set +a; cd /tier1/htx_boonhan/services/slurm-logs`):
   - `sbatch --export=ALL /tier1/htx_boonhan/services/pipelines/slurm/install-robodojo.sbatch` (Isaac Sim 5.1, Isaac Lab, curobo, XPolicyLab, ffmpeg, conda shim; ~1 h on a cold cache),
   - `sbatch --export=ALL .../install-policy-env.sbatch ACT`, the same for `DP` and `demo_policy`,
   - `sbatch --export=ALL .../download-robodojo.sbatch assets`, `... ckpt ACT`, `... data stack_bowls` (39 GB, 32 GB and the task's episodes from the RoboDojo hub).
   All editable installs point at `PROJECT_ROOT` from `slurm.env`; after moving the checkout, run the installers again and they re-point.
5. **Run the loop below** from Gitea. Every step was re-run after the wipe; the "Result" lines were reproduced.

What the fresh run caught, all fixed in `ctl.sh` or the scripts: dockerd creating a directory where Caddy's CA file would be (Caddy now starts first), the pipelines checkout not being recreated, the `mlflow` bucket not being created (artifact uploads then fail with 500), the MLflow token minted before the plugin had migrated its database (rerun `up`), and the installers skipping editable installs that pointed at the old checkout (now path-aware). One rule it taught: **don't push to the `pipelines` repo while runs are in flight**. Gitea cancels in-progress runs on the ref when it moves, whatever the workflow (two runs were lost that way); `ingest-episodes` now asks not to be cancelled, but queue the pushes anyway.


- **Reach the stack.** Everything is `https://flywheel.<node IP>.sslip.io:8443` plus the `fiftyone.`, `rerun.` and `s3.` subdomains (one port). On the cluster network the names resolve by themselves; from a laptop, tunnel 8443 and map the four names to 127.0.0.1 in `/etc/hosts`, or route the node IP with `sshuttle` (services/README.md → Access). Trust `/ca.crt` once.
- **Log in once.** Keycloak (`/auth`) signs you into Gitea, Grafana, FiftyOne, Rerun, Prometheus and Loki. MLflow and the S3 API use the account and keys that `services/ctl.sh user add <name> <email>` gives you.
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

A dataset is a directory under `s3://raw/<dataset>/`, in whatever format it was collected in. Three ways in:

| Way | Command | Notes |
|---|---|---|
| S3 upload (laptop, robot, HF mirror) | `aws --endpoint-url $S3 s3 sync ./capture s3://raw/<dataset>/` | Objects get ETags and bucket events |
| Already on `/tier1` | `mv <dir> /tier1/htx_boonhan/services/versitygw/buckets/raw/<dataset>` | Instant rename, no copy (how the 18 TB of public data went in) |
| Download job | *(planned)* a workflow running `hf download --local-dir /buckets/raw/<dataset>` | needs `/buckets/raw` writable in job containers |

**Example.** The Galaxea Open-World set was moved in as 227 per-task archives, `raw/galaxea-open-world-r1lite/lerobot/<task>.tar.gz`. A newly collected episode is uploaded instead; to exercise that path, push a marker and read the prefix back:

```bash
echo "collected $(date -u +%FT%TZ) on r1lite" > collection.log
aws --endpoint-url $S3 s3 cp collection.log s3://raw/galaxea-open-world-r1lite/logs/collection.log
aws --endpoint-url $S3 s3 ls s3://raw/galaxea-open-world-r1lite/
```

Result: `PRE lerobot/`, `PRE logs/`, `lerobot_info.json`, `README.md`, `demo.mp4`. The file is also visible in the S3 web UI (`https://s3.<host>:8443/ui/`, log in with the access key) and as a plain file under `buckets/raw/`.

## 2. Process: `raw` → `processed` with Gitea Actions

Pipelines are code in `admin/pipelines` (`.gitea/workflows/*.yml` plus `fiftyone/*.py`, `xpolicylab/*.py`, `slurm/*`). Jobs run as containers on the compose network with the buckets mounted read-only at `/buckets` and `S3_ENDPOINT_URL=http://versitygw:7070`, so they read from the mount and write through S3. `services/ctl.sh up` seeds the repo from `services/` once; after that the repo's copies are what runs.

### 2a. Unpack the task archive

Workflow **`unpack-archives`**, inputs `archives = galaxea-open-world-r1lite/lerobot/Arrange_Fruits_20250819_011.tar.gz`, `dest = galaxea-open-world-r1lite`.

Result: `processed/galaxea-open-world-r1lite/Arrange_Fruits_20250819_011/` with `data/`, `meta/`, `videos/` (575 files) and a `.unpacked` marker so re-runs skip it. 88 s including the job's own setup.

### 2b. Canonical episodes

Every raw format gets one converter into the layout both viewers read:

```
processed/episodes/<dataset>/<subset>/<episode_id>/
    <camera>.mp4        one per camera, frames on a common clock
    episode.json        dataset, episode_id, source, format, robot, task, tasks, fps, frames, duration_s, cameras
    signals.parquet     long format: t, group ("observation.state.left_arm" …), index, value
```

Workflow **`episodes-lerobot`** (LeRobot v2 and v3; `episodes-mcap` does ROS 2 bags), inputs `dataset = galaxea-open-world-r1lite`, `path = galaxea-open-world-r1lite/Arrange_Fruits_*`, `bucket = processed`, `limit = 0`.

Result: `processed/episodes/galaxea-open-world-r1lite/Arrange_Fruits_20250819_011/episode_000000 … 000113`: 114 episodes, 4 cameras each, 22 signal groups. 87 s.

## 3. Verify: look at what was collected

Workflow **`ingest-episodes`**, input `dataset = galaxea-open-world-r1lite`. Two jobs:

1. `episode_rrd.py` writes one **Rerun** recording per episode to `processed/rerun/<dataset>/<episode_id>.rrd` (camera videos as video assets, every signal group as a scalar series, one timeline).
2. `ingest_episodes.py` loads the episodes into the grouped **FiftyOne** dataset `episodes/galaxea-open-world-r1lite`: one group per episode, one slice per camera, fields from `episode.json`, signal summaries, and `rerun_url`.

Result: 114 recordings written, 456 videos added (114 episodes × 4 cameras), 116 groups in the dataset (two from an earlier `Make_The_Bed` test). 77 s.

Then, in the browser:

- **FiftyOne** `https://fiftyone.<host>:8443/` → dataset `episodes/galaxea-open-world-r1lite`. Each tile is an episode; the slice selector switches camera. Filter by `task`, `duration_s`, `left_gripper_range`; play any clip.
- **Rerun**: a sample's `rerun_url` opens `https://rerun.<host>:8443/?url=…/data/<dataset>/<episode_id>.rrd`: all cameras and every joint/gripper signal scrubbing together (the recording is served on the same SSO session; checked: `206` partial content). Rerun also opens raw mcap or LeRobot directories directly (`rerun --save out.rrd <path>`) for data that hasn't been converted yet.

What to check here, because training inherits it: episode count matches the collection log, every episode has all cameras, durations are plausible, gripper signals actually move, task strings are right.

## 4. Train: XPolicyLab under Slurm, tracked in MLflow

XPolicyLab (`eval/system1/RoboDojo/XPolicyLab`) trains ~35 policies from one input format, its xspark HDF5 (`PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`). That tree is `processed/xpolicylab/`, and `eval/system1/RoboDojo/data` is a symlink to it. Training runs as Slurm GPU jobs (the node's Docker has no GPU runtime); Gitea submits them over SSH with a key locked to `services/slurm/slurm-submit` and streams the Slurm log into the Actions log.

### 4a. Training format

Workflow **`convert-xpolicylab`** (a Slurm CPU job), inputs `subsets = galaxea-open-world-r1lite/Arrange_Fruits_*`, `limit = 0`, `env_cfg = arx_x5`.

Galaxea LeRobot v2.1 → xspark: 6 arm joints + 1 gripper per arm (14-D, the same layout as XPolicyLab's `arx_x5`), EE poses reordered to `[x y z qw qx qy qz]`, the three cameras at 480×640 as JPEGs stamped with XPolicyLab's `XPL-RGB1` marker. Verified against `XPolicyLab.utils.data_loader.load`.

Result: `processed/xpolicylab/Galaxea/Arrange_Fruits_20250819_011/arx_x5/data/episode_0000000.hdf5 … 0000113`, 114 files of ~130 MB (110 064 frames), Slurm job `COMPLETED` in 16 min, log streamed into the Actions run.

### 4b. Train

Workflow **`train-xpolicylab`**, inputs `policy = ACT`, `bench = Galaxea`, `task = Arrange_Fruits_20250819_011`, `env_cfg = arx_x5`, `action = joint`, `seed = 0`, `extra = --num_epochs 30 --save_freq 30` (drop `extra` for the real 6000-epoch run; `policy = DP` with `training.num_epochs=…` for diffusion policy).

The Slurm script runs the policy's own `process_data.sh`, then its training command exactly as `policy/<P>/train.sh` would, through `xpolicylab/train_mlflow.py`, which hooks the policy's epoch summary (XPolicyLab prints no metrics itself), mirrors every epoch to MLflow, uploads the checkpoint directory as run artifacts (stored in `s3://mlflow`) and registers it as a model version.

Result: Slurm job `COMPLETED` in 10 min 47 s (most of it the policy's `process_data` decoding 330k JPEG frames; the 30 epochs take about a minute). Reproduced after the clean reinstall with the checkout on the NFS home: 14 min 28 s, same `val/loss 0.754`; the extra minutes were `process_data` writing 399 GB of decoded frames into the policy directory on NFS; those caches, the checkpoints and the evaluation results now live under `$STATE_DIR/xpolicylab` and `$STATE_DIR/robodojo` with symlinks in the checkout. MLflow run `ACT-Galaxea-Arrange_Fruits_20250819_011-arx_x5-joint-0`: 30 epochs of `train/loss`, `train/l1`, `train/kl`, `val/loss` (86.1 → 0.754); artifacts `policy_epoch_30_seed_0.ckpt` and `policy_last.ckpt` (336 MB each) plus `dataset_stats.pkl`; registered model **`ACT-Galaxea-Arrange_Fruits_20250819_011` version 1**.

### 4c. Look at the results

`https://<host>:8443/mlflow/` → experiment **xpolicylab**: curves, parameters (bench/task/env_cfg/seed/action_dim), the exact command, the Slurm job id. **Models** → the registered model, each version linked to its run and checkpoints. Compare runs across policies or seeds by selecting them.

Caveats: `env_cfg=arx_x5` is a stand-in label (Galaxea's r1lite has the same 14-D layout; a proper `r1lite` entry needs edits inside the XPolicyLab/RoboDojo submodules, i.e. a fork); the policies' pinned `torch==2.4.1` has no CUDA build for this aarch64 Blackwell node, so the envs use the cu128 index.

## 5. Evaluate and feed back

Workflow **`evaluate-xpolicylab`**, inputs `policy = ACT`, `task = stack_bowls`, `ckpt = RoboDojo-stack_bowls-arx_x5-joint-0`, `env_cfg = arx_x5_gpu`, `action = joint`, `seed = 0`, `eval_num = 5` (`policy = demo_policy`, `ckpt = demo`, `eval_num = 1` is the zero-action smoke test).

The Slurm GPU job (`slurm/evaluate-xpolicylab.sbatch`) runs XPolicyLab's `eval.sh` exactly as documented: a policy server in the policy's env and RoboDojo's eval client in the `robodojo` env (Isaac Sim 5.1 headless, both through the conda shim), on the GPU Slurm allocated. RoboDojo plays `eval_num` episodes of the task against its evaluation layouts, scores them and writes `eval_result/<bench>/<task>/<policy>/<env_cfg>/<seed>_ckpt_name=…/<timestamp>/_result.json` plus one mp4 per camera. `xpolicylab/eval_mlflow.py` then logs `eval/success_rate`, `eval/score` and `eval/episodes` **on the training run that produced the checkpoint** (found by name; a new `eval-…` run if there is none), uploads the result file and the videos as artifacts under `eval/<task>/`, and tags the registered model version with the scores. So the registry answers "how good is version N" directly.

This closes the loop on data that has a simulator: RoboDojo's own `stack_bowls` episodes (`raw/robodojo/stack_bowls`, pulled from the hub by `download-robodojo.sbatch data stack_bowls`) → promoted into `processed/xpolicylab/RoboDojo/stack_bowls/arx_x5` (`promote_xspark.py`) → trained (`train-xpolicylab`, `bench = RoboDojo`) → evaluated here → scores on the run and the model version. For Galaxea's r1lite there is no simulator, so its evaluation stays physical.

Result (reproduced after the clean reinstall, Gitea run 43): the 30-epoch ACT checkpoint (`RoboDojo-stack_bowls-arx_x5-joint-0`, trained in 5 min 54 s), 2 episodes of 800 steps, Slurm job `COMPLETED` in 10 min 32 s on one GB300; `eval/success_rate 0.0`, `eval/score 0.0` on the training run `ACT-RoboDojo-stack_bowls-arx_x5-joint-0`, six episode videos and `_result.json` under its `eval/stack_bowls/` artifacts, and model `ACT-RoboDojo-stack_bowls` v1 tagged `eval_stack_bowls_success_rate = 0.000`. A 30-epoch model is not expected to succeed; the point is the plumbing. The zero-action smoke test (`demo_policy`, 1 episode, 2 min 40 s) lands as a new `eval-demo_policy-…` run. A full evaluation uses the task's default episode count (25 to 50; leave `eval_num` empty).

What it took on this hardware, and why the job script does what it does:

- **The simulation runs on the GPU device (`env_cfg = *_gpu`).** RoboDojo defaults the Isaac Lab simulation device to CPU. On the GB300 nodes Isaac Sim 5.1 then never delivers Replicator camera frames (verified with Isaac Lab's own tiled camera: empty buffers on `cpu`, frames on `cuda:0`, regardless of annotator device, Fabric or the zero-delay render settings), and RoboDojo's capture kernel spins forever on the empty buffer. An `env_cfg` ending in `_gpu` is derived from its base config by the job (`device: cuda:0`), with the base's evaluation layouts and checkpoints aliased. Physics on the GPU may score slightly differently from the official CPU-physics numbers.
- **Two nodes can't run it.** On C01 and C02 the eval client's Kit viewport stopped rendering a frame during the afternoon of 2026-10-08 (first beside the ACT policy server, later with the zero-action demo policy as well, while C03 and C04 kept working), so Isaac Lab's viewport camera controller fails with "Accessed invalid null prim" while creating the environment. Software, GPU mode, scratch and extension order are identical on all four nodes; the cause was not found. The job excludes those two nodes (`#SBATCH --exclude`), and the shim makes the viewport camera placement best-effort anyway, since RoboDojo's cameras are separate render products.
- **RoboDojo assumes CPU tensors** in a few places (`np.asarray` on a tensor, `.numpy()`); `slurm/robodojo-shim/sitecustomize.py`, injected through `PYTHONPATH`, makes CUDA tensors convert transparently. Together with the `_gpu` config this avoids editing the submodule; the proper fix is a RoboDojo fork with a `device` setting and device-safe conversions.
- **No `ffmpeg` on the nodes.** RoboDojo streams camera frames through one; `install-robodojo.sbatch` installs `imageio-ffmpeg`'s static build into `$ROBODOJO_DIR/bin`, which the job puts on `PATH`.
- Already covered by `install-robodojo.sbatch`: aarch64 wheels for Isaac Sim 5.1 and torch cu128, `libgomp` preloaded, NVRTC 12.9 preloaded (torch's 12.8 doesn't know sm_103), user-space GL libraries, curobo built from source, robot configs rendered with absolute asset paths. The first Isaac Sim start on a node compiles the RTX pipelines (minutes); the caches under `$ROBODOJO_DIR/cache` are shared, so later starts take 15 s.

## The same loop for other data

| Raw format | Collect | Process → episodes | Train |
|---|---|---|---|
| LeRobot v2 (Galaxea) | archive or directory into `raw` | `unpack-archives` (if archived) → `episodes-lerobot` | `convert-xpolicylab` → `train-xpolicylab` |
| LeRobot v3 (HiFi-UMI-2K) | directory into `raw` | `episodes-lerobot` (cuts episodes out of the per-camera videos) | EE-space data; XPolicyLab's joint-space layout doesn't fit yet |
| ROS 2 mcap (h2rc) | bag directories into `raw` | `episodes-mcap` | no recorded actions in the bags; would need derived targets |
| anything else | into `raw` | write one `episodes_from_<format>.py` (see `fiftyone/episodes.py`) | one `convert_<format>_xspark.py` |

Adding a format is one converter; FiftyOne, Rerun and the training path need nothing new.

## Operations cheat-sheet

| Need | Where |
|---|---|
| Start / stop / update the stack | `services/ctl.sh up` / `down` (runs compose on the service node over ssh) |
| Add a person | `services/ctl.sh user add <name> <email>` → Keycloak (SSO), MLflow account, S3 key |
| Reach it from a laptop | tunnel 8443 + `/etc/hosts` for the four names, or `sshuttle -r <login> <node ip>/32`; trust `/ca.crt` once |
| State on disk | `/tier1/htx_boonhan/services/<service>/`; buckets under `versitygw/buckets/` |
| Pipelines code | Gitea `admin/pipelines`; Slurm's checkout at `/tier1/htx_boonhan/services/pipelines` |
| Slurm logs | `/tier1/htx_boonhan/services/slurm-logs/<job>-<id>.log` (also streamed into the Actions log) |
| Policy envs | `/tier1/htx_boonhan/services/envs/{act,dp}` (uv venvs) |
| Secrets | `services/.env` (gitignored); Slurm jobs read `/tier1/htx_boonhan/services/slurm.env` |
