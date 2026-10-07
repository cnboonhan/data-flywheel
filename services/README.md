# services

The **Store** column of the [architecture diagram](../architecture.html), as a Docker Compose stack behind a Caddy reverse proxy. For how a dataset travels through the whole loop, raw → processed → trained → registered, see [docs/flywheel.md](../docs/flywheel.md).

| Path | Service | Image |
|---|---|---|
| `/gitea` | Git hosting and Actions (CI/CD over the datasets) | `gitea/gitea`, `gitea/act_runner` |
| `/mlflow` | Model registry, experiment tracking | `ghcr.io/mlflow/mlflow` |
| `/grafana` | Dashboards (Prometheus and Loki pre-provisioned) | `grafana/grafana` |
| `/prometheus` | Metrics | `prom/prometheus` |
| `/loki` | Log API (`/loki/api/v1/push`, `/loki/api/v1/query_range`) | `grafana/loki` |
| `/s3` | S3 gateway web UI | `versity/versitygw` |
| `/ca.crt` | Root certificate of the local CA | `caddy` |
| port `8444` | S3 API | `versity/versitygw` |
| port `8445` | FiftyOne App (dataset browser) | `voxel51/fiftyone`, `mongo` |
| port `8446` | Rerun web viewer + episode recordings | `rerun-sdk` (built in `rerun/Dockerfile`) |

The web UIs listen on **port 8443**. Two services get their **own port**: the S3 API (8444) because SigV4 signs the request path, so it can't sit behind a prefix, and FiftyOne (8445) because its App redirects every sub-path to `/`.

All images are multi-arch (`linux/arm64` and `linux/amd64`), so the same files run on the GB300 node and on an Intel machine; nothing pins a platform.

## Run

```bash
cp services/.env.example services/.env   # set the S3 and Grafana secrets
services/ctl.sh up
services/ctl.sh ps
services/ctl.sh logs -f caddy
services/ctl.sh down
```

`ctl.sh` is a thin wrapper: it exports `.env`, derives the bcrypt hash Caddy needs, creates the state directories, and runs `docker compose`. Run from the login node it re-runs itself on `$SERVICE_NODE` (C2-GB300-02-C03) over ssh; the repo and state are on `/tier1`, so paths match. Slurm isn't involved. Containers belong to the node's dockerd with `restart: unless-stopped`, so they keep running after the script returns and come back after a reboot. On any other machine, set `SERVICE_NODE` to its hostname and `ctl.sh` runs compose locally.

Plain `docker compose` works too, on the node, with `.env` exported and `ADMIN_PASSWORD_HASH` set:

```bash
ssh C2-GB300-02-C03
cd /tier1/htx_boonhan/workspaces/data-flywheel/services
set -a; . ./.env; set +a
export ADMIN_PASSWORD_HASH=$(docker run --rm caddy:2.11 caddy hash-password --plaintext "$ADMIN_PASSWORD")
docker compose -p flywheel up -d
```

`ctl.sh up` picks up changes to `docker-compose.yml` and `.env`. The config files are bind-mounted, so after editing `caddy/Caddyfile`, `prometheus/prometheus.yml` or `loki/loki.yml` run `ctl.sh restart <service>`.

After editing `mlflow/Dockerfile`, run `ctl.sh up --build`; a plain `up` only builds the image when it's missing.

If a git operation deletes and recreates files under `services/` (switching to a branch without the directory, a rebase, `git stash`), the running containers keep the old, deleted inodes and start returning 404s or lose their config. Run `ctl.sh up --force-recreate` afterwards.

## Access

Everything is HTTPS on **one port, 8443**. Apps that can serve under a path are path-routed on `SERVICE_HOST`; the three that can't get a subdomain of it:

| URL | Service |
|---|---|
| `https://<SERVICE_HOST>:8443/` | index, `/gitea`, `/mlflow`, `/grafana`, `/prometheus`, `/loki`, `/auth` (Keycloak), `/oauth2` (SSO), `/ca.crt` |
| `https://fiftyone.<SERVICE_HOST>:8443/` | FiftyOne |
| `https://rerun.<SERVICE_HOST>:8443/` | Rerun viewer, recordings under `/data/` |
| `https://s3.<SERVICE_HOST>:8443/` | S3 API; the gateway's web UI at `/ui/` |

`SERVICE_HOST` is `flywheel.<node IP>.sslip.io`: a public wildcard DNS name that resolves to the node from anywhere (the cluster has no DNS for subdomains of the node). Single sign-on needs one stable hostname, which is why `localhost` is no longer an alias. Caddy runs a local CA (`local_certs`) and issues the certificates.

**From a machine that can reach the node directly:** open `https://<SERVICE_HOST>:8443/`.

**From your laptop through the login node:** forward 8443 and point the names at the tunnel in `/etc/hosts` (wildcards aren't supported there, so list the four):

```bash
ssh -L 8443:C2-GB300-02-C03:8443 <login-node>
# /etc/hosts on the laptop:
# 127.0.0.1 flywheel.10.80.81.30.sslip.io fiftyone.flywheel.10.80.81.30.sslip.io rerun.flywheel.10.80.81.30.sslip.io s3.flywheel.10.80.81.30.sslip.io
# then open https://flywheel.10.80.81.30.sslip.io:8443/
```

To make that permanent, add an entry to `~/.ssh/config` on your laptop. The `LocalForward` lines carry the same port mappings; everything else is whatever you already use to reach the login node.

```sshconfig
Host flywheel
    HostName <login-node>
    User <your-user>
    LocalForward 8443 C2-GB300-02-C03:8443
    LocalForward 8444 C2-GB300-02-C03:8444
    LocalForward 8445 C2-GB300-02-C03:8445
    LocalForward 8446 C2-GB300-02-C03:8446
    ServerAliveInterval 30
    ExitOnForwardFailure yes
```

Then:

```bash
ssh flywheel          # interactive shell, tunnel open while it lasts
ssh -N flywheel       # tunnel only, no shell (Ctrl-C to close)
ssh -fN flywheel      # tunnel in the background
```

`ExitOnForwardFailure` makes ssh fail immediately if 8443 or 8444 is already taken locally, instead of silently connecting without the tunnel. Node names resolve on the login node, so `C2-GB300-02-C03` in `LocalForward` doesn't need to resolve on your laptop.

**Trusting the CA** (once per device) removes the browser warning and lets CLI tools verify. Fetch it with `curl -k https://localhost:8443/ca.crt -o flywheel-ca.crt` through the tunnel, or copy `$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt` from the login node.

- macOS: `sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain flywheel-ca.crt`
- Ubuntu: `sudo cp flywheel-ca.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates`
- Per tool: `curl --cacert`, `AWS_CA_BUNDLE=flywheel-ca.crt`, `REQUESTS_CA_BUNDLE=flywheel-ca.crt` (MLflow, boto3), `SSL_CERT_FILE=flywheel-ca.crt` (Go tools)

The S3 web UI at `/s3/` runs in the browser and calls the S3 API on port 8444 directly. Its login page offers both `https://localhost:8444` (through the tunnel) and `https://<SERVICE_HOST>:8444` (direct); pick the one your browser can reach. If you haven't trusted the CA, open `https://localhost:8444/health` once and accept the certificate, because the UI's background requests can't show that prompt.

## Gitea and Actions

Gitea is the home of dataset pipelines: code lives in repos, and Gitea Actions (GitHub Actions-compatible workflows under `.gitea/workflows/`) runs them. `ctl.sh up` bootstraps it on first run:

- creates the admin user from `ADMIN_USER` / `ADMIN_PASSWORD`;
- generates a runner registration token into `$STATE_DIR/act_runner/token` and registers the `act_runner` container (name `$SERVICE_NODE`, labels `ubuntu-latest` and `python`);
- creates the private repo `admin/pipelines` with the secrets `S3_ACCESS_KEY` / `S3_SECRET_KEY` and the example workflow `gitea/examples/process-raw.yml`.

Jobs run as sibling containers through the node's Docker socket (`act_runner/config.yaml`). They join the compose network, so `S3_ENDPOINT_URL=http://versitygw:7070` and `AWS_DEFAULT_REGION` are preset in every job; the example lists the `raw` bucket with boto3 and writes a manifest into `processed`. Clone over HTTPS (SSH is disabled):

```bash
git -c http.sslCAInfo=flywheel-ca.crt clone https://<SERVICE_HOST>:8443/gitea/admin/pipelines.git
```

Reacting to new data: there is no event wiring yet. Workflows trigger on `push`, `schedule` (cron, e.g. poll the bucket) and `workflow_dispatch` (API or the Run button). The S3 gateway can post bucket events to a webhook (`--event-webhook-url`), so a small bridge that turns those into `workflow_dispatch` calls would make uploads trigger runs.

## Episodes: one layout, two viewers

Raw datasets arrive in any format (LeRobot v2/v3, ROS 2 mcap, tar archives…), and nothing can show "an episode" until something interprets the format. So every format gets a converter into **one canonical layout** in the processed bucket, and the viewers read only that:

```
processed/episodes/<dataset>/<...>/<episode_id>/
    <camera>.mp4        one per camera, frames on a common clock
    episode.json        dataset, episode_id, source, format, robot, task, tasks, fps, frames, duration_s, cameras{...}
    signals.parquet     long format: t (s), group ("observation.state.left_arm", "hdas.feedback_arm_left.position"…), index, value
```

| Converter (in `fiftyone/`) | Workflow | Handles |
|---|---|---|
| `episodes_from_lerobot.py` | `episodes-lerobot` | LeRobot v2 (per-episode mp4s uploaded as is) and v3 (episodes cut out of the per-camera mp4s with ffmpeg); signals from the parquet rows |
| `episodes_from_mcap.py` | `episodes-mcap` | ROS 2 bags: `CompressedImage` topics → mp4, JointState/IMU/Wrench topics → signals |
| `episodes.py` | | the layout and S3 writers the converters share |

Then `ingest-episodes` (one workflow, any dataset):
1. `episode_rrd.py` builds a **Rerun** recording per episode (`processed/rerun/<dataset>/<episode_id>.rrd`): the camera videos as video assets and every signal group as scalar series on one timeline, so you scrub cameras and joints together.
2. `ingest_episodes.py` loads the episodes into a **grouped FiftyOne dataset** `episodes/<dataset>`: one group per episode, one slice per camera, fields from `episode.json`, a few signal summaries, and `rerun_url`.

**Rerun** runs as its own service on **port 8446** behind the admin login (it has no login of its own); the recordings are served from the same origin at `/data/<dataset>/<episode_id>.rrd`, so the browser's login covers both. Open a recording with `https://localhost:8446/?url=https://localhost:8446/data/<dataset>/<episode_id>.rrd`, which is exactly the `rerun_url` field on each FiftyOne sample. Rerun also opens most raw robotics formats natively (`rerun --save out.rrd <bag.mcap>` or a LeRobot directory) for one-off inspection of data that hasn't been converted yet.

The earlier per-format ingests (`ingest-lerobot`, `convert-mcap`) still exist; the canonical path supersedes them.

## FiftyOne

[FiftyOne](https://github.com/voxel51/fiftyone) browses the datasets: videos, episodes, metadata, filters. The App is at `https://<SERVICE_HOST>:8445/` (or `https://localhost:8445/` through the tunnel) with the admin login. It shows what pipelines have *ingested* into its MongoDB (`mongo` service), and streams media from the bucket directories, which are mounted read-only at `/buckets` in the FiftyOne container and in every Actions job container (`act_runner/config.yaml`), so ingested filepaths resolve in both.

Ingest scripts live in `fiftyone/`, one per dataset format; the workflows in `gitea/examples/` call them. Jobs that run in the FiftyOne image can't use `actions/checkout` (no `node` there), so they fetch the repo archive from Gitea's API with a few lines of Python instead. `ctl.sh up` seeds both into the `pipelines` repo (`fiftyone/*.py`, `.gitea/workflows/*.yml`) once; from then on the repo's copies are what runs, so edit them there.

- `fiftyone/ingest_lerobot.py` + `ingest-lerobot.yml`: LeRobot v2 and v3. Inputs: FiftyOne dataset name, bucket, and a glob matching LeRobot roots (directories with `meta/info.json`). Skips videos already ingested.
  - **v3** (HiFi-UMI-2K, path `HiFi-UMI-2K/chunk-000*/part-*`): each part has one mp4 per camera holding ~1 100 episodes as time ranges. A sample is one such mp4 (`chunk`, `part`, `camera`) with an `episodes` field of temporal detections (task text, frame range, episode index); the saved view **episodes** turns those into one clip per episode per camera.
  - **v2** (Galaxea, bucket `processed`, path `galaxea-open-world-r1lite/*`): one mp4 per episode per camera, so a sample is an episode video (`subset` = the per-task dataset, `episode_index`, `camera`, `task`, `tasks`, `length`).
- `fiftyone/unpack_archives.py` + `unpack-archives.yml`: extracts tar archives from raw into processed through the S3 gateway (reads from the mount, writes via S3, so the objects get ETags and events). Galaxea ships one LeRobot v2 dataset per task as a tar.gz; unpack them, then ingest from `processed`. A `.unpacked` marker makes re-runs skip finished archives.
- `fiftyone/convert_mcap.py` + `fiftyone/ingest_videos.py` + `convert-mcap.yml`: h2rc (ROS 2 mcap bags). The first job decodes every `CompressedImage` camera topic of each episode and pipes the JPEG frames through ffmpeg (from the `imageio-ffmpeg` wheel) into H.264 mp4s at `processed/h2rc/<same path>/<camera>.mp4`, plus an `episode.json` with task, day, frame counts and fps. The second job ingests those as plain video samples (`task`, `day`, `episode`, `camera`, `fps`, `frames`). Inputs: an episode-directory glob and an optional limit; converted episodes are skipped on re-runs. About 10 s per episode.

## XPolicyLab training (Slurm)

Training data for XPolicyLab (`eval/system1/RoboDojo/XPolicyLab`) is its "xspark v1.0" HDF5: one file per episode under `PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`. That tree lives in the processed bucket as `processed/xpolicylab/`, and `eval/system1/RoboDojo/data` is a symlink to it (ignored by git). Jobs run under Slurm on the `raus_manual` partition (GB300 nodes); Docker there has no GPU runtime, so training doesn't run in Actions containers.

- `xpolicylab/convert_galaxea_xspark.py` + `slurm/convert-galaxea-xspark.sbatch`: Galaxea (LeRobot v2.1, unpacked in `processed/galaxea-open-world-r1lite/`) → xspark. State/action = 6 arm joints + 1 gripper per arm (14-D, same as XPolicyLab's `arx_x5`), EE poses reordered to `[x y z qw qx qy qz]`, the three cameras resized to 480x640 and stored as JPEGs stamped with XPolicyLab's `XPL-RGB1` marker (its `decode_image_bit` swaps channels on unmarked JPEGs). Verified against `XPolicyLab.utils.data_loader.load`. ~0.4 s per episode.
- `xpolicylab/train_mlflow.py` + `slurm/train-xpolicylab.sbatch`: runs a policy's `process_data.sh` and then its training command exactly as `policy/<P>/train.sh` would, but through a wrapper that mirrors the per-epoch losses to **MLflow** (experiment `xpolicylab`, run `<policy>-<bench>-<task>-<env_cfg>-<action>-<seed>`), uploads the checkpoint directory as run artifacts, and registers it in the **model registry** as a version of `<policy>-<bench>-<task>`. MLflow's artifact store is the `mlflow` bucket on the S3 gateway (`--artifacts-destination s3://mlflow`, proxied, so clients need no S3 keys). XPolicyLab itself has no training dashboard: ACT computes epoch summaries and never prints them, DP writes `logs.json.txt` (its wandb calls are commented out). Supported: `ACT`, `DP`. DP's shipped task config has `agent_pos` commented out, so its own `train.sh` override fails; the Slurm script adds it with `+task.shape_meta.obs.agent_pos.*`.

```bash
cd /tier1/htx_boonhan/services/slurm-logs   # Slurm writes <job>-<id>.log here
EXP="ALL,MLFLOW_TRACKING_URI=https://$SERVICE_HOST:8443/mlflow,MLFLOW_TRACKING_USERNAME=$ADMIN_USER,MLFLOW_TRACKING_PASSWORD=$ADMIN_PASSWORD,MLFLOW_TRACKING_SERVER_CERT_PATH=<ca.crt>"
sbatch --export="$EXP,S3_ENDPOINT_URL=https://$SERVICE_HOST:8444,AWS_CA_BUNDLE=<ca.crt>,AWS_ACCESS_KEY_ID=$ADMIN_USER,AWS_SECRET_ACCESS_KEY=$ADMIN_PASSWORD" \
  /tier1/htx_boonhan/services/pipelines/slurm/convert-galaxea-xspark.sbatch 'galaxea-open-world-r1lite/*' 0 arx_x5
sbatch --export="$EXP" /tier1/htx_boonhan/services/pipelines/slurm/train-xpolicylab.sbatch ACT Galaxea Make_The_Bed_20250730_012 arx_x5 joint 0
sbatch --export="$EXP" /tier1/htx_boonhan/services/pipelines/slurm/train-xpolicylab.sbatch DP  Galaxea Make_The_Bed_20250730_012 arx_x5 joint 0
```

**From Gitea Actions:** the workflows `train-xpolicylab` and `convert-xpolicylab` submit these same scripts as Slurm jobs and stream the job log into the Actions log until it ends (`slurm/follow.sh`). They reach Slurm over SSH as you, with a dedicated key that `ctl.sh up` creates at `$STATE_DIR/act_runner/ssh/` and adds to `~/.ssh/authorized_keys` locked to the forced command `services/slurm/slurm-submit` (`restrict`, so no shell, no forwarding): it can only `sbatch` a script from the pipelines checkout at the commit being run, query a job's state, read its log, or cancel it. The private key and `user@SLURM_LOGIN_HOST` go into the pipelines repo as the Actions secrets `SLURM_SSH_KEY` / `SLURM_SSH_HOST`. Jobs get their MLflow/S3 credentials from `$STATE_DIR/slurm.env` (mode 600, written by `ctl.sh`), never from Gitea. Set `SLURM_LOGIN_HOST` in `.env` to enable all of this; leave it empty to disable.

The Slurm scripts find the Python next to them through `PIPELINES_ROOT`, a checkout of the Gitea `pipelines` repo at `/tier1/htx_boonhan/services/pipelines` (sbatch copies the script itself into the spool dir). `ctl.sh up` seeds `xpolicylab/*.py` and `slurm/*.sbatch` into that repo once; `git pull` the checkout after changing them there.

Policy environments are uv venvs at `/tier1/htx_boonhan/services/envs/<policy, lowercase>` (`act`, `dp`): Python 3.10, torch from the `cu128` index (aarch64 + Blackwell; the policies' pinned `torch==2.4.1` has no CUDA build for this node), the policy's `install.sh` packages with numpy/numba pins relaxed, `pip install -e` of the policy and of XPolicyLab, plus `mlflow-skinny`.

`env_cfg=arx_x5` is a stand-in: Galaxea's r1lite has the same 14-D layout, and a proper `r1lite` entry needs edits inside the XPolicyLab and RoboDojo submodules (`utils/robot/_robot_info.json`, `env_cfg/r1lite.yml`), i.e. a fork.

## Logins and single sign-on

**Keycloak** (`/auth/`, realm `flywheel`) is the identity provider. One login covers:

| Service | How |
|---|---|
| Gitea, Grafana | native OIDC ("Sign in with Keycloak"); accounts auto-register on first login, Grafana maps the `admins` group to Admin |
| FiftyOne, Rerun, Prometheus, Loki | no login of their own: Caddy asks **oauth2-proxy** (`/oauth2/`), which holds a session cookie for `.SERVICE_HOST`; without one the browser is sent to Keycloak and back |
| MLflow | its own accounts (basic auth): API clients such as training jobs can't do browser SSO. Same username and password, mirrored by `ctl.sh user add` |
| S3 API and its web UI | access keys per user (SigV4), issued by `ctl.sh user add`; no browser session involved |

The Keycloak admin console is at `/auth/admin/flywheel/console/` (the bootstrap admin is `ADMIN_USER`); `/oauth2/sign_out` ends the proxy session. Groups `admins` and `users`; oauth2-proxy admits both.

**Adding a person everywhere:**

```bash
services/ctl.sh user add <name> <email> [password]
```

Creates the Keycloak user (group `users`), the MLflow account, and an S3 access key pair (printed once), with one password for Keycloak and MLflow (random if not given). Gitea and Grafana accounts appear on their first Keycloak login. Idempotent.

The realm (clients, groups, mappers, the admin user) comes from `keycloak/realm.json.tmpl`, rendered by `ctl.sh up` with the secrets from `.env` and imported on Keycloak's first start only; later changes are made in the console (or wipe `$STATE_DIR/keycloak/db` to re-import).

## Logins (reference)

One admin login, `ADMIN_USER` / `ADMIN_PASSWORD` in `.env`, is applied everywhere by `docker-compose.yml`:

| Service | How the admin login is used |
|---|---|
| Gitea | admin user; registration is off, the admin creates accounts |
| MLflow | admin user of the basic-auth app; further users created by the admin |
| Grafana | admin user |
| S3 gateway (API and `/s3/` UI) | root account: access key = user, secret key = password |
| Prometheus, Loki, FiftyOne, Rerun | single sign-on at the proxy (Keycloak via oauth2-proxy, see above) |

Caveat when **changing** the password: the S3 gateway and the proxy pick it up on the next `ctl.sh up`, but MLflow and Grafana only read it the first time they create their admin user. After editing `.env`, also run:

```bash
services/ctl.sh exec grafana grafana cli admin reset-admin-password "$ADMIN_PASSWORD"
curl -u admin:<old> -X PATCH -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"<new>"}' https://<SERVICE_HOST>:8443/mlflow/api/2.0/mlflow/users/update-password
```

MLflow runs its [basic-auth app](https://mlflow.org/docs/latest/auth/) (`mlflow/basic_auth.ini`; the image in `mlflow/Dockerfile` adds the `mlflow[auth]` extra). The admin is created once, when the user store is empty; changing `MLFLOW_ADMIN_PASSWORD` later has no effect, use the update-password API instead. Every user can read everything (`default_permission = READ`) and gets MANAGE on what they create; the admin can grant per-experiment and per-model permissions.

Create a user (admin only; passwords need at least 12 characters):

```python
import mlflow
from mlflow.server import get_app_client

client = get_app_client("basic-auth", tracking_uri="https://<SERVICE_HOST>:8443/mlflow")
client.create_user(username="alice", password="...")
```

Clients authenticate with env vars, which also works for the tracking URI through the SSH tunnel:

```bash
export MLFLOW_TRACKING_URI=https://localhost:8443/mlflow
export MLFLOW_TRACKING_USERNAME=alice MLFLOW_TRACKING_PASSWORD=...
export MLFLOW_TRACKING_SERVER_CERT_PATH=flywheel-ca.crt
```

## State

Each service keeps its state under `$STATE_DIR/<service>` (default `/tier1/htx_boonhan/services`):

```
caddy/        CA and certificates (data/caddy/pki), config
versitygw/    buckets/ (objects), meta/ (object metadata sidecar), iam/ (users)
mlflow/       mlflow.db (sqlite), basic_auth.db, artifacts/
gitea/        data/ (repos, sqlite db, LFS), config/app.ini
act_runner/   .runner (registration), token, cache/
mongo/        FiftyOne's database (datasets, samples, saved views)
fiftyone/     FiftyOne config and cache
prometheus/   TSDB
loki/         chunks, index, compactor
grafana/      grafana.db, plugins
```

Containers run as `SERVICE_UID:SERVICE_GID` so these files stay owned by you. Don't bind-mount a path under `$STATE_DIR` that doesn't exist yet: dockerd creates it as root, and the service then can't write there.

Notes on `/tier1` (WekaFS):
- It has no extended attributes, so versitygw stores object metadata in `meta/` (`--sidecar`). Files placed into `buckets/` by hand show up as objects, but without an ETag.
- Prometheus upstream advises against network filesystems for its TSDB. It works on Weka; move `prometheus/` to local disk if it ever corrupts.
- MLflow uses sqlite on the same filesystem. Switch `--backend-store-uri` to Postgres if concurrent writers become a problem.

## S3 clients

```bash
export AWS_ACCESS_KEY_ID=$S3_ROOT_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$S3_ROOT_SECRET_KEY
export AWS_CA_BUNDLE=flywheel-ca.crt
aws --endpoint-url https://<SERVICE_HOST>:8444 s3 mb s3://datasets
```

Use `s3://` paths with `MLFLOW_S3_ENDPOINT_URL=https://<SERVICE_HOST>:8444` to log MLflow artifacts straight into the gateway.
