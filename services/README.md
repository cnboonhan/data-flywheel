# services

The **Store** column of the [architecture diagram](../architecture.html), as a Docker Compose stack behind a Caddy reverse proxy.

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

Everything is HTTPS. Caddy runs a local CA (`local_certs`) and issues a certificate for `$SERVICE_HOST` and `localhost`.

**From a machine that can reach the node directly:** open `https://<SERVICE_HOST>:8443/`.

**From your laptop through the login node:** forward both ports, then use `localhost`.

```bash
ssh -L 8443:C2-GB300-02-C03:8443 -L 8444:C2-GB300-02-C03:8444 -L 8445:C2-GB300-02-C03:8445 <login-node>
# then open https://localhost:8443/
```

To make that permanent, add an entry to `~/.ssh/config` on your laptop. The `LocalForward` lines carry the same port mappings; everything else is whatever you already use to reach the login node.

```sshconfig
Host flywheel
    HostName <login-node>
    User <your-user>
    LocalForward 8443 C2-GB300-02-C03:8443
    LocalForward 8444 C2-GB300-02-C03:8444
    LocalForward 8445 C2-GB300-02-C03:8445
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

## FiftyOne

[FiftyOne](https://github.com/voxel51/fiftyone) browses the datasets: videos, episodes, metadata, filters. The App is at `https://<SERVICE_HOST>:8445/` (or `https://localhost:8445/` through the tunnel) with the admin login. It shows what pipelines have *ingested* into its MongoDB (`mongo` service), and streams media from the bucket directories, which are mounted read-only at `/buckets` in the FiftyOne container and in every Actions job container (`act_runner/config.yaml`), so ingested filepaths resolve in both.

`gitea/examples/ingest-hifi-umi.yml` is the first ingestion pipeline. HiFi-UMI-2K is LeRobot v3: each part has one mp4 per camera holding ~1 100 episodes as time ranges. A FiftyOne sample is one such mp4 (`chunk`, `part`, `camera` fields) with an `episodes` field of temporal detections (task text, frame range, episode index), and the saved view **episodes** turns those into one clip per episode per camera. Run it from the repo's Actions tab with a chunk glob; it skips videos already ingested.

## Logins

One admin login, `ADMIN_USER` / `ADMIN_PASSWORD` in `.env`, is applied everywhere by `docker-compose.yml`:

| Service | How the admin login is used |
|---|---|
| Gitea | admin user; registration is off, the admin creates accounts |
| MLflow | admin user of the basic-auth app; further users created by the admin |
| Grafana | admin user |
| S3 gateway (API and `/s3/` UI) | root account: access key = user, secret key = password |
| Prometheus, Loki, FiftyOne | HTTP basic auth at the proxy (`caddy/Caddyfile`); `ctl.sh` turns the password into the bcrypt hash Caddy wants |

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
