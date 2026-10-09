# services

The **Store** column of the [architecture diagram](../architecture.html): Docker Compose behind Caddy, one port. How data moves through it: [docs/flywheel.md](../docs/flywheel.md). Settings come from `.env` (copy `.env.example`); `$SERVICE_HOST`, `$CADDY_PORT`, `$SERVICE_NODE`, `$STATE_DIR` below are its values.

| Service | Folder | URL |
|---|---|---|
| Caddy: TLS, routing, local CA | [caddy/](caddy/README.md) | `https://$SERVICE_HOST:$CADDY_PORT/` |
| Keycloak + oauth2-proxy: single sign-on | [keycloak/](keycloak/README.md) | `/auth`, `/oauth2` |
| Gitea + Actions runner: pipelines | [gitea/](gitea/README.md), [act_runner/](act_runner/README.md) | `/gitea` |
| MLflow: tracking, model registry | [mlflow/](mlflow/README.md) | `/mlflow` |
| Grafana, Loki | [grafana/](grafana/README.md), [loki/](loki/README.md) | `/grafana`, `/loki` |
| Versity S3 gateway: buckets | [versitygw/](versitygw/README.md) | `https://s3.$SERVICE_HOST:$CADDY_PORT/` |
| FiftyOne (+ Mongo): dataset browser | [fiftyone/](fiftyone/README.md) | `https://fiftyone.$SERVICE_HOST:$CADDY_PORT/` |
| Rerun: episode viewer | [rerun/](rerun/README.md) | `https://rerun.$SERVICE_HOST:$CADDY_PORT/` |
| Triton: model serving from the MLflow registry (all GPUs of the node) | [triton/](triton/README.md), Gitea `sync-triton` | `https://triton.$SERVICE_HOST:$CADDY_PORT/` (bearer token) |
| Slurm jobs: train, evaluate (per model and embodiment) | [slurm/](slurm/README.md) | `sbatch` by hand |

## Run

```bash
cp services/.env.example services/.env     # fill in the secrets
services/ctl.sh up                         # start or update, provisions everything on first run
services/ctl.sh ps
services/ctl.sh logs -f caddy
services/ctl.sh restart <service>          # after editing a bind-mounted config (Caddyfile, loki.yml)
services/ctl.sh up --build                 # after editing mlflow/, rerun/ or triton/Dockerfile
services/ctl.sh up --force-recreate        # after git deleted and recreated files under services/
services/ctl.sh setup                      # rebuild the policy and RoboDojo envs (Gitea workflow setup-envs); up does it when needed
services/ctl.sh down
```

`ctl.sh` runs `docker compose` on `$SERVICE_NODE` (over ssh from anywhere else). Run it a second time if MLflow was still starting when the job token was minted (it says so). Fresh install step by step: [docs/flywheel.md](../docs/flywheel.md#0-before-you-start).

## Access

From a machine that can reach the node: `https://$SERVICE_HOST:$CADDY_PORT/`. Through a login node:

```bash
ssh -L $CADDY_PORT:$SERVICE_NODE:$CADDY_PORT <login-node>
# /etc/hosts on the laptop (the subdomains have no wildcard there, list all five):
# 127.0.0.1 $SERVICE_HOST fiftyone.$SERVICE_HOST rerun.$SERVICE_HOST s3.$SERVICE_HOST triton.$SERVICE_HOST
```

Trust the CA once per device (also removes the browser warning):

```bash
curl -k https://$SERVICE_HOST:$CADDY_PORT/ca.crt -o flywheel-ca.crt
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain flywheel-ca.crt   # macOS
sudo cp flywheel-ca.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates           # Ubuntu
export AWS_CA_BUNDLE=flywheel-ca.crt REQUESTS_CA_BUNDLE=flywheel-ca.crt SSL_CERT_FILE=flywheel-ca.crt  # per tool
```

## Users

```bash
services/ctl.sh user add <name> <email> [password]   # Keycloak user, MLflow access token, S3 key pair (printed once)
```

One Keycloak login covers every web UI; API clients use the MLflow token and the S3 keys. Details: [keycloak/](keycloak/README.md).

## State

`$STATE_DIR/<service>`, owned by `$SERVICE_UID:$SERVICE_GID`. Buckets, policy envs, caches, checkpoints and evaluation results also live there (`versitygw/buckets`, `envs`, `uv-cache`, `xpolicylab`, `robodojo`), so the checkout holds code only. `ctl.sh up` creates every directory it mounts; a path dockerd has to create ends up root-owned.
