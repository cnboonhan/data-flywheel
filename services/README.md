# services

The **Store** column of the [architecture diagram](../README.md#architecture): Docker Compose behind Caddy on one port, run on `$SERVICE_NODE`. `$SERVICE_HOST`, `$CADDY_PORT`, `$SERVICE_NODE`, `$STATE_DIR` below are the values in `.env`. How data moves through it: [the flywheel](../README.md#the-flywheel-end-to-end).

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
| Triton: model serving from the MLflow registry | [triton/](triton/README.md) | `https://triton.$SERVICE_HOST:$CADDY_PORT/` (bearer token) |
| Slurm jobs: train, evaluate | [slurm/](slurm/README.md) | `sbatch` by hand |

## Run

1. Fill in the secrets.
   ```bash
   cp services/.env.example services/.env
   ```
2. Start or update the stack (provisions everything on first run; from any node, it ssh-es to `$SERVICE_NODE`).
   ```bash
   services/ctl.sh up
   ```
3. Operate it.
   ```bash
   services/ctl.sh ps
   services/ctl.sh logs -f caddy
   services/ctl.sh restart <service>          # after editing a bind-mounted config (Caddyfile, loki.yml)
   services/ctl.sh up --build                 # after editing mlflow/, rerun/ or triton/Dockerfile
   services/ctl.sh up --force-recreate        # after git deleted and recreated files under services/
   services/ctl.sh setup                      # rebuild the Slurm-side envs (Gitea setup-envs); up does it when needed
   services/ctl.sh down
   ```

**Notes**
- Run `up` a second time if it says MLflow was still starting when the job token was minted.

### Fresh install

1. Clone and set the push URL: [Setup](../README.md#setup). For IsaacLab-Arena's nested submodules (SSH URLs the cluster can't reach):
   ```bash
   git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive eval/system2/IsaacLab-Arena
   ```
2. Copy `services/.env` (or fill in `.env.example`) and run `services/ctl.sh up` (about 5 min). On a clean state it starts Caddy first and waits for its CA, then the rest, then provisions Gitea ([gitea/](gitea/README.md)), Keycloak, Grafana, the MLflow job token, `$STATE_DIR/slurm.env` and the `$STATE_DIR/pipelines` checkout.
3. Trust the CA: [Access](#access).
4. Wait for `setup-envs`, which `up` dispatches: ~1 h for the envs on a cold cache, up to a day for the RoboDojo data ([gitea/setup/](gitea/setup/README.md)).

## Access

1. Reach the stack. From a machine that can reach the node, open `https://$SERVICE_HOST:$CADDY_PORT/`. Otherwise tunnel through a login node and map the names on your laptop (no wildcard there, so list all five):
   ```bash
   ssh -L $CADDY_PORT:$SERVICE_NODE:$CADDY_PORT <login-node>
   echo "127.0.0.1 $SERVICE_HOST fiftyone.$SERVICE_HOST rerun.$SERVICE_HOST s3.$SERVICE_HOST triton.$SERVICE_HOST" | sudo tee -a /etc/hosts
   ```
2. Trust the CA once per device (also removes the browser warning).
   ```bash
   curl -k https://$SERVICE_HOST:$CADDY_PORT/ca.crt -o flywheel-ca.crt
   sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain flywheel-ca.crt   # macOS
   sudo cp flywheel-ca.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates           # Ubuntu
   export AWS_CA_BUNDLE=flywheel-ca.crt REQUESTS_CA_BUNDLE=flywheel-ca.crt SSL_CERT_FILE=flywheel-ca.crt  # per tool
   ```

## Users

1. Create an account: Keycloak user, MLflow access token and S3 key pair, printed once. One Keycloak login covers every web UI ([keycloak/](keycloak/README.md)); S3 access per bucket: [versitygw/](versitygw/README.md).
   ```bash
   services/ctl.sh user add <name> <email> [password]
   ```

## State

Everything lives in `$STATE_DIR/<service>`, owned by `$SERVICE_UID:$SERVICE_GID`, including buckets, envs, caches, checkpoints and evaluation results (`versitygw/buckets`, `envs`, `uv-cache`, `xpolicylab`, `robodojo`); the checkout holds code only. `ctl.sh up` creates every directory it mounts, because a path dockerd has to create ends up root-owned.
