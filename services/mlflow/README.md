# mlflow

Experiment tracking and model registry at `/mlflow`, built from `Dockerfile` with the [mlflow-oidc-auth](https://github.com/mlflow-oidc/mlflow-oidc-auth) plugin: Keycloak login for the UI (client `mlflow`), personal access tokens for API clients, per-experiment/model permissions in `$STATE_DIR/mlflow/oidc_auth.db`, admin UI at `/mlflow/oidc/ui/`. Group `admins` = admin; everyone else gets `EDIT` where nothing is set. Artifacts go to `s3://mlflow` on the gateway through the server (clients need no S3 keys).

```bash
# token: Profile > Tokens at /mlflow/oidc/ui/user, or `ctl.sh user add`; ctl.sh up re-mints the admin's into $STATE_DIR/slurm.env
export MLFLOW_TRACKING_URI=https://$SERVICE_HOST:$CADDY_PORT/mlflow
export MLFLOW_TRACKING_USERNAME=<name> MLFLOW_TRACKING_PASSWORD=mlf_...
export MLFLOW_TRACKING_SERVER_CERT_PATH=flywheel-ca.crt MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false
curl -u <name>:mlf_... https://$SERVICE_HOST:$CADDY_PORT/mlflow/api/2.0/mlflow/registered-models/search
services/ctl.sh up --build   # after editing Dockerfile
```

Inside the compose network MLflow serves at the root; the `/mlflow` prefix comes from Caddy's `X-Forwarded-Prefix`. Backend store: sqlite in `$STATE_DIR/mlflow` (switch to Postgres if writers contend).
