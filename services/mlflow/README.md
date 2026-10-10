# mlflow

Experiment tracking and model registry at `/mlflow`, built from `Dockerfile` with the [mlflow-oidc-auth](https://github.com/mlflow-oidc/mlflow-oidc-auth) plugin. Artifacts go to `s3://mlflow` through the server, so clients need no S3 keys.

1. Get a token: Profile > Tokens at `/mlflow/oidc/ui/user`, or from [`ctl.sh user add`](../README.md#users).
2. Point a client at the server.
   ```bash
   export MLFLOW_TRACKING_URI=https://$SERVICE_HOST:$CADDY_PORT/mlflow
   export MLFLOW_TRACKING_USERNAME=<name> MLFLOW_TRACKING_PASSWORD=mlf_...
   export MLFLOW_TRACKING_SERVER_CERT_PATH=flywheel-ca.crt MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false
   curl -u <name>:mlf_... https://$SERVICE_HOST:$CADDY_PORT/mlflow/api/2.0/mlflow/registered-models/search
   ```
3. Apply an edit to `Dockerfile`.
   ```bash
   services/ctl.sh up --build
   ```

**Notes**
- Login: Keycloak for the UI (client `mlflow`), personal access tokens for API clients. Group `admins` = admin; everyone else gets `EDIT` where no permission is set. Per-experiment and per-model permissions live in `$STATE_DIR/mlflow/oidc_auth.db`; admin UI at `/mlflow/oidc/ui/`.
- `ctl.sh up` re-mints the admin's token into `$STATE_DIR/slurm.env` for Slurm jobs.
- Inside the compose network MLflow serves at the root; the `/mlflow` prefix comes from Caddy's `X-Forwarded-Prefix`.
- Backend store: sqlite in `$STATE_DIR/mlflow`; switch to Postgres if writers contend.
- Naming of training runs and models: [slurm/](../slurm/README.md#naming-in-mlflow). Serving registered versions: [triton/](../triton/README.md).
