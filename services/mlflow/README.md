# mlflow

Experiment tracking and model registry at `/mlflow`, built from `Dockerfile` with the [mlflow-oidc-auth](https://github.com/mlflow-oidc/mlflow-oidc-auth) plugin. Artifacts go to `s3://mlflow` through the server, so clients need no S3 keys.

1. Get a personal access token: open `/mlflow/oidc/ui/user` and go to Profile > Tokens, or use the one [`ctl.sh user add`](../README.md#users) printed.
2. Point your client at the server with that token, and check it works.
   ```bash
   export MLFLOW_TRACKING_URI=https://$SERVICE_HOST:$CADDY_PORT/mlflow
   export MLFLOW_TRACKING_USERNAME=<name> MLFLOW_TRACKING_PASSWORD=mlf_...
   export MLFLOW_TRACKING_SERVER_CERT_PATH=flywheel-ca.crt MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false
   curl -u <name>:mlf_... https://$SERVICE_HOST:$CADDY_PORT/mlflow/api/2.0/mlflow/registered-models/search
   ```
3. After editing `Dockerfile`, rebuild.
   ```bash
   services/ctl.sh up --build
   ```

**Notes**
- Log in to the UI with Keycloak; use a personal access token for API clients. Members of `admins` are admins; everyone else gets `EDIT` wherever no permission is set. Set per-experiment and per-model permissions in the admin UI at `/mlflow/oidc/ui/` (stored in `$STATE_DIR/mlflow/oidc_auth.db`).
- Slurm jobs use the admin token that `ctl.sh up` writes to `$STATE_DIR/slurm.env`; rerun `up` if it stops working.
- From another container on the compose network, use `http://mlflow:5000` (no `/mlflow` prefix; Caddy adds it with `X-Forwarded-Prefix`).
- The backend store is sqlite in `$STATE_DIR/mlflow`; switch to Postgres if concurrent writers start failing.
- For how training runs and models are named, see [slurm/](../slurm/README.md#naming-in-mlflow); to serve a registered version, see [triton/](../triton/README.md).
