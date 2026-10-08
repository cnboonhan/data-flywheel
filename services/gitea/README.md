# gitea

Git hosting and Gitea Actions. `ctl.sh up` creates the admin user, the private repo `admin/pipelines` (workflows from the folders below, flattened into `.gitea/workflows/`; setup and adapter scripts into `setup/`, `adapter/`; scripts from `../fiftyone`, `../xpolicylab`, `../slurm`), its secrets (`S3_ACCESS_KEY`, `S3_SECRET_KEY`, `SLURM_SSH_KEY`, `SLURM_SSH_HOST`, `MLFLOW_TOKEN`, optional `HF_TOKEN`) and variables (`RERUN_BASE`, `SERVICE_URL`, `MLFLOW_USERNAME`, `STATE_DIR`, `FLYWHEEL_ROOT`, `USER_LOCAL`, `RUN_UID`, `RUN_GID`), and registers the runner ([act_runner/](../act_runner/README.md)). Seeding only adds missing files; edit the repo afterwards.

```bash
git -c http.sslCAInfo=flywheel-ca.crt clone https://$SERVICE_HOST:$CADDY_PORT/gitea/admin/pipelines.git   # SSH is disabled
# run a workflow (UI: Actions > workflow > Run workflow)
curl -u $ADMIN_USER:$ADMIN_PASSWORD -H 'Content-Type: application/json' \
  -d '{"ref":"main","inputs":{"repo":"simple-world-lab/HiFi-UMI-2K","include":"*","dest":"HiFi-UMI-2K","limit":"0"}}' \
  https://$SERVICE_HOST:$CADDY_PORT/gitea/api/v1/repos/admin/pipelines/actions/workflows/download-datasets-hf.yml/dispatches
```

Workflows live in one folder per stage of the architecture, each with its own README:

- [setup/](setup/README.md): environments the pipelines need (dispatched by `ctl.sh up`)
- [ingest/](ingest/README.md): Upload and Sync
- [adapter/](adapter/README.md): conversion into the training format (`processed/xpolicylab/`)
- [clean/](clean/README.md): Data Cleaning
- [validate/](validate/README.md): Data Validation
- [mix/](mix/README.md): Data Mixing

Workflow file names must be unique across stages: Gitea only reads a flat `.gitea/workflows/`.

Jobs run as containers on the compose network with `/buckets` mounted read-only and `S3_ENDPOINT_URL=http://versitygw:7070`. Triggers: `workflow_dispatch`, `push`, `schedule`; nothing fires on new objects yet (the gateway can post bucket events to a webhook).

Pushing to `admin/pipelines` cancels in-flight runs of every workflow; land changes between runs.
