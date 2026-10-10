# gitea

Git hosting and Gitea Actions. The pipelines live in the private repo `admin/pipelines`, seeded by `ctl.sh up` from the stage folders here.

| Stage folder | Workflows |
|---|---|
| [setup/](setup/README.md) | environments the Slurm jobs need (dispatched by `ctl.sh up`) |
| [ingest/](ingest/README.md) | Upload and Sync: Hugging Face downloads, FiftyOne mirror of `raw` |
| [adapter/](adapter/README.md) | conversion into the training format (`processed/xpolicylab/`) |
| [serve/](serve/README.md) | MLflow registry → Triton |
| [clean/](clean/README.md), [validate/](validate/README.md), [mix/](mix/README.md) | Data Cleaning, Validation, Mixing (none yet) |

1. Run a workflow: UI Actions > workflow > Run workflow, or the API.
   ```bash
   curl -u $ADMIN_USER:$ADMIN_PASSWORD -H 'Content-Type: application/json' \
     -d '{"ref":"main","inputs":{"repo":"simple-world-lab/HiFi-UMI-2K","include":"*","dest":"HiFi-UMI-2K","limit":"0"}}' \
     https://$SERVICE_HOST:$CADDY_PORT/gitea/api/v1/repos/admin/pipelines/actions/workflows/download-datasets-hf.yml/dispatches
   ```
2. Change a workflow: edit it here, copy it into the pipelines checkout and push (seeding only adds missing files, so the repo's copy is what runs).
   ```bash
   cp services/gitea/<stage>/<workflow>.yml $STATE_DIR/pipelines/.gitea/workflows/
   git -C $STATE_DIR/pipelines commit -am "<what changed>" && git -C $STATE_DIR/pipelines push
   ```
3. Clone the repo elsewhere (HTTPS only; SSH is disabled).
   ```bash
   git -c http.sslCAInfo=flywheel-ca.crt clone https://$SERVICE_HOST:$CADDY_PORT/gitea/admin/pipelines.git
   ```

**Notes**
- `ctl.sh up` creates the admin user, the repo (workflows flattened into `.gitea/workflows/`, setup and adapter scripts into `setup/`, `adapter/`), its secrets (`S3_ACCESS_KEY`, `S3_SECRET_KEY`, `MLFLOW_TOKEN`, optional `HF_TOKEN`, `RERUN_RAW_TOKEN`) and variables (`RERUN_BASE`, `SERVICE_URL`, `MLFLOW_USERNAME`, `STATE_DIR`, `FLYWHEEL_ROOT`, `USER_LOCAL`, `RUN_UID`, `RUN_GID`), and registers the runner ([act_runner/](../act_runner/README.md)).
- Workflow file names must be unique across stages: Gitea reads a flat `.gitea/workflows/`.
- Jobs run as containers on the compose network with `/buckets` mounted read-only and `S3_ENDPOINT_URL=http://versitygw:7070`.
- Triggers: `workflow_dispatch`, `push`, `schedule`. Nothing fires on new objects yet (the gateway can post bucket events to a webhook).
- Pushing to `admin/pipelines` cancels in-flight runs of every workflow; push between runs.
