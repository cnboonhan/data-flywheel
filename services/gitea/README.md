# gitea

Gitea hosts the pipelines and runs them with Gitea Actions. Every workflow lives in the private repo `admin/pipelines`, which `ctl.sh up` seeds from the stage folders here.

| Stage folder | Workflows |
|---|---|
| [setup/](setup/README.md) | environments the Slurm jobs need (dispatched by `ctl.sh up`) |
| [ingest/](ingest/README.md) | Upload and Sync: Hugging Face downloads, FiftyOne mirror of `raw` |
| [adapter/](adapter/README.md) | Data Conversion: raw datasets into the training format (`processed/xpolicylab/`) |
| [serve/](serve/README.md) | MLflow registry → Triton |
| [clean/](clean/README.md), [mix/](mix/README.md) | Data Cleaning, Mixing (none yet) |

## Run a workflow

1. Open `https://$SERVICE_HOST:$CADDY_PORT/gitea/admin/pipelines`, go to **Actions**, pick the workflow and click **Run workflow**. Fill in its inputs; each workflow's header explains them.
2. Or start it from a shell. Put the workflow's inputs in `inputs`; this example downloads HiFi-UMI-2K.
   ```bash
   curl -u $ADMIN_USER:$ADMIN_PASSWORD -H 'Content-Type: application/json' \
     -d '{"ref":"main","inputs":{"repo":"simple-world-lab/HiFi-UMI-2K","include":"*","dest":"HiFi-UMI-2K","limit":"0"}}' \
     https://$SERVICE_HOST:$CADDY_PORT/gitea/api/v1/repos/admin/pipelines/actions/workflows/download-datasets-hf.yml/dispatches
   ```
3. Follow the run under **Actions**; click it to see each step's log.

## Change or add a workflow

1. Edit the workflow in its stage folder here (or add a new one there). Give it a file name no other stage uses: Gitea reads all workflows from one flat `.gitea/workflows/` folder.
2. Copy it into the pipelines checkout and push. Do this between runs, because a push cancels every workflow that is running.
   ```bash
   cp services/gitea/<stage>/<workflow>.yml $STATE_DIR/pipelines/.gitea/workflows/
   git -C $STATE_DIR/pipelines add .gitea/workflows && git -C $STATE_DIR/pipelines commit -m "<what changed>" && git -C $STATE_DIR/pipelines push
   ```
   `ctl.sh up` only adds files that are missing from the repo, so the repo's copy is the one that runs until you push yours.
3. Commit the same change here as well, so the checkout and the repo stay alike.

## Work on the repo from elsewhere

Clone over HTTPS (SSH is disabled), trusting the stack's CA ([Access](../README.md#access)):

```bash
git -c http.sslCAInfo=flywheel-ca.crt clone https://$SERVICE_HOST:$CADDY_PORT/gitea/admin/pipelines.git
```

## What a workflow can rely on

- **Where it runs:** each job is a container on the services' network, with the buckets mounted read-only at `/buckets` and the S3 gateway at `S3_ENDPOINT_URL=http://versitygw:7070`. The runner is described in [act_runner/](../act_runner/README.md).
- **Secrets:** `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `MLFLOW_TOKEN`, and `HF_TOKEN` and `RERUN_RAW_TOKEN` when they are set in `services/.env`. `ctl.sh up` keeps them in sync.
- **Variables:** `RERUN_BASE`, `SERVICE_URL`, `MLFLOW_USERNAME`, `STATE_DIR`, `FLYWHEEL_ROOT`, `USER_LOCAL`, `RUN_UID`, `RUN_GID`.
- **Helper scripts:** setup and adapter scripts are in the repo under `setup/` and `adapter/`.
- **Triggers:** use `workflow_dispatch`, `push` or `schedule`. Nothing can fire on new objects in a bucket yet (the gateway could post bucket events to a webhook).
