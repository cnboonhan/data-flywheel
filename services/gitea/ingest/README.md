# ingest

Upload and Sync: brings data and models into the store.

| Workflow | Does |
|---|---|
| `download-datasets-hf` | a public Hugging Face dataset (or subset) → `raw/open_datasets/<dest>/`, file by file, resumable; known datasets in the header |
| `download-models-hf` | a Hugging Face model → MLflow registry (artifacts in the `mlflow` bucket, experiment `hf-models`, version tagged with the hub commit); known models in the header |

Gated repos need `HF_TOKEN` in `services/.env` (synced to the Gitea secret by `ctl.sh up`).
