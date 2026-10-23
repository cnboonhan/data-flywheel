# serve

Model serving stage. Each workflow serves the registered models whose MLflow alias names a version, and stops the rest; dispatch it after setting or removing an alias.

| Workflow | Alias | Serves on |
|---|---|---|
| [`sync-triton`](../../triton/gitea_ci/sync-triton.yml) (kept with Triton, in `triton/gitea_ci/`) | `triton` | Triton: image models, robot policies ([triton/](../../triton/README.md#serve-a-model)) |
| [`sync-dynamo`](../../dynamo/gitea_ci/sync-dynamo.yml) (kept with Dynamo, in `dynamo/gitea_ci/`) | `dynamo` | Dynamo: language models ([dynamo/](../../dynamo/README.md#serve-swap-and-stop-models)) |
