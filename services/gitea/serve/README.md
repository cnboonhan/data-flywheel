# serve

Model serving stage. Each workflow serves what its spec repo in Gitea says, at the MLflow registered versions the spec names, and stops what was removed. A push to the spec repo dispatches it through the repo's webhook; an admin can also dispatch it by hand ([gitea/](../README.md#run-a-workflow)).

| Workflow | Spec repo | Serves on |
|---|---|---|
| [`sync-triton`](../../triton/gitea_ci/sync-triton.yml) (kept with Triton, in `triton/gitea_ci/`) | `triton` (a Triton model repository) | Triton: image models, robot policies ([triton/](../../triton/README.md#serve-a-model)) |
| [`sync-dynamo`](../../dynamo/gitea_ci/sync-dynamo.yml) (kept with Dynamo, in `dynamo/gitea_ci/`) | `dynamo` (`compose.yaml`) | Dynamo: language models ([dynamo/](../../dynamo/README.md#serve-swap-and-stop-models)) |
