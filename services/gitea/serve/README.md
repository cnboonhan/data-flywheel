# serve

Model serving stage. `sync-triton` (every 10 min, or dispatch it) loads every registered model whose MLflow alias `triton` names a version into Triton, and unloads the rest.

1. Serve or unload a model: [triton/](../../triton/README.md#serve-a-model).
