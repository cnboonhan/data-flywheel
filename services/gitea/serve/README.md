# serve

Model serving stage. `sync-triton` (every 10 min, or dispatch it) loads every registered model whose MLflow alias `triton` names a version into Triton, and unloads the rest.

1. To serve or unload a model, set or remove its MLflow alias as described in [triton/](../../triton/README.md#serve-a-model).
