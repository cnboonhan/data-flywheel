# serve

Model serving stage. `sync-triton.yml` keeps Triton in step with the MLflow registry: every registered model whose alias
`triton` names a version is packaged and loaded, models without it are unloaded. Every 10 minutes and on demand. See
[../../triton/README.md](../../triton/README.md).
