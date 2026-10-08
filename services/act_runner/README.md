# act_runner

Gitea Actions runner (`config.yaml`): labels `ubuntu-latest` and `python`, jobs as sibling containers through the node's Docker socket (`DOCKER_GID` in `.env`), joined to the compose network with `$STATE_DIR/versitygw/buckets` at `/buckets` (read-only).

Registration: `ctl.sh up` writes the token to `$STATE_DIR/act_runner/token` and `.runner`. Jobs don't reach Slurm; GPU jobs are submitted by hand ([slurm/](../slurm/README.md)).

```bash
services/ctl.sh logs -f act_runner
services/ctl.sh up --force-recreate act_runner   # after changing config.yaml
```
