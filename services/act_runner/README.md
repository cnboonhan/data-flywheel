# act_runner

Gitea Actions runner (`config.yaml`): labels `ubuntu-latest` and `python`; jobs run as sibling containers through the node's Docker socket (`DOCKER_GID` in `.env`), on the compose network, with `$STATE_DIR/versitygw/buckets` at `/buckets` (read-only).

1. To follow the jobs the runner picks up, tail its log.
   ```bash
   services/ctl.sh logs -f act_runner
   ```
2. After editing `config.yaml` (labels, mounts, capacity), recreate the runner.
   ```bash
   services/ctl.sh up --force-recreate act_runner
   ```

**Notes**
- `ctl.sh up` registers the runner and keeps its token in `$STATE_DIR/act_runner/token` and `.runner`; delete both to register it again.
- Don't expect jobs to reach Slurm or a GPU: submit GPU work by hand ([slurm/](../slurm/README.md)).
