# act_runner

Gitea Actions runner (`config.yaml`): labels `ubuntu-latest` and `python`; jobs run as sibling containers through the node's Docker socket (`DOCKER_GID` in `.env`), on the compose network, with `$STATE_DIR/versitygw/buckets` at `/buckets` (read-only); `sync-triton` also mounts the checkout, read-only. A job that must act on the host doesn't mount it: it logs in over ssh with a key whose only command is a script in [`tools/bin/`](../../tools/bin/README.md).

1. To follow the jobs the runner picks up, tail its log.
   ```bash
   services/ctl.sh logs -f act_runner
   ```
2. After editing `config.yaml` (labels, mounts, capacity), recreate the runner: remove its container and let `up` create it again (`up --force-recreate act_runner` would recreate every service).
   ```bash
   services/ctl.sh rm -sf act_runner && services/ctl.sh up
   ```

**Notes**
- `ctl.sh up` registers the runner and keeps its token in `$STATE_DIR/act_runner/token` and `.runner`; delete both to register it again.
- Don't expect jobs to reach Slurm or a GPU: submit GPU work by hand ([slurm/](../slurm/README.md)).
