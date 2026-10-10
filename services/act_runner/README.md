# act_runner

Gitea Actions runner (`config.yaml`): labels `ubuntu-latest` and `python`; jobs run as sibling containers through the node's Docker socket (`DOCKER_GID` in `.env`), on the compose network, with `$STATE_DIR/versitygw/buckets` at `/buckets` (read-only).

1. Follow a job.
   ```bash
   services/ctl.sh logs -f act_runner
   ```
2. Apply a change to `config.yaml`.
   ```bash
   services/ctl.sh up --force-recreate act_runner
   ```

**Notes**
- `ctl.sh up` registers it: the token goes to `$STATE_DIR/act_runner/token` and `.runner`.
- Jobs don't reach Slurm; GPU jobs are submitted by hand ([slurm/](../slurm/README.md)).
