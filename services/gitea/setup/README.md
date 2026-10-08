# setup

Builds the environments the pipelines run in. Not a pipeline stage: `ctl.sh up` dispatches it when an env is missing or was built for another checkout, `ctl.sh setup` forces it.

| File | Does |
|---|---|
| `setup-envs.yml` | runs both installers in a job container on the service node, `$STATE_DIR`, the checkout and `~/.local` mounted at their host paths, commands under your uid (`setpriv`) |
| `install-robodojo.sh` | RoboDojo eval env `$ENVS_DIR/robodojo` (Isaac Sim 5.1, Isaac Lab, curobo, XPolicyLab, ffmpeg, conda shim), sim assets into `$ROBODOJO_DIR/Assets`, RoboDojo data (sim 1.85 TB, real 273 GB) into `raw/open_datasets/robodojo{,_real}/` |
| `install-policy-env.sh <ACT\|DP\|demo_policy>` | policy env `$ENVS_DIR/<policy>` |

```bash
services/ctl.sh setup                                                  # rebuild all (idempotent; data download resumes)
set -a; . $STATE_DIR/slurm.env; set +a; bash install-policy-env.sh ACT   # by hand, on any node
```

Each installer writes the checkout path to `<env>/.flywheel-setup`; that is what `ctl.sh up` compares.
