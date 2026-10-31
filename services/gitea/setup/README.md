# setup

Builds the environments the Slurm jobs run in. Not a pipeline stage: `ctl.sh up` dispatches it when an env is missing or was built for another checkout.

| File | Does |
|---|---|
| `setup-envs.yml` | logs in to the service node over ssh; the key may only run [`tools/bin/setup-envs`](../../../tools/bin/README.md), which runs the three installers there side by side as the service account (input `dry_run`: check them, build nothing) |
| `install-robodojo.sh` | RoboDojo eval env `$ENVS_DIR/robodojo` (Isaac Sim 5.1, Isaac Lab, curobo, XPolicyLab, ffmpeg, the `conda-shim/` here), sim assets into `$ROBODOJO_DIR/Assets`, RoboDojo data (sim 1.85 TB, real 273 GB) into `raw/open_datasets/robodojo{,_real}/` |
| `install-policy-env.sh <ACT\|DP>` | policy env `$ENVS_DIR/<policy>` |

1. To rebuild every env, run setup (safe to rerun; the data download resumes where it stopped). Expect up to a day on a cold start, most of it the RoboDojo data.
   ```bash
   services/ctl.sh up --setup
   ```
2. To build one env by hand, run its installer on any node.
   ```bash
   set -a; . $STATE_DIR/flywheel.env; set +a; bash services/gitea/setup/install-policy-env.sh ACT
   ```

**Notes**
- `ctl.sh up` rebuilds an env when its `<env>/.flywheel-setup` doesn't name the current checkout and the env's own path, so after moving or recloning the checkout, or moving `STATE_DIR`, the next `up` rebuilds.
