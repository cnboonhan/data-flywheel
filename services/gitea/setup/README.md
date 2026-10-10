# setup

Builds the environments the Slurm jobs run in. Not a pipeline stage: `ctl.sh up` dispatches it when an env is missing or was built for another checkout.

| File | Does |
|---|---|
| `setup-envs.yml` | runs the installers in a job container on the service node, with `$STATE_DIR`, the checkout and `~/.local` mounted at their host paths and commands under your uid (`setpriv`) |
| `install-robodojo.sh` | RoboDojo eval env `$ENVS_DIR/robodojo` (Isaac Sim 5.1, Isaac Lab, curobo, XPolicyLab, ffmpeg, the `conda-shim/` here), sim assets into `$ROBODOJO_DIR/Assets`, RoboDojo data (sim 1.85 TB, real 273 GB) into `raw/open_datasets/robodojo{,_real}/` |
| `install-policy-env.sh <ACT\|DP>` | policy env `$ENVS_DIR/<policy>` |

1. Rebuild everything (idempotent; the data download resumes).
   ```bash
   services/ctl.sh setup
   ```
2. Or build one env by hand, on any node.
   ```bash
   set -a; . $STATE_DIR/slurm.env; set +a; bash services/gitea/setup/install-policy-env.sh ACT
   ```

**Notes**
- Each installer writes the checkout path to `<env>/.flywheel-setup`; that is what `ctl.sh up` compares.
