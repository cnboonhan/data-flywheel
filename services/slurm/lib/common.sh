# Sourced by train.sbatch and evaluate.sbatch. Paths come from slurm.env (`ctl.sh up` writes it); the defaults match it.
# sbatch runs a spooled copy of the job script, so the checkout comes from FLYWHEEL_ROOT, not the script's location.
SLURM_DIR=${FLYWHEEL_ROOT:-$HOME/workspaces/data-flywheel}/services/slurm
LIB=$SLURM_DIR/lib
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/workspaces/data-flywheel/eval/system1/RoboDojo}
ENVS_DIR=${ENVS_DIR:-/tier1/htx_boonhan/services/envs}
ROBODOJO_DIR=${ROBODOJO_DIR:-/tier1/htx_boonhan/services/robodojo}
BUCKETS_DIR=${BUCKETS_DIR:-/tier1/htx_boonhan/services/versitygw/buckets}
DATA_ROOT=${DATA_ROOT:-/tier1/htx_boonhan/services/xpolicylab}
XPL=$PROJECT_ROOT/XPolicyLab
export BUCKETS_DIR FLYWHEEL_ROOT
export PYTHONPATH="$LIB${PYTHONPATH:+:$PYTHONPATH}"   # embodiment.py, flywheel_mlflow.py
# Multipart uploads hand out presigned URLs on the compose-internal S3 host, which nodes can't reach.
export MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false

# load_model <model> <train|eval>: source models/<model>/<stage>.sh, which sets ENV (its env under ENVS_DIR, with
# mlflow and pyyaml) and defines the stage function (train or evaluate). Activating ENV is left to the caller.
load_model() {
  local dir=$SLURM_DIR/models/${1,,}
  [[ -f $dir/$2.sh ]] || { echo "no $2 recipe for model $1 (models/${1,,}/$2.sh); known: $(cd "$SLURM_DIR/models" && ls -d */ | tr -d / | tr '\n' ' ')" >&2; exit 2; }
  MODEL_DIR=$dir
  # shellcheck source=/dev/null
  source "$dir/$2.sh"
  [[ -n ${ENV:-} && -f $ENVS_DIR/$ENV/bin/activate ]] || { echo "model $1: env '${ENV:-}' missing under $ENVS_DIR (Gitea setup-envs)" >&2; exit 2; }
}

# load_embodiment <name>: EMB_NAME, EMB_DATA_ENV_CFG, EMB_XPL_ENV_CFG, EMB_EVAL_ENV_CFG from embodiments/<name>.yaml.
load_embodiment() {
  local vars; vars=$("$ENVS_DIR/$ENV/bin/python" "$LIB/embodiment.py" "$1") || exit 2
  eval "$vars"
}
