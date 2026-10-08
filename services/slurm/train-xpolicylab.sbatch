#!/usr/bin/env bash
#SBATCH --job-name=train-xpolicylab
#SBATCH --partition=raus_manual
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log
#
# Train one XPolicyLab policy on an xspark dataset, metrics mirrored to MLflow.
#   sbatch train-xpolicylab.sbatch <policy> <bench> <task> <env_cfg> <action> <seed> [extra train args]
#   e.g. sbatch train-xpolicylab.sbatch ACT Galaxea Make_The_Bed_20250730_012 arx_x5 joint 0 --num_epochs 100
#
# Runs the policy's own process_data.sh, then its training command exactly as
# policy/<P>/train.sh would, but through xpolicylab/train_mlflow.py so the
# losses appear at /mlflow. Needs: a uv env at ENVS_DIR/<policy, lowercase>
# (see services/README.md), MLFLOW_TRACKING_URI/USERNAME/PASSWORD and
# MLFLOW_TRACKING_SERVER_CERT_PATH exported by the submitter, and
# PROJECT_ROOT/data -> processed/xpolicylab.
set -euo pipefail
policy=${1:?policy, e.g. ACT or DP}; bench=${2:?bench}; task=${3:?task}; env_cfg=${4:?env_cfg}; action=${5:?joint|ee}; seed=${6:-0}
shift 6 || true
extra=("$@")

PIPELINES_ROOT=${PIPELINES_ROOT:-/tier1/htx_boonhan/services/pipelines}
PROJECT_ROOT=${PROJECT_ROOT:-/tier1/htx_boonhan/workspaces/data-flywheel/eval/system1/RoboDojo}
ENVS_DIR=${ENVS_DIR:-/tier1/htx_boonhan/services/envs}
ROBODOJO_DIR=${ROBODOJO_DIR:-/tier1/htx_boonhan/services/robodojo}
XPL=$PROJECT_ROOT/XPolicyLab
BUCKETS_DIR=${BUCKETS_DIR:-/tier1/htx_boonhan/services/versitygw/buckets}
# XPolicyLab reads its xspark tree as PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/; that tree is processed/xpolicylab.
# Bulky, regenerable outputs stay out of the checkout (it may live on a slow filesystem such as the NFS home):
# the policies' decoded-frame caches, checkpoints and DP's zarr data go under $STATE_DIR/xpolicylab/<policy>/,
# evaluation results under $ROBODOJO_DIR/eval_result; the checkout gets symlinks (all gitignored upstream).
DATA_ROOT=${DATA_ROOT:-/tier1/htx_boonhan/services/xpolicylab}
link_out() {   # link_out <path in checkout> <path on /tier1>
  [[ -L $1 ]] && return 0
  mkdir -p "$2"
  if [[ -d $1 ]]; then cp -a "$1/." "$2/" && rm -rf "${1:?}"; fi
  ln -s "$2" "$1"
}
link_out "$PROJECT_ROOT/eval_result" "$ROBODOJO_DIR/eval_result"
for p in ACT DP; do
  link_out "$XPL/policy/$p/processed_data" "$DATA_ROOT/$p/processed_data"
  link_out "$XPL/policy/$p/checkpoints" "$DATA_ROOT/$p/checkpoints"
done
link_out "$XPL/policy/DP/data" "$DATA_ROOT/DP/data"
if [[ ! -e $PROJECT_ROOT/data ]]; then
  ln -s "$BUCKETS_DIR/processed/xpolicylab" "$PROJECT_ROOT/data"
  ex=$(git -C "$PROJECT_ROOT" rev-parse --git-path info/exclude 2>/dev/null) && { grep -qx /data "$ex" 2>/dev/null || echo /data >> "$ex"; }   # keep the submodule clean
fi
run="$policy-$bench-$task-$env_cfg-$action-$seed"
setting="$bench-$task-$env_cfg-$action"

source "$ENVS_DIR/${policy,,}/bin/activate"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
# Multipart uploads hand the client presigned part URLs on the compose-internal
# S3 host, which nodes can't reach; single PUTs go through the MLflow proxy.
export MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false
cd "$XPL/policy/$policy"
echo "[$(date)] $run on $(hostname), GPU $CUDA_VISIBLE_DEVICES"

bash process_data.sh "$bench" "$task" "$env_cfg" "$action"

action_dim=$(bash "$XPL/utils/get_action_dim.sh" "$PROJECT_ROOT" "$env_cfg")
case $policy in
  ACT) ckpt_dir="$XPL/policy/ACT/checkpoints/$setting-$seed" ;;
  DP)  ckpt_dir="$XPL/policy/DP/checkpoints/$setting-$seed" ;;   # robotworkspace.py saves there regardless of cwd
esac
wrap=(python "$PIPELINES_ROOT/xpolicylab/train_mlflow.py" --policy "$policy" --run "$run" --ckpt-dir "$ckpt_dir"
      --param "bench=$bench" --param "task=$task" --param "env_cfg=$env_cfg" --param "action=$action" --param "seed=$seed" --param "action_dim=$action_dim")

case $policy in
  ACT)
    export ACT_ACTION_DIM=$action_dim
    exec "${wrap[@]}" -- imitate_episodes.py --bench_name "$bench" --task_name "$task" --ckpt_setting "$setting" \
      --ckpt_dir "$ckpt_dir" --policy_class ACT --kl_weight 10 --chunk_size 50 \
      --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 --num_epochs 6000 --lr 1e-5 --save_freq 6000 --seed "$seed" "${extra[@]}"
    ;;
  DP)
    export HYDRA_FULL_ERROR=1
    exec "${wrap[@]}" -- train.py --config-name=robot_dp.yaml bench_name="$bench" task.name="$task" \
      "task.shape_meta.action.shape=[$action_dim]" \
      "+task.shape_meta.obs.agent_pos.shape=[$action_dim]" "+task.shape_meta.obs.agent_pos.type=low_dim" \
      task.dataset.zarr_path="data/$setting.zarr" training.debug=False training.seed="$seed" training.device=cuda:0 \
      exp_name="$task-robot_dp-train" logging.mode=offline setting="$env_cfg" "${extra[@]}"
    ;;
  *) echo "no recipe for policy $policy" >&2; exit 2 ;;
esac
