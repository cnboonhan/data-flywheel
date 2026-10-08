#!/usr/bin/env bash
#SBATCH --job-name=evaluate-xpolicylab
#SBATCH --partition=raus_manual
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
#
# Evaluate a policy checkpoint in the RoboDojo simulator (Isaac Sim, headless)
# and record the result in MLflow.
#   sbatch evaluate-xpolicylab.sbatch <policy> <task> <ckpt> [env_cfg] [action] [seed] [eval_num]
#   e.g. sbatch evaluate-xpolicylab.sbatch ACT stack_bowls stack_bowls arx_x5 joint 0 5
#        sbatch evaluate-xpolicylab.sbatch demo_policy stack_bowls demo arx_x5 joint 0 1   # zero-action smoke test
# Runs policy/<P>/eval.sh exactly as XPolicyLab documents it: a policy server
# in the policy's env, RoboDojo's eval client in the `robodojo` env, both
# activated through the conda shim (the scripts hard-code conda). Then
# eval_mlflow.py logs success_rate/score to the checkpoint's MLflow run.
set -euo pipefail
policy=${1:?policy}; task=${2:?task}; ckpt=${3:?ckpt}; env_cfg=${4:-arx_x5}; action=${5:-joint}; seed=${6:-0}; eval_num=${7:-}
PIPELINES_ROOT=${PIPELINES_ROOT:-/tier1/htx_boonhan/services/pipelines}
PROJECT_ROOT=${PROJECT_ROOT:-/tier1/htx_boonhan/workspaces/data-flywheel/eval/system1/RoboDojo}
ENVS_DIR=${ENVS_DIR:-/tier1/htx_boonhan/services/envs}
ROBODOJO_DIR=${ROBODOJO_DIR:-/tier1/htx_boonhan/services/robodojo}
XPL=$PROJECT_ROOT/XPolicyLab

export PATH="$ENVS_DIR/conda-shim/bin:$HOME/.local/bin:$PATH"
export ENVS_DIR
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TERM=xterm-256color
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ $(uname -m) == aarch64 ]] && export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"   # Isaac Sim on ARM
# torch 2.7 bundles NVRTC 12.8, which doesn't know the GB300 (sm_103); its runtime-compiled
# kernels (erfinv and friends, used by curobo) then fail with "invalid value for --gpu-architecture".
# Preloading the env's NVRTC 12.9 (same soname) makes torch use that one instead.
nvrtc=$(ls "$ENVS_DIR/robodojo"/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 2>/dev/null | head -1)
[[ -n $nvrtc ]] && export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$nvrtc"
# System libs the node lacks (libGLU.so.1 for RTX's MDL stack), extracted from Ubuntu packages into user space.
export LD_LIBRARY_PATH="$ROBODOJO_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ -n $eval_num ]] && export EVAL_NUM=$eval_num
# Isaac Sim's caches on shared storage, so the first start isn't repeated per node.
export XDG_CACHE_HOME=$ROBODOJO_DIR/cache XDG_DATA_HOME=$ROBODOJO_DIR/share HOME_NV=$ROBODOJO_DIR/nv
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
[[ -e $PROJECT_ROOT/Assets ]] || ln -s "$ROBODOJO_DIR/Assets" "$PROJECT_ROOT/Assets"
# Robot configs are rendered from *_tmp.yml with absolute paths through $PROJECT_ROOT/Assets; redo after moving the repo.
[[ -f $PROJECT_ROOT/Assets/Robots/x5/curobo.yml ]] || (cd "$PROJECT_ROOT" && python3 utils/update_embodiment_config_path.py </dev/null | tail -1)

# Pretrained checkpoints downloaded from the RoboDojo hub land in
# $ROBODOJO_DIR/ckpt/RoboDojo/<P>/; XPolicyLab looks under policy/<P>/checkpoints/.
if [[ -d $ROBODOJO_DIR/ckpt/RoboDojo/$policy ]]; then
  mkdir -p "$XPL/policy/$policy/checkpoints"
  for d in "$ROBODOJO_DIR/ckpt/RoboDojo/$policy"/*/; do
    [[ -e $XPL/policy/$policy/checkpoints/$(basename "$d") ]] || ln -s "${d%/}" "$XPL/policy/$policy/checkpoints/$(basename "$d")"
  done
fi

echo "[$(date)] eval $policy on $task ckpt=$ckpt ($env_cfg/$action/seed $seed, EVAL_NUM=${EVAL_NUM:-default}) on $(hostname)"
cd "$XPL/policy/$policy"
bash eval.sh RoboDojo "$task" "$ckpt" "$env_cfg" "$action" "$seed" 0 0 "${policy,,}" robodojo

source "$ENVS_DIR/act/bin/activate"   # any env with mlflow + pandas
python "$PIPELINES_ROOT/xpolicylab/eval_mlflow.py" --policy "$policy" --bench RoboDojo --task "$task" --ckpt "$ckpt" \
  --env-cfg "$env_cfg" --action "$action" --seed "$seed" --result-root "$PROJECT_ROOT/eval_result"
