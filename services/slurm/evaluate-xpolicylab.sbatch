#!/usr/bin/env bash
#SBATCH --job-name=evaluate-xpolicylab
#SBATCH --partition=raus_manual
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
# On C01 and C02 the eval client's Kit viewport stopped rendering a frame during the afternoon of
# 2026-10-08 (first with the ACT policy server beside it, later with the zero-action demo policy too);
# Isaac Lab then can't place its viewport camera. Identical software, GPU mode and scratch on all four
# nodes; cause not found. C03 and C04 work.
#SBATCH --exclude=C2-GB300-02-C01,C2-GB300-02-C02
#
# Evaluate a policy checkpoint in the RoboDojo simulator (Isaac Sim, headless)
# and record the result in MLflow.
#   sbatch evaluate-xpolicylab.sbatch <policy> <task> <ckpt> [env_cfg] [action] [seed] [eval_num]
#   e.g. sbatch evaluate-xpolicylab.sbatch ACT stack_bowls stack_bowls arx_x5_gpu joint 0 5
# Runs policy/<P>/eval.sh exactly as XPolicyLab documents it: a policy server
# in the policy's env, RoboDojo's eval client in the `robodojo` env, both
# activated through the conda shim (the scripts hard-code conda). Then
# eval_mlflow.py logs success_rate/score to the checkpoint's MLflow run.
set -euo pipefail
policy=${1:?policy}; task=${2:?task}; ckpt=${3:?ckpt}; env_cfg=${4:-arx_x5_gpu}; action=${5:-joint}; seed=${6:-0}; eval_num=${7:-}
# sbatch runs a spooled copy of this script, so the checkout comes from slurm.env (FLYWHEEL_ROOT).
SERVICES_DIR=${FLYWHEEL_ROOT:-$HOME/workspaces/data-flywheel}/services
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/workspaces/data-flywheel/eval/system1/RoboDojo}
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
  # keep the link out of the submodule's status (not every upstream .gitignore covers a link in place of the dir)
  local repo; repo=$(git -C "$(dirname "$1")" rev-parse --show-toplevel 2>/dev/null) || return 0
  local rel=${1#"$repo"/} ex; ex=$(git -C "$repo" rev-parse --git-path info/exclude)
  grep -qx "/$rel" "$ex" 2>/dev/null || echo "/$rel" >> "$ex"
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

export PATH="$ENVS_DIR/conda-shim/bin:$ROBODOJO_DIR/bin:$HOME/.local/bin:$PATH"   # robodojo/bin: ffmpeg (RoboDojo streams camera video through it)
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

# GB300: Isaac Sim 5.1 delivers no Replicator camera frames while the simulation runs on the CPU
# device (RoboDojo's default), so evaluate with the environment on cuda:0. An env_cfg ending in _gpu
# is derived from its base config on the fly; the layout and checkpoint names stay those of the base.
export PYTHONPATH="$SERVICES_DIR/slurm/robodojo-shim${PYTHONPATH:+:$PYTHONPATH}"   # CUDA tensors -> numpy in RoboDojo
if [[ $env_cfg == *_gpu ]]; then
  base=${env_cfg%_gpu}
  if [[ ! -f $PROJECT_ROOT/env_cfg/$env_cfg.yml ]]; then
    sim=$("$ENVS_DIR/robodojo/bin/python" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["config"]["sim"])' "$PROJECT_ROOT/env_cfg/$base.yml")   # the node python has no yaml
    { cat "$PROJECT_ROOT/env_cfg/sim/$sim.yml"; echo "device: cuda:0"; } > "$PROJECT_ROOT/env_cfg/sim/${sim}_gpu.yml"
    sed "s/^config_name: $base\$/config_name: $env_cfg/; s/^  sim: $sim\$/  sim: ${sim}_gpu/" "$PROJECT_ROOT/env_cfg/$base.yml" > "$PROJECT_ROOT/env_cfg/$env_cfg.yml"
    ex=$(git -C "$PROJECT_ROOT" rev-parse --git-path info/exclude 2>/dev/null) && { grep -qx "env_cfg/$env_cfg.yml" "$ex" 2>/dev/null || printf 'env_cfg/%s.yml\nenv_cfg/sim/%s_gpu.yml\n' "$env_cfg" "$sim" >> "$ex"; }
  fi
  [[ -e $ROBODOJO_DIR/Assets/Eval_Layout/RoboDojo/$env_cfg ]] || ln -s "$base" "$ROBODOJO_DIR/Assets/Eval_Layout/RoboDojo/$env_cfg"
  for d in "$XPL/policy/$policy/checkpoints"/RoboDojo-*-"$base"-*; do
    [[ -e $d ]] || continue
    alias=${d/-$base-/-$env_cfg-}; [[ -e $alias ]] || ln -s "$(basename "$d")" "$alias"
  done
fi

# Pretrained checkpoints downloaded from the RoboDojo hub land in
# $ROBODOJO_DIR/ckpt/RoboDojo/<P>/; XPolicyLab looks under policy/<P>/checkpoints/.
if [[ -d $ROBODOJO_DIR/ckpt/RoboDojo/$policy ]]; then
  mkdir -p "$XPL/policy/$policy/checkpoints"
  for d in "$ROBODOJO_DIR/ckpt/RoboDojo/$policy"/*/; do
    [[ -e $XPL/policy/$policy/checkpoints/$(basename "$d") ]] || ln -s "${d%/}" "$XPL/policy/$policy/checkpoints/$(basename "$d")"
  done
fi

# eval.sh sets CUDA_VISIBLE_DEVICES itself from its policy_gpu/env_gpu arguments, which would
# override Slurm's allocation and land on whatever physical GPU 0 is (shared with other jobs).
# Hand it the GPU Slurm gave us; policy server and simulator share it.
gpu=${CUDA_VISIBLE_DEVICES%%,*}          # simulator
policy_gpu=${CUDA_VISIBLE_DEVICES##*,}   # policy server: its own GPU when the job has two (sbatch --gres=gpu:2), else shared
echo "[$(date)] eval $policy on $task ckpt=$ckpt ($env_cfg/$action/seed $seed, EVAL_NUM=${EVAL_NUM:-default}) on $(hostname), GPU $gpu (policy server GPU $policy_gpu)"
started=$(date +%s)
cd "$XPL/policy/$policy"
bash eval.sh RoboDojo "$task" "$ckpt" "$env_cfg" "$action" "$seed" "$policy_gpu" "$gpu" "${policy,,}" robodojo

source "$ENVS_DIR/act/bin/activate"   # any env with mlflow + pandas
python "$SERVICES_DIR/xpolicylab/eval_mlflow.py" --policy "$policy" --bench RoboDojo --task "$task" --ckpt "$ckpt" \
  --env-cfg "$env_cfg" --action "$action" --seed "$seed" --result-root "$PROJECT_ROOT/eval_result" --since "$started"
