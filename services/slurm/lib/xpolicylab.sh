# Sourced by the ACT/DP recipes: XPolicyLab inside the RoboDojo submodule, with bulky outputs kept off the checkout.
# XPolicyLab reads its xspark tree as PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/; that tree is processed/xpolicylab.
# Decoded-frame caches, checkpoints and DP's zarr data go under DATA_ROOT/<policy>/, evaluation results under
# ROBODOJO_DIR/eval_result; the checkout gets symlinks, kept out of the submodule's status.

link_out() {   # link_out <path in checkout> <path on /tier1>
  [[ -L $1 ]] && return 0
  mkdir -p "$2"
  if [[ -d $1 ]]; then cp -a "$1/." "$2/" && rm -rf "${1:?}"; fi
  ln -s "$2" "$1"
  local repo; repo=$(git -C "$(dirname "$1")" rev-parse --show-toplevel 2>/dev/null) || return 0
  local rel=${1#"$repo"/} ex; ex=$(git -C "$repo" rev-parse --git-path info/exclude)
  grep -qx "/$rel" "$ex" 2>/dev/null || echo "/$rel" >> "$ex"
}

xpolicylab_links() {
  local p ex
  link_out "$PROJECT_ROOT/eval_result" "$ROBODOJO_DIR/eval_result"
  for p in ACT DP; do
    link_out "$XPL/policy/$p/processed_data" "$DATA_ROOT/$p/processed_data"
    link_out "$XPL/policy/$p/checkpoints" "$DATA_ROOT/$p/checkpoints"
  done
  link_out "$XPL/policy/DP/data" "$DATA_ROOT/DP/data"
  if [[ ! -e $PROJECT_ROOT/data ]]; then
    ln -s "$BUCKETS_DIR/processed/xpolicylab" "$PROJECT_ROOT/data"
    ex=$(git -C "$PROJECT_ROOT" rev-parse --git-path info/exclude 2>/dev/null) && { grep -qx /data "$ex" 2>/dev/null || echo /data >> "$ex"; }
  fi
}

# xpolicylab_single_dataset: XPolicyLab trains on one <bench>/<task>; sets BENCH and TASK from DATA.
xpolicylab_single_dataset() {
  [[ ${#DATA[@]} == 1 && $DATA != *[*?[]* && $DATA == */* ]] || { echo "$MODEL trains on one <bench>/<task> (got: ${DATA[*]}); mixes need a model that reads hdf5 directly" >&2; exit 2; }
  [[ -n $EMB_XPL_ENV_CFG ]] || { echo "embodiment $EMB_NAME has no xpolicylab.env_cfg" >&2; exit 2; }
  BENCH=${DATA%%/*}; TASK=${DATA#*/}
  [[ $EMB_DATA_ENV_CFG == "$EMB_XPL_ENV_CFG" ]] || { echo "XPolicyLab reads data/<bench>/<task>/<env_cfg>: data_env_cfg ($EMB_DATA_ENV_CFG) must equal xpolicylab.env_cfg ($EMB_XPL_ENV_CFG)" >&2; exit 2; }
}

# xpolicylab_eval <POLICY>: XPolicyLab policy/<POLICY>/eval.sh in RoboDojo (Isaac Sim headless), then eval_mlflow.py.
# Uses TASK, CKPT, ACTION, SEED, EVAL_NUM and EMB_EVAL_ENV_CFG / EMB_XPL_ENV_CFG.
xpolicylab_eval() {
  local policy=$1 env_cfg=$EMB_EVAL_ENV_CFG base=$EMB_XPL_ENV_CFG sim d alias ex nvrtc
  [[ -n $env_cfg ]] || { echo "embodiment $EMB_NAME has no xpolicylab.eval_env_cfg: RoboDojo can't simulate it" >&2; exit 2; }
  xpolicylab_links
  export PATH="$ENVS_DIR/conda-shim/bin:$ROBODOJO_DIR/bin:$HOME/.local/bin:$PATH"   # robodojo/bin: ffmpeg (RoboDojo streams camera video through it)
  export ENVS_DIR
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TERM=xterm-256color
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
  [[ $(uname -m) == aarch64 ]] && export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"   # Isaac Sim on ARM
  # torch 2.7 bundles NVRTC 12.8, which doesn't know the GB300 (sm_103); its runtime-compiled
  # kernels (erfinv and friends, used by curobo) then fail with "invalid value for --gpu-architecture".
  # Preloading the env's NVRTC 12.9 (same soname) makes torch use that one instead.
  nvrtc=$(ls "$ENVS_DIR/robodojo"/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 2>/dev/null | head -1)
  [[ -n $nvrtc ]] && export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$nvrtc"
  # System libs the node lacks (libGLU.so.1 for RTX's MDL stack), extracted from Ubuntu packages into user space.
  export LD_LIBRARY_PATH="$ROBODOJO_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  [[ -n ${EVAL_NUM:-} ]] && export EVAL_NUM
  # Isaac Sim's caches on shared storage, so the first start isn't repeated per node.
  export XDG_CACHE_HOME=$ROBODOJO_DIR/cache XDG_DATA_HOME=$ROBODOJO_DIR/share HOME_NV=$ROBODOJO_DIR/nv
  mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME"
  [[ -e $PROJECT_ROOT/Assets ]] || ln -s "$ROBODOJO_DIR/Assets" "$PROJECT_ROOT/Assets"
  # Robot configs are rendered from *_tmp.yml with absolute paths through $PROJECT_ROOT/Assets; redo after moving the repo.
  [[ -f $PROJECT_ROOT/Assets/Robots/x5/curobo.yml ]] || (cd "$PROJECT_ROOT" && python3 utils/update_embodiment_config_path.py </dev/null | tail -1)

  # GB300: Isaac Sim 5.1 delivers no Replicator camera frames while the simulation runs on the CPU
  # device (RoboDojo's default), so evaluate with the environment on cuda:0. An env_cfg ending in _gpu
  # is derived from its base config on the fly; the layout and checkpoint names stay those of the base.
  export PYTHONPATH="$LIB/robodojo-shim${PYTHONPATH:+:$PYTHONPATH}"   # CUDA tensors -> numpy in RoboDojo
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
  local gpu=${CUDA_VISIBLE_DEVICES%%,*}          # simulator
  local policy_gpu=${CUDA_VISIBLE_DEVICES##*,}   # policy server: its own GPU when the job has two (sbatch --gres=gpu:2), else shared
  echo "[$(date)] eval $policy on $TASK ckpt=$CKPT ($env_cfg/$ACTION/seed $SEED, EVAL_NUM=${EVAL_NUM:-default}) on $(hostname), GPU $gpu (policy server GPU $policy_gpu)"
  local started; started=$(date +%s)
  (cd "$XPL/policy/$policy" && bash eval.sh RoboDojo "$TASK" "$CKPT" "$env_cfg" "$ACTION" "$SEED" "$policy_gpu" "$gpu" "${policy,,}" robodojo)

  # shellcheck source=/dev/null
  source "$ENVS_DIR/$ENV/bin/activate"   # after eval.sh, which activates its own envs through the conda shim
  python "$LIB/eval_mlflow.py" --model "$policy" --bench RoboDojo --task "$TASK" --ckpt "$CKPT" \
    --env-cfg "$env_cfg" --action "$ACTION" --seed "$SEED" --result-root "$PROJECT_ROOT/eval_result" --since "$started"
}
