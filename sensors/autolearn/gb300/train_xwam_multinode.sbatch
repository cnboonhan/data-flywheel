#!/bin/bash
# train_xwam.sbatch over several GB300 nodes (4 GPUs each, InfiniBand between them): one torchrun per node
# (gb300/train_xwam_node.sh, via srun), rendezvous on the first node. X-WAM's train_sft.py takes the node count from
# WORLD_SIZE / GPUs per node, so Lightning's num_nodes follows the allocation; the global batch is
# batch_size_per_gpu x 4 x nodes (128 on 8 nodes at upstream's 4 per GPU).
#   [XWAM_RESUME_CKPT=$STATE/xwam/<run>/checkpoints[/<name>.ckpt]] [XWAM_TOTAL_SAMPLES=1500000 [XWAM_PRIOR_BATCH=16]] \
#     sbatch -N <nodes> --export=ALL sensors/autolearn/gb300/train_xwam_multinode.sbatch <xwam dataset> <run name> [overrides...]
# Resuming restores the step and optimizer; num_training_steps is the run's total, so the cosine schedule continues from
# the restored step over the new total. XWAM_RESUME_CKPT as a checkpoints directory resumes from its newest complete
# checkpoint as of the job's start (another job may still be writing there). XWAM_TOTAL_SAMPLES sets num_training_steps
# so the run sees that many samples in all: the steps so far at XWAM_PRIOR_BATCH, the rest at this job's global batch
# (batch_size_per_gpu 4, upstream's default, x 4 GPUs x nodes), rounded to a multiple of 1000 so a checkpoint lands on
# the last step (save_interval 1000). The choice goes to $STATE/xwam/<run>/resume-<job>.txt.
#SBATCH -p raus -N8 --ntasks-per-node=1 -c64 --gres=gpu:4 --mem=800G -t 120:00:00 -J xwam-train
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/xwam-train-%j.log
set -euo pipefail
export A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
STATE=/tier1/mistral/flywheel/autolearn
data=$(realpath "$1"); run=$2; shift 2
out=$STATE/xwam/$run; mkdir -p "$out"
export X=$A/../../eval/system1/RoboDojo/XPolicyLab/policy/X_WAM/X-WAM
export CUDA_HOME=$STATE/cuda-12.8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PATH=$STATE/venv-xwam/bin:$PATH PYTHONPATH=$A/adapter/xwam/shim:$X
# compile caches on each node's local disk, not shared storage (Triton/Inductor lock files under concurrent nodes)
export XWAM_NODE_CACHE=/tmp/$USER/xwam-cache
export TRITON_CACHE_DIR=$XWAM_NODE_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$XWAM_NODE_CACHE/inductor
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export XWAM_ZERO_STAGE=${XWAM_ZERO_STAGE:-0}     # see train_xwam.sbatch
export XWAM_DATA_CONFIG=$out/data_config.yaml
python - "$X/configs/data/robodojo.yaml" "$data" "$XWAM_DATA_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
base, data, dst = sys.argv[1:]
c = OmegaConf.load(base)
c.dataset_path = data
c.statistics = OmegaConf.load(f"{data}/quantile.yaml")
c.video_size = [256, 256]       # the 512x512 cameras at the converted videos' size (upstream's 256x320 would stretch them)
OmegaConf.save(c, dst)
PY
if [[ -d ${XWAM_RESUME_CKPT:-} && ${XWAM_RESUME_CKPT%/} == */checkpoints ]]; then
  # the newest checkpoint whose weights file is no longer growing (unchanged over STABLE_S seconds: a 99 GB save takes ~1 min)
  STABLE_S=120
  # and as large as the largest: every complete save of this model is the same size, one cut off mid-write is smaller
  full=$(stat -c %s "$XWAM_RESUME_CKPT"/*-step=*.ckpt/checkpoint/mp_rank_00_model_states.pt 2>/dev/null | sort -n | tail -1)
  for ck in $(ls -d "$XWAM_RESUME_CKPT"/*-step=*.ckpt 2>/dev/null | sed -E 's/.*-step=([0-9]+)\.ckpt$/\1 &/' | sort -rn | cut -d' ' -f2); do
    f=$ck/checkpoint/mp_rank_00_model_states.pt
    [[ -s $f ]] || continue
    a=$(stat -c %s "$f"); sleep "$STABLE_S"; b=$(stat -c %s "$f")
    (( a == b && b == full && $(date +%s) - $(stat -c %Y "$f") >= STABLE_S )) && { XWAM_RESUME_CKPT=$ck; break; }
  done
  [[ $XWAM_RESUME_CKPT == *.ckpt ]] || { echo "no complete checkpoint in $XWAM_RESUME_CKPT"; exit 1; }
fi
export XWAM_RESUME_CKPT
steps_ov=()
if [[ -n ${XWAM_TOTAL_SAMPLES:-} ]]; then
  done_steps=0; [[ ${XWAM_RESUME_CKPT:-} =~ -step=([0-9]+)\.ckpt$ ]] && done_steps=${BASH_REMATCH[1]}
  global=$((4 * 4 * SLURM_NNODES))
  total=$(( (done_steps + (XWAM_TOTAL_SAMPLES - done_steps * ${XWAM_PRIOR_BATCH:-16}) / global + 500) / 1000 * 1000 ))
  steps_ov=("num_training_steps=$total")
fi
printf 'resume %s\nnum_training_steps %s\nglobal batch %s\n' "${XWAM_RESUME_CKPT:-none}" "${total:-as given}" "${global:-}" \
  | tee "$out/resume-$SLURM_JOB_ID.txt"
export XWAM_MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export XWAM_PORT=29640      # the rendezvous port on the first node; any free port, fixed so every node agrees
ov=("dataset=xwam_behavior" "exp_root=$STATE/xwam" "exp_name=$run"
    "wan_checkpoint_dir=$STATE/models/diffsynth/Wan-AI/Wan2.2-TI2V-5B"
    "pretrained_checkpoint=$STATE/models/X-WAM-checkpoints/pretrained/checkpoints/last.ckpt")
echo "nodes $SLURM_JOB_NODELIST master $XWAM_MASTER zero stage $XWAM_ZERO_STAGE resume ${XWAM_RESUME_CKPT:-none} data $data run $out"
srun --ntasks-per-node=1 --kill-on-bad-exit=1 "$A/gb300/train_xwam_node.sh" "${ov[@]}" "$@" "${steps_ov[@]}"
