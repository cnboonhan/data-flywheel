#!/bin/bash
# Train FastWAM (XPolicyLab's policy/FastWAM, on Wan2.2-TI2V-5B) on a BEHAVIOR-1K collection, on one GB300 node, with
# DeepSpeed the way FastWAM's scripts/train_zero1.sh launches it (ZeRO stage 0, see below).
#   [FASTWAM_STATS=<dataset_stats.json>] sbatch --export=ALL sensors/autolearn/gb300/train_fastwam.sbatch <fastwam dataset> <run name> [hydra overrides...]
# The dataset comes from adapter/lerobot_to_fastwam.py, its text embeddings and the ActionDiT backbone from
# gb300/prep_fastwam.sbatch. XPolicyLab's train.sh is not used: it takes the action size from XPolicyLab's robot
# registry, which has no R1 Pro; the same dimension overrides are passed here for the 20-value joint vector (arms,
# grippers, torso). Runs, stats and checkpoints go under $STATE/fastwam/<run name>/, not into the submodule.
#SBATCH -p raus -N1 -n1 -c64 --gres=gpu:4 --mem=800G -t 24:00:00 -J fastwam-train
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/fastwam-train-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
STATE=/tier1/mistral/flywheel/autolearn
data=$(realpath "$1"); run=$2; shift 2
out=$STATE/fastwam/$run; mkdir -p "$out"
F=$A/../../eval/system1/RoboDojo/XPolicyLab/policy/FastWAM/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$STATE/models/diffsynth DIFFSYNTH_SKIP_DOWNLOAD=true CUDA_HOME=$STATE/cuda-12.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
export PATH=$STATE/venv-fastwam/bin:$PATH PYTHONPATH=$F:$F/src
ngpu=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")      # the GPUs Slurm allocated
dim=20      # lerobot_to_fastwam.py's joint vector
port=$((29500 + SLURM_JOB_ID % 1000))   # torch.distributed rendezvous; the default 29500 clashes when two jobs share a node
ov=("task=robotwin_uncond_3cam_384_1e-4" "output_dir=$out/run" "hydra.run.dir=$out/run" "model.redirect_common_files=false"
    "model.action_dit_pretrained_path=$STATE/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt")
for split in train val; do
  ov+=("data.$split.dataset_dirs=[$data]" "data.$split.text_embedding_cache_dir=$STATE/fastwam/text_embeds/$(basename "$data")")
  for group in action state; do ov+=("data.$split.shape_meta.$group.0.raw_shape=$dim" "data.$split.shape_meta.$group.0.shape=$dim"); done
  ov+=("data.$split.processor.action_output_dim=$dim" "data.$split.processor.proprio_output_dim=$dim")
done
# normalization stats: $FASTWAM_STATS when set (e.g. one with the standard deviations floored: a joint that barely moves,
# like the R1 Pro's 3rd and 4th torso targets, z-scores to ~1e5 and the loss turns NaN); else the training set computes
# them into the run directory and the validation set reads them from there
if [[ -n ${FASTWAM_STATS:-} ]]; then
  ov+=("data.train.pretrained_norm_stats=$FASTWAM_STATS" "data.val.pretrained_norm_stats=$FASTWAM_STATS")
else
  ov+=("data.train.pretrained_norm_stats=null" "data.val.pretrained_norm_stats=$out/run/dataset_stats.json")
fi
cd "$F"
echo "node $(hostname) GPUs $CUDA_VISIBLE_DEVICES data $data run $out"
# FastWAM's launcher (scripts/train_zero1.sh) changed in two ways, both tested on this dataset at lr 1e-5:
# - DeepSpeed ZeRO stage 0, not 1: under ZeRO-1 the first optimizer step turns video_expert.patch_embedding.weight NaN
#   (finite gradients); stage 0 keeps the whole optimizer on each GPU, ~100 GB for the 6B trainable parameters;
# - fp32 weights (mixed_precision=no): with bf16 both losses climb from the first few hundred steps (video 0.5 -> 1.6
#   by step 800, action loss above 1e4 at lr 1e-4); in fp32 both fall (video 0.31 -> 0.25 over 500 steps), ~10% slower
sed "s|ds_zero0.json|$A/adapter/fastwam/ds_zero0.json|" "$A/adapter/fastwam/accelerate_zero0.yaml" > "$out/accelerate.yaml"
# a /dev/shm of its own: logind removes the user's /dev/shm files whenever their last session on the node ends
unshare --user --map-root-user --ipc --mount -- bash -c 'mount -t tmpfs -o size=256g tmpfs /dev/shm && exec "$0" "$@"' \
  accelerate launch --config_file "$out/accelerate.yaml" --num_processes "$ngpu" --main_process_port "$port" "$A/adapter/fastwam/train.py" "wandb.name=$run" "mixed_precision=no" "${ov[@]}" "$@"
