#!/bin/bash
# FastWAM's one-off preparation on one GB300 GPU, everything under $STATE/fastwam/: the ActionDiT backbone interpolated
# from Wan2.2-TI2V-5B's DiT (scripts/preprocess_action_dit_backbone.py), and the umT5 text embeddings of a dataset's task
# instructions (scripts/precompute_text_embeds.py), which FastWAM's training reads from a cache.
#   sbatch sensors/autolearn/gb300/prep_fastwam.sbatch <fastwam dataset (adapter/lerobot_to_fastwam.py output)>
# Wan2.2-TI2V-5B (from the MLflow registry) sits in $STATE/models/diffsynth/Wan-AI/ as DiffSynth expects.
#SBATCH -p raus -N1 -n1 -c16 --gres=gpu:1 --mem=200G -t 02:00:00 -J fastwam-prep
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/fastwam-prep-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
STATE=/tier1/mistral/flywheel/autolearn
data=$(realpath "$1"); name=$(basename "$data")
F=$A/../../eval/system1/RoboDojo/XPolicyLab/policy/FastWAM/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$STATE/models/diffsynth DIFFSYNTH_SKIP_DOWNLOAD=true CUDA_HOME=$STATE/cuda-12.8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
dit=$STATE/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
# redirect_common_files false: load the text encoder and VAE from Wan2.2's own .pth files (in FastWAM's hash registry)
# rather than DiffSynth's converted safetensors, which are published on ModelScope only
mcfg=$STATE/fastwam/fastwam_model.yaml
mkdir -p "$STATE/fastwam"; sed 's/^redirect_common_files: true/redirect_common_files: false/' "$F/configs/model/fastwam.yaml" > "$mcfg"
cd "$F"
if [[ ! -f $dit ]]; then
  "$STATE/venv-fastwam/bin/python" scripts/preprocess_action_dit_backbone.py --model-config "$mcfg" \
    --output "$dit" --device cuda --dtype bfloat16
fi
PYTHONPATH=$F:$F/src "$STATE/venv-fastwam/bin/python" scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4 model.redirect_common_files=false \
  "data.train.dataset_dirs=[$data]" "data.train.text_embedding_cache_dir=$STATE/fastwam/text_embeds/$name" \
  "data.val.dataset_dirs=[$data]" "data.val.text_embedding_cache_dir=$STATE/fastwam/text_embeds/$name" \
  "hydra.run.dir=$STATE/fastwam/hydra/prep-$SLURM_JOB_ID"
ls -la "$dit" "$STATE/fastwam/text_embeds/$name"
