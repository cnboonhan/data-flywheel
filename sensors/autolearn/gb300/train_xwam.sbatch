#!/bin/bash
# Fine-tune X-WAM (XPolicyLab's policy/X_WAM, on Wan2.2-TI2V-5B) on a BEHAVIOR-1K collection, on one GB300 node, from the
# pretrained X-WAM checkpoint (sharinka0715/X-WAM-checkpoints pretrained/, Apache-2.0; gb300/setup_xwam.sh fetches it).
#   [XWAM_ZERO_STAGE=0] sbatch --export=ALL sensors/autolearn/gb300/train_xwam.sbatch <xwam dataset> <run name> [overrides...]
# The dataset comes from adapter/xwam/lerobot_to_xwam.py, its quantile.yaml from X-WAM's scripts/compute_stats.py.
# XPolicyLab's train.sh is not used: it writes into the submodule (checkpoints/, data/) and reads its fixed data config;
# adapter/xwam/train_xwam.py runs the same scripts/train_sft.py with this run's data config. Overrides are X-WAM's
# OmegaConf keys (lr=, num_training_steps=, save_interval=, batch_size_per_gpu=, ...). Runs go to $STATE/xwam/<run name>/.
#SBATCH -p raus -N1 -n1 -c64 --gres=gpu:4 --mem=800G -t 24:00:00 -J xwam-train
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/xwam-train-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
STATE=/tier1/mistral/flywheel/autolearn
data=$(realpath "$1"); run=$2; shift 2
out=$STATE/xwam/$run; mkdir -p "$out"
X=$A/../../eval/system1/RoboDojo/XPolicyLab/policy/X_WAM/X-WAM
export CUDA_HOME=$STATE/cuda-12.8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PATH=$STATE/venv-xwam/bin:$PATH PYTHONPATH=$A/adapter/xwam/shim:$X
# ZeRO stage 0 by default: FastWAM (the same Wan2.2 backbone) turned a weight NaN on its first step under ZeRO-1 and trains
# under stage 0; the full optimizer state per GPU (~110 GB for the ~6.5B parameters) fits a GB300's 279 GB
export XWAM_ZERO_STAGE=${XWAM_ZERO_STAGE:-0}
# the data config: upstream's robodojo.yaml with this dataset, its statistics and the cameras' square shape
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
ngpu=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")      # the GPUs Slurm allocated
ov=("dataset=xwam_behavior" "exp_root=$STATE/xwam" "exp_name=$run"
    "wan_checkpoint_dir=$STATE/models/diffsynth/Wan-AI/Wan2.2-TI2V-5B"
    "pretrained_checkpoint=$STATE/models/X-WAM-checkpoints/pretrained/checkpoints/last.ckpt")
cd "$X"
echo "node $(hostname) GPUs $CUDA_VISIBLE_DEVICES zero stage $XWAM_ZERO_STAGE data $data run $out"
# a /dev/shm of its own: logind removes the user's /dev/shm files whenever their last session on the node ends
unshare --user --map-root-user --ipc --mount -- bash -c 'mount -t tmpfs -o size=256g tmpfs /dev/shm && exec "$0" "$@"' \
  torchrun --nnodes=1 --nproc_per_node="$ngpu" --master_port=$((29600 + RANDOM % 1000)) \
  "$A/adapter/xwam/train_xwam.py" "${ov[@]}" "$@"
