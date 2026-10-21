#!/usr/bin/env bash
# X-WAM (XPolicyLab's policy/X_WAM: a cross-embodiment world action model on Wan2.2-TI2V-5B, end-effector actions) on
# the GB300 (arm64), under $STATE, idempotent. X-WAM is no package: train_xwam.sbatch and serve_xwam.py put its source
# on PYTHONPATH, so nothing is written into the submodule. Its requirements.txt minus decord (no arm64 wheels:
# adapter/xwam/shim/decord stands in, on PyAV) and flash-attn (X-WAM falls back to torch's SDPA without it), on torch
# 2.7.1 cu128 as the FastWAM venv, plus av, pyarrow (the shim, adapter/xwam/lerobot_to_xwam.py) and tensorboard (train_sft.py's logger) (X-WAM was tested on 2.8; 2.7 has the flex_attention it imports). DeepSpeed runs nvcc
# at start: CUDA_HOME is the CUDA 12.8 toolkit setup.sh fastwam installs into $STATE/cuda-12.8. Also downloads the
# pretrained X-WAM checkpoint (sharinka0715/X-WAM-checkpoints pretrained/, Apache-2.0, 39 GB) into $STATE/models.
#   sensors/autolearn/gb300/setup_xwam.sh
set -euo pipefail
A="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE=${STATE:-/tier1/mistral/flywheel/autolearn}
X=$A/../../eval/system1/RoboDojo/XPolicyLab/policy/X_WAM/X-WAM
V=$STATE/venv-xwam
PIP=(uv pip install --python "$V/bin/python")
if [[ ! -f $V/.xwam-ok ]]; then
  uv venv --allow-existing --python 3.10 "$V"
  "${PIP[@]}" --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
  grep -v -E '^\s*(#|$)|^(decord|flash-attn)\b' "$X/requirements.txt" > "$STATE/xwam-requirements-arm64.txt"
  "${PIP[@]}" -r "$STATE/xwam-requirements-arm64.txt" av pyarrow tensorboard 'torch==2.7.1' 'torchvision==0.22.1' \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  [[ -x $STATE/cuda-12.8/bin/nvcc ]] || { echo "no $STATE/cuda-12.8: run setup.sh fastwam first (it installs the toolkit)" >&2; exit 1; }
  # X-WAM's own modules are not imported here: modules/t5.py asks CUDA for a device at import, so only a GPU node can
  CUDA_HOME=$STATE/cuda-12.8 PYTHONPATH=$A/adapter/xwam/shim:$X "$V/bin/python" -c \
    'import torch, lightning, deepspeed, decord, av, diffusers, transformers; print("xwam ok, torch", torch.__version__, "deepspeed", deepspeed.__version__, "lightning", lightning.__version__)' \
    && touch "$V/.xwam-ok"
fi
# the pretrained X-WAM checkpoint (Apache-2.0), the fine-tune's init
if [[ ! -f $STATE/models/X-WAM-checkpoints/pretrained/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt ]]; then
  HF_TOKEN=${HF_TOKEN:-$(grep -E '^HF_TOKEN=' "$A/../../services/.env" | cut -d= -f2- | tr -d '"')} \
    uvx --from huggingface_hub hf download sharinka0715/X-WAM-checkpoints --include 'pretrained/*' --local-dir "$STATE/models/X-WAM-checkpoints"
fi
