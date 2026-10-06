#!/usr/bin/env bash
# Download the Hugging Face repos listed below: DATASETS go to datasets/, CHECKPOINTS to checkpoints/.
#   bash scripts/download.sh              # pick from a checkbox list, see size + time, confirm, download
#   bash scripts/download.sh --dry-run    # only show access, size left and estimated time
#   bash scripts/download.sh --all --yes  # non-interactive: download everything available
# Entry format: "<model|dataset> <repo_id> <include globs, comma-separated or *> <local folder name>"
# (the repo type is the Hugging Face repo type; some checkpoints live in dataset repos)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASETS=(
  "dataset OpenGalaxea/Galaxea-Open-World-Dataset * galaxea-open-world"
  "dataset simple-world-lab/HiFi-UMI-2K * hifi-umi-2k"
)

CHECKPOINTS=(
  # Example:
  # "model Qwen/Qwen3-VL-2B-Instruct * qwen3-vl-2b-instruct"
  # GR00T-N1.7: official RoboDojo fine-tune (lives in the RoboDojo dataset repo)
  "dataset RoboDojo-Benchmark/RoboDojo ckpt/RoboDojo/GR00T_N17/* robodojo-gr00t_n17"
  # GR00T-N1.7 VLM backbone, loaded in full at startup (gated: accept terms + hf auth login)
  "model nvidia/Cosmos-Reason2-2B * cosmos-reason2-2b"
  # OpenWAM: benchmark fine-tunes
  "model OpenWAM/OpenWAM-Alpha-Sim-RoboDojo * openwam-sim-robodojo"
  "model OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full * openwam-sim-robotwin-full"
)

ENTRIES=()
for entry in "${DATASETS[@]}"; do ENTRIES+=("datasets ${entry}"); done
for entry in "${CHECKPOINTS[@]}"; do ENTRIES+=("checkpoints ${entry}"); done

if [ ${#ENTRIES[@]} -eq 0 ]; then
  echo "No repos listed in ${BASH_SOURCE[0]}; add entries to DATASETS or CHECKPOINTS."
  exit 0
fi

exec uv run --quiet --script "${ROOT}/scripts/hf_download.py" --root "${ROOT}" "$@" "${ENTRIES[@]}"
