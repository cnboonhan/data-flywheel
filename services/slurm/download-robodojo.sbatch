#!/usr/bin/env bash
#SBATCH --job-name=download-robodojo
#SBATCH --partition=raus_manual
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
#
# Fetch parts of the RoboDojo Hugging Face dataset (RoboDojo-Benchmark/RoboDojo).
#   sbatch download-robodojo.sbatch assets                 -> $ROBODOJO_DIR/Assets (sim assets)
#   sbatch download-robodojo.sbatch ckpt GR00T_N17         -> $ROBODOJO_DIR/ckpt/GR00T_N17 (pretrained checkpoints)
#   sbatch download-robodojo.sbatch data stack_bowls       -> s3://raw/robodojo/stack_bowls (xspark episodes, via the S3 gateway)
# RoboDojo's own scripts use git-lfs, which the cluster lacks; this uses the
# hub client with hf_transfer. Needs HF_TOKEN for gated files and, for data,
# S3_ENDPOINT_URL / AWS_* (from slurm.env).
set -euo pipefail
what=${1:?assets|ckpt|data}; name=${2:-}
ROBODOJO_DIR=${ROBODOJO_DIR:-/tier1/htx_boonhan/services/robodojo}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tier1/htx_boonhan/services/uv-cache}
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf() { uv run -q --python 3.12 --with "huggingface_hub[hf_transfer]" hf "$@"; }
repo=(RoboDojo-Benchmark/RoboDojo --repo-type dataset)

case $what in
  assets)
    hf download "${repo[@]}" --include "Assets/**" --local-dir "$ROBODOJO_DIR"
    ls "$ROBODOJO_DIR/Assets"
    # The robot configs ship as *_tmp.yml templates; RoboDojo's helper writes the real
    # ones with absolute asset paths (it must run from a dir containing Assets/Robots).
    PROJECT_ROOT=${PROJECT_ROOT:-$HOME/workspaces/data-flywheel/eval/system1/RoboDojo}
    [[ -e $PROJECT_ROOT/Assets ]] || ln -s "$ROBODOJO_DIR/Assets" "$PROJECT_ROOT/Assets"
    (cd "$PROJECT_ROOT" && python3 utils/update_embodiment_config_path.py </dev/null | tail -2) ;;
  ckpt)
    hf download "${repo[@]}" --include "ckpt/RoboDojo/${name:?policy}/**" --local-dir "$ROBODOJO_DIR"
    find "$ROBODOJO_DIR/ckpt/RoboDojo/$name" -maxdepth 3 -type d ;;
  data)
    tmp=$ROBODOJO_DIR/download/$name; mkdir -p "$tmp"
    hf download "${repo[@]}" --include "data/RoboDojo/${name:?task}/**" --local-dir "$tmp"
    uv run -q --python 3.12 --with awscli aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync "$tmp/data/RoboDojo/$name" "s3://raw/robodojo/$name" --no-progress | tail -2
    rm -rf "$tmp" ;;
esac
