#!/usr/bin/env bash
#SBATCH --job-name=convert-galaxea-xspark
#SBATCH --partition=raus_manual
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
#
# Convert unpacked Galaxea task datasets into XPolicyLab xspark HDF5 episodes.
#   sbatch convert-galaxea-xspark.sbatch <subsets glob> [limit] [env_cfg]
# Expects S3_ENDPOINT_URL, AWS_* and AWS_CA_BUNDLE in the environment (the
# submitter exports them). Reads from the bucket directory and writes through
# the S3 gateway to processed/xpolicylab/.
#
# sbatch copies this file to the spool dir, so sibling scripts are found via
# PIPELINES_ROOT: a checkout of the pipelines repo on shared storage.
set -euo pipefail
subsets=${1:?subsets glob, e.g. 'galaxea-open-world-r1lite/*'}
limit=${2:-0}
env_cfg=${3:-r1lite}
PIPELINES_ROOT=${PIPELINES_ROOT:-/tier1/htx_boonhan/services/pipelines}

export UV_CACHE_DIR=${UV_CACHE_DIR:-/tier1/htx_boonhan/services/uv-cache}
export PATH="$HOME/.local/bin:$PATH"   # uv; jobs submitted over the restricted ssh command inherit a minimal PATH
cd /tier1/htx_boonhan/services
exec uv run -q --python 3.12 --with h5py,pyarrow,pandas,numpy,opencv-python-headless,av,boto3 \
  python "$PIPELINES_ROOT/xpolicylab/convert_galaxea_xspark.py" \
  --root /tier1/htx_boonhan/services/versitygw/buckets/processed \
  --subsets "$subsets" --limit "$limit" --env-cfg "$env_cfg"
