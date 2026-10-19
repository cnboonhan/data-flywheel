#!/usr/bin/env bash
# Test NVLink between every pair of GPUs on one tray, in Docker as the Dynamo engines run: services/dynamo/nvlink_test/run.sh <tray>
# Prints one line per pair (pair.py). The tray's GPUs should be idle: the copies take 2 GiB of two GPUs at a time.
set -euo pipefail
tray=${1:?usage: run.sh <tray>}
here=$(cd "$(dirname "$0")" && pwd)
docker=/cm/local/apps/docker/current/bin/docker   # not on PATH in a non-interactive ssh session on the trays
image=${IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0-deepseek-v4.1-flash-dev.2}   # any image with torch; this one is on the trays

ssh -o BatchMode=yes "$tray" "$docker run --rm --device nvidia.com/gpu=all --ipc host -v $here:/t:ro --entrypoint bash $image -c '
  n=\$(nvidia-smi -L | wc -l)
  for a in \$(seq 0 \$((n - 1))); do for b in \$(seq 0 \$((n - 1))); do
    [ \$a = \$b ] || python3 /t/pair.py \$a \$b 2>&1 | grep -E \"^[0-9]+->\"
  done; done'"
