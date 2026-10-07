#!/usr/bin/env bash
# Submit a Slurm script from this repo over SSH and follow it to completion.
#   SSH_KEY=<private key> SSH_HOST=user@host bash slurm/follow.sh <script.sbatch> [args...]
# The remote side is the restricted slurm-submit command (services/slurm/
# slurm-submit in data-flywheel). Streams the job log into this log, exits
# with 0 only if the job COMPLETED. Cancels the job if this step is killed.
set -euo pipefail
script=${1:?script}; shift
sha=${GITHUB_SHA:-$(git rev-parse HEAD)}

key=$(mktemp); trap 'rm -f "$key"' EXIT
printf '%s\n' "$SSH_KEY" > "$key"; chmod 600 "$key"
remote() { ssh -i "$key" -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR "$SSH_HOST" "$@"; }

job=$(remote sbatch "$sha" "$script" "$@" 2>&1) || { echo "submit failed:"; echo "$job"; exit 1; }
[[ $job =~ ^[0-9]+$ ]] || { echo "submit failed: $job"; exit 1; }
echo "slurm job $job: $script $* (pipelines@${sha:0:8})"
trap 'remote cancel "$job" || true; rm -f "$key"' INT TERM

offset=0
while :; do
  chunk=$(remote log "$job" "$offset" || true)
  if [[ -n $chunk ]]; then
    printf '%s\n' "$chunk"
    offset=$((offset + ${#chunk} + 1))
  fi
  state=$(remote status "$job")
  case $state in
    PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|UNKNOWN) sleep 20 ;;
    *) break ;;
  esac
done
chunk=$(remote log "$job" "$offset" || true); [[ -n $chunk ]] && printf '%s\n' "$chunk"
echo "slurm job $job ended: $state"
[[ $state == COMPLETED ]]
