#!/usr/bin/env bash
# Run the Codex agent on a folder (services/codex-agent/README.md): it works there autonomously, with the stack's LLM.
#   services/codex-agent/run.sh <folder> [prompt...]   one run on the prompt, else $AGENT_PROMPT, else <folder>/TASK.md
#   services/codex-agent/run.sh <folder> tui           Codex's interactive UI on the folder
# Env: CODEX_MODEL (default kimi-k3), LLM_URL (default the stack's gateway), CODEX_AGENT_IMAGE. Runs on any node with
# Docker and the image (build it there: docker build -t flywheel-codex-agent:0.162.1 services/codex-agent); as you, so
# what it writes in the folder is yours. Nothing but the folder and the stack's CA is mounted.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
folder=$(realpath "${1:?usage: run.sh <folder> [prompt...|tui]}"); shift
[[ -d $folder ]] || { echo "$folder is not a folder" >&2; exit 2; }
# the stack's CA and gateway: the quickstart's FLYWHEEL_CA and LLM_URL, else the admin's services/.env
[[ -f $here/../.env ]] && { set -a; . "$here/../.env"; set +a; }
ca=${FLYWHEEL_CA:-${STATE_DIR:+$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt}}
[[ -f $ca ]] || { echo "no CA: set FLYWHEEL_CA (quickstart section 1)" >&2; exit 2; }
llm=${LLM_URL:-${SERVICE_HOST:+https://llm.$SERVICE_HOST:$CADDY_PORT/v1}}
tty=(); [[ -t 0 && -t 1 ]] && tty=(-it)
exec docker run --gpus all --rm "${tty[@]}" --user "$(id -u):$(id -g)" \
  -v "$folder:/workspace" -v "$ca:/ca/root.crt:ro" \
  -e CODEX_MODEL="${CODEX_MODEL:-}" -e LLM_URL="$llm" \
  -e AGENT_PROMPT="${AGENT_PROMPT:-}" -e AGENT_TASK_FILE="${AGENT_TASK_FILE:-TASK.md}" \
  "${CODEX_AGENT_IMAGE:-flywheel-codex-agent:0.162.1}" "$@"
