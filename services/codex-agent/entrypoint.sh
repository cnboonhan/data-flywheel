#!/usr/bin/env bash
# The agent container's entry (services/codex-agent/README.md): Codex in yolo mode on /workspace, the mounted folder.
#   <image> [prompt...]   one autonomous run (`codex exec`) on the prompt: the arguments, else $AGENT_PROMPT, else the
#                         file $AGENT_TASK_FILE in /workspace (default TASK.md)
#   <image> tui           Codex's interactive UI, same config (needs `docker run -it`)
# Env: CODEX_MODEL (default the config's, kimi-k3), LLM_URL (the gateway's /v1; default the config's), and the stack's CA
# mounted at /ca/root.crt (the gateway's certificate is Caddy's own).
set -euo pipefail
export CODEX_HOME=${CODEX_HOME:-$HOME/.codex}   # sessions and logs, under /home/agent: writable for any uid (Dockerfile)
mkdir -p "$CODEX_HOME"
cp /etc/codex/config.toml "$CODEX_HOME/config.toml"
if [[ -f /ca/root.crt ]]; then   # trust the stack's CA beside the system's: for Codex (Rust) and its Node wrapper
  cat /etc/ssl/certs/ca-certificates.crt /ca/root.crt > /tmp/ca-bundle.crt
  export SSL_CERT_FILE=/tmp/ca-bundle.crt NODE_EXTRA_CA_CERTS=/ca/root.crt
fi
over=()
[[ -n ${CODEX_MODEL:-} ]] && over+=(-c "model=\"$CODEX_MODEL\"")
[[ -n ${LLM_URL:-} ]] && over+=(-c "model_providers.selfhosted.base_url=\"$LLM_URL\"")
cd /workspace
if [[ ${1:-} == tui ]]; then
  exec codex "${over[@]}" --dangerously-bypass-approvals-and-sandbox
fi
prompt="$*"
[[ -n $prompt ]] || prompt=${AGENT_PROMPT:-}
[[ -n $prompt || ! -f ${AGENT_TASK_FILE:-TASK.md} ]] || prompt=$(cat "${AGENT_TASK_FILE:-TASK.md}")
[[ -n $prompt ]] || { echo "no task: give a prompt, AGENT_PROMPT, or a ${AGENT_TASK_FILE:-TASK.md} in the folder" >&2; exit 2; }
exec codex exec "${over[@]}" --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C /workspace "$prompt"
