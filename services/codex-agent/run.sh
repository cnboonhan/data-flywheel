#!/usr/bin/env bash
# Run the Codex agent on a folder (services/codex-agent/README.md): OpenAI's Codex CLI (Apache-2.0) in yolo mode on the
# stack's self-hosted LLM, working there autonomously. Builds its image on first use.
#   services/codex-agent/run.sh <folder> [prompt...]   one run on the prompt, else $AGENT_PROMPT, else <folder>/TASK.md
#   services/codex-agent/run.sh <folder> tui           Codex's interactive UI on the folder
# <folder>/.env is sourced and passed to the agent: it overrides the defaults below for this run.
set -euo pipefail
folder=$(realpath "${1:?usage: run.sh <folder> [prompt...|tui]}")
shift
[[ -d $folder ]] || { echo "$folder is not a folder" >&2; exit 2; }

envs=()
if [[ -f $folder/.env ]]; then
  set -a; . "$folder/.env"; set +a
  while IFS= read -r name; do envs+=(-e "$name"); done \
    < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=.*/\2/p' "$folder/.env")
fi
CODEX_MODEL=${CODEX_MODEL:-kimi-k3}                                     # any model the gateway serves
LLM_URL=${LLM_URL:-https://llm.flywheel.10.80.81.30.sslip.io:8443/v1}   # the stack's LLM gateway (no token)
FLYWHEEL_CA=${FLYWHEEL_CA:-$HOME/.flywheel/ca.crt}                      # the quickstart's path for the stack's CA
image=${CODEX_AGENT_IMAGE:-flywheel-codex-agent:0.162.1-noble}

# Trust on first use: the stack's CA comes from the gateway itself (https://<host>/ca.crt) the first time, then from
# $FLYWHEEL_CA. The agent trusts the host's CAs (as update-ca-certificates saved them) plus that one.
if [[ ! -s $FLYWHEEL_CA ]]; then
  ca_url=$(sed -E 's#^(https://)(llm\.)?([^/]+).*#\1\3/ca.crt#' <<<"$LLM_URL")
  mkdir -p "$(dirname "$FLYWHEEL_CA")"
  curl -sfk "$ca_url" -o "$FLYWHEEL_CA.new" && openssl x509 -in "$FLYWHEEL_CA.new" -noout 2>/dev/null \
    || { rm -f "$FLYWHEEL_CA.new"; echo "no CA at $ca_url: set FLYWHEEL_CA (quickstart section 1)" >&2; exit 2; }
  mv "$FLYWHEEL_CA.new" "$FLYWHEEL_CA"
  echo "trusting $ca_url on first use, saved to $FLYWHEEL_CA: $(openssl x509 -in "$FLYWHEEL_CA" -noout -subject -fingerprint -sha256)" >&2
fi
bundle=$(dirname "$FLYWHEEL_CA")/codex-agent-ca-bundle.crt
cat /etc/ssl/certs/ca-certificates.crt "$FLYWHEEL_CA" > "$bundle"

# The image: Codex 0.162.1 (the newest release at least two weeks old when added) on Node 22, with the tools such work
# usually needs. Ubuntu 24.04 like the cluster nodes, so environments built there run inside it too (README, On Slurm):
# libgomp and the X/GL libraries are what Isaac Sim loads from the system, libgfortran what open3d's aarch64 wheel does,
# and the Vulkan and EGL entries name the NVIDIA driver libraries that the container runtime mounts in.
# HOME is world-writable as the container runs as you, and Codex won't put its helpers under /tmp.
docker image inspect "$image" &>/dev/null || docker build -t "$image" - <<'EOF'
FROM node:22.20.0-bookworm-slim AS node
FROM ubuntu:24.04
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl jq ripgrep python3 python3-venv python3-pip build-essential less procps openssl ffmpeg \
      libgomp1 libglib2.0-0t64 libx11-6 libxext6 libxrender1 libxt6t64 libxrandr2 libxi6 libxinerama1 libxcursor1 \
      libsm6 libice6 libgl1 libegl1 libglu1-mesa libopengl0 libvulkan1 libxkbcommon0 libnss3 libgfortran5 \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @openai/codex@0.162.1 \
 && pip install --no-cache-dir --break-system-packages uv==0.12.8 \
 && git config --system --add safe.directory '*' \
 && mkdir -m 1777 /home/agent /home/agent/.codex \
 && mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
 && echo '{"file_format_version": "1.0.1", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.4.312"}}' \
      > /usr/share/vulkan/icd.d/nvidia_icd.json \
 && echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV HOME=/home/agent
WORKDIR /workspace
EOF

# Yolo: no confirmations and no Codex sandbox (the container is the boundary: it sees only the folder).
codex=(codex -c model_provider=selfhosted -c "model=\"$CODEX_MODEL\""
  -c model_providers.selfhosted.name=selfhosted -c "model_providers.selfhosted.base_url=\"$LLM_URL\""
  -c model_providers.selfhosted.wire_api=responses -c model_providers.selfhosted.requires_openai_auth=false
  -c tui.screen_reader_detection_done=true)
if [[ ${1:-} == tui ]]; then
  codex+=(--dangerously-bypass-approvals-and-sandbox)
else
  prompt="$*"
  [[ -n $prompt ]] || prompt=${AGENT_PROMPT:-}
  task=$folder/${AGENT_TASK_FILE:-TASK.md}
  [[ -n $prompt || ! -f $task ]] || prompt=$(cat "$task")
  [[ -n $prompt ]] || { echo "no task: give a prompt, AGENT_PROMPT, or a $task" >&2; exit 2; }
  codex+=(exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "$prompt")
fi

tty=(); [[ -t 0 && -t 1 ]] && tty=(-it)
gpus=(); nvidia-smi &>/dev/null && gpus=(--gpus all)
exec docker run --rm "${tty[@]}" "${gpus[@]}" --user "$(id -u):$(id -g)" -v "$folder:/workspace" \
  -v "$bundle:/etc/ssl/certs/ca-certificates.crt:ro" -e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "${envs[@]}" "$image" "${codex[@]}"
