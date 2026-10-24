# An autonomous coding agent on the stack's self-hosted LLM (services/codex-agent/README.md): OpenAI's Codex CLI
# (Apache-2.0) in yolo mode, working in whatever folder is mounted at /workspace, with the tools such work usually
# needs (git, Python with uv, ripgrep, curl, jq, build tools). Codex 0.162.1: the newest release at least two weeks
# old when this was written; the Node 22 base is what the Codex npm package runs on.
FROM node:22.20.0-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl jq ripgrep python3 python3-venv python3-pip build-essential less procps \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @openai/codex@0.162.1 \
 && pip install --no-cache-dir --break-system-packages uv==0.12.8 \
 && git config --system --add safe.directory '*' \
 && mkdir -m 1777 /home/agent   # the folder belongs to the host user, not the container's; HOME for any uid (Codex won't
                                # put its helper commands under /tmp)
COPY config.toml /etc/codex/config.toml
COPY entrypoint.sh /usr/local/bin/codex-agent
ENV HOME=/home/agent CODEX_HOME=/home/agent/.codex
WORKDIR /workspace
ENTRYPOINT ["codex-agent"]
