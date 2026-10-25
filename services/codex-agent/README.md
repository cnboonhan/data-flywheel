# codex-agent

An autonomous coding agent on the stack's self-hosted LLM: OpenAI's [Codex CLI](https://github.com/openai/codex) (Apache-2.0) in a container, pointed at the [LLM gateway](../llm-gateway/README.md) (`kimi-k3` by default), working in a folder you mount, in yolo mode (no confirmations, no Codex sandbox: the container is the boundary and sees only that folder). Give it a task; it reads, edits, runs commands and tests in the folder until done. Everything is in [`run.sh`](run.sh), including the image, which it builds on first use.

## Run

1. Put the task in the folder as `TASK.md` (or pass it as arguments), and any settings or variables for the agent in `<folder>/.env`.
2. Run the agent on the folder, on a node with Docker:
   ```bash
   services/codex-agent/run.sh <folder>                           # the task in <folder>/TASK.md
   services/codex-agent/run.sh <folder> "fix the failing tests"   # the task as arguments
   services/codex-agent/run.sh <folder> tui                       # Codex's interactive UI on the folder
   ```
3. Read the result in the folder; the run's log is the command's output. Expect a small task (a script, run, its output saved) in under a minute.

| Setting | Default | What it does |
|---|---|---|
| `CODEX_MODEL` | `kimi-k3` | any model the gateway serves (`curl $LLM_URL/models`) |
| `LLM_URL` | the stack's gateway | the OpenAI-compatible endpoint (`/v1`) |
| `FLYWHEEL_CA` | `~/.flywheel/ca.crt` | the stack's CA, as in the [quickstart](../../quickstart.md); fetched from the gateway on first use |
| `AGENT_PROMPT`, `AGENT_TASK_FILE` | `TASK.md` | the task when no arguments are given |
| `CODEX_AGENT_IMAGE` | `flywheel-codex-agent:0.162.1-noble` | the image; `docker rmi` it to rebuild |

Set them in the environment or in `<folder>/.env`, which wins. Every variable in `<folder>/.env` also reaches the agent's environment.

## On Slurm

Import the image once on a node with Docker, then run it with Pyxis:

```bash
enroot import -o codex-agent.sqsh dockerd://flywheel-codex-agent:0.162.1-noble   # ENROOT_*_PATH under /tmp/$USER
srun --gpus=1 --container-image=codex-agent.sqsh --no-container-mount-home --container-mounts=<src>:<dst>[:ro] \
  --export=USER=$USER,NVIDIA_VISIBLE_DEVICES=all,NVIDIA_DRIVER_CAPABILITIES=all <command>
```

## Notes

- **Certificates:** the agent trusts the host's CAs (`/etc/ssl/certs/ca-certificates.crt`, as `update-ca-certificates` saves them) plus `FLYWHEEL_CA`. If that file is missing, `run.sh` downloads it from `https://<gateway>/ca.crt` and prints its fingerprint (trust on first use); compare it with the admin's if in doubt.
- **What it can touch:** the mounted folder (as you: what it writes is yours), the GPUs when the node has any, and the network. Nothing else of the host is mounted, but it runs any command the model decides on there, so point it at a copy or a git checkout you can reset, never at shared data.
- **Pinned:** Codex 0.162.1 (the newest release at least two weeks old when added), Node 22.20.0, uv 0.12.8; bump them in `run.sh`, change the image tag and run again.
