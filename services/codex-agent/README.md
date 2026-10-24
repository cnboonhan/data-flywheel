# codex-agent

An autonomous coding agent on the stack's self-hosted LLM: OpenAI's [Codex CLI](https://github.com/openai/codex) (Apache-2.0) in a container, pointed at the [LLM gateway](../llm-gateway/README.md) (`kimi-k3` by default), working in a folder you mount, in yolo mode (no confirmations, no Codex sandbox: the container is the boundary and sees only that folder). Give it a task; it reads, edits, runs commands and tests in the folder until done, as ASPIRE's agents do in [autolearn](../../sensors/autolearn/README.md).

## Run

1. Build the image on a node with Docker (once, and after editing anything here):
   ```bash
   docker build -t flywheel-codex-agent:0.162.1 services/codex-agent
   ```
2. Put the task in the folder as `TASK.md` (or pass it as arguments), then run the agent on the folder:
   ```bash
   services/codex-agent/run.sh <folder>                        # the task in <folder>/TASK.md
   services/codex-agent/run.sh <folder> "fix the failing tests"   # the task as arguments
   services/codex-agent/run.sh <folder> tui                     # Codex's interactive UI on the folder
   ```
3. Read the result in the folder; the run's log is the command's output. Expect a small task (a script, run, its output saved) in under a minute.

| Setting | Default | What it does |
|---|---|---|
| `CODEX_MODEL` | `kimi-k3` | any model the gateway serves (`curl $LLM_URL/models`) |
| `LLM_URL` | the stack's gateway | the OpenAI-compatible endpoint (`/v1`) |
| `FLYWHEEL_CA` | the admin's `services/.env` | the stack's CA ([quickstart](../../quickstart.md)) |
| `AGENT_PROMPT`, `AGENT_TASK_FILE` | `TASK.md` | the task when no arguments are given |
| `CODEX_AGENT_IMAGE` | `flywheel-codex-agent:0.162.1` | the image |

## Notes

- **Config:** [`config.toml`](config.toml) is the quickstart's Codex config for the self-hosted LLM plus yolo mode (`approval_policy = "never"`, `sandbox_mode = "danger-full-access"`); `CODEX_MODEL` and `LLM_URL` override it per run.
- **What it can touch:** the mounted folder (as you: what it writes is yours) and the network. Nothing else of the host is mounted, but it runs any command the model decides on there, so point it at a copy or a git checkout you can reset, never at shared data.
- **Pinned:** Codex 0.162.1 (the newest release at least two weeks old when added), Node 22.20.0, uv 0.12.8; bump them in the [`Dockerfile`](Dockerfile) and rebuild.
