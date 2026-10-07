# eval/system2

Agentic eval environments (system 2).

| Path | Contents |
|---|---|
| `IsaacLab-Arena` | [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) git submodule (Isaac Sim 6.1, uv `.venv`) |
| `environments/` | Environment specs (YAML) generated with `scripts/arena_envgen.sh` |

## Generate environments

[`scripts/arena_envgen.sh`](../../scripts/arena_envgen.sh) runs Arena's agentic environment generation through a local OpenAI-compatible proxy (CLIProxyAPI):

```bash
bash scripts/arena_envgen.sh cli --mode resolve --prompt "..."   # write a YAML spec
bash scripts/arena_envgen.sh cli --mode build --env_spec eval/system2/environments/<env>.yaml
bash scripts/arena_envgen.sh gui                                  # live editor on http://localhost:8501
```

Specs go to `environments/` unless `--out_dir` is given. Set `OPENAI_API_KEY` (the proxy's client key); `ARENA_PROXY_BASE_URL` (default `http://127.0.0.1:8317/v1`) and `ARENA_PROXY_MODEL` (default `claude-sonnet-5-5`) are optional.

**Gotchas**
- **Arena `uv.lock`:** upstream's lockfile is stale. Regenerate it locally and keep it out of commits:
  ```bash
  cd eval/system2/IsaacLab-Arena && uv lock && uv sync --extra dev
  git update-index --skip-worktree uv.lock   # run once per clone; undo with --no-skip-worktree before pulling
  ```
