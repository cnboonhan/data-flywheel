# sim/isaaclab_arena

Helpers around the [IsaacLab-Arena](../../eval/system2/README.md) submodule. Nothing here modifies the submodule; everything hooks in from outside.

| Path | Purpose |
|---|---|
| `envgen.sh` | Runs Arena's agentic environment generation (CLI runner or Streamlit GUI) through a local OpenAI-compatible proxy |
| `cliproxy/sitecustomize.py` | Adds a `cliproxy` entry to Arena's fixed list of inference endpoints. Loaded via `PYTHONPATH` by `envgen.sh`; registers through an import hook so it also applies inside the GUI's separate process |

## Generate environments

```bash
bash sim/isaaclab_arena/envgen.sh cli --mode resolve --prompt "..."   # write a YAML spec
bash sim/isaaclab_arena/envgen.sh cli --mode build --env_spec eval/system2/environments/<env>.yaml
bash sim/isaaclab_arena/envgen.sh gui                                  # live editor on http://localhost:8501
```

Specs go to [`eval/system2/environments/`](../../eval/system2/environments/) unless `--out_dir` is given.

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | The proxy's client key (required) | |
| `ARENA_PROXY_BASE_URL` | Proxy endpoint | `http://127.0.0.1:8317/v1` |
| `ARENA_PROXY_MODEL` | Model name the proxy serves | `claude-sonnet-5-5` |

The proxy is [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI), which exposes a CLI-authenticated model over the OpenAI API. `build` mode only needs Isaac Sim (it instantiates an existing spec); `resolve` and the GUI need the proxy running.
