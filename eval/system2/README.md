# eval/system2

Agentic eval environments (system 2).

| Path | Contents |
|---|---|
| `IsaacLab-Arena` | [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) git submodule (Isaac Sim 6.1, uv `.venv`) |
| `environments/` | Environment specs (YAML) generated with [`sim/isaaclab_arena/envgen.sh`](../../sim/isaaclab_arena/README.md). Example: `franka_mug_to_bowl_splat_scene.yaml`, on a [splat background](../../sim/isaaclab_arena/README.md#splat-backgrounds) |

## Generate environments

See [`sim/isaaclab_arena/`](../../sim/isaaclab_arena/README.md): `envgen.sh` runs Arena's agentic environment generation through a local LLM proxy and writes specs to `environments/`.

## Install

```bash
bash eval/system2/setup-arena.sh   # IsaacLab-Arena/.venv, Isaac Sim 6.1; x86_64 or aarch64
```

**Gotchas**
- **Arena `pyproject.toml` and `uv.lock`:** upstream locks for Linux x86_64 only and its lockfile is stale. `setup-arena.sh` adds the current platform to `[tool.uv] environments`, regenerates `uv.lock` and marks both `skip-worktree`, so they stay out of commits. Before pulling Arena: `git -C eval/system2/IsaacLab-Arena update-index --no-skip-worktree pyproject.toml uv.lock && git -C eval/system2/IsaacLab-Arena checkout pyproject.toml uv.lock`, then rerun the script.
