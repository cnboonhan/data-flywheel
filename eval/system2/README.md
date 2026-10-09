# eval/system2

Agentic eval environments (system 2).

| Path | Contents |
|---|---|
| `IsaacLab-Arena` | [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) git submodule (Isaac Sim 6.1, uv `.venv`) |
| `environments/` | Environment specs (YAML) generated with [`sim/isaaclab_arena/envgen.sh`](../../sim/isaaclab_arena/README.md). Example: `franka_mug_to_bowl_splat_scene.yaml`, on a [splat background](../../sim/isaaclab_arena/README.md#splat-backgrounds) |

## Generate environments

See [`sim/isaaclab_arena/`](../../sim/isaaclab_arena/README.md): `envgen.sh` runs Arena's agentic environment generation through a local LLM proxy and writes specs to `environments/`.

**Gotchas**
- **Arena `uv.lock`:** upstream's lockfile is stale. Regenerate it locally and keep it out of commits:
  ```bash
  cd eval/system2/IsaacLab-Arena && uv lock && uv sync --extra dev
  git update-index --skip-worktree uv.lock   # run once per clone; undo with --no-skip-worktree before pulling
  ```
