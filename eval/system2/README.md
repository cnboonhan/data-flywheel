# eval/system2

Agentic eval environments (system 2).

| Path | Contents |
|---|---|
| `IsaacLab-Arena` | [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) git submodule (Isaac Sim 6.1, uv `.venv`) |

## Install

Install Arena into `IsaacLab-Arena/.venv` (Isaac Sim 6.1; x86_64 or aarch64).

```bash
bash eval/system2/setup-arena.sh
```

Before pulling Arena, restore its lock files, then rerun the script:

```bash
git -C eval/system2/IsaacLab-Arena update-index --no-skip-worktree pyproject.toml uv.lock
git -C eval/system2/IsaacLab-Arena checkout pyproject.toml uv.lock
```

**Notes**
- Upstream locks for Linux x86_64 only and its lockfile is stale. `setup-arena.sh` adds the current platform to `[tool.uv] environments`, regenerates `uv.lock` and marks both files `skip-worktree`, so they stay out of commits.
