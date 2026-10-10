# eval

Evaluation benchmarks, all git submodules. Fetch them:

```bash
git submodule update --init --recursive
```

| Benchmark | Path | Simulator | Env |
|---|---|---|---|
| [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) | [`system1/RoboDojo`](system1/README.md) | Isaac Sim 5.1 | conda `RoboDojo` |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) | [`system1/RoboTwin`](system1/README.md) | SAPIEN | conda `RoboTwin` |
| [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) | [`system2/IsaacLab-Arena`](system2/README.md) | Isaac Sim 6.1 | uv `.venv` |

- [`system1/`](system1/README.md): action models (VLA policies)
- [`system2/`](system2/README.md): agentic models

**Notes**
- **Driver:** NVIDIA 580-open is required, because Isaac Sim 5.1 crashes on 595. It's held with `apt-mark hold`.
- **Upstream code:** don't commit inside the submodules. Changes belong on a fork; this repo only records each submodule's commit.
