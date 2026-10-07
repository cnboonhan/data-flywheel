# sim

Simulation tooling that sits on top of the benchmarks in `eval/`.

| Path | Contents |
|---|---|
| [`isaaclab_arena/`](isaaclab_arena/README.md) | IsaacLab-Arena helpers: LLM-driven environment generation through a local proxy (`envgen.sh`, `cliproxy/`) |
| [`nurec/`](nurec/README.md) | NVIDIA NuRec real-to-sim pipeline: capture → Gaussian-splat reconstruction (3DGRUT, git submodule at `nurec/3dgrut`) → sim-ready USD → IsaacLab-Arena background |
