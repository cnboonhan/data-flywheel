# sim

Simulation tooling that sits on top of the benchmarks in `eval/`.

| Path | Contents |
|---|---|
| [`isaaclab_arena/`](isaaclab_arena/README.md) | IsaacLab-Arena helpers: LLM-driven environment generation through a local proxy (`envgen.sh`, `cliproxy/`) |
| [`worldgen/`](worldgen/README.md) | World-generation loop prototype: Nova Carter explores a NuRec room under Nav2 + slam_toolbox (ROS 2 in Docker), recording the stereo bag for reconstruction |
| [`nurec/`](nurec/README.md) | NVIDIA NuRec real-to-sim pipeline: capture → Gaussian-splat reconstruction (3DGRUT, git submodule at `nurec/3dgrut`) → sim-ready USD → IsaacLab-Arena background |
