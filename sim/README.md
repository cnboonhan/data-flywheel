# sim

Simulation tooling that sits on top of the benchmarks in `eval/`.

| Path | Contents |
|---|---|
| [`splat/`](splat/README.md) | Gaussian splats of real environments with 3DGRUT (submodule, uv): COLMAP images in, Isaac Sim `ParticleField` USD out |
| [`isaaclab_arena/`](isaaclab_arena/README.md) | IsaacLab-Arena helpers: LLM-driven environment generation through a local proxy (`envgen.sh`, `hooks/`), splat backgrounds |
| [`pano_arena/`](pano_arena/README.md) | IsaacLab-Arena scenes from one panorama: metric depth, interactable objects as 3D assets, the room as a splat background (uv, step by step) |
