# sim

Simulation tooling that sits on top of the benchmarks in `eval/`.

| Path | Contents |
|---|---|
| [`colmap_splat/`](colmap_splat/README.md) | Gaussian splats of real environments with 3DGRUT (submodule, uv): COLMAP images in, Isaac Sim `ParticleField` USD out |
| [`isaaclab_arena/`](isaaclab_arena/README.md) | IsaacLab-Arena helpers: LLM-driven environment generation through a local proxy (`envgen.sh`, `hooks/`), splat backgrounds |
| [`pano_splat/`](pano_splat/README.md) | IsaacLab-Arena scenes from one panorama: metric depth, interactable objects, movable objects erased, actions rendered as end states, a browser viewer to walk the room and edit objects (uv, step by step) |
