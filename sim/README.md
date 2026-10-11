# sim

Simulation tooling that sits on top of the benchmarks in `eval/`.

| Path | Contents |
|---|---|
| [`colmap_splat/`](colmap_splat/README.md) | Gaussian splats of real environments with 3DGRUT (submodule, uv): COLMAP images in, Isaac Sim `ParticleField` USD out |
| [`isaaclab_arena/`](isaaclab_arena/README.md) | IsaacLab-Arena helpers: LLM-driven environment generation through a local proxy (`envgen.sh`, `hooks/`), splat backgrounds |
| [`pano_ideate/`](pano_ideate/README.md) | Interactions worked out on one panorama in the browser (walk, pick up, place, open and close objects, each end state rendered; models on Triton), exported as a session zip |
| [`asset_gen/`](asset_gen/README.md) | Simulation assets of a session's objects: meshes with physics (USD, GLB); its background goes to `colmap_splat` |
