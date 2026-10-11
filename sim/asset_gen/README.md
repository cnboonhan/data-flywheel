# sim/asset_gen

Simulation assets of the objects in a [`pano_ideate`](../pano_ideate/README.md) session: each object in the session (acted on, or added with **Add to session**) becomes a textured mesh with physics, generated freely rather than reconstructed: a VLM says what the object most likely is, an image editor draws a clean, typical one, an image-to-3D model makes its shape, and the editor paints renders of that shape for its texture. The session sets only its size and place. The room around them becomes a splat in [`colmap_splat`](../colmap_splat/README.md#from-a-pano_ideate-session).

1. In the pano_ideate viewer, act on the objects, then click **Download session** (`<scene>_session.zip`).
2. Set up the model endpoints (below).
3. Generate the objects.
   ```bash
   uv run sim/asset_gen/objects.py <scene>_session.zip
   ```

Each object lands in `datasets/asset_gen/<scene>/objects/<asset>/` (`--out` changes the root):

| File | What |
|---|---|
| `model.glb` | the textured mesh, glTF (y up) |
| `model.usda`, `texture.png` | the mesh as USD (z up, metres): a rigid body with convex-decomposition collision, a mass from its volume, a `UsdPreviewSurface` reading the texture (1024 px) |
| `views/` | the session's view (`cutout`), the `drawing`, and each side's grey render and painting (`render_<side>`, `paint_<side>`); check these first when an object looks wrong |
| `pose.json` | where it stands in the session's scene frame (`scene.json`) and how it is turned, its size and mass, how well its silhouette matched, and what it was generated as (`generated_as`: name, description, colours) |

Both meshes are in the object's own frame, origin at the centre of its base. On launch a checklist asks which objects to generate: tick with space, start with Enter; objects generated before start unticked. `--asset <name>` (repeat for more; a name as in the zip's `objects/`, such as `02_chair`, or a label, such as `chair`) picks them without the checklist, and a run outside a terminal generates every one; `--seed <n>` draws another object when one comes out wrong.

## Model endpoints

| Variable | Default model | Used for | Licence |
|---|---|---|---|
| `VLM_ENDPOINT` | Qwen3.5-122B-A10B | what the object is, how a typical one looks, its colours in words | Apache-2.0 |
| `EDIT_ENDPOINT` | Qwen-Image-Edit-2511 | drawing the object, painting renders of its shape | Apache-2.0 |
| `SEGMENT_ENDPOINT` | sam2.1-hiera-large | cutting the drawing out | Apache-2.0 |
| `MESH_ENDPOINT` | TripoSG | the shape, from the drawing | MIT |

Leave a variable unset to use the default model on the flywheel Triton, or set it to another model name there or to a full URL, `https://<host:port>/v2/models/<model>`. The flywheel Triton is `TRITON_URL`, else `triton.$SERVICE_HOST:$CADDY_PORT`, with `TRITON_TOKEN` and `AWS_CA_BUNDLE`, set as in [quickstart.md](../../quickstart.md).

## Notes

- **Generated, not reconstructed:** the object is a typical one of its kind in the session's colours, not a copy: printed labels, dents and dirt are dropped, and hidden parts are invented. The session's view is only a loose reference for the drawing.
- **Texture** (`texture.py`): TripoSG makes shape only. The mesh is rendered grey from five sides (the drawing's own view, its sides and back, and from above), and the editor paints each render, told in words which side it shows and the drawing's colours (the VLM's reading of the drawing). It is given only the render: given the drawing too, it redraws the drawing's pose instead. A painting whose outline overlaps its render under 0.8 is dropped (good ones overlap 0.84–0.98). The paintings are baked into a 1024 px texture over an [xatlas](https://github.com/jpcy/xatlas) (MIT) unwrap, weighted by how squarely each view sees the surface; what none sees (an underside) takes the nearest painted point's colour. The paintings ask for flat light, so Isaac Sim's lighting isn't doubled by painted highlights.
- Expect ~4 min per object: two VLM calls, six edits, six SAM calls, TripoSG, the bake (~20 s).
- **Size:** an object seen only in part (a chair behind a table) is scaled to the part the session saw; check `size_m` in `pose.json`.
- The fit: the mesh is scaled to the object's measured height, turned about the vertical axis in 5° steps until its silhouette from the capture point best overlaps the object's last-seen 3D points (`silhouette_overlap` in `pose.json`; low values mean a poor turn), and stood where it was last seen.
- The mass assumes 400 kg/m³ over the convex hull, clamped to 0.02–50 kg; set it in `model.usda` when you know it.
- Articulated objects (in the zip's `articulated/`: each state's view and area) are not generated yet.
