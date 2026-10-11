# sim/asset_gen

Simulation assets of the objects in a [`pano_ideate`](../pano_ideate/README.md) session: each object that was picked up or placed becomes a coloured mesh with physics, generated freely rather than reconstructed: a VLM says what the object most likely is, an image editor draws a clean, typical one, and an image-to-3D model makes its shape. The session sets only its size and place. The room around them becomes a splat in [`colmap_splat`](../colmap_splat/README.md#from-a-pano_ideate-session).

1. In the pano_ideate viewer, act on the objects, then click **Download session** (`<scene>_session.zip`).
2. Set up the model endpoints (below).
3. Generate the objects.
   ```bash
   uv run sim/asset_gen/objects.py <scene>_session.zip
   ```

Each object lands in `datasets/asset_gen/<scene>/objects/<asset>/` (`--out` changes the root):

| File | What |
|---|---|
| `model.glb` | the mesh with vertex colours, glTF (y up) |
| `model.usda` | the mesh as USD (z up, metres): a rigid body with convex-decomposition collision, a mass from its volume, its colours as `displayColor` |
| `views/` | the session's view (`cutout`) and the drawings (`front`, `back`); check these first when an object looks wrong |
| `pose.json` | where it stands in the session's scene frame (`scene.json`) and how it is turned, its size and mass, how well its silhouette matched, and what it was generated as (`generated_as`) |

Both meshes are in the object's own frame, origin at the centre of its base. `--asset <name>` (as in the zip's `objects/`) generates only that one; `--seed <n>` draws another object when one comes out wrong.

## Model endpoints

| Variable | Default model | Used for | Licence |
|---|---|---|---|
| `VLM_ENDPOINT` | Qwen3.5-122B-A10B | what the object is and how a typical one looks | Apache-2.0 |
| `EDIT_ENDPOINT` | Qwen-Image-Edit-2511, with the camera-angle LoRA | drawing the object, front and back | Apache-2.0 (LoRA [fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) Apache-2.0) |
| `SEGMENT_ENDPOINT` | sam2.1-hiera-large | cutting the drawings out | Apache-2.0 |
| `MESH_ENDPOINT` | TripoSG | the shape, from the front drawing | MIT |

Leave a variable unset to use the default model on the flywheel Triton, or set it to another model name there or to a full URL, `https://<host:port>/v2/models/<model>`. The flywheel Triton is `TRITON_URL`, else `triton.$SERVICE_HOST:$CADDY_PORT`, with `TRITON_TOKEN` and `AWS_CA_BUNDLE`, set as in [quickstart.md](../../quickstart.md). An editor served elsewhere needs the LoRA loaded as `angles` ([Triton](../../services/triton/README.md#serve-a-model)).

## Notes

- **Generated, not reconstructed:** the object is a typical one of its kind in the session's colours, not a copy: printed labels, dents and dirt are dropped, and hidden parts are invented. The session's view is only a loose reference for the drawing.
- **Colours:** each drawing's camera is found by matching the mesh's silhouette to the drawing's (the back drawing's turn is the LoRA's, about 180°), and each vertex takes the colour where a drawing sees it; vertices neither sees take the nearest coloured one's. TripoSG makes shape only.
- The fit: the mesh is scaled to the object's measured height, turned about the vertical axis in 5° steps until its silhouette from the capture point best overlaps the object's last-seen 3D points (`silhouette_overlap` in `pose.json`; low values mean a poor turn), and stood where it was last seen.
- The mass assumes 400 kg/m³ over the convex hull, clamped to 0.02–50 kg; set it in `model.usda` when you know it.
- Articulated objects (in the zip's `articulated/`: each state's view and area) are not generated yet.
