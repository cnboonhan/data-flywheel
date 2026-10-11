# sim/asset_gen

Simulation assets of the objects in a [`pano_ideate`](../pano_ideate/README.md) session: each object that was picked up or placed becomes a textured mesh with physics, generated from every view of it the session recorded. The room around them becomes a splat in [`colmap_splat`](../colmap_splat/README.md#from-a-pano_ideate-session).

1. In the pano_ideate viewer, act on the objects, then click **Download session** (`<scene>_session.zip`).
2. Set up the model endpoint (below).
3. Generate the objects.
   ```bash
   uv run sim/asset_gen/objects.py <scene>_session.zip
   ```

Each object lands in `datasets/asset_gen/<scene>/objects/<asset>/` (`--out` changes the root):

| File | What |
|---|---|
| `model.glb` | the mesh, glTF (y up) |
| `model.usda` | the mesh as USD (z up, metres): a rigid body with convex-decomposition collision, a mass from its volume, `texture.png` through UsdPreviewSurface |
| `pose.json` | where it stands in the session's scene frame (`scene.json`) and how it is turned, its size and mass, how well its silhouette matched |

Both meshes are in the object's own frame, origin at the centre of its base. `--asset <name>` (as in the zip's `objects/`) generates only that one.

## Model endpoint

| Variable | Default | Model | Licence |
|---|---|---|---|
| `MESH_ENDPOINT` | `TRELLIS.2` on the flywheel Triton | [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), image to textured mesh | code MIT; see Notes |

Set the variable to a model name on the flywheel Triton or to a full URL, `https://<host:port>/v2/models/<model>`. The flywheel Triton is `TRITON_URL`, else `triton.$SERVICE_HOST:$CADDY_PORT`, with `TRITON_TOKEN` and `AWS_CA_BUNDLE`, set as in [quickstart.md](../../quickstart.md). To serve a model there, register it and set its `triton` alias ([Triton](../../services/triton/README.md#serve-a-model)).

## Notes

- **TRELLIS.2 is not served yet.** Its image encoder, Meta's DINOv3, is gated under a custom licence that needs Hugging Face access approval, and its GLB texture baking uses nvdiffrast, licensed for non-commercial use only. Until it is, `objects.py` stops at the first object with Triton's error.
- The fit: the mesh is scaled to the object's measured height, turned about the vertical axis in 5° steps until its silhouette from the capture point best overlaps the object's last-seen 3D points (`silhouette_overlap` in `pose.json`; low values mean a poor turn), and stood where it was last seen.
- More views make better meshes: each pick-up in the viewer records the object from where it stood, so picking an object up after moving it adds a view.
- The mass assumes 400 kg/m³ over the convex hull, clamped to 0.02–50 kg; set it in `model.usda` when you know it.
- Articulated objects (in the zip's `articulated/`: each state's view and area) are not generated yet.
