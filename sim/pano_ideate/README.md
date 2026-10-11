# sim/pano_ideate

Work out the interactions a scene needs from one equirectangular panorama: walk around it in the browser, pick objects up, put them back, open and close doors and drawers, and see each end state rendered. Then download the session and turn it into simulation assets:

1. Open the viewer, upload a panorama, act on its objects (below).
2. Click **Download session**: `<scene>_session.zip` holds the interacted objects as last seen (every view, their 3D points, each state of an articulated one), the room with them erased, and `interactions.json`, the interactions in order.
3. Generate the objects with [`asset_gen`](../asset_gen/README.md).
4. Generate the background splat with [`colmap_splat`](../colmap_splat/README.md#from-a-pano_ideate-session).

Every model runs on [Triton](../../services/triton/README.md), so this folder runs anywhere that reaches it, a laptop included, without a GPU. Each script is a uv script; data lives in `datasets/pano_ideate/<scene>/` (gitignored).

## Set up the model endpoints

| Variable | Default model | Used for | Licence |
|---|---|---|---|
| `VLM_ENDPOINT` | Qwen3.5-122B-A10B | listing objects, planning actions | Apache-2.0 |
| `EDIT_ENDPOINT` | Qwen-Image-Edit-2511 | rendering end states, removing and placing objects | Apache-2.0 |
| `DEPTH_ENDPOINT` | moge-3-vitg | metric depth | MIT |
| `DETECT_ENDPOINT` | llmdet_large | finding objects | Apache-2.0 |
| `SEGMENT_ENDPOINT` | sam2.1-hiera-large | outlining objects | Apache-2.0 |

All five are served on the flywheel Triton; to serve them elsewhere, see [Triton](../../services/triton/README.md#serve-a-model). Leave a variable unset to use the default model on the flywheel Triton, or set it to another model name there or to a full URL, `https://<host:port>/v2/models/<model>`. Reach the flywheel Triton as in [quickstart.md](../../quickstart.md) steps 1 to 3: the environment with `TRITON_TOKEN`, the CA, and from a laptop the SSH tunnel. Its URL is `triton.$SERVICE_HOST:$CADDY_PORT` unless `TRITON_URL` says otherwise.

## Walk around a scene and change it

Start the viewer, then open `http://localhost:8765`.

```bash
uv run sim/pano_ideate/serve.py
```

- **Add a panorama** in the panel: any 360° image twice as wide as tall. It runs steps 1 and 2 (about 4 min) with a progress bar, then opens.
- **Move:** drag to look, WASD or the arrow keys to move, scroll to zoom. The eye stays within 0.4 m of the capture point, shown as a circle on the floor; further away the holes behind objects show.
- **Act:** hover to see an object (movable in orange, articulated in cyan), click it and type what happens ("pick it up", "open it"). Expect about 30 s per rendered change, 1 s for a state seen before.
- **Inventory:** what you pick up waits there; pick an item, click where it goes, and it is placed (about 35 s).
- **Edits:** stack, step back through them (click one, or `[` and `]`), Undo, Reset.
- **Download session** in the panel: the zip for steps 3 and 4 above. It erases the interacted objects still in the room first (about 30 s each, nearby ones together).

Notes:
- Each scene's state is saved to `<scene>/edits/` after every edit and resumed on restart; Reset deletes it. `session.py <scene>` writes the session zip from it without the viewer.
- An articulated object remembers each state's view, contents included: close the fridge with a bowl inside, open it, and the bowl is there. Changes made in that area while it was in the other state are lost.
- `--host 0.0.0.0` serves other machines; `--radius` changes the walking limit.

## Run the steps from the command line

The viewer runs these itself; run them by hand to check or redo one.

| Step | Script | Writes |
|---|---|---|
| 1. Depth and scene frame | `depth.py` | `depth/` |
| 2. Objects | `objects.py` | `objects/` |
| An action's end state | `act.py` | `act/<action>/` |
| The session, from the viewer's edits | `session.py` | `session.zip` |

1. Put the panorama at `datasets/pano_ideate/<scene>/source.jpg`. The two samples are Poly Haven HDRIs (CC0):
   ```bash
   for p in "kitchen blinds" "office poly_haven_studio"; do set -- $p; mkdir -p datasets/pano_ideate/$1
     curl -fL -o datasets/pano_ideate/$1/source.jpg "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/$2.jpg"; done
   ```
2. Estimate depth and the scene frame (about 2 min, on the server). Check `depth/floor.png`: walls straight, floor flat; expect the camera 1.2 m above the floor for `office`, 1.4 m for `kitchen`.
   ```bash
   uv run sim/pano_ideate/depth.py datasets/pano_ideate/office
   ```
3. Find the objects (about 1 min). Check `objects/labels.jpg`: movable objects in orange, articulated ones in cyan.
   ```bash
   uv run sim/pano_ideate/objects.py datasets/pano_ideate/office
   ```
4. Render an action's end state. Check `act/<action>/crops/<n>.jpg` (each object's view: before, outlined, the editor's output, after).
   ```bash
   uv run sim/pano_ideate/act.py datasets/pano_ideate/kitchen "open the fridge"
   ```

Notes:
- The scene frame is Arena's: z up, metres, origin on the floor below the capture point, +x towards the middle of the panorama. The scale is MoGe's estimate; `depth.py --camera_height <m>` sets it.
- Object kinds: movable = could be picked up, pushed or carried and likely to move from one day to the next; articulated = stays in place but has a part to open, close or turn; static = the rest. Movable also needs at most 1.5 m across and 1 cm thick (`--max_size`, `--min_thickness`).
- Expect duplicates (one fridge as several entries), odd labels and the occasional object that isn't there; `act.py` and the viewer merge an object's entries when they plan.
- Detections and checks are cached in `objects/`; `objects.py --redetect` asks the models again.
- `act.py` reuses `act/<action>/plan.json` on a rerun (`--replan` asks again). A removal pastes back the object's surroundings, shading-matched; a new state pastes back only what changed near the object.
