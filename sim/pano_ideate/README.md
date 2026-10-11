# sim/pano_ideate

Turn one 360° photo of a room into a simulation scene: walk around the room in the browser, pick objects up, put them back, open and close doors and drawers, and see each end state rendered. Then download the session and generate its objects as textured meshes with physics and the room around them as a Gaussian splat. Every model runs on [Triton](../../services/triton/README.md), so the viewer and the object generation run anywhere that reaches it, a laptop included, without a GPU; only the splat training needs one.

## From a panorama to a simulation scene

1. Set up the model endpoints (below).
2. Start the viewer, open `http://localhost:8765` and add the panorama in the panel (about 4 min).
   ```bash
   uv run sim/pano_ideate/serve.py
   ```
3. Act on the objects: click one and type what happens ("pick it up", "open the fridge"). For an object you want in the simulation without acting on it, click it and **Add to session**.
4. Click **Download session** (`<scene>_session.zip`): it erases the session's objects from the room first, about 30 s for each group of nearby ones.
5. Generate the objects ([asset_gen](../asset_gen/README.md)), ticking the ones you want in the checklist (about 4 min each).
   ```bash
   uv run sim/asset_gen/objects.py <scene>_session.zip
   ```
6. Generate the background splat ([colmap_splat](../colmap_splat/README.md#from-a-pano_ideate-session)): write its dataset anywhere (~7 min), train it on a machine with an NVIDIA GPU (~3 min on a GB300).
   ```bash
   uv run sim/colmap_splat/from_pano.py <scene>_session.zip datasets/colmap_splat/<scene>
   uv run sim/colmap_splat/train.py datasets/colmap_splat/<scene> datasets/colmap_splat/<scene>/runs --floor
   ```

What you get, all in the session's scene frame (z up, metres, origin on the floor below the capture point):

| | Where | What |
|---|---|---|
| Objects | `datasets/asset_gen/<scene>/objects/<asset>/model.usda` | a textured rigid body per object, origin at the centre of its base |
| Layout | `layout.json` in the session zip | each object's x, y now (null if picked up and not put back) |
| Interactions | `interactions.json` in the session zip | the pick-ups, placements and state changes, in order |
| Background | `datasets/colmap_splat/<scene>/runs/.../scene.usda` | the room without the objects, for Isaac Sim / Arena |

Check before using a scene:
- **The background:** look at the zip's `background/pano.jpg`. The export erases an object again on its own when the VLM still sees it, and logs `WARNING: the <object> may still be in the background` when that fails too; an object left in the background appears twice in the scene.
- **Each object:** look at its `views/` (the drawing and the painted sides). `--seed <n>` draws another one.
- **Heights:** `layout.json` has no height and an object seen only in part (a chair behind a table) is scaled to the part seen: stand each asset on the surface below it and check its size.

Each script is a uv script; data lives in `datasets/pano_ideate/<scene>/` (gitignored).

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
- **Add to session** on an object's card makes it an asset without acting on it: it is erased from the background and generated like the interacted ones. Click again to take it out.
- **Download session** in the panel: the zip for steps 5 and 6 above. It erases the session's objects still in the room first (about 30 s for each group of nearby ones, plus a retry for each the VLM still sees).

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
