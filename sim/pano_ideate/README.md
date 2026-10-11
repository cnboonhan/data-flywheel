# sim/pano_splat

IsaacLab-Arena scenes from a single equirectangular panorama: metric depth, movable objects (chairs, bags, items on surfaces) erased so the sim can place its own, the rest of the room as a Gaussian-splat background. Each step is a uv script that reads and writes `datasets/pano_splat/<scene>/` (gitignored), so its output can be checked before the next step runs.

| Step | Script | Writes |
|---|---|---|
| 1. Depth and frame | `depth.py` | `depth/` |
| 2. Objects | `objects.py` | `objects/` |
| 3. Erase movable objects | `inpaint.py` | `inpaint/` |
| Render an action's end state | `act.py` | `act/<action>/` |
| Walk around and edit objects | `serve.py` | `edits/` |

## Sample panoramas

Two indoor HDRIs from [Poly Haven](https://polyhaven.com/hdris/indoor) (CC0), 8192×4096 tonemapped JPGs:

| Scene | Poly Haven asset | Contents |
|---|---|---|
| `kitchen` | [`blinds`](https://polyhaven.com/a/blinds) | camera just above a counter: jars, a bowl, a tissue box; cabinets, drawers, oven, fridge |
| `office` | [`poly_haven_studio`](https://polyhaven.com/a/poly_haven_studio) | camera on a tripod mid-room: office chairs, a bin, a heater; desks, double doors, a shelf unit |

Fetch them into `datasets/pano_splat/<scene>/source.jpg`.

```bash
for p in "kitchen blinds" "office poly_haven_studio"; do set -- $p; mkdir -p datasets/pano_splat/$1
  curl -fL -o datasets/pano_splat/$1/source.jpg "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/$2.jpg"; done
```

Any other panorama works the same way: put it at `datasets/pano_splat/<scene>/source.jpg`, equirectangular, level.

## 1. Depth and frame

Run [MoGe-3](https://github.com/microsoft/MoGe) metric depth (ViT-g, MIT) on the panorama and set up the scene frame (about 3 min and 15 GB of RAM per scene; the model downloads on first use).

```bash
uv run sim/pano_splat/depth.py datasets/pano_splat/office
```

Check `depth/floor.png` (top-down view, coloured by height, camera as a red dot): walls should be straight and the floor flat. Expect a camera height of 1.2 m for `office` and 1.4 m for `kitchen`.

Notes:
- The frame is Arena's: z up, origin on the floor below the camera, +x towards the middle of the panorama.
- The scale is MoGe's estimate. When you know the camera height, pass `--camera_height <m>`; it rescales everything.
- `points.ply` (coloured points in the scene frame) opens in any point-cloud viewer, such as MeshLab or Rerun.
- MoGe runs on 20 views of `--view_size` (1024 px) and merges them at the panorama's full width; most of the time and memory is the merge. Pass `--merge_width 1920` for about 1 min, with blurrier object edges.

## 2. Objects

Find the objects and mark which are interactable: Qwen3.5-122B-A10B lists and boxes the objects in each view with their kind, LLMDet adds the ones it missed, SAM 2.1 masks them, and Qwen checks each interactable candidate again on its own (all Apache-2.0; LLMDet and SAM download on first use). The VLM runs on [Triton](../../services/triton/README.md): source `slurm.env` first. With `--local_vlm` it loads here instead and needs about 245 GB of GPU memory, one B300; on a smaller GPU add `--vlm Qwen/Qwen3-VL-8B-Instruct`.

```bash
set -a; . $STATE_DIR/slurm.env; set +a
uv run sim/pano_splat/objects.py datasets/pano_splat/office
```

Check `objects/labels.jpg`: movable objects in orange, articulated ones in cyan, each with its label; everything else is dimmed. `objects/objects.json` lists every object with its kind, 3D centre and size (scene frame) and mask. Expect about 1 min per scene.

Notes:
- Kinds: movable = could be picked up, pushed or carried and likely to be moved from one day to the next (chairs, bags, bins, boxes, items on surfaces); articulated = stays in place but has a part to open, close or turn (doors, drawers, cupboard and appliance doors, lids, taps); static = everything else (tables, desks, counters, shelves, sofas, rugs, pet furniture, built-ins, pets). Movable also needs at most `--max_size` (1.5 m) across and at least `--min_thickness` (1 cm) thick.
- Detections of the same object from overlapping views are merged in the panorama and again in 3D (same label, overlapping footprint); runs of identical boxes, which the VLM sometimes produces, are dropped.
- Detections are cached in `objects/detections.json` and the per-object checks in `objects/checks.json`, so changing the thresholds reruns only segmentation and merging. Pass `--redetect` to ask the models again.
- Expect duplicates (one fridge as several entries), odd labels and the occasional object that isn't there; inspect `labels.jpg` before the next steps.

## 3. Erase movable objects

Erase the movable objects from the panorama (the models, ~60 GB, download on first use).

```bash
uv run sim/pano_splat/inpaint.py datasets/pano_splat/kitchen
```

Check `inpaint/before_after.jpg`; `inpaint/pano.jpg` and `inpaint/distance.npy` are the cleaned panorama and its depth.

Notes:
- Small holes use LaMa. Each connected part of a hole over `--large` (3%) of its view goes on its own, tinted red, to Qwen-Image-Edit-2511 with an object-removal LoRA and the 8-step Lightning LoRA (all Apache-2.0). With several objects marked at once, or a red box instead of a tint, the LoRA leaves objects in place.
- Keep Qwen's transformer in bf16: 4-bit quantisation turns its output to grain. With 48 GB of GPU memory or more it runs 40 steps with guidance. Below that it is streamed from CPU memory (needs ~40 GB of free RAM) and uses the 8-step Lightning LoRA (expect ~3 min per large hole on a 24 GB GPU).
- The depth behind each erased object comes from MoGe on the cleaned panorama, scaled to the original depth around the hole.

## Render an action's end state

Render what the panorama looks like after an action on its objects: picked-up or carried-off objects are gone, opened or closed ones are in their new state, and nothing else changes. Needs steps 1 and 2.

```bash
uv run sim/pano_splat/act.py datasets/pano_splat/kitchen "open the fridge"
```

Check `act/<action>/crops/<n>.jpg` (each changed object's view: before, outlined, the editor's output, after) and `before_after.jpg`; `pano.jpg` and `distance.npy` are the changed panorama and its depth.

Notes:
- Qwen3.5-122B-A10B picks the changed objects from step 2's list and words their end state; the plan is saved in `plan.json` and reused on the next run of the same action (pass `--replan` to ask again). Read it when the result is wrong: most errors are the wrong object or count.
- Qwen-Image-Edit-2511 (Apache-2.0, bf16, ~60 GB) renders each object in a perspective view, given a copy with the object outlined in red. For a removal, the object's surroundings go back onto the panorama, shading-matched and with its contact shadow, but other movable or articulated objects keep their pixels; for a new state, only the changed regions touching the object go back.
- The changed pixels get MoGe-3 depth from one view around each changed region, scaled to the old depth around it.
- Pass `--input inpaint` to act on the cleaned panorama.

## Walk around and edit objects

Serve a viewer that walks around each scene and changes objects on click (needs steps 1 and 2, the VLM on [Triton](../../services/triton/README.md) and one free B300 on the service node (`SERVICE_NODE`) for the image editor).

```bash
set -a; . $STATE_DIR/slurm.env; set +a
CUDA_VISIBLE_DEVICES=2 uv run sim/pano_splat/serve.py datasets/pano_splat --port 8765
```

Open it through a tunnel from your machine, then browse to `http://localhost:8765`.

```bash
ssh -L 8765:$SERVICE_NODE:8765 <login node>
```

Drag to look, move with WASD or the arrow keys, hover to see an object, click it and type what happens ("pick it up", "open it"). Objects you pick up go into the inventory; pick an item there and click where it goes to place it back. Undo and Reset go back.

Notes:
- The eye stays within `--radius` (0.4 m) of the capture point, shown as a circle on the floor; further away the holes behind objects and stretched surfaces show.
- Edits use `act.py`'s models and methods: expect about 30 s per rendered edit (nearly all of it the image edit), 30-40 s per placement, and about a second to restore a state seen before. Without `TRITON_URL`, or with `--local_vlm`, the VLM loads in the server instead and needs a whole second GPU (`--vlm_gpu`).
- Each edit's result is the new current panorama, and every object remembers how it was last seen, in 3D:
  - A movable object's memory is its coloured points, saved when it is picked up. Placing moves them onto the clicked surface and draws their outline from the capture point (sized, and hidden behind what is in front); the editor draws the object there from its cutout, and the points give its depth. It stays the same object, so picking it up again updates its memory.
  - An articulated object has one memory per state (closed, open), its area with whatever is inside. Going back to a state it has been in restores that view in about a second, without the editor: close the fridge with a bowl inside, open it, and the bowl is there. Changes made in that area while it was in the other state are lost.
- The state is saved to `<scene>/edits/` after every edit and resumed on restart; Reset deletes it. Undo works within a session.
