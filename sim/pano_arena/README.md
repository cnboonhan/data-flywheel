# sim/pano_arena

IsaacLab-Arena scenes from a single equirectangular panorama: metric depth, interactable objects cut out and turned into 3D assets, the rest of the room as a Gaussian-splat background. Each step is a uv script that reads and writes `datasets/pano_arena/<scene>/` (gitignored), so its output can be checked before the next step runs.

| Step | Script | Writes |
|---|---|---|
| 1. Depth and frame | `depth.py` | `depth/` |

## Sample panoramas

Two indoor HDRIs from [Poly Haven](https://polyhaven.com/hdris/indoor) (CC0), 8192×4096 tonemapped JPGs:

| Scene | Poly Haven asset | Contents |
|---|---|---|
| `kitchen` | [`blinds`](https://polyhaven.com/a/blinds) | camera just above a counter: jars, a bowl, a tissue box; cabinets, drawers, oven, fridge |
| `office` | [`poly_haven_studio`](https://polyhaven.com/a/poly_haven_studio) | camera on a tripod mid-room: office chairs, a bin, a heater; desks, double doors, a shelf unit |

Fetch them into `datasets/pano_arena/<scene>/source.jpg`.

```bash
for p in "kitchen blinds" "office poly_haven_studio"; do set -- $p; mkdir -p datasets/pano_arena/$1
  curl -fL -o datasets/pano_arena/$1/source.jpg "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/$2.jpg"; done
```

Any other panorama works the same way: put it at `datasets/pano_arena/<scene>/source.jpg`, equirectangular, level.

## 1. Depth and frame

Run [MoGe-2](https://github.com/microsoft/MoGe) metric depth on the panorama and set up the scene frame (about 1 min per scene; the model downloads on first use).

```bash
uv run sim/pano_arena/depth.py datasets/pano_arena/office
```

Check `depth/floor.png` (top-down view, coloured by height, camera as a red dot): walls should be straight and the floor flat. Expect a camera height of 1.2 m for `office` and 1.3 m for `kitchen`.

Notes:
- The frame is Arena's: z up, origin on the floor below the camera, +x towards the middle of the panorama.
- The scale is MoGe's estimate. When you know the camera height, pass `--camera_height <m>`; it rescales everything.
- `points.ply` (coloured points in the scene frame) opens in any point-cloud viewer, such as MeshLab or Rerun.
