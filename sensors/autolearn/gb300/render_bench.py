"""How fast a GB300 GPU renders for Genie Sim: steps per second of Isaac Sim 5.1 headless, with Genie Sim's renderer.

    /isaac-sim/python.sh render_bench.py --renderer RealTimePathTracing --cameras 3 --width 1280 --height 720 --out r.json

Opens a scene (Isaac Sim's full warehouse from the cloud assets by default), adds N cameras at the given resolution,
then times world.step(render=True): first alone, then with every camera's RGB fetched each step, as data collection
does. Prints one RESULT line and writes the numbers as JSON. Genie Sim's settings (geniesim.exp.kit,
data_collector_server.py): RealTimePathTracing with rt2, DLSS and DLSS-G off, no async rendering.
"""
import argparse
import json
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--renderer", default="RealTimePathTracing", choices=["RealTimePathTracing", "RaytracedLighting", "PathTracing"])
p.add_argument("--cameras", type=int, default=3)   # Genie Sim's G2 has a head camera and one per wrist
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--scene", default="/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")   # under the assets root
p.add_argument("--warmup", type=int, default=30)    # steps before timing: shader compilation and the renderer settling
p.add_argument("--steps", type=int, default=100)
p.add_argument("--out", default="")
args = p.parse_args()

from isaacsim import SimulationApp   # noqa: E402

app = SimulationApp({
    "headless": True,
    "renderer": args.renderer,
    "extra_args": [
        "--/persistent/rtx/modes/rt2/enabled=true",   # Genie Sim's data collector
        "--/rtx/post/dlss/execMode=0", "--/rtx-transient/dlssg/enabled=false",   # geniesim.exp.kit: no DLSS (no NGX on GB300 either)
        # no NGX at all on data-centre GPUs: FXAA and no DL denoiser, else every frame fails (IsaacLab-Arena's GB300 hook)
        "--/rtx/post/aa/op=2", "--/rtx-transient/post/aa/limitedOps=false", "--/rtx-transient/dldenoiser/enabled=false",
        "--/renderer/gpuEnumeration/glInterop/enabled=false",
        "--/crashreporter/enabled=false",
    ],
})
app._carb_settings.set("/omni/replicator/asyncRendering", False)
app._carb_settings.set("/app/asyncRendering", False)

import numpy as np   # noqa: E402
from isaacsim.core.api import World   # noqa: E402
from isaacsim.core.utils.stage import open_stage   # noqa: E402
from isaacsim.sensors.camera import Camera   # noqa: E402
from isaacsim.storage.native import get_assets_root_path   # noqa: E402

t0 = time.time()
root = get_assets_root_path()
scene = args.scene if "://" in args.scene else root + args.scene
assert open_stage(scene), f"could not open {scene}"
world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)   # Genie Sim's rates
cams = []
for i in range(args.cameras):   # spread around the warehouse floor, looking along +x at shelf height
    cam = Camera(prim_path=f"/World/bench_cam_{i}", position=np.array([-6.0 + 3 * i, -2.0 + i, 1.5]),
                 orientation=np.array([1, 0, 0, 0]), frequency=30, resolution=(args.width, args.height))
    cam.initialize()
    cams.append(cam)
world.reset()
load_s = time.time() - t0

for _ in range(args.warmup):
    world.step(render=True)
t = time.time()
for _ in range(args.steps):
    world.step(render=True)
render_sps = args.steps / (time.time() - t)
t = time.time()
got = 0
for _ in range(args.steps):
    world.step(render=True)
    for cam in cams:
        rgba = cam.get_rgba()
        got += int(rgba is not None and getattr(rgba, "size", 0) > 0)
capture_sps = args.steps / (time.time() - t)

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().split("\n")[0]
result = {"gpu": gpu, "renderer": args.renderer, "cameras": args.cameras, "resolution": [args.width, args.height],
          "scene": scene, "load_s": round(load_s, 1), "render_steps_per_s": round(render_sps, 2),
          "render_and_capture_steps_per_s": round(capture_sps, 2), "frames_captured": got, "frames_expected": args.steps * args.cameras}
print("RESULT " + json.dumps(result), flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
app.close()
sys.exit(0)
