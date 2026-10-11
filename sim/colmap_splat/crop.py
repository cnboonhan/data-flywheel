"""Copy a 3DGRUT export's gaussians, dropping those more than RADIUS m from their median.

    sim/colmap_splat/3dgrut/.venv/bin/python sim/colmap_splat/crop.py <export.usdz> <out.usdc> <radius>

3DGRUT keeps a shell of background gaussians out to ~1000 km. In Isaac Sim's path tracer that shell occludes the dome
light and samples stochastically: meshes go black on top and their shading flickers frame to frame.
"""

import sys

import numpy as np
from pxr import Usd, Vt

PRIM = "/World/gaussians/Gaussians/gaussians"

src_path, out, radius = sys.argv[1], sys.argv[2], float(sys.argv[3])
src = Usd.Stage.Open(src_path)
p = src.GetPrimAtPath(PRIM)
pos = np.array(p.GetAttribute("positions").Get())
n = len(pos)
keep = np.linalg.norm(pos - np.median(pos, 0), axis=1) < radius

dst = Usd.Stage.CreateNew(out)
q = dst.DefinePrim(PRIM, p.GetTypeName())
for a in p.GetAttributes():
    v = a.Get()
    if v is None:
        continue
    b = q.CreateAttribute(a.GetName(), a.GetTypeName(), a.IsCustom())
    if hasattr(v, "__len__") and not isinstance(v, str) and len(v) >= n and len(v) % n == 0:   # per-gaussian
        arr = np.array(v)
        arr = arr.reshape(n, len(v) // n, *arr.shape[1:])[keep].reshape(-1, *arr.shape[1:])
        b.Set(type(v).FromNumpy(np.ascontiguousarray(arr)))
    else:
        b.Set(v)
q.GetAttribute("extent").Set(Vt.Vec3fArray([tuple(map(float, pos[keep].min(0))), tuple(map(float, pos[keep].max(0)))]))
dst.SetDefaultPrim(dst.GetPrimAtPath("/World"))
dst.GetRootLayer().Save()
print(f"kept {keep.sum()} of {n} gaussians within {radius:g} m -> {out}")
