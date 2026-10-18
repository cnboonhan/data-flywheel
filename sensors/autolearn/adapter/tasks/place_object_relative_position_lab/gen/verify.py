"""Check the generated instances without Isaac Sim. Needs pxr; the asset paths are the container's, so run
it in geniesim3 against a path under /workspace.

Per instance: scene.usda parses, every payload exists on disk and loads, the table top is at the height
the layout assumed, each object rests on the table (bottom within [0, 2] cm of the top) inside the top's
footprint, objects are >= MIN_GAP apart centre to centre, the strip where the instruction's target zone
lies is empty, and problems.json / instructions.json name prims that exist.

usage: python3 verify.py <llm_task/<sub_task> dir>
"""

import json
import os
import sys

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import SPEC  # noqa: E402

TABLE_TOP = 0.73  # benchmark_table_019 mesh: z in [-0.265, 0.465] about its origin, placed at z = 0.265
TABLE_HALF = (0.30, 0.50)  # the top's half extents in x, y
MIN_GAP = 0.12  # m between object centres, the layout brief's minimum
REST_TOL = (0.0, 0.02)  # object bottom relative to the table top: on it, at most the 1 cm clearance + rounding
# Target strip: the active object must end up within |other axis| < 0.1 of B (aligned_*|0.1) and more than
# 0.05 past it along the relation axis; a 0.20 m reach past B leaves room for a 10 cm object.
STRIP_ALONG, STRIP_ACROSS = (0.05, 0.20), 0.10
AXIS = {"leftof": (1, +1), "rightof": (1, -1), "topof": (0, +1), "bottomof": (0, -1)}


def world_bounds(stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    out = {}
    for prim in stage.GetPrimAtPath("/World/Objects").GetChildren():
        ops = prim.GetMetadata("payload")  # postprocess.py writes an explicit list; the generator prepends
        payload = (ops.explicitItems or ops.prependedItems)[0].assetPath
        if not os.path.exists(payload):
            raise SystemExit(f"{prim.GetName()}: payload {payload} does not exist")
        if not prim.GetPrimAtPath("entity"):
            raise SystemExit(f"{prim.GetName()}: payload {payload} did not load (/entity missing)")
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        out[prim.GetName()] = (np.array(rng.GetMin()), np.array(rng.GetMax()), payload)
    return out


def check_instance(inst, task):
    stage = Usd.Stage.Open(os.path.join(inst, "scene.usda"), Usd.Stage.LoadAll)
    bounds = world_bounds(stage)
    tables = [k for k, (_, _, p) in bounds.items() if "/table/" in p]
    assert len(tables) == 1, tables
    lo, hi, _ = bounds[tables[0]]
    assert abs(hi[2] - TABLE_TOP) < 1e-3 and abs(lo[2]) < 1e-3, f"table z {lo[2]:.3f}..{hi[2]:.3f}"
    kinematic = stage.GetAttributeAtPath(f"/World/Objects/{tables[0]}/entity.physics:kinematicEnabled")
    assert kinematic and kinematic.Get() is True, "table is not kinematic"

    objs = {k: v for k, v in bounds.items() if k != tables[0]}
    centres = {k: (lo + hi) / 2 for k, (lo, hi, _) in objs.items()}
    for k, (lo, hi, _) in objs.items():
        rest = lo[2] - TABLE_TOP
        assert REST_TOL[0] - 1e-6 <= rest <= REST_TOL[1], f"{k}: bottom {rest:+.3f} m from the table top"
        assert abs(lo[0]) <= TABLE_HALF[0] and abs(hi[0]) <= TABLE_HALF[0], f"{k}: off the table in x"
        assert abs(lo[1]) <= TABLE_HALF[1] and abs(hi[1]) <= TABLE_HALF[1], f"{k}: off the table in y"
    names = list(objs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = float(np.linalg.norm(centres[a][:2] - centres[b][:2]))
            assert d >= MIN_GAP, f"{a} and {b} are {d:.3f} m apart"

    with open(os.path.join(inst, "scene_info.json")) as f:
        layout = json.load(f)["layout"]
    by_asset = {v["usd"]: k for k, v in layout.items()}
    active, passive = by_asset[task["active"]], by_asset[task["passive"]]
    axis, sign = AXIS[task["relation"]]
    other = 1 - axis
    b = centres[passive]
    for k, (lo, hi, _) in objs.items():
        if k in (active, passive):
            continue
        along = sorted(sign * (np.array([lo[axis], hi[axis]]) - b[axis]))
        across = sorted(np.array([lo[other], hi[other]]) - b[other])
        overlaps = along[1] > STRIP_ALONG[0] and along[0] < STRIP_ALONG[1] and across[1] > -STRIP_ACROSS and across[0] < STRIP_ACROSS
        assert not overlaps, f"{k} sits in the target strip {task['relation']} {passive}"

    with open(os.path.join(inst, "problems.json")) as f:
        text = json.dumps(json.load(f))
    for name in (active, passive):
        assert name in text, f"problems.json does not mention {name}"
        assert stage.GetPrimAtPath(f"/World/Objects/{name}"), f"{name} is not a prim"
    with open(os.path.join(inst, "instructions.json")) as f:
        assert json.load(f)["instructions"][0]["instruction"] == task["instruction"]
    print(f"{os.path.basename(inst)}: ok  {len(objs)} objects, {task['instruction']!r}, {active} {task['relation']} {passive}")


def main(root):
    for n, task in enumerate(SPEC):
        check_instance(os.path.join(root, str(n)), task)


if __name__ == "__main__":
    main(sys.argv[1])
