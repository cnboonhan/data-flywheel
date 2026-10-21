"""The door room's structure as BEHAVIOR assets: a floor slab (floors/flyflr) and a wall either side of the door's
frame (walls/flywal), our own boxes, imported with OmniGibson's pipeline like assets/latch_door.

    venv-behavior/bin/python make_room_assets.py [--dataset flywheel-assets]

Both are single fixed bodies with their origin at their bounding box's centre (BEHAVIOR's convention), z up. The
wall model's gap fits the latch door's frame (1.0 m wide with its jambs); build_door_room.py places them so the
floor's top is z = 0 and the door frame sits in the gap.
"""
import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="flywheel-assets")
p.add_argument("--floor-size", type=float, default=7.0)       # m, square: the walls span it (0.5 + 3.0 each side), so
#                                                                the only way across is the door
p.add_argument("--floor-thickness", type=float, default=0.1)   # m, a slab; only its top face matters
p.add_argument("--wall-length", type=float, default=3.0)       # m, each side of the door's frame
p.add_argument("--wall-height", type=float, default=2.5)       # m: a common ceiling height
p.add_argument("--wall-thickness", type=float, default=0.12)   # m: the latch door's frame depth, so jambs sit flush
p.add_argument("--opening", type=float, default=1.0)           # m: the latch door's frame width (0.9 + 2 x 0.05)
args = p.parse_args()


def xyz(*v):
    return " ".join(f"{x:.6f}" for x in v)


def urdf(name, boxes, rgba, mass):
    """One fixed link of boxes [(size, centre)], visual and collision."""
    robot = ET.Element("robot", name=name)
    m = ET.SubElement(robot, "material", name="m")
    ET.SubElement(m, "color", rgba=rgba)
    link = ET.SubElement(robot, "link", name="base_link")
    i = ET.SubElement(link, "inertial")
    ET.SubElement(i, "origin", xyz=xyz(0, 0, 0))
    ET.SubElement(i, "mass", value=f"{mass:.3f}")
    ET.SubElement(i, "inertia", ixx="1", iyy="1", izz="1", ixy="0", ixz="0", iyz="0")   # fixed base: unused
    for size, at in boxes:
        for kind in ("visual", "collision"):
            e = ET.SubElement(link, kind)
            ET.SubElement(e, "origin", xyz=xyz(*at), rpy=xyz(0, 0, 0))
            ET.SubElement(ET.SubElement(e, "geometry"), "box", size=xyz(*size))
            if kind == "visual":
                ET.SubElement(e, "material", name="m")
    return ET.ElementTree(robot)


tmp = Path(tempfile.mkdtemp())
F, FT = args.floor_size, args.floor_thickness
floor = urdf("flyflr", [((F, F, FT), (0, 0, 0))], "0.62 0.6 0.56 1", 1000.0)
L, WH, WT, O = args.wall_length, args.wall_height, args.wall_thickness, args.opening
walls = urdf("flywal", [((WT, L, WH), (0, s * (O / 2 + L / 2), 0)) for s in (-1, 1)], "0.9 0.89 0.86 1", 500.0)
paths = {}
for name, tree in (("flyflr", floor), ("flywal", walls)):
    paths[name] = tmp / f"{name}.urdf"
    tree.write(paths[name])

import omnigibson as og  # noqa: E402

og.launch()
import carb  # noqa: E402

carb.settings.get_settings().set("/crashreporter/enabled", False)
import omnigibson.utils.asset_conversion_utils as acu  # noqa: E402

for category, model in (("floors", "flyflr"), ("walls", "flywal")):
    _, usd, _ = acu.import_og_asset_from_urdf(category=category, model=model, dataset_name=args.dataset,
                                              urdf_path=str(paths[model]), collision_method=None, overwrite=True)
    print(category, model, usd, flush=True)
og.shutdown()
sys.exit(0)
