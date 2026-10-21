"""A hinged door with a lever handle, written as a URDF for OmniGibson's object importer (import_latch_door.py).

    python make_latch_door.py --out <dir> [--width 0.9 --height 2.1 --thickness 0.04 ...]

Our own geometry (boxes and cylinders), so it carries no dataset licence. Object frame: x the door's normal, y its
width, z up; the origin at the centre of the frame's bounding box, as BEHAVIOR's door models have it.

Links and joints:
- base_link: the frame, two jambs and a head;
- leaf (revolute j_leaf about z at the hinge edge, 0 .. swing): the slab and a rose on each face;
- handle (revolute j_handle about the door's normal through the spindle, -travel .. 0): the spindle and a lever on
  each face, horizontal at rest and pointing towards the hinge; pressing a lever down turns the joint negative.
Every part is a box or cylinder, and every link's collision is explicit boxes, so the importer has nothing to
decompose (the frame is not convex: a single hull would close the doorway).
"""
import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--name", default="latch_door")
p.add_argument("--width", type=float, default=0.9)        # m, clear opening: a common single interior door
p.add_argument("--height", type=float, default=2.1)       # m, clear opening
p.add_argument("--thickness", type=float, default=0.04)   # m, leaf: a standard 40 mm door slab
p.add_argument("--frame-width", type=float, default=0.05)   # m, jamb and head face width
p.add_argument("--frame-depth", type=float, default=0.12)   # m, jamb depth through the wall opening
p.add_argument("--gap", type=float, default=0.004)        # m, leaf-to-frame clearance at each edge
p.add_argument("--floor-gap", type=float, default=0.01)   # m, leaf bottom above the floor
p.add_argument("--swing-deg", type=float, default=100.0)  # leaf travel from closed; past 90 so it can be held wide open
p.add_argument("--leaf-mass", type=float, default=20.0)   # kg: a solid-core 40 mm interior door is ~20-30 kg
p.add_argument("--handle-height", type=float, default=1.0)   # m above the floor: the usual lever height
p.add_argument("--backset", type=float, default=0.065)       # m, free edge to spindle: a common latch backset
p.add_argument("--lever-length", type=float, default=0.13)   # m, spindle to tip
p.add_argument("--lever-bar", type=float, default=0.02)      # m, the lever's square section
p.add_argument("--standoff", type=float, default=0.05)       # m, face to lever: room for fingers
p.add_argument("--travel-deg", type=float, default=45.0)     # lever travel down from horizontal
p.add_argument("--handle-mass", type=float, default=0.4)     # kg, both levers and the spindle
args = p.parse_args()

W, H, T = args.width, args.height, args.thickness
FW, FD, G = args.frame_width, args.frame_depth, args.gap
leaf_w, leaf_h = W - 2 * G, H - args.floor_gap - G
z0 = -(H + FW) / 2                      # the floor, with the origin at the frame box's centre

robot = ET.Element("robot", name=args.name)
for mat, rgba in {"frame": "0.92 0.91 0.88 1", "leaf": "0.55 0.38 0.24 1", "steel": "0.75 0.75 0.78 1",
                  "rose": "0.6 0.6 0.62 1"}.items():
    m = ET.SubElement(robot, "material", name=mat)
    ET.SubElement(m, "color", rgba=rgba)


def xyz(*v):
    return " ".join(f"{x:.6f}" for x in v)


def box(link, size, at, material, rpy=(0, 0, 0), collide=True):
    for kind in ("visual", "collision") if collide else ("visual",):
        e = ET.SubElement(link, kind)
        ET.SubElement(e, "origin", xyz=xyz(*at), rpy=xyz(*rpy))
        ET.SubElement(ET.SubElement(e, "geometry"), "box", size=xyz(*size))
        if kind == "visual":
            ET.SubElement(e, "material", name=material)


def cylinder(link, radius, length, at, material, rpy):
    e = ET.SubElement(link, "visual")
    ET.SubElement(e, "origin", xyz=xyz(*at), rpy=xyz(*rpy))
    ET.SubElement(ET.SubElement(e, "geometry"), "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")
    ET.SubElement(e, "material", name=material)


def inertial(link, mass, size):
    """A solid box's inertia about its centre (the link frame is offset: the importer uses the given origin)."""
    a, b, c = size
    i = ET.SubElement(link, "inertial")
    ET.SubElement(i, "mass", value=f"{mass:.4f}")
    ET.SubElement(i, "inertia", ixx=f"{mass * (b * b + c * c) / 12:.6f}", iyy=f"{mass * (a * a + c * c) / 12:.6f}",
                  izz=f"{mass * (a * a + b * b) / 12:.6f}", ixy="0", ixz="0", iyz="0")
    return i


# base_link: the frame
base = ET.SubElement(robot, "link", name="base_link")
for side in (-1, 1):
    box(base, (FD, FW, H), (0, side * (W / 2 + FW / 2), z0 + H / 2), "frame")
box(base, (FD, W + 2 * FW, FW), (0, 0, z0 + H + FW / 2), "frame")
inertial(base, 20.0, (FD, W + 2 * FW, H + FW)).insert(0, ET.Element("origin", xyz=xyz(0, 0, 0)))

# leaf: hinge on the -y jamb, at the leaf's -x face; the link frame sits on the hinge axis at the leaf's bottom
hinge = (-T / 2, -W / 2 + G, z0 + args.floor_gap)
leaf = ET.SubElement(robot, "link", name="leaf")
centre = (T / 2, leaf_w / 2, leaf_h / 2)
box(leaf, (T, leaf_w, leaf_h), centre, "leaf")
spindle = (T / 2, leaf_w - args.backset, args.handle_height - args.floor_gap)
for face in (-1, 1):   # a rose on each face (visual only: it sits flat on the slab)
    cylinder(leaf, 0.025, 0.008, (spindle[0] + face * (T / 2 + 0.004), spindle[1], spindle[2]), "rose", (0, math.pi / 2, 0))
li = inertial(leaf, args.leaf_mass, (T, leaf_w, leaf_h))
li.insert(0, ET.Element("origin", xyz=xyz(*centre)))
j = ET.SubElement(robot, "joint", name="j_leaf", type="revolute")
ET.SubElement(j, "origin", xyz=xyz(*hinge))
ET.SubElement(j, "parent", link="base_link")
ET.SubElement(j, "child", link="leaf")
ET.SubElement(j, "axis", xyz="0 0 1")           # opening turns the free edge towards -x
ET.SubElement(j, "limit", lower="0", upper=f"{math.radians(args.swing_deg):.6f}", effort="1000", velocity="5")
ET.SubElement(j, "dynamics", damping="0", friction="0")

# handle: the frame on the spindle, x along the door's normal; levers point towards the hinge (-y)
hl = ET.SubElement(robot, "link", name="handle")
bar, L = args.lever_bar, args.lever_length
out = T / 2 + args.standoff
box(hl, (2 * out + bar, bar, bar), (0, 0, 0), "steel")
for face in (-1, 1):
    box(hl, (bar, L, bar), (face * out, -L / 2, 0), "steel")
hi = inertial(hl, args.handle_mass, (2 * out, L, bar))
hi.insert(0, ET.Element("origin", xyz=xyz(0, -L / 4, 0)))
j = ET.SubElement(robot, "joint", name="j_handle", type="revolute")
ET.SubElement(j, "origin", xyz=xyz(*spindle))
ET.SubElement(j, "parent", link="leaf")
ET.SubElement(j, "child", link="handle")
ET.SubElement(j, "axis", xyz="-1 0 0")          # about -x: a negative angle takes the lever tips (at -y) down
ET.SubElement(j, "limit", lower=f"{-math.radians(args.travel_deg):.6f}", upper="0", effort="50", velocity="10")
ET.SubElement(j, "dynamics", damping="0", friction="0")

args.out.mkdir(parents=True, exist_ok=True)
ET.indent(robot)
path = args.out / f"{args.name}.urdf"
ET.ElementTree(robot).write(path, xml_declaration=True, encoding="utf-8")
print(path)
