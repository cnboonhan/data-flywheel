"""The door room as a BEHAVIOR scene: the floor, two walls and the latching door (assets/latch_door), in BEHAVIOR's
scene layout so OmniGibson loads it as InteractiveTraversableScene(scene_model="door_room",
dataset_name="flywheel-assets").

    venv-behavior/bin/python build_door_room.py [--dataset-root $STATE/behavior-data/flywheel-assets]

Writes scenes/door_room/json/door_room_best.json and scenes/door_room/layout/*.png. Needs no simulator: the JSON is
written in the shape OmniGibson saves (house_single_floor_best.json): each object's init_info and its root pose and
joint positions. The door stands in the wall line x = 0, its leaf swinging into -x (the corridor) about the hinge on
the -y jamb, the lever near the free (+y) edge; the entryway is +x.

Maps, BEHAVIOR's conventions (maps/map_base.py, traversable_map.py, segmentation_map.py): square PNGs, 0.01 m a pixel,
the world origin at the centre, row = y, column = x; trav 255 free, 0 blocked; semseg the room category's line in
behavior-1k-assets/metadata/room_categories.txt (1-based), insseg one id per room instance, 0 for no room.
- floor_trav_0: the floor less the walls, the frame and the closed leaf: no way across.
- floor_trav_no_obj_0 and floor_trav_no_door_0: walls and frame only, the doorway free.
- floor_trav_open_door_0: as no_door, with the leaf blocked where it stands open at 90 degrees.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

STATE = Path("/tier1/mistral/flywheel/autolearn")
p = argparse.ArgumentParser()
p.add_argument("--dataset-root", type=Path, default=STATE / "behavior-data" / "flywheel-assets")
p.add_argument("--b1k-root", type=Path, default=STATE / "behavior-data" / "behavior-1k-assets")
p.add_argument("--map-size", type=float, default=8.0)   # m, square, a margin past the 7 m floor
args = p.parse_args()

# the assets' dimensions (make_room_assets.py, make_latch_door.py defaults)
FLOOR, FLOOR_T = 7.0, 0.1
WALL_L, WALL_H, WALL_T, OPENING = 3.0, 2.5, 0.12, 1.0
DOOR_W, DOOR_FRAME_W, DOOR_T, DOOR_BOX_H = 0.9, 0.05, 0.04, 2.15   # clear width, jamb face, leaf, frame box height

ROOMS = {"corridor": -1, "entryway": +1}   # room type -> the side of the wall line (sign of x)

scene_dir = args.dataset_root / "scenes" / "door_room"
(scene_dir / "json").mkdir(parents=True, exist_ok=True)
(scene_dir / "layout").mkdir(parents=True, exist_ok=True)


def obj(name, category, model, rooms):
    return {"class_module": "omnigibson.objects.dataset_object", "class_name": "DatasetObject",
            "args": {"name": name, "category": category, "model": model, "dataset_name": "flywheel-assets",
                     "scale": [1.0, 1.0, 1.0], "fixed_base": True, "visual_only": False, "in_rooms": rooms,
                     "bddl_object_scope": None, "expected_file_hash": None}}


def fixed_state(pos):
    return {"is_asleep": False, "root_link": {"pos": list(pos), "ori": [0.0, 0.0, 0.0, 1.0]}, "non_kin": {}}


init_info = {
    "floors_flyflr_0": obj("floors_flyflr_0", "floors", "flyflr", ["corridor_0", "entryway_0"]),
    "walls_flywal_0": obj("walls_flywal_0", "walls", "flywal", ""),
    "door_flylat_0": obj("door_flylat_0", "door", "flylat", ["corridor_0", "entryway_0"]),
}
registry = {
    "floors_flyflr_0": fixed_state((0.0, 0.0, -FLOOR_T / 2)),          # top face at z = 0
    "walls_flywal_0": fixed_state((0.0, 0.0, WALL_H / 2)),
    "door_flylat_0": {"is_asleep": False,
                      "root_link": {"pos": [0.0, 0.0, DOOR_BOX_H / 2], "ori": [0.0, 0.0, 0.0, 1.0],
                                    "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
                      "joint_pos": [0.0, 0.0], "joint_vel": [0.0, 0.0], "non_kin": {}},   # j_leaf, j_handle closed
}
b1k_versions = json.loads((args.b1k_root / "scenes/house_single_floor/json/house_single_floor_best.json").read_text())["versions"]
scene = {"versions": b1k_versions, "metadata": {"task": {}},
         "state": {"pos": [0.0, 0.0, 0.0], "ori": [0.0, 0.0, 0.0, 1.0],
                   "registry": {"system_registry": {}, "object_registry": registry}},
         "objects_info": {"init_info": init_info}}
(scene_dir / "json" / "door_room_best.json").write_text(json.dumps(scene, indent=1))

# --- layout maps
RES = 0.01
n = int(round(args.map_size / RES))
ys, xs = np.mgrid[0:n, 0:n]
X, Y = (xs - n / 2) * RES, (ys - n / 2) * RES       # row = y, column = x (map_base.world_to_map)


def box(x0, x1, y0, y1):
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


floor = box(-FLOOR / 2, FLOOR / 2, -FLOOR / 2, FLOOR / 2)
half_t = WALL_T / 2
walls = box(-half_t, half_t, OPENING / 2, OPENING / 2 + WALL_L) | box(-half_t, half_t, -OPENING / 2 - WALL_L, -OPENING / 2)
jambs = box(-half_t, half_t, DOOR_W / 2, OPENING / 2) | box(-half_t, half_t, -OPENING / 2, -DOOR_W / 2)
leaf_closed = box(-half_t, half_t, -DOOR_W / 2, DOOR_W / 2)
hinge_y = -DOOR_W / 2
leaf_open = box(-DOOR_W, 0.0, hinge_y, hinge_y + DOOR_T)              # open 90 degrees into -x


def trav(blocked):
    m = np.zeros((n, n), np.uint8)
    m[floor & ~blocked] = 255
    return m


maps = {"floor_trav_0": trav(walls | jambs | leaf_closed),
        "floor_trav_no_obj_0": trav(walls | jambs),
        "floor_trav_no_door_0": trav(walls | jambs),
        "floor_trav_open_door_0": trav(walls | jambs | leaf_open)}
cats = (args.b1k_root / "metadata" / "room_categories.txt").read_text().split()
ins = np.zeros((n, n), np.uint8)
sem = np.zeros((n, n), np.uint8)
for i, (room, side) in enumerate(ROOMS.items(), start=1):
    region = floor & (side * X > half_t)
    ins[region] = i
    sem[region] = cats.index(room) + 1
maps["floor_insseg_0"], maps["floor_semseg_0"] = ins, sem
for name, m in maps.items():
    cv2.imwrite(str(scene_dir / "layout" / f"{name}.png"), m)
print(scene_dir, {k: int((v > 0).sum()) for k, v in maps.items()})
