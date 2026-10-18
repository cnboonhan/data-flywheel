# Scene-Language program for place_object_relative_position_lab (geniesim_generator "Path B": written by
# hand, no LLM service). app.py imports `root_scene` from this file and compiles it to scene.usda,
# scene_info.json and graph.*; generate.sh runs it once per LAB_INSTANCE.

from helper import *

"""
scene_name: lab_table_with_everyday_objects
description: benchmark_table_019 at the origin with five everyday objects resting on its top; one of four
             hand-written layouts, chosen by the LAB_INSTANCE environment variable (0-3).
"""

INSTANCE = int(os.environ.get("LAB_INSTANCE", "0"))

TABLE = "benchmark_table_019"  # the table every shipped place_object_relative_position instance uses

# 0.01 m gap between an object's lowest point and the table top: enough that the convex collision hulls
# never start interpenetrating, small enough that the 1-frame settle does not move anything. The shipped
# instances start their objects ~0.15 m up and let them fall, which rolls round fruit.
CLEARANCE = 0.01

# Object pool: (asset id, keywords). All ids exist in ASSETS_INDEX; all are 3-11 cm, graspable top-down by
# the G2 omnipicker (13.6 cm opening); cup, sponge, orange and medicine bottle are the shipped set's.
CUP = ("benchmark_cup_105", ["paper cup", "benchmark_cup_105"])
SPONGE = ("benchmark_bowl_sponge_000", ["sponge", "benchmark_bowl_sponge_000"])
ORANGE = ("benchmark_orange_004", ["orange", "benchmark_orange_004"])
LEMON = ("benchmark_lemon_030", ["lemon", "benchmark_lemon_030"])
MOUSE = ("benchmark_mouse_000", ["computer mouse", "benchmark_mouse_000"])
BOTTLE = ("benchmark_medicine_bottle_000", ["medicine bottle", "benchmark_medicine_bottle_000"])
CUBE = ("benchmark_rubik_cube_001", ["rubik cube", "benchmark_rubik_cube_001"])

# Layouts in the table frame (table centre = origin, +x away from the robot, +y = robot's left). Positions
# follow the shipped instances' band, x in [-0.25, 0.05] and |y| <= 0.28 (table top is 0.6 x 1.0 m), every
# pair >= 0.20 m apart centre to centre (>= 0.12 m asked for), and the strip where the instruction's target
# zone lies (left/right of B along y, top/bottom of B along x) is kept empty. yaw in degrees about z.
#   instance 0: "place the sponge to the left of the paper cup"      -> free strip at y > cup.y
#   instance 1: "place the orange to the right of the lemon"         -> free strip at y < lemon.y
#   instance 2: "place the paper cup to the top of the sponge"       -> free strip at x > sponge.x
#   instance 3: "place the lemon to the bottom of the orange"        -> free strip at x < orange.x
LAYOUTS = [
    [  # 0
        (CUP, (-0.08, -0.06), 10),
        (SPONGE, (-0.22, 0.18), -15),
        (ORANGE, (0.00, -0.24), 0),
        (MOUSE, (-0.22, -0.20), 20),
        (LEMON, (0.02, 0.22), 35),
    ],
    [  # 1
        (LEMON, (-0.12, 0.06), 25),
        (ORANGE, (0.00, 0.24), 0),
        (CUP, (-0.24, -0.22), -5),
        (BOTTLE, (0.02, -0.26), 15),
        (SPONGE, (-0.24, 0.24), 10),
    ],
    [  # 2
        (SPONGE, (-0.18, 0.02), 5),
        (CUP, (-0.22, -0.24), 0),
        (MOUSE, (0.02, 0.22), -30),
        (LEMON, (-0.20, 0.26), 40),
        (CUBE, (0.04, -0.20), 20),
    ],
    [  # 3
        (ORANGE, (-0.06, -0.04), 0),
        (LEMON, (0.00, 0.24), -20),
        (SPONGE, (-0.24, 0.22), 15),
        (BOTTLE, (0.02, -0.26), 0),
        (MOUSE, (-0.24, -0.26), -10),
    ],
]


@register()
def workspace_table() -> Shape:
    """The table, standing on the floor: helper.usd() lifts the asset so its lowest point is at z = 0."""
    return library_call("usd", oid=TABLE, keywords=["workspace_table", "table", "white", "marble_top"])


@register()
def place_on_table(oid: str, keywords: list, xy: tuple, yaw_deg: float, table_top: float) -> Shape:
    """Rotate an asset about z, then set its centre to `xy` and its lowest point to table_top + CLEARANCE."""
    shape = library_call("usd", oid=oid, keywords=keywords + ["on_desk", "left" if xy[1] > 0 else "right"])
    shape = transform_shape(shape, rotation_matrix(math.radians(yaw_deg), (0, 0, 1), compute_shape_center(shape)))
    info = get_object_info(shape)
    center, bottom = np.asarray(info["center"]), info["min"][2]
    offset = np.array([xy[0] - center[0], xy[1] - center[1], table_top + CLEARANCE - bottom])
    return transform_shape(shape, translation_matrix(offset))


@register()
def lab_table_with_everyday_objects() -> Shape:
    table = library_call("workspace_table")
    table_top = float(get_object_info(table)["max"][2])
    objects = [
        library_call("place_on_table", oid=oid, keywords=list(keywords), xy=xy, yaw_deg=yaw, table_top=table_top)
        for (oid, keywords), xy, yaw in LAYOUTS[INSTANCE]
    ]
    return concat_shapes(table, *objects)


@register()
def root_scene() -> Shape:
    return library_call("lab_table_with_everyday_objects")
