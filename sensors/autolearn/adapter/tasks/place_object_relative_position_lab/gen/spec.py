"""The hand-written task per instance: which object goes where relative to which. checks.py turns this into
instructions.json / problems.json once the generator has assigned object ids; verify.py checks the layout
leaves the target strip free. Asset ids identify the objects (each appears once per layout)."""

# relation: ADER RelativePositionChecker names, evaluated in the robot's frame (robot faces +x):
#   leftof = A at +y of B, rightof = -y, topof = A further from the robot (+x), bottomof = nearer (-x).
SPEC = [
    {
        "instruction": "place the sponge to the left of the paper cup",
        "active": "benchmark_bowl_sponge_000",
        "passive": "benchmark_cup_105",
        "relation": "leftof",
    },
    {
        "instruction": "place the orange to the right of the lemon",
        "active": "benchmark_orange_004",
        "passive": "benchmark_lemon_030",
        "relation": "rightof",
    },
    {
        "instruction": "place the paper cup to the top of the sponge",
        "active": "benchmark_cup_105",
        "passive": "benchmark_bowl_sponge_000",
        "relation": "topof",
    },
    {
        "instruction": "place the lemon to the bottom of the orange",
        "active": "benchmark_lemon_030",
        "passive": "benchmark_orange_004",
        "relation": "bottomof",
    },
]

# Instances beyond the four hand-written ones are drawn from this pool with a fixed seed: a task (A, B, relation)
# and a layout of 5 objects in the shipped band with every pair >= 0.20 m apart and the target strip free.
# LAB_INSTANCES (default 4) sets how many instances exist in all; generate.sh, checks.py and verify.py read it.
import os
import random

POOL = {  # asset id -> (name used in the instruction, keywords for the generator)
    "benchmark_cup_105": ("paper cup", ["paper cup", "benchmark_cup_105"]),
    "benchmark_bowl_sponge_000": ("sponge", ["sponge", "benchmark_bowl_sponge_000"]),
    "benchmark_orange_004": ("orange", ["orange", "benchmark_orange_004"]),
    "benchmark_lemon_030": ("lemon", ["lemon", "benchmark_lemon_030"]),
    "benchmark_mouse_000": ("computer mouse", ["computer mouse", "benchmark_mouse_000"]),
    "benchmark_medicine_bottle_000": ("medicine bottle", ["medicine bottle", "benchmark_medicine_bottle_000"]),
    "benchmark_rubik_cube_001": ("rubik cube", ["rubik cube", "benchmark_rubik_cube_001"]),
}
WORDS = {"leftof": "left", "rightof": "right", "topof": "top", "bottomof": "bottom"}
X_RANGE, Y_RANGE = (-0.25, 0.05), (-0.28, 0.28)   # the shipped instances' band on the 0.6 x 1.0 m table top
MIN_GAP = 0.17                                    # centre to centre: the widest object is 11 cm, and 0.20 leaves some instances no layout
STRIP = (-0.01, 0.26, 0.16)                       # verify.py's strip (0.05-0.20 past B, +-0.10 across) on object extents, widened by a 6 cm half-size for centres
N_OBJECTS = 5


def _strip_free(xy, b, relation):
    dx, dy = xy[0] - b[0], xy[1] - b[1]
    along, across = {"leftof": (dy, dx), "rightof": (-dy, dx), "topof": (dx, dy), "bottomof": (-dx, dy)}[relation]
    return not (STRIP[0] <= along <= STRIP[1] and abs(across) <= STRIP[2])


def random_instance(n):
    """Task and layout for instance n >= 4, deterministic in n: B is placed first, then A and the fillers, each
    at least MIN_GAP from the others and outside the target strip past B."""
    for draw in range(100):     # a draw that packs badly is re-drawn, still deterministic in n
        rng = random.Random(1000 + n + 100000 * draw)
        ids = rng.sample(sorted(POOL), N_OBJECTS)
        active, passive = ids[0], ids[1]
        relation = rng.choice(sorted(WORDS))
        placed = []
        for oid in [passive, active] + ids[2:]:
            for _ in range(5000):
                xy = (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE))
                clear = all(((xy[0] - q[0]) ** 2 + (xy[1] - q[1]) ** 2) ** 0.5 >= MIN_GAP for _, q, _ in placed)
                if clear and (oid == passive or _strip_free(xy, placed[0][1], relation)):
                    placed.append((oid, xy, rng.uniform(-40, 40)))
                    break
        if len(placed) == N_OBJECTS:
            break
    else:
        raise SystemExit(f"instance {n}: no layout found")
    task = {"instruction": f"place the {POOL[active][0]} to the {WORDS[relation]} of the {POOL[passive][0]}",
            "active": active, "passive": passive, "relation": relation}
    layout = [((oid, POOL[oid][1]), xy, yaw) for oid, xy, yaw in placed]
    return task, layout


NUM_INSTANCES = int(os.environ.get("LAB_INSTANCES", "4"))
for _n in range(4, NUM_INSTANCES):
    SPEC.append(random_instance(_n)[0])

# Same limits as the shipped place_object_relative_position instances:
FOLLOW_STEPS = 300  # steps allowed for the gripper to reach the active object's Follow box
PLACE_STEPS = 600  # steps allowed for the MixedRules (relation + alignment) to pass
# Follow box [x, y, z] around the active object's centre; the shipped instances use 0.10-0.25 per axis
# depending on the object, 0.15 for the fruit. 0.15 covers every object here (tallest is 11 cm).
FOLLOW_BBOX = "[0.15,0.15,0.15]"
# aligned_* tolerance, as shipped: |offset along the other axis| < 0.1 m.
ALIGN_TOL = 0.1
# leftof/rightof check the offset along y, so alignment is along x, and vice versa.
ALIGN_AXIS = {"leftof": "aligned_x", "rightof": "aligned_x", "topof": "aligned_y", "bottomof": "aligned_y"}
