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
