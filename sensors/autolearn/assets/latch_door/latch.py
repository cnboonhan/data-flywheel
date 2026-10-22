"""Closer, latch and lever return for doors whose asset has its own handle joint (the latching door asset, flylat).

    from latch import fit, press
    door = fit(obj, load(spring_doors.yaml), seed=instance)     # after the scene has loaded
    press(door, 40)                                              # a hand holding the lever down; 0 lets go

The same behaviour as spring_doors (assets/spring_doors), whose config and closed-end rule it reuses, but the lever
is the asset's j_handle joint, a DOF of the door's own articulation, instead of a body built at runtime. Every
physics step applies three torques, because OmniGibson zeroes joint stiffness on load and its joint drives don't
move scene objects (spring_doors' notes):
- leaf: the closer, -K (q - closed) - K tau qdot; while latched (within latch_band of closed and the lever above
  -unlatch_deg) the latch's stiff spring instead;
- handle: the return spring towards preload_deg above its 0 stop (the joint limit), so the lever rests level.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spring_doors"))


def load(path):
    """spring_doors' config loader (the shared spring_doors.yaml, with `extends`)."""
    import spring_doors

    return spring_doors.load(path)

_doors = []
_subscription = None


def fit(obj, cfg, seed=0, leaf_joint="j_leaf", handle_joint="j_handle", stiffness=None, time_constant=None, subscribe=True):
    """Make the door self-closing and latching; returns its entry (push it with entry["push"] = N m). With
    subscribe=False the caller steps it (spring_doors.apply runs these entries from its own physics callback)."""
    import torch as th
    from spring_doors import closed_position   # imported here: spring_doors imports this module

    rng = random.Random(seed)
    k = stiffness if stiffness is not None else rng.uniform(*cfg["stiffness"])
    tau = time_constant if time_constant is not None else rng.uniform(*cfg["time_constant"])
    h = cfg["handles"]
    leaf, handle = obj.joints[leaf_joint], obj.joints[handle_joint]
    view = leaf._articulation_view
    dofs = list(leaf.dof_indices) + list(handle.dof_indices)   # [leaf, lever]
    for j in (leaf, handle):
        j.friction = 0.0
    view.set_gains(kds=th.zeros(1, 2), joint_indices=dofs)
    span = math.radians(h["max_deg"] + h["preload_deg"])
    door = {"obj": obj, "view": view, "dofs": dofs, "closed": closed_position(leaf), "k": k, "d": k * tau,
            "push": 0.0, "latched": True, "angle": 0.0, "open": 0.0, "released": False,
            "asset_handle": True, "handle": {"angle": 0.0},     # the shape spring_doors.state() reads
            "kh": h["return_torque"] / span,                         # N m / rad: return_torque at full travel
            "dh": h["return_damping"] * 180 / math.pi,                 # yaml gives N m s / deg
            "target": math.radians(h["preload_deg"]), "rest": math.radians(h["preload_deg"]),
            "unlatch": math.radians(h["unlatch_deg"]), "band": h["latch_band"],
            "kl": h["latch_stiffness"], "dl": h["latch_damping"]}
    if subscribe:
        _doors.append(door)
        _subscribe()
    return door


def press(door, degrees):
    """Hold the lever turned down by `degrees` (0 lets it spring back)."""
    door["target"] = -math.radians(degrees) if degrees else door["rest"]


def step_door(d):
    """One physics step of one door: the closer or latch torque on the leaf, the return spring on the lever."""
    import torch as th

    q = d["view"].get_joint_positions(joint_indices=d["dofs"])
    qd = d["view"].get_joint_velocities(joint_indices=d["dofs"])
    if q is None or qd is None:      # the view is invalid for a step while the stage reloads or renders only
        return
    q, qd = q[0], qd[0]
    leaf, lever = float(q[0]), float(q[1])
    d["angle"] = d["handle"]["angle"] = lever
    d["open"] = abs(leaf - d["closed"])
    # a latch holds until the lever releases it, however hard the door is pulled (a spring latch judged by the band
    # alone could be yanked through it: a held-out run opened the door with the lever at -17 degrees); the band is only
    # where the bolt catches again, with the lever up
    if d["latched"]:
        if lever <= -d["unlatch"]:
            d["latched"], d["released"] = False, True      # unlatched by the lever, not merely still open
    elif d["open"] < d["band"] and lever > -d["unlatch"]:
        d["latched"] = True
    k, c = (d["kl"], d["dl"]) if d["latched"] else (d["k"], d["d"])
    t_leaf = -k * (leaf - d["closed"]) - c * float(qd[0]) + d["push"]
    t_lever = -d["kh"] * (lever - d["target"]) - d["dh"] * float(qd[1])
    d["view"].set_joint_efforts(th.tensor([[t_leaf, t_lever]]), joint_indices=d["dofs"])


def _step(dt):
    for d in _doors:
        step_door(d)


def _subscribe():
    global _subscription
    if _subscription is None:
        import omni.physx

        _subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(_step)


def clear():
    global _subscription
    _doors.clear()
    _subscription = None
