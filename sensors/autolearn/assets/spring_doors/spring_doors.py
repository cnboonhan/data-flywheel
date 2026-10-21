"""Self-closing, latching doors for BEHAVIOR-1K scenes: a door closer on each chosen hinge, and a lever handle that has
to be pressed down to unlatch the closed door.

Closer. OmniGibson loads every dataset object's hinge with no spring (objects/dataset_object.py), so a door stays
wherever it is left. A closer here is a torque applied every physics step, tau = -K (q - closed) - D qdot, with
D = K * time constant, K and the time constant drawn per task instance from spring_doors.yaml. Two things OmniGibson
does instead do not work for scene doors:
- its joint drive: gains and a closed target set on the hinge are accepted but never move the door (even K 200);
- its default hinge friction (0.2) and drive damping (an acceleration drive, kd ~2.9): with them the door creeps
  (0.90 -> 0.65 rad in 8 s at K 20), so a sprung door has both removed and the closer's own damping instead.

Handle. BEHAVIOR's door models have no turning handle (4 of 83 have a handle part, fixed to the leaf) and their USD is
encrypted, so the handle is built here: a rigid lever on each face, horizontal at rest and pointing towards the hinge,
both on one spindle through the leaf, on a revolute joint to the leaf with a return spring. Pressing a lever down
past the unlatch angle frees the closed door; otherwise a closed door is held by the latch (a stiff spring at the
closed position). The latch catches again whenever the closer brings the door back within the latch band.

    BEHAVIOR_SPRING_DOORS=sensors/autolearn/assets/spring_doors/spring_doors.yaml   (adapter/run_behavior.py applies it)

or from code, after the scene has loaded:

    from spring_doors import apply, load
    sprung = apply(env.scene, load(path), seed=instance)
"""
import math
import random
from pathlib import Path

import yaml

HANDLES_ROOT = "/World/flywheel_door_handles"
_doors = []            # one dict per sprung hinge, read every physics step
_subscription = None   # the physics-step subscription; dropping it stops the closers


def load(path):
    """A config, with `extends: <yaml beside it>` merged under it (top-level keys, and `handles` key by key)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("extends", None)
    if base:
        merged = load(path.parent / base)
        handles = {**(merged.get("handles") or {}), **(cfg.pop("handles", None) or {})}
        merged.update(cfg)
        if handles:
            merged["handles"] = handles
        cfg = merged
    return cfg


def state(name):
    """The sprung door `name`'s last physics-step reading: hinge angle from closed (rad, >= 0 open), lever turn (deg,
    negative pressed), latched, and whether the latch has been released since apply(); None if not sprung."""
    for door in _doors:
        if door["obj"].name == name:
            h = door["handle"]
            return {"open_rad": door.get("open", 0.0), "lever_deg": math.degrees(h["angle"]) if h else None,
                    "latched": door.get("latched"), "released": door.get("released", False)}
    return None


def closed_position(joint):
    """The hinge's closed end: the limit nearer zero. BEHAVIOR doors open from 0 to +/- their swing."""
    lo, hi = float(joint.lower_limit), float(joint.upper_limit)
    return lo if abs(lo) <= abs(hi) else hi


def spring(obj, joint, stiffness, time_constant):
    """Make one hinge self-closing; returns its entry. The damping is K * time_constant, not a damping ratio of the
    leaf's inertia: the doors close several times faster than m w^2 / 3 predicts, while K * time_constant sets the
    closing time as measured (spring_doors.yaml)."""
    import torch as th

    view, dofs = joint._articulation_view, joint.dof_indices
    joint.friction = 0.0
    view.set_gains(kds=th.tensor([[0.0]]), joint_indices=dofs)
    door = {"obj": obj, "joint": joint, "view": view, "dofs": dofs, "k": stiffness, "d": stiffness * time_constant,
            "closed": closed_position(joint), "handle": None, "push": 0.0}
    _doors.append(door)
    _subscribe()
    return door


def add_handle(door, cfg):
    """A lever handle on both faces of the door's leaf (the hinge's child link), on a revolute joint with a return
    spring; the door is latched while closed and the lever is above the unlatch angle."""
    import omnigibson as og
    import omnigibson.lazy as lazy
    import omnigibson.utils.transform_utils as T
    import torch as th

    obj, joint = door["obj"], door["joint"]
    pxr = lazy.pxr
    leaf_path = joint.prim.GetRelationship("physics:body1").GetTargets()[0]
    leaf = next(l for l in obj.links.values() if l.prim_path == str(leaf_path))
    leaf_pos, leaf_quat = leaf.get_position_orientation()
    R_leaf = T.quat2mat(leaf_quat)
    hinge_local = th.tensor(list(joint.prim.GetAttribute("physics:localPos1").Get()), dtype=th.float32)
    hinge = leaf_pos + R_leaf @ hinge_local

    # the leaf in its own frame: the vertical axis, the thin one (the door's normal), the long horizontal one (width);
    # the slab's faces measured on the hinge half, where no model handle sits; the model's own handle, if any, is the
    # vertices sticking out past the slab on the free half: it is flattened onto the face and its centre is where the
    # lever goes
    geo = _leaf_geometry(obj, leaf, R_leaf, leaf_pos, hinge_local, cfg)
    n_axis, w_axis, v_axis = geo["axes"]
    a, b = geo["slab"]
    thickness = b - a
    to_free = 1.0 if geo["free_w"] > float(hinge_local[w_axis]) else -1.0
    if geo["handle_wv"] is not None:
        w, v = geo["handle_wv"]
    else:
        w = geo["free_w"] - to_free * cfg["backset"]
        v = geo["bottom_v"] + cfg["height"]
    local = th.zeros(3)
    local[n_axis], local[w_axis], local[v_axis] = (a + b) / 2, w, v
    spindle = leaf_pos + R_leaf @ local

    # handle frame: x along the spindle (the door's normal), y along the lever towards the hinge, z up
    up = R_leaf[:, v_axis] * (1.0 if float(R_leaf[2, v_axis]) > 0 else -1.0)
    y = -to_free * R_leaf[:, w_axis]
    x = th.linalg.cross(y, up)
    R_handle = th.stack([x, y, up], dim=1)
    handle_quat = T.mat2quat(R_handle)

    R_rel = R_leaf.T @ R_handle                       # the handle (= joint) frame in the leaf's frame
    pos_in_leaf = R_leaf.T @ (spindle - leaf_pos)
    stage = og.sim.stage
    path = f"{HANDLES_ROOT}/{obj.name}_{joint.prim.GetName()}"
    built = stage.GetPrimAtPath(path).IsValid()   # a later instance in the same scene keeps the handle it has
    with og.sim.editing_usd():
        if not built:
            pxr.UsdGeom.Xform.Define(stage, HANDLES_ROOT)
            body = pxr.UsdGeom.Xform.Define(stage, path)
            body.AddTranslateOp().Set(pxr.Gf.Vec3d(*[float(v) for v in spindle]))
            qx, qy, qz, qw = [float(v) for v in handle_quat]
            body.AddOrientOp().Set(pxr.Gf.Quatf(qw, qx, qy, qz))
            pxr.UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            pxr.UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(cfg["mass"])
            length, bar = cfg["lever_length"], cfg["lever_thickness"]
            out = thickness / 2 + cfg["standoff"]
            parts = {"spindle": ((0.0, 0.0, 0.0), (2 * out + bar, bar, bar)),
                     "lever_a": ((out, length / 2, 0.0), (bar, length, bar)),
                     "lever_b": ((-out, length / 2, 0.0), (bar, length, bar))}
            for name, (centre, size) in parts.items():
                cube = pxr.UsdGeom.Cube.Define(stage, f"{path}/{name}")
                cube.CreateSizeAttr(1.0)
                cube.AddTranslateOp().Set(pxr.Gf.Vec3d(*centre))
                cube.AddScaleOp().Set(pxr.Gf.Vec3f(*size))
                cube.CreateDisplayColorAttr([pxr.Gf.Vec3f(0.75, 0.75, 0.78)])   # brushed steel
                pxr.UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            # the joint in the same edit as the body: PhysX ignores limits set after it has parsed the joint
            rev = pxr.UsdPhysics.RevoluteJoint.Define(stage, f"{path}/joint")
            rev.CreateBody0Rel().SetTargets([leaf_path])
            rev.CreateBody1Rel().SetTargets([pxr.Sdf.Path(path)])
            qx, qy, qz, qw = [float(v) for v in T.mat2quat(R_rel)]
            rev.CreateLocalPos0Attr(pxr.Gf.Vec3f(*[float(v) for v in pos_in_leaf]))
            rev.CreateLocalRot0Attr(pxr.Gf.Quatf(qw, qx, qy, qz))
            rev.CreateLocalPos1Attr(pxr.Gf.Vec3f(0.0, 0.0, 0.0))
            rev.CreateLocalRot1Attr(pxr.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            rev.CreateAxisAttr("X")
            rev.CreateLowerLimitAttr(-float(cfg["max_deg"]))    # pressing down turns the lever about -x
            rev.CreateUpperLimitAttr(0.0)                       # the horizontal stop
            drive = pxr.UsdPhysics.DriveAPI.Apply(rev.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            # N m per degree: return_torque at max_deg; the target sits preload_deg above the horizontal stop, so the
            # lever rests firmly on the stop instead of sagging under its own weight
            drive.CreateStiffnessAttr(cfg["return_torque"] / (cfg["max_deg"] + cfg["preload_deg"]))
            drive.CreateDampingAttr(cfg["return_damping"])
            drive.CreateTargetPositionAttr(float(cfg["preload_deg"]))

    if built:
        drive = pxr.UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"{path}/joint"), "angular")
    door["handle"] = {"path": path, "drive": drive, "R_rel0": R_rel, "leaf": leaf, "angle": 0.0, "rest": float(cfg["preload_deg"]),
                      "unlatch": math.radians(cfg["unlatch_deg"]), "band": cfg["latch_band"],
                      "k": cfg["latch_stiffness"], "d": cfg["latch_damping"]}
    return {"handle": path, "height": round(float(v - geo["bottom_v"]), 3), "thickness": round(thickness, 3),
            "model_handle": geo["handle_wv"] is not None, "collision_off": geo["collision_off"]}


def _leaf_geometry(obj, leaf, R_leaf, leaf_pos, hinge_local, cfg):
    """The leaf's axes and slab in its own frame, and the model's own handle (flattened here), from its meshes."""
    import numpy as np
    import omnigibson as og
    import omnigibson.lazy as lazy

    pxr = lazy.pxr
    cache = pxr.UsdGeom.XformCache()
    R, t = R_leaf.numpy().astype(np.float64), leaf_pos.numpy().astype(np.float64)
    meshes = []
    for m in leaf.visual_meshes.values():
        pts = m.prim.GetAttribute("points").Get()
        if pts is None:
            continue
        M = np.array(cache.GetLocalToWorldTransform(m.prim), dtype=np.float64).T      # mesh -> world
        P = np.asarray(pts, dtype=np.float64)
        L = ((P @ M[:3, :3].T + M[:3, 3]) - t) @ R                                   # mesh -> leaf frame
        meshes.append((m.prim, M, L))
    allp = np.concatenate([L for _, _, L in meshes])
    v_axis = int(np.argmax(np.abs(R[2, :])))                                       # leaf axis closest to world up
    others = [k for k in range(3) if k != v_axis]
    ext = allp.max(0) - allp.min(0)
    n_axis = min(others, key=lambda k: ext[k])
    w_axis = max(others, key=lambda k: ext[k])
    hinge_w = float(hinge_local[w_axis])
    lo_w, hi_w = allp[:, w_axis].min(), allp[:, w_axis].max()
    free_w = hi_w if abs(hi_w - hinge_w) > abs(lo_w - hinge_w) else lo_w
    mid_w = (hinge_w + free_w) / 2
    on_hinge_half = (allp[:, w_axis] - mid_w) * (hinge_w - mid_w) > 0
    a, b = np.percentile(allp[on_hinge_half, n_axis], [1, 99])                     # the slab's two faces
    margin = cfg["handle_margin"]
    handle_wv = None
    with og.sim.editing_usd():
        for prim, M, L in meshes:
            out = ((L[:, n_axis] < a - margin) | (L[:, n_axis] > b + margin)) & ((L[:, w_axis] - mid_w) * (free_w - mid_w) > 0)
            if not out.any():
                continue
            if handle_wv is None:
                handle_wv = (float(np.median(L[out, w_axis])), float(np.median(L[out, v_axis])))
            L2 = L.copy()
            L2[out, n_axis] = np.clip(L2[out, n_axis], a, b)                           # flat onto the face
            W = L2 @ R.T + t
            Pm = (W - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
            prim.GetAttribute("points").Set(pxr.Vt.Vec3fArray.FromNumpy(Pm.astype(np.float32)))
    # collision pieces wholly outside the slab are the old handle's: off, so they can't block a hand
    off = []
    for name, c in leaf.collision_meshes.items():
        pts = c.prim.GetAttribute("points").Get() if c.prim.HasAttribute("points") else None
        if pts is None:
            continue
        M = np.array(cache.GetLocalToWorldTransform(c.prim), dtype=np.float64).T
        L = ((np.asarray(pts, dtype=np.float64) @ M[:3, :3].T + M[:3, 3]) - t) @ R
        if L[:, n_axis].min() > b + margin or L[:, n_axis].max() < a - margin:
            with og.sim.editing_usd():
                c.prim.GetAttribute("physics:collisionEnabled").Set(False)
            off.append(name)
    return {"axes": (n_axis, w_axis, v_axis), "slab": (float(a), float(b)), "free_w": float(free_w),
            "bottom_v": float(allp[:, v_axis].min()), "handle_wv": handle_wv, "collision_off": off}


def handle_angle(handle):
    """The lever's turn from horizontal, radians, negative when pressed down."""
    import omni.physx
    import omnigibson.utils.transform_utils as T
    import torch as th

    tf = omni.physx.get_physx_interface().get_rigidbody_transformation(handle["path"])
    r = tf["rotation"]
    R_handle = T.quat2mat(th.tensor([r[0], r[1], r[2], r[3]], dtype=th.float32))
    _, leaf_quat = handle["leaf"].get_position_orientation()
    R = handle["R_rel0"].T @ (T.quat2mat(leaf_quat).T @ R_handle)
    return math.atan2(float(R[2, 1]), float(R[1, 1]))


def press(door, degrees):
    """Hold the lever turned down by `degrees` (0 releases it), as a hand would; for tests and scripted demos."""
    import omnigibson as og

    h = door["handle"]
    with og.sim.editing_usd():
        h["drive"].GetTargetPositionAttr().Set(-float(degrees) if degrees else h["rest"])


def _step(dt):
    for door in _doors:
        view, dofs = door["view"], door["dofs"]
        q = view.get_joint_positions(joint_indices=dofs)
        qd = view.get_joint_velocities(joint_indices=dofs)
        if q is None or qd is None:      # the view is invalid for a step while the stage reloads or renders only
            continue
        k, d, h = door["k"], door["d"], door["handle"]
        door["open"] = abs(float(q[0, 0]) - door["closed"])
        if h is not None:
            h["angle"] = handle_angle(h)
            door["latched"] = abs(float(q[0, 0]) - door["closed"]) < h["band"] and h["angle"] > -h["unlatch"]
            if door["latched"]:
                k, d = h["k"], h["d"]
            elif door["open"] < h["band"] and h["angle"] <= -h["unlatch"]:
                door["released"] = True     # unlatched by the lever, not merely still open
        view.set_joint_efforts(-k * (q - door["closed"]) - d * qd + door["push"], joint_indices=dofs)


def _subscribe():
    global _subscription
    if _subscription is None:
        import omni.physx

        _subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(_step)


def clear():
    """Stop every closer and latch (before a new instance or scene); handles already built stay in the stage."""
    global _subscription
    _doors.clear()
    _subscription = None


def apply(scene, cfg, seed=0):
    """Spring (and, with `handles`, fit levers to) the doors in `scene` per `cfg`, replacing any earlier closers;
    returns one dict per sprung hinge."""
    from omnigibson.utils.constants import JointType

    clear()
    rng = random.Random(seed)
    types = {"revolute": JointType.JOINT_REVOLUTE, "prismatic": JointType.JOINT_PRISMATIC}
    wanted = {types[t] for t in cfg.get("joint_types", ["revolute"])}
    categories = set(cfg.get("categories", ["door"]))
    names = set(cfg.get("objects") or [])       # empty: every door of those categories
    handles = cfg.get("handles") or {}
    sprung = []
    for obj in sorted(scene.objects, key=lambda o: o.name):
        if getattr(obj, "category", None) not in categories or (names and obj.name not in names) \
                or rng.random() >= float(cfg.get("fraction", 1.0)):
            continue
        k = rng.uniform(*cfg["stiffness"])
        tau = rng.uniform(*cfg["time_constant"])
        for name, joint in obj.joints.items():
            if joint.joint_type not in wanted:
                continue
            door = spring(obj, joint, k, tau)
            row = {"object": obj.name, "joint": name, "stiffness": round(k, 3), "damping": round(door["d"], 3),
                   "closed": round(door["closed"], 4)}
            if handles.get("enabled"):
                row.update(add_handle(door, handles))
            sprung.append(row)
    return sprung
