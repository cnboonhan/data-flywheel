"""Where the omnipicker's fingers are, from the simulator: load the G2 robot USD headless, drive each gripper open
then closed, step physics so the linkage settles, and print the finger links' origins and mesh bounds in the
gripper-centre-link frame. Run inside the geniesim3 container with Isaac's python.
"""
import sys
import numpy as np
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "enable_crashreporter": False})
from pxr import Usd, UsdGeom, Gf  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

USD = "/geniesim_assets/robot/G2_omnipicker/robot_fix.usda"
OPEN, CLOSED = 0.785, 0.0   # the USD joint runs 0..45 deg; the benchmark reports it negated

world = World(stage_units_in_meters=1.0)
add_reference_to_stage(USD, "/genie")
world.reset()
robot = SingleArticulation("/genie")
robot.initialize()
names = list(robot.dof_names)
print("dofs", len(names), [n for n in names if "gripper" in n], flush=True)
stage = omni.usd.get_context().get_stage()
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)


def world_T(path):
    prim = stage.GetPrimAtPath(path)
    m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T
    return m  # column-major Gf -> row-major 4x4 with translation in [:3, 3]


def bounds_in(path, T_inv):
    prim = stage.GetPrimAtPath(path)
    pts = []
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            M = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T
            v = np.array(mesh.GetPointsAttr().Get(), dtype=float)
            if len(v) == 0:
                continue
            vw = (M[:3, :3] @ v.T).T + M[:3, 3]
            vc = (T_inv[:3, :3] @ vw.T).T + T_inv[:3, 3]
            pts.append(vc)
    if not pts:
        return None
    pts = np.concatenate(pts)
    return pts


hold = robot.get_joint_positions()
for side, idx in (("r", "8"),):
    ctrl = [f"idx{idx}1_gripper_{side}_outer_joint1", f"idx{int(idx) - 1}1_gripper_{side}_inner_joint1"]
    ids = [names.index(n) for n in ctrl if n in names]
    for label, val in (("open", OPEN), ("half", 0.4), ("closed", CLOSED)):
        target = np.array(hold, dtype=float)
        target[ids] = val
        robot.set_joint_positions(target)
        for _ in range(150):
            robot.apply_action(ArticulationAction(joint_positions=target))
            world.step(render=False)
        q = robot.get_joint_positions()
        fj = {n: round(float(q[i]), 3) for i, n in enumerate(names) if f"gripper_{side}" in n}
        print(f"\n=== gripper_{side} {label}: finger joints {fj}", flush=True)
        Tc = world_T(f"/genie/gripper_{side}_center_link")
        Tc_inv = np.linalg.inv(Tc)
        for l in ["base_link", "camera_link"] + [f"{k}_link{i}" for k in ("outer", "inner") for i in (1, 2, 3, 4)]:
            path = f"/genie/gripper_{side}_{l}"
            if not stage.GetPrimAtPath(path).IsValid():
                print("  missing", path)
                continue
            o = Tc_inv @ world_T(path)
            pts = bounds_in(path, Tc_inv)
            ext = "" if pts is None else f" mesh x[{pts[:, 0].min() * 100:+.1f},{pts[:, 0].max() * 100:+.1f}] y[{pts[:, 1].min() * 100:+.1f},{pts[:, 1].max() * 100:+.1f}] z[{pts[:, 2].min() * 100:+.1f},{pts[:, 2].max() * 100:+.1f}] cm"
            print(f"  {l:12s} origin {np.round(o[:3, 3] * 100, 1)} cm{ext}", flush=True)
        # the fingertips: the mesh points with the largest z (furthest along the approach) on each finger assembly
        for k in ("outer", "inner"):
            allp = [bounds_in(f"/genie/gripper_{side}_{k}_link{i}", Tc_inv) for i in (1, 2, 3, 4)]
            allp = np.concatenate([p for p in allp if p is not None])
            tip = allp[allp[:, 2].argmax()]
            band = allp[allp[:, 2] > tip[2] - 0.01]
            print(f"  {k}: tip {np.round(tip * 100, 1)} cm; top-1cm band x[{band[:, 0].min() * 100:+.1f},{band[:, 0].max() * 100:+.1f}] y[{band[:, 1].min() * 100:+.1f},{band[:, 1].max() * 100:+.1f}]", flush=True)
app.close()
