"""Kinematics check against the simulator itself: set the G2 USD's joints to saved configurations and compare the
centre links' poses (relative to base_link) with the URDF FK the adapter plans on."""
import json
import sys
import numpy as np
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "enable_crashreporter": False})
from pxr import Usd, UsdGeom  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

world = World(stage_units_in_meters=1.0)
add_reference_to_stage("/geniesim_assets/robot/G2_omnipicker/robot_fix.usda", "/genie")
world.reset()
robot = SingleArticulation("/genie")
robot.initialize()
names = list(robot.dof_names)
stage = omni.usd.get_context().get_stage()
print("arm/waist dofs:", [n for n in names if "arm" in n or "body" in n], flush=True)


def T(path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T


lim = robot.get_dof_limits() if hasattr(robot, "get_dof_limits") else None
for cfg in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/configs.json")):
    q = np.array(robot.get_joint_positions(), dtype=float)
    missing = []
    for n, v in cfg["joints"].items():
        if n in names:
            q[names.index(n)] = v
        else:
            missing.append(n)
    robot.set_joint_positions(q)
    for _ in range(30):
        robot.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=False)
    q2 = np.array(robot.get_joint_positions(), dtype=float)
    drift = {n: round(float(q2[names.index(n)] - v), 3) for n, v in cfg["joints"].items() if n in names and abs(q2[names.index(n)] - v) > 0.01}
    Tb = np.linalg.inv(T("/genie/base_link"))
    out = []
    for arm, link in (("primary", "gripper_l_center_link"), ("secondary", "gripper_r_center_link")):
        p = (Tb @ T(f"/genie/{link}"))[:3, 3]
        u = np.array(cfg["urdf_centre"][arm])
        out.append(f"{arm}: usd {np.round(p, 3)} urdf {np.round(u, 3)} diff {np.round((p - u) * 100, 1)} cm")
    print(f"\n{cfg['name']}: missing {missing}; joints that didn't hold {drift}\n  " + "\n  ".join(out), flush=True)
app.close()
