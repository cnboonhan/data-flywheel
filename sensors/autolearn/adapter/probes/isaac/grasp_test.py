"""Grasp unit test for the omnipicker, no perception or IK: the G2 USD is loaded headless with its base fixed, a
5 cm cube is placed between the right fingers at a chosen offset from the centre link, the gripper closes, and the
cube's pose relative to the centre link is read after closing and after the fingers are held closed while gravity
acts on the cube (the arm doesn't move; a held cube stays put, an expelled one falls). Prints one row per trial.
"""
import itertools
import numpy as np
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "enable_crashreporter": False})
from pxr import Usd, UsdGeom, UsdPhysics, Gf  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

USD = "/geniesim_assets/robot/G2_omnipicker/robot_fix.usda"
OPEN, CLOSED = 0.785, 0.0
SIDE, IDX = "r", "8"
CUBE = 0.05

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120)
add_reference_to_stage(USD, "/genie")
world.reset()
robot = SingleArticulation("/genie")
robot.initialize()
names = list(robot.dof_names)
stage = omni.usd.get_context().get_stage()
ctrl = [names.index(f"idx{IDX}1_gripper_{SIDE}_outer_joint1"), names.index(f"idx{int(IDX) - 1}1_gripper_{SIDE}_inner_joint1")]
hold = np.array(robot.get_joint_positions(), dtype=float)


def centre_T():
    prim = stage.GetPrimAtPath(f"/genie/gripper_{SIDE}_center_link")
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T


def drive(val, steps, pin=None):
    t = hold.copy()
    t[ctrl] = val
    for _ in range(steps):
        if pin is not None:          # the table holds the cube while the fingers close: pin it in place
            cube.set_world_pose(*pin)
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        robot.apply_action(ArticulationAction(joint_positions=t))
        world.step(render=False)


scene = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))


def joint():
    return float(robot.get_joint_positions()[ctrl[0]])


cube = None
# cube centre in the centre-link frame: along the finger axis (x), across it (y), along the approach (z; + is past
# the centre link, towards the tips), and the cube's yaw about the approach axis
for dx, dy, dz, yaw in [(0, 0, -0.01, 0), (0, 0, -0.02, 0), (0, 0, -0.03, 0), (0, 0, -0.04, 0), (0, 0, -0.05, 0), (0, 0, -0.06, 0),
                        (0, 0, -0.03, 30), (0, 0, -0.03, 45), (0.02, 0, -0.03, 0), (0.04, 0, -0.03, 0), (0, 0.02, -0.03, 0), (0, 0.04, -0.03, 0)]:
    drive(OPEN, 120)
    Tc = centre_T()
    # "down" for the cube is the approach direction, as when the gripper lifts a cube off a table
    g = Tc[:3, 2] / np.linalg.norm(Tc[:3, 2])
    scene.GetGravityDirectionAttr().Set(Gf.Vec3f(*[float(v) for v in g]))
    scene.GetGravityMagnitudeAttr().Set(9.81)
    Rz = np.array([[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0], [np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0], [0, 0, 1]])
    pos = Tc[:3, 3] + Tc[:3, :3] @ np.array([dx, dy, dz])
    R = Tc[:3, :3] @ Rz
    # rotation -> quaternion wxyz
    tr = np.trace(R)
    s = np.sqrt(max(tr + 1, 1e-9)) * 2
    q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if cube is None:
        cube = world.scene.add(DynamicCuboid("/World/cube", name="cube", position=pos, orientation=q, size=CUBE, mass=0.05))
    else:
        cube.set_world_pose(pos, q)
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
    # hold the cube still in the open fingers for a moment (kinematic for the placement), then release and close
    cube.set_default_state(position=pos, orientation=q)
    for _ in range(5):
        cube.set_world_pose(pos, q)
        cube.set_linear_velocity(np.zeros(3))
        world.step(render=False)
    drive(CLOSED, 120, pin=(pos, q))
    j_close = joint()
    p1, _ = cube.get_world_pose()
    rel1 = np.linalg.inv(Tc) @ np.append(p1, 1)
    drive(CLOSED, 240)   # two seconds with gravity on the cube
    j_hold = joint()
    p2, _ = cube.get_world_pose()
    rel2 = np.linalg.inv(Tc) @ np.append(p2, 1)
    held = np.linalg.norm(rel2[:3] - rel1[:3]) < 0.01
    print(f"cube at x{dx * 100:+.0f} y{dy * 100:+.0f} z{dz * 100:+.1f} yaw{yaw:3d}: joint after close {j_close:.3f}, after 2 s {j_hold:.3f}; "
          f"cube rel centre after close {np.round(rel1[:3] * 100, 1)} cm, after 2 s {np.round(rel2[:3] * 100, 1)} cm -> {'HELD' if held else 'lost'}", flush=True)
app.close()
