"""Which joint is which: move one right-arm joint at a time in the benchmark and check the head camera against FK.

For each joint, the arm goes from home to home + delta on that joint alone (a 40-row chunk), the reported joints are
checked against the command, and the FK gripper base and centre link are projected into the head depth image: the
depth the camera sees there must match FK's depth (a solid link) when the model and the simulator agree on that
joint, and shows the table or the floor when they don't.

    aspire/.venv-pyroki/bin/python adapter/probes/joint_probe.py [--arm secondary] [--delta 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # the adapter
from geniesim_adapter import ARM_JOINTS, CorobotBridge, G2IkProvider, G2Model, GenieSimAdapter, EE_LINKS  # noqa: E402
from cap_harness.geometry import quaternion_wxyz_to_matrix  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--arm", default="secondary")
p.add_argument("--delta", type=float, default=0.5)
p.add_argument("--port", type=int, default=8999)
p.add_argument("--urdf", type=Path, default=HERE.parent / "GenieSimAssets/robot/curobo_robot/assets/robot/G2/G2_omnipicker_fixed_dual.urdf")
p.add_argument("--usda", type=Path, default=HERE.parent / "GenieSimAssets/robot/G2_omnipicker/robot_fix.usda")
args = p.parse_args()

model = G2Model(args.urdf, args.usda)
bridge = CorobotBridge(args.port)
bridge.start()
adapter = GenieSimAdapter(bridge, model, suite="joint_probe")
obs = adapter.reset()
side = 0 if args.arm == "primary" else 1
base_link = "gripper_l_base_link" if side == 0 else "gripper_r_base_link"
home = np.asarray(adapter._raw["states"]["arm_joint_states"], dtype=float)[7 * side:7 * side + 7].copy()
print("home", np.round(home, 3), flush=True)


def check(label):
    obs = adapter.get_observation()
    cam = obs.cameras["head"]
    cam_from_base = np.linalg.inv(np.r_[np.c_[quaternion_wxyz_to_matrix(cam.camera_pose.quaternion_wxyz), cam.camera_pose.position], [[0, 0, 0, 1]]])
    joints = adapter._joints()
    got = np.asarray(adapter._raw["states"]["arm_joint_states"], dtype=float)[7 * side:7 * side + 7]
    out = []
    for name, link in (("centre", EE_LINKS[args.arm]), ("base", base_link)):
        pos = model.fk(joints, link)[:3, 3]
        pc = cam_from_base @ np.r_[pos, 1.0]
        uv = cam.intrinsics @ (pc[:3] / pc[2])
        u, v = int(uv[0]), int(uv[1])
        if 0 <= v < cam.depth_m.shape[0] and 0 <= u < cam.depth_m.shape[1]:
            win = cam.depth_m[max(0, v - 8):v + 9, max(0, u - 8):u + 9]
            near = float(np.min(win[win > 0.05])) if np.any(win > 0.05) else -1
            out.append(f"{name} px({u},{v}) fk {pc[2]:.3f} seen {float(cam.depth_m[v, u]):.3f} near17 {near:.3f}")
        else:
            out.append(f"{name} off-image")
    print(f"{label}: joints {np.round(got, 3)} | " + " | ".join(out), flush=True)
    return got


check("home")
for j in range(7):
    target = home.copy()
    delta = args.delta if home[j] + args.delta < model.limits[args.arm][1][j] - 0.05 else -args.delta
    target[j] += delta
    rows = np.linspace(home, target, 40)
    adapter._exchange(adapter._chunk(40, {args.arm: target}, None, rows))
    adapter._exchange(adapter._chunk(10))
    got = check(f"joint{j + 1} {delta:+.1f}")
    print(f"   command echo error {np.round(got - target, 3)}", flush=True)
    rows = np.linspace(target, home, 40)
    adapter._exchange(adapter._chunk(40, {args.arm: home}, None, rows))
adapter.finish_episode()
