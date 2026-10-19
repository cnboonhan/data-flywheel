# Does the arm lag its reported joints? One big move, then four gripper commands (each snapshots the cameras with
# --debug and holds for ~25 ticks): if the rendered gripper keeps moving between snapshots while the reported pose
# stays put, the benchmark's observation comes back before the arm has arrived.
import math

report = {}
state = get_robot_state()
base = state.base_frame
arm = "secondary"
open_gripper(arm=arm)
home = state.end_effector_poses[arm]
t = math.radians(20.0)
z = (math.sin(t), 0.0, -math.cos(t))
y = (0.0, 1.0, 0.0)
x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
m00, m01, m02 = x[0], y[0], z[0]
m10, m11, m12 = x[1], y[1], z[1]
m20, m21, m22 = x[2], y[2], z[2]
s = math.sqrt(m00 + m11 + m22 + 1.0) * 2
q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
up = move_to_pose({"position": (float(home.position[0]), float(home.position[1]), float(home.position[2]) + 0.15),
                   "quaternion_wxyz": [float(v) for v in home.quaternion_wxyz], "frame": base}, tolerance=0.03, max_steps=200, arm=arm)
far = move_to_pose({"position": (0.80, -0.10, 0.80), "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
report["far"] = (bool(far.ok), None if far.error is None else str(far.error.message)[:60])
poses = []
for i in range(4):
    open_gripper(arm=arm)
    poses.append([round(float(v), 4) for v in get_robot_state().end_effector_poses[arm].position])
report["ee_after_each_snapshot"] = poses
print(report)
result = report
