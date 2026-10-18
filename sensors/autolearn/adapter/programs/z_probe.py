# Where are the fingertips relative to the gripper-centre link? Descend over an empty patch of table next to the
# block, open then closed, until the arm stalls on the table; the stall height minus the table height is the tips'
# extension beyond the centre link in that state.
import math
report = {}
state = get_robot_state()
base = state.base_frame
language = (get_observation().task_context.language or "").lower()
arm = "primary" if "left" in language else "secondary"
report["arm"] = arm
sign = 1.0 if arm == "primary" else -1.0
# approach: 20 deg tilt towards +x, fingers along base y (the quaternion the pick programs use)
def quat_from_columns(x, y, z):
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    t = m00 + m11 + m22
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
t = math.radians(20.0)
z = (math.sin(t), 0.0, -math.cos(t))
y = (0.0, 1.0, 0.0)
x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
q = quat_from_columns(x, y, z)
# the table height from the head camera's depth: the lowest band of valid points in front of the robot
obs = get_observation()
cam = obs.cameras["head"]
depth = cam.depth_m
h, w = len(depth), len(depth[0])
rows = [depth[i] for i in range(0, h, 4)]
vals = []
# project a sparse grid of pixels to base-frame points through the public API (a mask of the lower half of the image)
mask = [[(i > h // 2) and (j % 8 == 0) and (i % 8 == 0) for j in range(w)] for i in range(h)]
cloud = mask_to_point_cloud(mask, "head", target_frame=base)
pts = [p for p in cloud.points if 0.3 < float(p[2]) < 1.5]
zs = sorted([float(p[2]) for p in pts])
table = zs[len(zs) // 2]     # the median height of the lower image half is the table top
report["table_z"] = table
# an empty spot: 0.65 m ahead, 0.25 m to the arm's side
spot = (0.65, sign * 0.25, table)
for label, g in (("open", 1.0), ("closed", 0.0)):
    set_gripper(g, arm=arm)
    pre = move_to_pose({"position": (spot[0], spot[1], table + 0.12), "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
    stall = None
    last = None
    for k in range(12):
        target_z = table + 0.10 - 0.01 * k
        r = move_to_pose({"position": (spot[0], spot[1], target_z), "quaternion_wxyz": q, "frame": base}, tolerance=0.006, max_steps=120, arm=arm)
        ee = get_robot_state().end_effector_poses[arm]
        last = (target_z, float(ee.position[2]), bool(r.ok))
        if not r.ok or (float(ee.position[2]) - target_z) > 0.008:
            stall = last
            break
    report[label] = {"pregrasp_ok": bool(pre.ok), "stall": stall, "last": last, "gripper": float(get_robot_state().gripper_positions[arm])}
    if stall is not None:
        report[label]["tips_beyond_centre_cm"] = (stall[1] - table) * 100
    move_to_pose({"position": (spot[0], spot[1], table + 0.12), "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
print(report)
result = report
