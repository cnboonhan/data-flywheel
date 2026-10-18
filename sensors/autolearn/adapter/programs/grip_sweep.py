# Calibrate the omnipicker's pinch point: put the gripper-centre link at the localized block's centre, then at eight
# spots 3 cm around it, closing at each; the spot where the fingers stop short of closed is where the block really is.
report = {}
state = get_robot_state()
base = state.base_frame
try:
    language = get_observation().task_context.language.lower()
except Exception:
    language = ""
arm = "primary" if "left" in language else "secondary"
color = "red"
for word in ["red", "yellow", "blue", "green", "purple", "orange", "black", "white", "pink"]:
    if word in language:
        color = word
        break
report["arm"], report["color"] = arm, color
# approach tilted 20 deg towards +x (straight down is just out of reach), fingers along base y
import math
t = math.radians(20.0)
z = (math.sin(t), 0.0, -math.cos(t))
y = (0.0, 1.0, 0.0)
x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])


def quat_from_columns(x, y, z):
    # wxyz quaternion of the rotation whose columns are x, y, z (right-handed, unit)
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


q = quat_from_columns(x, y, z)

open_gripper(arm=arm)
target = localize_object(color + " block", camera_name="head", target_frame=base)
report["localized"] = bool(target.ok)
if target.ok:
    points = [p for p in target.point_cloud.points if 0.3 < float(p[2]) < 1.5 and float(p[0]) == float(p[0])]   # finite, on the table
    zs = sorted([float(p[2]) for p in points])
    top = zs[9 * len(zs) // 10]
    face = [p for p in points if float(p[2]) > top - 0.012] or points[-1:]
    cx = sum(float(p[0]) for p in face) / len(face)
    cy = sum(float(p[1]) for p in face) / len(face)
    report["block"] = (cx, cy, top)
    gz = top - 0.03
    pre = move_to_pose({"position": (cx, cy, top + 0.12), "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
    report["pregrasp"] = (bool(pre.ok), None if pre.error is None else str(pre.error.message)[:60])
    sweep = []
    for dx, dy in [(0, 0), (0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03), (0.03, 0.03), (0.03, -0.03), (-0.03, 0.03), (-0.03, -0.03)]:
        open_gripper(arm=arm)
        up = move_to_pose({"position": (cx + dx, cy + dy, gz + 0.06), "quaternion_wxyz": q, "frame": base}, tolerance=0.015, max_steps=200, arm=arm)
        down = move_to_pose({"position": (cx + dx, cy + dy, gz), "quaternion_wxyz": q, "frame": base}, tolerance=0.012, max_steps=200, arm=arm)
        closed = close_gripper(arm=arm)
        g = float(get_robot_state().gripper_positions[arm])
        err = None
        for r in (up, down, closed):
            if not r.ok:
                err = (str(r.status), None if r.error is None else str(r.error.message)[:70], bool(r.terminated))
                break
        sweep.append((dx, dy, bool(up.ok and down.ok and closed.ok), round(g, 3), err))
        if g > 0.12 or (err is not None and err[2]):
            break
    report["sweep"] = sweep
    best = max(sweep, key=lambda e: e[3])
    report["best"] = best
    if best[3] > 0.12:
        lift = move_to_pose({"position": (cx + best[0], cy + best[1], gz + 0.15), "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
        report["lift"] = (bool(lift.ok), float(get_robot_state().gripper_positions[arm]))
print(report)
result = report
