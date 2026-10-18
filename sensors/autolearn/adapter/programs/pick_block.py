# Pick the block the instruction names, with the arm it names: SAM3 on the head camera locates it, the gripper
# comes straight down onto it, closes, and lifts. A hand-written first program for Genie Sim's pick_block_color.
import math

report = {}
state = get_robot_state()
base = state.base_frame
try:
    language = get_observation().task_context.language.lower()
except Exception:
    language = ""
report["language"] = language
arm = "primary" if "left" in language else "secondary"
color = "red"
for word in ["red", "yellow", "blue", "green", "purple", "orange", "black", "white", "pink"]:
    if word in language:
        color = word
        break
report["arm"], report["color"] = arm, color


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


def approach_quat(tilt_deg):
    # gripper +z (approach) pointing down, tilted by tilt_deg towards +x (away from the robot); fingers along base y
    t = math.radians(tilt_deg)
    z = (math.sin(t), 0.0, -math.cos(t))
    y = (0.0, 1.0, 0.0)
    x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
    return quat_from_columns(x, y, z)


opened = open_gripper(arm=arm)
target = localize_object(color + " block", camera_name="head", target_frame=base)
report["localized"] = bool(target.ok)
if not target.ok:
    report["localize_error"] = None if target.error is None else str(target.error.message)
    result = report
else:
    points = [p for p in target.point_cloud.points if 0.3 < float(p[2]) < 1.5 and float(p[0]) == float(p[0])]   # finite, on the table
    zs = sorted([float(p[2]) for p in points])
    n = len(points)
    top = zs[9 * n // 10]
    # the top face alone (within 1 cm of the top): its centroid is the block's x, y; the visible side faces would bias it
    face = [p for p in points if float(p[2]) > top - 0.012] or points[-1:]
    cx = sum(float(p[0]) for p in face) / len(face)
    cy = sum(float(p[1]) for p in face) / len(face)
    report["block"] = (cx, cy, top, n, len(face))
    done = False
    q = None
    for tilt in (0.0, 25.0, 45.0):
        q = approach_quat(tilt)
        pre = move_to_pose({"position": (cx, cy, top + 0.13), "quaternion_wxyz": q, "frame": base},
                           tolerance=0.02, max_steps=300, arm=arm)
        report["pregrasp_tilt_%d" % int(tilt)] = (bool(pre.ok), None if pre.error is None else str(pre.error.message))
        if pre.ok:
            break
        q = None
    if q is not None:
        # re-localize from the wrist camera, 13 cm above the block: a head-camera estimate at 0.7 m is a few cm off
        wrist = "left_wrist" if arm == "primary" else "right_wrist"
        near = localize_object(color + " block", camera_name=wrist, target_frame=base)
        report["wrist_localized"] = bool(near.ok)
        if near.ok:
            pts = near.point_cloud.points
            nz = sorted([float(p[2]) for p in pts])
            ntop = nz[9 * len(nz) // 10]
            nface = [p for p in pts if float(p[2]) > ntop - 0.012]
            ncx = sum(float(p[0]) for p in nface) / len(nface)
            ncy = sum(float(p[1]) for p in nface) / len(nface)
            report["block_from_wrist"] = (ncx, ncy, ntop, len(pts), len(nface))
            report["head_vs_wrist_cm"] = ((ncx - cx) * 100, (ncy - cy) * 100, (ntop - top) * 100)
            cx, cy, top = ncx, ncy, ntop
        # descend in stages: the gripper-centre link sits beyond the finger pads, so the pads only reach the block
        # when the centre is lower than the block's middle; stop at the first depth where the fingers close on it
        attempts = []
        for depth in (0.02, 0.035, 0.05):
            open_gripper(arm=arm)
            down = move_to_pose({"position": (cx, cy, top - depth), "quaternion_wxyz": q, "frame": base},
                                tolerance=0.012, max_steps=300, arm=arm)
            closed = close_gripper(arm=arm)
            g = float(get_robot_state().gripper_positions[arm])
            attempts.append((depth, bool(down.ok), None if down.error is None else str(down.error.message)[:60], g))
            if down.ok and g > 0.12:
                break
        report["attempts"] = attempts
        lift = move_to_pose({"position": (cx, cy, top + 0.15), "quaternion_wxyz": q, "frame": base},
                            tolerance=0.02, max_steps=300, arm=arm)
        report["lift"] = (bool(lift.ok), None if lift.error is None else str(lift.error.message))
        report["gripper_after_lift"] = float(get_robot_state().gripper_positions[arm])
        done = True
    report["done"] = done
    result = report
print(report)
