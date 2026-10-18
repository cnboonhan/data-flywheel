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


HUES = {"red": 0.0, "orange": 30.0, "yellow": 55.0, "green": 120.0, "blue": 215.0, "purple": 265.0, "pink": 330.0}


def hue_of(rgb):
    r, g, b = [float(c) / 255.0 for c in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 0.08:
        return None
    if mx == r:
        h = 60.0 * (((g - b) / (mx - mn)) % 6)
    elif mx == g:
        h = 60.0 * ((b - r) / (mx - mn) + 2)
    else:
        h = 60.0 * ((r - g) / (mx - mn) + 4)
    return h


def hue_distance(h, target):
    d = abs(h - target) % 360.0
    return min(d, 360.0 - d)


opened = open_gripper(arm=arm)
# SAM3 confuses colour names (it handed us the blue block for "purple"): take every candidate it offers and keep the
# one whose pixels actually have the colour's hue
obs0 = get_observation()
rgb_img = obs0.cameras["head"].rgb
seg = segment_text("head", color + " block")
chosen = None
candidates = []
if seg.ok:
    for s in seg.segmentations:
        m = s.mask
        ys = [i for i in range(len(m)) if any(m[i])]
        if not ys:
            continue
        xs = [j for i in ys[:: max(1, len(ys) // 8)] for j in range(len(m[i])) if m[i][j]]
        cy_px, cx_px = sum(ys) / len(ys), sum(xs) / len(xs)
        pix = rgb_img[int(cy_px)][int(cx_px)]
        h = hue_of(pix)
        dist = 999.0 if h is None else hue_distance(h, HUES.get(color, 0.0))
        candidates.append((round(float(s.score), 3), int(cx_px), int(cy_px), [int(c) for c in pix], None if h is None else round(h), round(dist)))
        if dist < 40.0 and (chosen is None or dist < chosen[0]):
            chosen = (dist, s)
report["candidates"] = candidates
if chosen is None:
    report["localized"] = False
    report["localize_error"] = "no candidate with the right hue"
    result = report
else:
    cloud = mask_to_point_cloud(chosen[1].mask, "head", target_frame=base)
    report["localized"] = True
    points = [p for p in cloud.points if 0.3 < float(p[2]) < 1.5 and float(p[0]) == float(p[0])]   # finite, on the table
    zs = sorted([float(p[2]) for p in points])
    n = len(points)
    top = zs[9 * n // 10]
    # the top face alone (within 1 cm of the top): its centroid is the block's x, y; the visible side faces would bias it
    face = [p for p in points if float(p[2]) > top - 0.012] or points[-1:]
    cx = sum(float(p[0]) for p in face) / len(face)
    cy = sum(float(p[1]) for p in face) / len(face)
    report["block"] = (cx, cy, top, n, len(face))
    # The omnipicker's centre link floats ~2.7 cm beyond the open fingertips (probed: it descends below the table
    # top without stalling), and the pads sit behind the tips; so to put the pads at the block's middle the centre
    # link goes PAD_OFFSET beyond the middle along the approach axis, about 1 cm below the table for a 5 cm block.
    PAD_OFFSET = 0.035
    mid = (cx, cy, top - 0.025)
    # The planner interpolates in joint space and knows nothing about the table: from the home pose the gripper
    # can sweep low across it and knock blocks about. So: straight up first, then over the block at height, then down.
    home = get_robot_state().end_effector_poses[arm]
    up = move_to_pose({"position": (float(home.position[0]), float(home.position[1]), float(home.position[2]) + 0.15),
                       "quaternion_wxyz": [float(v) for v in home.quaternion_wxyz], "frame": base},
                      tolerance=0.03, max_steps=200, arm=arm)
    report["lift_from_home"] = bool(up.ok)
    done = False
    q = None
    for tilt in (0.0, 20.0, 35.0):
        q = approach_quat(tilt)
        t = math.radians(tilt)
        axis = (math.sin(t), 0.0, -math.cos(t))
        grasp = tuple(mid[i] + PAD_OFFSET * axis[i] for i in range(3))
        high = tuple(grasp[i] - 0.25 * axis[i] for i in range(3))
        pregrasp = tuple(grasp[i] - 0.12 * axis[i] for i in range(3))
        pre = move_to_pose({"position": high, "quaternion_wxyz": q, "frame": base}, tolerance=0.03, max_steps=300, arm=arm)
        report["high_tilt_%d" % int(tilt)] = (bool(pre.ok), None if pre.error is None else str(pre.error.message))
        if pre.ok:
            break
        q = None
    if q is not None:
        # the block should still be where it was: the head pixel of the estimate must show its colour
        obs = get_observation()
        cam = obs.cameras["head"]
        K = cam.intrinsics
        cw, cxq, cyq, czq = [float(v) for v in cam.camera_pose.quaternion_wxyz]
        Rm = ((1 - 2 * (cyq * cyq + czq * czq), 2 * (cxq * cyq - cw * czq), 2 * (cxq * czq + cw * cyq)),
              (2 * (cxq * cyq + cw * czq), 1 - 2 * (cxq * cxq + czq * czq), 2 * (cyq * czq - cw * cxq)),
              (2 * (cxq * czq - cw * cyq), 2 * (cyq * czq + cw * cxq), 1 - 2 * (cxq * cxq + cyq * cyq)))
        ct = [float(v) for v in cam.camera_pose.position]
        dv = [mid[i] - ct[i] for i in range(3)]
        pc = [sum(Rm[r][k] * dv[r] for r in range(3)) for k in range(3)]
        u = int(float(K[0][0]) * pc[0] / pc[2] + float(K[0][2]))
        v = int(float(K[1][1]) * pc[1] / pc[2] + float(K[1][2]))
        rgb = [int(c) for c in cam.rgb[v][u]]
        report["block_pixel_after_approach"] = (u, v, rgb)
        pre = move_to_pose({"position": pregrasp, "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
        report["pregrasp"] = (bool(pre.ok), None if pre.error is None else str(pre.error.message)[:60])
        report["grasp_target"] = grasp
        # the finger pads sit ~2.5 cm behind the gripper-centre link (URDF), so the centre goes 4.5 cm below the
        # block's top for the pads to meet its middle; one attempt, the episode is only ~330 steps
        def progress():
            try:
                return [(p["name"], p["score"]) for p in get_observation().task_context.metadata["progress"]["subtasks"]]
            except Exception:
                return None
        down = move_to_pose({"position": grasp, "quaternion_wxyz": q, "frame": base},
                            tolerance=0.012, max_steps=300, arm=arm)
        report["grasp_move"] = (bool(down.ok), None if down.error is None else str(down.error.message)[:60])
        report["ee_at_grasp"] = [float(v) for v in get_robot_state().end_effector_poses[arm].position]
        closed = close_gripper(arm=arm)
        g = float(get_robot_state().gripper_positions[arm])
        report["gripper_after_close"] = g
        report["progress_after_close"] = progress()
        lift = move_to_pose({"position": (grasp[0], grasp[1], grasp[2] + 0.15), "quaternion_wxyz": q, "frame": base},
                            tolerance=0.02, max_steps=300, arm=arm)
        report["lift"] = (bool(lift.ok), None if lift.error is None else str(lift.error.message))
        report["gripper_after_lift"] = float(get_robot_state().gripper_positions[arm])
        report["progress_after_lift"] = progress()
        done = True
    report["done"] = done
    result = report
print(report)
