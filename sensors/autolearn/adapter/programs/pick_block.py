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


def approach_frame(tilt_deg, yaw_deg):
    # gripper +z (approach) pointing down, tilted by tilt_deg towards +x (away from the robot); the fingers (gripper x)
    # along base -x turned by yaw_deg about base z, so they straddle a yawed cube's faces rather than its corners
    t = math.radians(tilt_deg)
    z = (math.sin(t), 0.0, -math.cos(t))
    s_, c_ = math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))
    y = (-s_, c_, 0.0)
    d = sum(y[i] * z[i] for i in range(3))
    y = [y[i] - d * z[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in y))
    y = tuple(v / n for v in y)
    x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
    return quat_from_columns(x, y, z), x, z


def face_yaw(face):
    # the yaw of a square face from its top-face points: the rotation whose axis-aligned bounding box is smallest,
    # folded into (-45, 45] so the fingers turn the least from their default
    best = (None, 0.0)
    for k in range(30):
        a = math.radians(3.0 * k)
        ca, sa = math.cos(a), math.sin(a)
        us = [ca * float(p[0]) + sa * float(p[1]) for p in face]
        vs = [-sa * float(p[0]) + ca * float(p[1]) for p in face]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best[0] is None or area < best[0]:
            best = (area, 3.0 * k)
    yaw = best[1]
    return yaw - 90.0 if yaw > 45.0 else yaw


# the scene's blocks: the "green" one is teal (hue ~170), the "purple" one navy (~245); matched within 40 deg
HUES = {"red": 0.0, "orange": 30.0, "yellow": 55.0, "green": 160.0, "blue": 215.0, "purple": 255.0, "pink": 330.0}


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


def localize(camera_name):
    # SAM3 confuses colour names (it handed us the blue block for "purple"): take every candidate it offers and
    # keep the one whose pixels actually have the colour's hue; then the block's top face from the depth cloud
    obs0 = get_observation()
    rgb_img = obs0.cameras[camera_name].rgb
    seg = segment_text(camera_name, color + " block")
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
    if chosen is None:
        return None, candidates
    cloud = mask_to_point_cloud(chosen[1].mask, camera_name, target_frame=base)
    points = [p for p in cloud.points if 0.3 < float(p[2]) < 1.5 and float(p[0]) == float(p[0])]   # finite, on the table
    if len(points) < 20:
        return None, candidates
    zs = sorted([float(p[2]) for p in points])
    n = len(points)
    top = zs[9 * n // 10]
    # the top face alone (within 1 cm of the top): its centroid is the block's x, y; the visible side faces would bias it
    face = [p for p in points if float(p[2]) > top - 0.012] or points[-1:]
    cx = sum(float(p[0]) for p in face) / len(face)
    cy = sum(float(p[1]) for p in face) / len(face)
    return (cx, cy, top, n, len(face), face), candidates


# The head camera is the reference: the adapter holds the head joints at zero so its FK pose is right (its block
# tops agree with the table height a stalled descent measures); the wrist camera's estimates wander by several cm
# with the viewpoint, so they're recorded for comparison only.
wrist = "left_wrist" if arm == "primary" else "right_wrist"
head_est, report["candidates"] = localize("head")
wrist_est, report["candidates_wrist"] = localize(wrist)
report["block_head"] = None if head_est is None else head_est[:5]
report["block_wrist"] = None if wrist_est is None else wrist_est[:5]
est = head_est if head_est is not None else wrist_est
if est is None:
    report["localized"] = False
    report["localize_error"] = "no candidate with the right hue"
    result = report
else:
    cx, cy, top, n, nface, face = est
    report["localized"] = True
    report["block"] = (cx, cy, top, n, nface)
    # The omnipicker's centre link is its pinch point: the fingertips swing in a near-horizontal arc and meet 1 cm
    # past it (adapter/isaac/finger_probe.py). The grasp test (adapter/isaac/grasp_test.py) holds a face-aligned
    # 5 cm cube when the centre link is 1-3 cm past the cube's centre and loses it at 4 cm or more, or at any
    # 30 deg yaw; so the centre link goes PAD_OFFSET past the cube's middle, 3.5 cm below its top, the tips
    # 0.5 cm above the table. No lateral offset: the pinch is symmetric about the centre link.
    PAD_OFFSET = 0.01
    LATERAL = 0.0
    yaw = face_yaw(face)
    report["yaw"] = yaw

    def targets(est, tilt):
        # the grasp, pregrasp and high poses for a block estimate; the descent aims 1 cm deeper than the hold
        # window so a low estimate still stalls on the table with the tips 0.5-2.5 cm above it
        cx, cy, top = est[0], est[1], est[2]
        q, xg, axis = approach_frame(tilt, yaw)
        mid = (cx, cy, top - 0.025)
        grasp = tuple(mid[i] + (PAD_OFFSET + 0.01) * axis[i] + LATERAL * xg[i] for i in range(3))
        return q, axis, grasp, tuple(grasp[i] - 0.25 * axis[i] for i in range(3)), tuple(grasp[i] - 0.12 * axis[i] for i in range(3))
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
        q, axis, grasp, high, pregrasp = targets(est, tilt)
        pre = move_to_pose({"position": high, "quaternion_wxyz": q, "frame": base}, tolerance=0.03, max_steps=300, arm=arm)
        report["high_tilt_%d" % int(tilt)] = (bool(pre.ok), None if pre.error is None else str(pre.error.message))
        if not pre.ok:
            q = None
            continue
        # the block should still be where it was: the head pixel of the estimate must show its colour
        obs = get_observation()
        cam = obs.cameras["head"]
        K = cam.intrinsics
        cw, cxq, cyq, czq = [float(v) for v in cam.camera_pose.quaternion_wxyz]
        Rm = ((1 - 2 * (cyq * cyq + czq * czq), 2 * (cxq * cyq - cw * czq), 2 * (cxq * czq + cw * cyq)),
              (2 * (cxq * cyq + cw * czq), 1 - 2 * (cxq * cxq + czq * czq), 2 * (cyq * czq - cw * cxq)),
              (2 * (cxq * czq - cw * cyq), 2 * (cyq * czq + cw * cxq), 1 - 2 * (cxq * cxq + cyq * cyq)))
        ct = [float(v) for v in cam.camera_pose.position]
        mid = (est[0], est[1], est[2] - 0.025)
        dv = [mid[i] - ct[i] for i in range(3)]
        pc = [sum(Rm[r][k] * dv[r] for r in range(3)) for k in range(3)]
        u = int(float(K[0][0]) * pc[0] / pc[2] + float(K[0][2]))
        v = int(float(K[1][1]) * pc[1] / pc[2] + float(K[1][2]))
        rgb = [int(c) for c in cam.rgb[v][u]]
        report["block_pixel_after_approach"] = (u, v, rgb)
        # re-localize from the head (no sim steps spent): the block may have been nudged on the way
        again, unused = localize("head")
        if again is not None:
            report["block_from_high"] = again[:3]
            est = again
            q, axis, grasp, high, pregrasp = targets(est, tilt)
        wrist_again, unused = localize(wrist)
        report["block_from_high_wrist"] = None if wrist_again is None else wrist_again[:3]
        pre = move_to_pose({"position": pregrasp, "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
        report["pregrasp_tilt_%d" % int(tilt)] = (bool(pre.ok), None if pre.error is None else str(pre.error.message)[:60])
        wrist_again, unused = localize(wrist)
        report["block_from_pregrasp_wrist"] = None if wrist_again is None else wrist_again[:3]
        report["grasp_target"] = grasp
        down = move_to_pose({"position": grasp, "quaternion_wxyz": q, "frame": base},
                            tolerance=0.012, max_steps=300, arm=arm)
        report["grasp_tilt_%d" % int(tilt)] = (bool(down.ok), None if down.error is None else str(down.error.message)[:60])
        # the grasp pose at the edge of reach: the arm hasn't moved, so try a steeper tilt from here
        if not down.ok and down.error is not None and "IK" in str(down.error.message):
            q = None
            continue
        break
    if q is not None:
        def progress():
            try:
                return [(p["name"], p["score"]) for p in get_observation().task_context.metadata["progress"]["subtasks"]]
            except Exception:
                return None
        report["ee_at_grasp"] = [float(v) for v in get_robot_state().end_effector_poses[arm].position]
        closed = close_gripper(arm=arm)
        g = float(get_robot_state().gripper_positions[arm])
        report["gripper_after_close"] = g
        # a held 5 cm cube stops the fingers at 0.29-0.45 of the way (grasp test and episodes); below 0.1 they closed on nothing
        report["held"] = 0.25 < g < 0.7
        report["progress_after_close"] = progress()
        # a short lift first: the gripper reading after it tells whether the block came along
        lift = move_to_pose({"position": (grasp[0], grasp[1], grasp[2] + 0.04), "quaternion_wxyz": q, "frame": base},
                            tolerance=0.015, max_steps=200, arm=arm)
        report["gripper_after_short_lift"] = float(get_robot_state().gripper_positions[arm])
        lift = move_to_pose({"position": (grasp[0], grasp[1], grasp[2] + 0.15), "quaternion_wxyz": q, "frame": base},
                            tolerance=0.02, max_steps=300, arm=arm)
        report["lift"] = (bool(lift.ok), None if lift.error is None else str(lift.error.message))
        report["gripper_after_lift"] = float(get_robot_state().gripper_positions[arm])
        report["progress_after_lift"] = progress()
        done = True
    report["done"] = done
    result = report
print(report)
