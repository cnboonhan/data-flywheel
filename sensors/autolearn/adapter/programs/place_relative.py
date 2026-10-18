# Place object A to the left/right/top/bottom of object B (Genie Sim g2op_spatial_place_object_relative_position).
# Learned, not hand-written: an ASPIRE-style fix loop (a coding agent, 5 attempts, brief in README "Learn") grew it
# from pick_block.py; this version scored end to end on 3 of 4 benchmark episodes and held the object in 4 of 4.
# Built from adapter/programs/pick_block.py: SAM3 on the head camera locates A and B by the instruction's names,
# the gripper comes down onto A across its narrow side, closes, lifts, carries A above B's position offset in the
# instructed direction (robot base frame: left +y, right -y, top +x, bottom -x), lowers it to the table and opens.
import math

report = {}
state = get_robot_state()
base = state.base_frame
try:
    language = get_task_context().language.lower()
except Exception:
    language = ""
report["language"] = language


def parse(lang):
    # "place the <A> to the <dir> of the <B>" -> (A, dir, B); anything unparsed falls back to the raw words
    lang = lang.strip().rstrip(".")
    head, sep, rest = lang.partition(" to the ")
    name_a = head[head.find("the ") + 4:] if "the " in head else head
    direction, sep2, name_b = rest.partition(" of ")
    name_b = name_b[4:] if name_b.startswith("the ") else name_b
    d = "left"
    for w in ("left", "right", "top", "bottom"):
        if w in direction:
            d = w
            break
    return name_a.strip(), d, name_b.strip()


name_a, direction, name_b = parse(language)
report["parsed"] = (name_a, direction, name_b)
# the instructed direction in the base frame; the checker wants > 5 cm along it and < 10 cm across it
DIRS = {"left": (0.0, 1.0), "right": (0.0, -1.0), "top": (1.0, 0.0), "bottom": (-1.0, 0.0)}
OFFSET = 0.17   # m from B's centre: clear of a 9 cm B plus a 8 cm A, with 12 cm to spare before 5 cm minimum


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
    # along base -x turned by yaw_deg about base z
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


def narrow_yaw(face):
    # the yaw (deg) of the object's narrow side from its upper points: the rotation whose axis-aligned bounding box
    # is smallest, then the box axis with the smaller extent; the fingers (gripper x) lie along that axis.
    # Returns (yaw of narrow axis, narrow extent, wide extent); yaw in (-90, 90]
    best = None
    for k in range(60):
        a = math.radians(3.0 * k)
        ca, sa = math.cos(a), math.sin(a)
        us = [ca * float(p[0]) + sa * float(p[1]) for p in face]
        vs = [-sa * float(p[0]) + ca * float(p[1]) for p in face]
        eu, ev = max(us) - min(us), max(vs) - min(vs)
        if best is None or eu * ev < best[0]:
            best = (eu * ev, 3.0 * k, eu, ev)
    area, a, eu, ev = best
    yaw = a if eu <= ev else a + 90.0
    while yaw > 90.0:
        yaw -= 180.0
    return yaw, min(eu, ev), max(eu, ev)


def localize(camera_name, name, avoid=None):
    # every SAM3 candidate for the name, as (score, cx, cy, top, bottom, narrow, wide, yaw, n, face) from its depth
    # cloud in the base frame; the best score among tabletop-sized ones at least 4 cm from `avoid` wins
    seg = segment_text(camera_name, name)
    cands = []
    if not seg.ok:
        return None, cands
    for s in seg.segmentations:
        try:
            cloud = mask_to_point_cloud(s.mask, camera_name, target_frame=base)
        except Exception:      # a mask with no valid depth (the robot's own body, the far background) raises
            continue
        pts = [p for p in cloud.points if 0.3 < float(p[2]) < 1.5 and float(p[0]) == float(p[0])]
        if len(pts) < 20:
            continue
        zs = sorted(float(p[2]) for p in pts)
        n = len(zs)
        top, bottom = zs[9 * n // 10], zs[n // 20]
        body = [p for p in pts if float(p[2]) > top - 0.06]
        xs, ys = [float(p[0]) for p in body], [float(p[1]) for p in body]
        if max(max(xs) - min(xs), max(ys) - min(ys)) > 0.25 or top - bottom > 0.16:
            continue    # a table or the robot's own gripper (SAM3 offers those for any prompt), not an object
        face = [p for p in pts if float(p[2]) > top - 0.03] or body
        cx = sum(float(p[0]) for p in face) / len(face)
        cy = sum(float(p[1]) for p in face) / len(face)
        yaw, narrow, wide = narrow_yaw(face)
        if avoid is not None and math.hypot(cx - avoid[1], cy - avoid[2]) < 0.04:
            continue
        cands.append((round(float(s.score), 3), cx, cy, top, bottom, round(narrow, 3), round(wide, 3), yaw, n, face))
    if not cands:
        return None, cands
    best = cands[0]
    for c in cands:
        if c[0] > best[0]:
            best = c
    return best, cands


def brief(c):
    return None if c is None else [round(float(v), 3) for v in c[:9]]


# The episode's first frame is taken before the head has settled at the pose the adapter holds (in this scene it
# pitches by several degrees over the first ticks while the image looks alike), so estimates from it are biased by
# ~6 cm in x and 3 cm in z (attempt 3). They only pick the arm; everything else uses estimates taken after the
# first motions.
est_a0, cands_a0 = localize("head", name_a)
est_b0, cands_b0 = localize("head", name_b, avoid=est_a0)
report["a_first_frame"] = brief(est_a0)
report["b_first_frame"] = brief(est_b0)
report["a_candidates_first"] = [brief(c) for c in cands_a0]
report["b_candidates_first"] = [brief(c) for c in cands_b0]


def progress():
    try:
        return [(p["name"], p["score"]) for p in get_observation().task_context.metadata["progress"]["subtasks"]]
    except Exception:
        return None


if est_a0 is None:
    report["localized"] = False
    result = report
else:
    dx, dy = DIRS[direction]
    # the arm on the side of the whole job: the midpoint of A and the target (left arm = primary = +y)
    ty0 = est_a0[2] if est_b0 is None else est_b0[2] + OFFSET * dy
    arm = "primary" if 0.5 * (est_a0[2] + ty0) >= 0.0 else "secondary"
    report["arm"] = arm
    opened = open_gripper(arm=arm)
    # The planner interpolates in joint space and knows nothing about the table: straight up first, then over the
    # object at height, then down.
    home = get_robot_state().end_effector_poses[arm]
    up = move_to_pose({"position": (float(home.position[0]), float(home.position[1]), float(home.position[2]) + 0.10),
                       "quaternion_wxyz": [float(v) for v in home.quaternion_wxyz], "frame": base},
                      tolerance=0.03, max_steps=200, arm=arm)
    report["lift_from_home"] = bool(up.ok)
    est_a, cands_a = localize("head", name_a)
    est_b, cands_b = localize("head", name_b, avoid=est_a)
    report["a_head"] = brief(est_a)
    report["b_head"] = brief(est_b)
    report["a_candidates"] = [brief(c) for c in cands_a]
    report["b_candidates"] = [brief(c) for c in cands_b]
    if est_a is None:
        est_a = est_a0
    if est_b is None:
        est_b = est_b0
    report["localized"] = True
    target = None if est_b is None else (est_b[1] + OFFSET * dx, est_b[2] + OFFSET * dy)
    report["target"] = target
    # The omnipicker's centre link is its pinch point; a face-aligned 5 cm cube is held when the centre link is
    # 2.5-4.5 cm below its top (grasp_test.py). The descent aims 1 cm deeper than the 3.5 cm hold so a low
    # estimate still ends inside the window; the fingers pass through the table, so nothing stalls.
    PAD_OFFSET = 0.01
    yaw = est_a[7]

    def targets(est, tilt, yaw=yaw):
        q, xg, axis = approach_frame(tilt, yaw)
        mid = (est[1], est[2], est[3] - 0.025)
        grasp = tuple(mid[i] + (PAD_OFFSET + 0.01) * axis[i] for i in range(3))
        return q, axis, grasp, tuple(grasp[i] - 0.20 * axis[i] for i in range(3)), tuple(grasp[i] - 0.10 * axis[i] for i in range(3))
    q = None
    est = est_a
    used_tilt = 0.0
    for tilt in (0.0, 20.0, 35.0, 50.0):
        q, axis, grasp, high, pregrasp = targets(est, tilt)
        pre = move_to_pose({"position": high, "quaternion_wxyz": q, "frame": base}, tolerance=0.03, max_steps=300, arm=arm)
        report["high_tilt_%d" % int(tilt)] = (bool(pre.ok), None if pre.error is None else str(pre.error.message)[:60])
        if not pre.ok:
            q = None
            continue
        # re-localize from the head (no sim steps spent): a new estimate counts only within 5 cm of the old one,
        # since with the arm over the object the head can lose it and hand back a look-alike
        again, unused = localize("head", name_a)
        if again is not None:
            report["a_from_high"] = brief(again)
            if max(abs(again[i] - est[i]) for i in (1, 2, 3)) < 0.05:
                est = again
                q, axis, grasp, high, pregrasp = targets(est, tilt)
            else:
                report["relocalize_rejected"] = True
        pre = move_to_pose({"position": pregrasp, "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
        report["pregrasp_tilt_%d" % int(tilt)] = (bool(pre.ok), None if pre.error is None else str(pre.error.message)[:60])
        report["grasp_target"] = grasp
        down = move_to_pose({"position": grasp, "quaternion_wxyz": q, "frame": base}, tolerance=0.012, max_steps=300, arm=arm)
        report["grasp_tilt_%d" % int(tilt)] = (bool(down.ok), None if down.error is None else str(down.error.message)[:60])
        if not down.ok and down.error is not None and "IK" in str(down.error.message):
            q = None
            continue
        used_tilt = tilt
        break
    report["progress_at_grasp"] = progress()
    if q is None:
        report["done"] = False
        result = report
    else:
        report["ee_at_grasp"] = [round(float(v), 4) for v in get_robot_state().end_effector_poses[arm].position]
        closed = close_gripper(arm=arm)
        g = float(get_robot_state().gripper_positions[arm])
        report["gripper_after_close"] = g
        # a held 5 cm cube stops the fingers at 0.29-0.45 of the way; an empty close goes below 0.1
        report["held"] = 0.12 < g < 0.75
        if not report["held"]:
            # the fingers closed beside the object (a 3 cm block was pushed out in attempt 3): once more across the
            # other axis, from a fresh head estimate if the head still sees it near the old one
            open_gripper(arm=arm)
            again, unused = localize("head", name_a)
            if again is not None and max(abs(again[i] - est[i]) for i in (1, 2, 3)) < 0.05:
                est = again
            yaw = yaw + 90.0 if yaw <= 0.0 else yaw - 90.0
            q, axis, grasp, high, pregrasp = targets(est, used_tilt, yaw)
            pre = move_to_pose({"position": pregrasp, "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
            down = move_to_pose({"position": grasp, "quaternion_wxyz": q, "frame": base}, tolerance=0.012, max_steps=300, arm=arm)
            report["regrasp"] = (bool(pre.ok), bool(down.ok), None if down.error is None else str(down.error.message)[:60])
            closed = close_gripper(arm=arm)
            g = float(get_robot_state().gripper_positions[arm])
            report["gripper_after_regrasp"] = g
            report["held"] = 0.12 < g < 0.75
        lift = move_to_pose({"position": (grasp[0], grasp[1], grasp[2] + 0.12), "quaternion_wxyz": q, "frame": base},
                            tolerance=0.02, max_steps=200, arm=arm)
        report["lift"] = (bool(lift.ok), None if lift.error is None else str(lift.error.message)[:60])
        report["gripper_after_lift"] = float(get_robot_state().gripper_positions[arm])
        report["progress_after_lift"] = progress()
        placed = False
        if target is not None:
            # carry A at height to above the target spot, lower it until its grasp height (the table is flat), open
            for tilt in (used_tilt, 0.0, 20.0, 35.0, 50.0):
                qc, axis, dummy_g, dummy_h, dummy_p = targets(est, tilt, yaw)
                carry = (target[0], target[1], grasp[2] + 0.12)
                mv = move_to_pose({"position": carry, "quaternion_wxyz": qc, "frame": base}, tolerance=0.03, max_steps=300, arm=arm)
                report["carry_tilt_%d" % int(tilt)] = (bool(mv.ok), None if mv.error is None else str(mv.error.message)[:60])
                if not mv.ok and mv.error is not None and "IK" in str(mv.error.message):
                    continue
                low = (target[0], target[1], grasp[2] + 0.01)
                mv = move_to_pose({"position": low, "quaternion_wxyz": qc, "frame": base}, tolerance=0.015, max_steps=300, arm=arm)
                report["place_tilt_%d" % int(tilt)] = (bool(mv.ok), None if mv.error is None else str(mv.error.message)[:60])
                report["ee_at_place"] = [round(float(v), 4) for v in get_robot_state().end_effector_poses[arm].position]
                report["gripper_before_open"] = float(get_robot_state().gripper_positions[arm])
                open_gripper(arm=arm)
                rise = move_to_pose({"position": (low[0], low[1], low[2] + 0.12), "quaternion_wxyz": qc, "frame": base},
                                    tolerance=0.03, max_steps=200, arm=arm)
                report["rise"] = bool(rise.ok)
                placed = True
                break
        report["placed"] = placed
        report["progress_after_place"] = progress()
        # where A and B ended up, from the head, for the record
        fa, unused = localize("head", name_a)
        fb, unused = localize("head", name_b, avoid=fa)
        report["a_final"] = brief(fa)
        report["b_final"] = brief(fb)
        report["done"] = True
        result = report
print(report)
