# Open the latching, self-closing door (opening_door on door_room, run_behavior.py; assets/latch_door, door_room.yaml;
# v13: a grasp retry on a miss, hops to 1.30 rad by the hand) with the Galaxea R1 Pro: find the lever with SAM3, drive in until it is in the
# right arm's reach, pinch the bar diagonally (fingers front-top and back-bottom), press it 40 degrees to free the
# latch, pull 16 degrees with the arm, then back the base along the handle's arc in 0.12 m hops, the hand holding the
# lever, re-tensioning the arm between hops. The door swings towards the robot; the hinge is on the robot's right.
# Frames: everything is in odom (the base at reset), x forward, y left, z up.
# State after 13 versions on dev instances 36-43: the lever unlatches on 8 of 8 (the witness reads the lever at -45
# degrees), the door reaches 0.37-0.83 rad on 7 of 8, none the 1.05 rad the task asks: the bar walks out of the
# diagonal pinch towards the fingertips with each hop (gripper 0.47 -> 0.37 -> 0.31 -> 0.24 -> 0 over four hops), a
# true hook (fingers fore and aft, the hand pointing along the bar) has no IK at the lever's height, and lifting the
# hand against the lever's stop (v13) or pressing it down (v08) lost the bar sooner. Next: a hook on the leaf's free
# edge once the door is 0.4 rad open (edge_quat, read_edge: untested, the hand pointing right), or the left arm.
import math

report = {"stage": "start"}
ARM = "secondary"     # the right arm: the lever is on the robot's right, on the side of the hinge
GRIP_OPEN = 0.7       # see the grasp


def fmt(v, nd=3):
    try:
        return [round(float(x), nd) for x in v]
    except TypeError:
        return round(float(v), nd)


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def mat_to_quat(xa, ya, za):
    """wxyz quaternion of the rotation whose columns are the hand's x, y, z axes."""
    m00, m01, m02 = xa[0], ya[0], za[0]
    m10, m11, m12 = xa[1], ya[1], za[1]
    m20, m21, m22 = xa[2], ya[2], za[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    n = math.sqrt(sum(x * x for x in q))
    return [x / n for x in q]


def grasp_quat(yaw):
    """The hand pointing along `yaw` (horizontal), fingers closing vertically: hand z = approach, hand y = up (the
    hanging hand's z is its approach and its y the finger line: the probe, reset pose)."""
    fwd = [math.cos(yaw), math.sin(yaw), 0.0]
    up = [0.0, 0.0, 1.0]
    return mat_to_quat(cross(up, fwd), up, fwd)


def grip_state():
    return round(float(get_robot_state().gripper_positions[ARM]), 3)


def approach_vec(yaw, pitch_deg):
    """The hand's approach: along `yaw`, tipped down by `pitch_deg`."""
    p = math.radians(pitch_deg)
    return [math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), -math.sin(p)]


def pitched_quat(yaw, pitch_deg):
    """The hand approaching along approach_vec, its finger line in the vertical plane of the approach, square to it:
    at pitch 0 the fingers close above and below a bar that runs across the robot, at 45 front-top and back-bottom."""
    za = approach_vec(yaw, pitch_deg)
    p = math.radians(pitch_deg)
    ya = [math.sin(p) * math.cos(yaw), math.sin(p) * math.sin(yaw), math.cos(p)]
    return mat_to_quat(cross(ya, za), ya, za)


def along_quat(th):
    """The hand pointing along the lever's bar towards its spindle (+y when the door is closed, turning with the
    door's angle `th`), the fingers fore and aft: pulling the door (along the arc, -x at first) presses the bar on
    the rear finger, a hook that holds however far the door turns; the bar can leave only along the hand's approach,
    which the arc never asks for. v04-v06's diagonal pinch let the bar slide out of the fingertips on the pull."""
    za = [-math.sin(th), math.cos(th), 0.0]
    ya = [math.cos(th), math.sin(th), 0.0]
    return mat_to_quat(cross(ya, za), ya, za)


def edge_quat(th):
    """The hand pointing along the leaf towards the hinge (-y when the door is closed, turning with its angle `th`),
    the fingers fore and aft: they straddle the leaf at its free edge, and pulling along the arc presses the leaf on
    the rear finger, a hook that holds through base hops. The right hand pointing right is the arm's natural reach,
    where the lever hook (pointing left, along_quat) had no IK."""
    za = [math.sin(th), -math.cos(th), 0.0]
    ya = [math.cos(th), math.sin(th), 0.0]
    return mat_to_quat(cross(ya, za), ya, za)


def read_edge():
    """The leaf's free edge from the head camera: the door cloud's points at hand height nearest the robot."""
    frame = get_robot_state().base_frame
    seg = segment_text("head", "door")
    if not seg.ok:
        return None
    best = None
    for sg in seg.segmentations:
        pts = [p for p in mask_to_point_cloud(sg, "head", target_frame=frame).points if 0.8 < float(p[2]) < 1.2]
        if len(pts) > 200 and (best is None or float(sg.score) > best[0]):
            best = (float(sg.score), pts)
    if best is None:
        return None
    pts = best[1]
    xs = sorted(float(p[0]) for p in pts)
    ex = xs[max(1, len(xs) // 50)]                    # the 2nd percentile: the edge, without stray near points
    near = [p for p in pts if float(p[0]) < ex + 0.03]
    ys = sorted(float(p[1]) for p in near)
    return [ex, ys[len(ys) // 2]]


def topdown_quat(yaw):
    """The hand pointing down, its finger line along `yaw` (horizontal): hand z down, hand y along yaw. At yaw 0 the
    fingers close fore and aft, around a bar that runs across the robot."""
    down = [0.0, 0.0, -1.0]
    line = [math.cos(yaw), math.sin(yaw), 0.0]
    return mat_to_quat(cross(line, down), line, down)


def hand_pos():
    return [float(v) for v in get_robot_state().end_effector_poses[ARM].position]


def cloud_stats(pts):
    n = len(pts)
    xs = sorted(float(p[0]) for p in pts)
    ys = sorted(float(p[1]) for p in pts)
    zs = sorted(float(p[2]) for p in pts)
    k = max(1, n // 20)
    return {"n": n, "med": [xs[n // 2], ys[n // 2], zs[n // 2]],
            "lo": [xs[k], ys[k], zs[k]], "hi": [xs[-k - 1], ys[-k - 1], zs[-k - 1]]}


def best_cloud(prompts, min_pts=30):
    """The highest-scoring SAM3 mask over the prompts on the head camera, as point-cloud stats in odom."""
    frame = get_robot_state().base_frame
    best = None
    for prompt in prompts:
        seg = segment_text("head", prompt)
        if not seg.ok:
            continue
        for s in seg.segmentations:
            if best is not None and float(s.score) <= best[0]:
                continue
            pts = mask_to_point_cloud(s, "head", target_frame=frame).points
            if len(pts) >= min_pts:
                best = (float(s.score), prompt, cloud_stats(pts))
    return best


def reach(tag, pos, q, tol=0.01, steps=150):
    """move_to_pose; if refused, the IK solution seeded at the current joints driven in joint space."""
    frame = get_robot_state().base_frame
    pose = Pose(position=pos, quaternion_wxyz=q, frame=frame)
    r = move_to_pose(pose, tolerance=tol, max_steps=steps, arm=ARM)
    if not r.ok:
        cur = [float(v) for v in get_robot_state().joint_positions[ARM]]
        ik = solve_ik(pose, seed_joints=cur, arm=ARM)
        if ik.ok:
            r = move_to_joints([float(v) for v in ik.joint_positions], tolerance=0.02, max_steps=steps, arm=ARM)
    print(tag, "target", fmt(pos), "ok:", r.ok, "status:", str(r.status), "hand:", fmt(hand_pos()))
    return r


def door_angle(H):
    """The leaf's opening (rad) read from the head camera: a line fit of x against y over the door's points near the
    hinge's height band, measured as the turn from the closed face (constant x)."""
    best = None
    frame = get_robot_state().base_frame
    seg = segment_text("head", "door")
    if not seg.ok:
        return None
    for s in seg.segmentations:
        pts = mask_to_point_cloud(s, "head", target_frame=frame).points
        if len(pts) > 500 and (best is None or float(s.score) > best[0]):
            best = (float(s.score), pts)
    if best is None:
        return None
    pts = [p for p in best[1] if 0.6 < float(p[2]) < 1.6]
    n = len(pts)
    if n < 100:
        return None
    my = sum(float(p[1]) for p in pts) / n
    mx = sum(float(p[0]) for p in pts) / n
    syy = sum((float(p[1]) - my) ** 2 for p in pts)
    sxy = sum((float(p[1]) - my) * (float(p[0]) - mx) for p in pts)
    if syy < 1e-6:
        return None
    return round(math.atan(abs(sxy / syy)), 3)


# Block 1: find the lever and the door; drive in until the lever is in reach.
lever = best_cloud(["door handle", "lever handle"])
door = best_cloud(["door"], min_pts=500)
if lever is None or door is None:
    print("lever or door not seen:", lever is not None, door is not None)
    report["stage"] = "perception"
else:
    L = lever[2]
    face_x = door[2]["med"][0]
    # the bar runs from the spindle (the end nearer the free edge, +y) towards the hinge (-y)
    spindle_y, tip_y = L["hi"][1], L["lo"][1]
    grip = [L["lo"][0], (spindle_y + tip_y) / 2.0, L["med"][2]]       # the bar's front face, mid-length
    print("lever", lever[1], "score", round(lever[0], 3), "lo", fmt(L["lo"]), "hi", fmt(L["hi"]),
          "door face x", round(face_x, 3), "grip", fmt(grip))
    # REACH_X 0.40, SHOULDER_Y 0.25: from the reach map (torso upright) a forward-pointing hand reaches 1.1 m high at
    # 0.3-0.5 m ahead; the right shoulder is ~0.25 m right of the base centre (the hanging hands at y -+0.252)
    REACH_X, SHOULDER_Y = 0.40, 0.25
    gx, gy = grip[0] - REACH_X, grip[1] + SHOULDER_Y
    r = behavior.navigate_to_pose(gx, gy, 0.0, planner="servo", max_steps=400)
    print("drive in to", fmt([gx, gy]), "ok:", r.ok, "base:", fmt(behavior.get_base_pose()))
    lever2 = best_cloud(["door handle", "lever handle"])
    if lever2 is not None:
        L = lever2[2]
        spindle_y, tip_y = L["hi"][1], L["lo"][1]
        grip = [L["lo"][0], (spindle_y + tip_y) / 2.0, L["med"][2]]
        print("lever re-read", fmt(grip), "spindle y", round(spindle_y, 3), "tip y", round(tip_y, 3))
    report["grip"] = fmt(grip)

    # Block 2: hook the bar: the hand comes in from beyond the lever's tip, sliding along the bar towards the spindle
    # with the fingers fore and aft (along_quat). The diagonal pinch stays as the fallback where that has no IK.
    bar_x = L["lo"][0] + 0.012       # the bar's centre: its front face plus half its 2 cm section
    bar_z = L["med"][2]
    frame = get_robot_state().base_frame
    mode = None
    q_along = along_quat(0.0)
    pre_along = [bar_x, tip_y - 0.08, bar_z]
    if solve_ik(Pose(position=pre_along, quaternion_wxyz=q_along, frame=frame), arm=ARM).ok and \
            solve_ik(Pose(position=[bar_x, grip[1], bar_z], quaternion_wxyz=q_along, frame=frame), arm=ARM).ok:
        mode = "along"
    pitch = None
    if mode is None:
        for p_deg in (45, 30, 60):
            qq = pitched_quat(0.0, p_deg)
            a = approach_vec(0.0, p_deg)
            pre = [bar_x - 0.10 * a[0], grip[1], bar_z - 0.10 * a[2]]
            if solve_ik(Pose(position=pre, quaternion_wxyz=qq, frame=frame), arm=ARM).ok and \
                    solve_ik(Pose(position=[bar_x, grip[1], bar_z], quaternion_wxyz=qq, frame=frame), arm=ARM).ok:
                pitch = p_deg
                mode = "diagonal"
                break
    print("grasp mode", mode, "pitch", pitch)
    report["grasp_mode"] = mode
    report["pitch"] = pitch
    if mode is None:
        mode, pitch = "diagonal", 45
    # GRIP_OPEN 0.7: the fingers ~3.5 cm either side of the hand point (5 cm travel each): clear of the 2 cm bar
    # (v05 at 0.4, 2 cm, the rear finger hit the bar and the fingers closed on nothing on 7 of 8 seeds) and of the door
    # face 6 cm behind the bar's centre (v04 fully open, 5 cm, the rear finger met the face: the hand stopped 7 cm short)
    set_gripper(GRIP_OPEN, arm=ARM)
    if mode == "along":
        gripq = q_along
        reach("pregrasp", pre_along, gripq)
        reach("grasp", [bar_x, grip[1], bar_z], gripq, tol=0.008)
    else:
        gripq = pitched_quat(0.0, pitch)
        a = approach_vec(0.0, pitch)
        reach("pregrasp", [bar_x - 0.10 * a[0], grip[1], bar_z - 0.10 * a[2]], gripq)
        reach("grasp", [bar_x + 0.005 * a[0], grip[1], bar_z + 0.005 * a[2]], gripq, tol=0.008)
    close_gripper(arm=ARM)
    report["grip_closed"] = grip_state()
    print("gripper closed; opening", grip_state(), "(0 closed .. 1 open) hand:", fmt(hand_pos()))
    # MISS 0.1: the fingers closed on nothing (the 2 cm bar holds them at ~0.3): back off, read the lever again from
    # here and grasp once more (door_room v12 seed 43: closed to 0.0 at the first try, the press then turned the lever
    # with the knuckles but nothing held for the pull)
    MISS = 0.1
    if grip_state() < MISS:
        set_gripper(GRIP_OPEN, arm=ARM)
        hp = hand_pos()
        a = [-math.sin(0.0), math.cos(0.0), 0.0] if mode == "along" else approach_vec(0.0, pitch)
        reach("miss: back off", [hp[0] - 0.10 * a[0], hp[1] - 0.10 * a[1], hp[2] - 0.10 * a[2]], gripq, tol=0.02, steps=90)
        lever3 = best_cloud(["door handle", "lever handle"])
        if lever3 is not None:
            L = lever3[2]
            spindle_y, tip_y = L["hi"][1], L["lo"][1]
            grip = [L["lo"][0], (spindle_y + tip_y) / 2.0, L["med"][2]]
            bar_x, bar_z = L["lo"][0] + 0.012, L["med"][2]
            print("lever re-read after miss", fmt(grip), "spindle y", round(spindle_y, 3), "tip y", round(tip_y, 3))
        if mode == "along":
            reach("regrasp", [bar_x, grip[1], bar_z], gripq, tol=0.008)
        else:
            reach("regrasp", [bar_x + 0.005 * a[0], grip[1], bar_z + 0.005 * a[2]], gripq, tol=0.008)
        close_gripper(arm=ARM)
        report["grip_closed"] = grip_state()
        print("gripper closed again; opening", grip_state(), "hand:", fmt(hand_pos()))


    def hand_quat(th):
        return along_quat(th) if mode == "along" else pitched_quat(th, pitch)

    # Block 3: press the lever down 40 degrees about the spindle: the grip point (r from the spindle) moves down by
    # r sin 40 and towards the spindle by r (1 - cos 40).
    rad = abs(grip[1] - spindle_y)
    press = math.radians(40.0)
    pressed = [bar_x, grip[1] + rad * (1 - math.cos(press)), bar_z - rad * math.sin(press)]
    reach("press", pressed, gripq, tol=0.01)
    report["grip_pressed"] = grip_state()
    print("after press: gripper opening", grip_state())

    # Block 4: pull the door off its latch with the lever, the arm alone: the diagonal pinch holds to ~0.29 rad
    # (v04-v07) and slides off the bar under base hops (v06-v08), so this pull only has to get the door clear of the
    # latch band, and the edge is hooked next.
    free_y = spindle_y + 0.07
    hinge_y = min(free_y - 0.90, door[2]["lo"][1])
    H = [face_x, hinge_y]
    v0 = [pressed[0] - H[0], pressed[1] - H[1]]
    report["hinge"] = fmt(H)
    last_ok = 0
    for deg in range(4, 20, 4):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        tgt = [H[0] + c * v0[0] - s * v0[1], H[1] + s * v0[0] + c * v0[1], pressed[2]]
        r = reach("pull " + str(deg), tgt, hand_quat(th), tol=0.02, steps=90)
        print("   gripper", grip_state())
        if not r.ok:
            break
        last_ok = deg
    report["pulled_deg"] = last_ok

    # Block 5: pull the door the rest of the way by backing the base along the handle's arc, the hand holding the
    # pressed lever. A hop along the arc's tangent (perpendicular to the bar, whatever the door's angle) drags the door
    # and the bar stays in the fingers (v11: one 0.2 m hop took the door from 0.28 to ~0.45 rad on 7 of 8 seeds); the
    # hops of v06-v08 aimed the base at an offset point, part sideways, and slid the hand along the bar and off its tip.
    # Between hops a short arm pull re-tensions the arm on the arc. The hand's own angle about the hinge is the door's.
    # HOP 0.12 m (~8 degrees of door per hop), HAND_YAW_MAX 40 degrees (the wrist's comfortable turn, hand yaw then
    # lags the door and the pinch twists a little); DONE 1.30 rad: the hand's angle about the estimated hinge runs
    # ahead of the door's (door_room v12 seed 40: hand 67 degrees when the door was at 59, 1.02 rad, 0.03 short of the
    # witness's 1.05), so the hops go on to 1.30 by the hand, ~1.15 at the door; 16 hops cover it; all untuned.
    HOP, HAND_YAW_MAX, DONE = 0.12, math.radians(40.0), 1.30
    radius = math.hypot(v0[0], v0[1])
    hops = 0
    for k in range(16):
        hp = hand_pos()
        th_h = math.atan2(-(hp[0] - H[0]), hp[1] - H[1])
        print("hop round", k, "door angle from the hand", round(math.degrees(th_h), 1), "deg, gripper", grip_state())
        if th_h >= DONE or grip_state() < 0.1:
            break
        bx, by, byaw = behavior.get_base_pose()
        r = behavior.navigate_to_pose(bx - HOP * math.cos(th_h), by - HOP * math.sin(th_h), 0.0, planner="servo", max_steps=100)
        hops += 1
        print("   hop", hops, "ok:", r.ok, "base:", fmt(behavior.get_base_pose()), "gripper", grip_state(), "hand:", fmt(hand_pos()))
        if not r.ok:
            break
        hp = hand_pos()
        th_h = math.atan2(-(hp[0] - H[0]), hp[1] - H[1])
        th_n = th_h + math.radians(4.0)
        tgt = [H[0] - radius * math.sin(th_n), H[1] + radius * math.cos(th_n), pressed[2]]
        reach("re-tension " + str(k), tgt, hand_quat(min(th_n, HAND_YAW_MAX)), tol=0.03, steps=60)
    hp = hand_pos()
    report["door_from_hand_deg"] = round(math.degrees(math.atan2(-(hp[0] - H[0]), hp[1] - H[1])), 1)
    report["hops"] = hops
    report["stage"] = "pulled"
    thl = min(math.atan2(-(hp[0] - H[0]), hp[1] - H[1]), HAND_YAW_MAX)
    open_gripper(arm=ARM)
    hp = hand_pos()
    am = [-math.sin(thl), math.cos(thl), 0.0] if mode == "along" else approach_vec(thl, pitch)
    reach("back off", [hp[0] - 0.12 * am[0], hp[1] - 0.12 * am[1], hp[2] - 0.12 * am[2]], hand_quat(thl), tol=0.03, steps=60)
print("report:", report)
