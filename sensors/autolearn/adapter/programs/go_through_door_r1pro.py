# Open the latching, self-closing door and go through it (going_through_door on door_room, run_behavior.py; assets/
# latch_door, door_room.yaml) with the Galaxea R1 Pro. The opening is open_door_r1pro.py's (v13): find the lever with
# SAM3, pinch it with the RIGHT hand, press it to free the latch, pull with base hops along the handle's arc. Then the
# hand-over, as a person does it: the LEFT hand hooks the leaf's free edge and holds the door against its closer, the
# right lets go of the near lever, goes round the edge and takes the lever on the far face (the two levers share a
# spindle; their poses follow from the hinge and the door's angle), the left lets go, the right pushes the door wider,
# and the base drives forward through the doorway with the right arm trailing on the far lever; when the arm runs out
# of reach the hand slides to the far face near the hinge and braces there until the base is clear of the leaf's
# sweep, then lets go. The door shuts behind; the base ends 0.6 m past the door plane (the witness).
# The hinge is on the robot's right as it faces the door, so the open leaf lies on its right on the way through.
# Frames: everything is in odom (the base at reset), x forward at reset, y left, z up.
import math

report = {"stage": "start"}
ARM = "secondary"     # the right arm opens: the lever is on the robot's right, on the side of the hinge
LEFT = "primary"      # the left arm holds the door open on the way through
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


def reach(tag, pos, q, tol=0.01, steps=150, arm=None):
    """move_to_pose; if refused, the IK solution seeded at the current joints driven in joint space."""
    arm = arm or ARM
    frame = get_robot_state().base_frame
    pose = Pose(position=pos, quaternion_wxyz=q, frame=frame)
    r = move_to_pose(pose, tolerance=tol, max_steps=steps, arm=arm)
    if not r.ok:
        cur = [float(v) for v in get_robot_state().joint_positions[arm]]
        ik = solve_ik(pose, seed_joints=cur, arm=arm)
        if ik.ok:
            r = move_to_joints([float(v) for v in ik.joint_positions], tolerance=0.02, max_steps=steps, arm=arm)
    print(tag, "target", fmt(pos), "ok:", r.ok, "status:", str(r.status), "hand:", fmt(hand_pos(arm)))
    return r


def hand_pos(arm=None):
    return [float(v) for v in get_robot_state().end_effector_poses[arm or ARM].position]


def grip_of(arm):
    return round(float(get_robot_state().gripper_positions[arm]), 3)


def face_quat(th):
    """The hand pointing into the door's far face (against its normal, which turns with the door's angle `th`), the
    fingers closing above and below: a pinch on the far lever's bar, or a fist pressed on the face."""
    za = [-math.cos(th), -math.sin(th), 0.0]
    up = [0.0, 0.0, 1.0]
    return mat_to_quat(cross(up, za), up, za)


def hop_to(tag, x, y, yaw, steps=150):
    r = behavior.navigate_to_pose(x, y, yaw, planner="servo", max_steps=steps)
    print(tag, "to", fmt([x, y, yaw]), "ok:", r.ok, "base:", fmt(behavior.get_base_pose()))
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


home = {a: [float(v) for v in get_robot_state().joint_positions[a]] for a in (ARM, LEFT)}   # the arms at reset: tucked
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
    # STALL_DEG 2: a hop that turns the door less than this did not move it (the latch caught again, or the pinch
    # slipped); two in a row end the pull rather than haul on a latched handle (v04 seeds 40/41 tipped the robot that
    # way at 55-58 degrees of tilt)
    STALL_DEG, stalled, last_th = 2.0, 0, None
    for k in range(16):
        hp = hand_pos()
        th_h = math.atan2(-(hp[0] - H[0]), hp[1] - H[1])
        print("hop round", k, "door angle from the hand", round(math.degrees(th_h), 1), "deg, gripper", grip_state())
        if th_h >= DONE or grip_state() < 0.1:
            break
        if last_th is not None and math.degrees(th_h - last_th) < STALL_DEG:
            stalled += 1
            if stalled >= 2:
                print("door not following the hand: pull stopped")
                report["pull_stalled"] = True
                break
        else:
            stalled = 0
        last_th = th_h
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
    th_h = math.atan2(-(hp[0] - H[0]), hp[1] - H[1])
    report["door_from_hand_deg"] = round(math.degrees(th_h), 1)
    report["hops"] = hops
    report["stage"] = "pulled"

    # Block 6: the hand-over. The door's angle from the right hand on the near lever: the hand's angle about the hinge
    # runs ahead of the leaf by atan(0.08 / 0.77), the bar being 0.08 m off the leaf's plane (face 0.02, standoff 0.05,
    # half the bar 0.01) at 0.77 m from the hinge (0.9 m leaf, 0.065 m backset, half the 0.13 m bar). The far lever's
    # bar centre is the mirror point, 0.08 m to the far side. LEVER_R 0.77, OFF 0.08, LEAF 0.9, HALF_T 0.02:
    # assets/latch_door/make_latch_door.py.
    LEVER_R, OFF, LEAF, HALF_T = 0.77, 0.08, 0.9, 0.02
    th = th_h - math.atan2(OFF, LEVER_R)
    report["door_rad_est"] = round(th, 3)
    cmeta = behavior.get_controller_metadata()
    torso0 = [float(v) for v in cmeta.get("torso_positions", [0.0, 0.0, 0.0, 0.0])]
    # OPEN_MIN 1.15 rad: below this the leaf's free edge still crosses the doorway's centre line and the hand-over
    # geometry (the edge ahead of the left shoulder, the far bar's tip reachable) does not hold; untuned
    OPEN_MIN = 1.15
    if th < OPEN_MIN:
        print("door only", round(th, 3), "rad open: no hand-over")
        report["stage"] = "stopped: door not open enough"

    def u_of(t):          # along the leaf, hinge to free edge
        return [-math.sin(t), math.cos(t), 0.0]

    def n_far(t):         # the far face's outward normal (the side away from the robot's start)
        return [math.cos(t), math.sin(t), 0.0]

    def on_leaf(t, d, off):
        u, n = u_of(t), n_far(t)
        return [H[0] + d * u[0] + off * n[0], H[1] + d * u[1] + off * n[1], bar_z]

    def hook_quats(t):
        """Hands pointing along the leaf towards the hinge, the fingers either side of the leaf (closing along its
        normal), both wrist rolls, level and tipped down 30 degrees: the IK picks."""
        u, n = u_of(t), n_far(t)
        out = []
        for pitch in (0.0, 30.0):
            pr = math.radians(pitch)
            za = [-u[0] * math.cos(pr), -u[1] * math.cos(pr), -math.sin(pr)]
            for sgn in (1.0, -1.0):
                ya = [sgn * n[0], sgn * n[1], 0.0]
                out.append(mat_to_quat(cross(ya, za), ya, za))
        return out

    def endon_quats(t):
        """Hands pointing along the leaf towards the hinge (the right arm's forward reach), the fingers closing above
        and below: a pinch on the far bar's tip, come at end-on from beyond the free edge; both rolls."""
        u = u_of(t)
        za = [-u[0], -u[1], 0.0]
        return [mat_to_quat(cross([0.0, 0.0, sgn], za), [0.0, 0.0, sgn], za) for sgn in (1.0, -1.0)]

    def feasible(pos, quats, arm):
        frame = get_robot_state().base_frame
        for q in quats:
            if solve_ik(Pose(position=pos, quaternion_wxyz=q, frame=frame), arm=arm).ok:
                return q
        return None

    # 6a. the LEFT hand hooks the leaf's free edge (HOOK_IN 0.03 m in from it) and holds the door against its closer.
    # The edge is on the robot's right (the hinge side); the torso turns TORSO_YAWS (rad, +left) to bring the left
    # shoulder round if the hook has no IK with the torso straight; the right hand holds the near lever throughout.
    # HOOK_Z 0.85 m: the left hand reaching across to the edge gets to 0.76-0.88 m, not the lever's 1.0 m (v04 seed
    # 41); the leaf is 2.1 m tall, so the hook can sit lower
    HOOK_IN, TORSO_YAWS, HOOK_Z = 0.03, (0.0, -0.5, 0.5, -0.9), 0.85
    edge = on_leaf(th, LEAF, 0.0)
    hook = on_leaf(th, LEAF - HOOK_IN, 0.0)
    pre_hook = on_leaf(th, LEAF + 0.10, 0.0)
    hook[2] = pre_hook[2] = HOOK_Z
    qe, yaw_used = None, None
    for yaw in (TORSO_YAWS if th >= OPEN_MIN else ()):
        if yaw != 0.0:
            tgt = list(torso0)
            tgt[3] = torso0[3] + yaw
            r = behavior.move_torso(tgt, max_steps=90)
            print("torso yaw", yaw, "ok:", r.ok, "right gripper", grip_state())
        qe = feasible(pre_hook, hook_quats(th), LEFT) and feasible(hook, hook_quats(th), LEFT)
        print("left hook IK at torso yaw", yaw, ":", qe is not None)
        if qe is not None:
            yaw_used = yaw
            break
    braced = False
    if qe is not None:
        set_gripper(GRIP_OPEN, arm=LEFT)
        reach("left pre-hook", pre_hook, qe, tol=0.015, arm=LEFT)
        reach("left hook", hook, qe, tol=0.01, arm=LEFT)
        close_gripper(arm=LEFT)
        braced = grip_of(LEFT) >= MISS
        print("left gripper closed on the edge; opening", grip_of(LEFT), "braced:", braced)
        if not braced:      # closed on nothing: the edge is a little further in than estimated; hook HOOK_IN more
            set_gripper(GRIP_OPEN, arm=LEFT)
            deeper = on_leaf(th, LEAF - 2 * HOOK_IN - 0.02, 0.0)
            deeper[2] = HOOK_Z
            reach("left hook deeper", deeper, qe, tol=0.01, arm=LEFT)
            close_gripper(arm=LEFT)
            braced = grip_of(LEFT) >= MISS
            print("left gripper closed again; opening", grip_of(LEFT), "braced:", braced)
    report["left_braced"] = braced
    report["torso_yaw"] = yaw_used

    # 6b. the right hand lets go of the near lever and takes the far lever end-on: off the near face, out beyond the
    # free edge (CLEAR_U 0.15 m past it, CLEAR_N 0.12 m off the near face), across to the far bar's line (OFF off the
    # far face) and in along the bar onto its tip (TIP_IN 0.035 m from the tip; the bar runs 0.705-0.835 m from the
    # hinge). The hand points along the bar towards the hinge, i.e. forward: the fingers-across grasp needed the
    # right hand pointing right at a point left of its shoulder, which had no IK (v03). Only with the door braced.
    CLEAR_U, CLEAR_N, TIP_IN = 0.15, 0.12, 0.035
    nf = n_far(th)
    far = False
    gf = on_leaf(th, LEVER_R - 0.065 + TIP_IN, OFF)
    qf = None
    if braced:
        open_gripper(arm=ARM)
        hp = hand_pos()
        reach("right off the lever", [hp[0] - CLEAR_N * nf[0], hp[1] - CLEAR_N * nf[1], hp[2]],
              hand_quat(min(th_h, HAND_YAW_MAX)), tol=0.03, steps=60)
        qf = feasible(gf, endon_quats(th), ARM)
        print("far lever end-on IK:", qf is not None)
        if qf is None:
            qf = endon_quats(th)[0]
        reach("round 1", on_leaf(th, LEAF + CLEAR_U, -CLEAR_N), qf, tol=0.03, steps=90)
        reach("round 2", on_leaf(th, LEAF + CLEAR_U, OFF), qf, tol=0.03, steps=90)
        set_gripper(GRIP_OPEN, arm=ARM)
        reach("far pregrasp", on_leaf(th, LEVER_R + 0.065 + 0.06, OFF), qf, tol=0.015, steps=120)
        reach("far grasp", gf, qf, tol=0.008, steps=120)
        close_gripper(arm=ARM)
        far = grip_of(ARM) >= MISS
        print("right gripper closed on the far lever; opening", grip_of(ARM), "took:", far)
        if not far:
            set_gripper(GRIP_OPEN, arm=ARM)
            reach("far back off", on_leaf(th, LEVER_R + 0.065 + 0.06, OFF), qf, tol=0.03, steps=60)
            reach("far grasp again", on_leaf(th, LEVER_R - 0.065 + TIP_IN + 0.02, OFF), qf, tol=0.008, steps=120)
            close_gripper(arm=ARM)
            far = grip_of(ARM) >= MISS
            print("right gripper closed again; opening", grip_of(ARM), "took:", far)
    report["right_far"] = far
    # 6c. the left lets go of the edge and tucks away; the torso straightens
    if braced:
        open_gripper(arm=LEFT)
        reach("left off the edge", pre_hook, qe, tol=0.03, steps=60, arm=LEFT)
        r = move_to_joints(home[LEFT], tolerance=0.05, max_steps=150, arm=LEFT)
        print("left arm home ok:", r.ok)
    report["stage"] = "handed over" if far else "hand-over failed"
    if not far and th >= OPEN_MIN:
        print("no hand-over: not driving through (the door would shut onto the base)")
        report["stage"] = "stopped: no hand-over"

    if far:
        # 6d. push the door wider by WIDEN 0.25 rad in two steps along the lever's arc (the far lever pushed against
        # the face's normal opens the door), so the leaf lies back along the wall and the doorway is clear for the base
        WIDEN = 0.25
        for k in (1, 2):
            t = th + WIDEN * k / 2
            reach("widen " + str(k), on_leaf(t, LEVER_R, OFF), qf, tol=0.02, steps=90)
        th2 = th + WIDEN
        gf, nf, u2 = on_leaf(th2, LEVER_R, OFF), n_far(th2), u_of(th2)

        # Block 7: through the doorway, forward, in HOP 0.15 m hops along a lane between the open leaf and the far
        # jamb. LANE: the base centre keeps the base's half-width BASE_HW 0.30 m plus CLEAR_M 0.05 m from the leaf
        # (its far side at HALF_T) and from the jamb (0.95 m from the hinge: 1.0 m opening, 0.05 m jamb). The right
        # hand stays on the far lever while the base moves only as long as the lever stays within REACH 0.55 m of the
        # right shoulder (SHOULDER_Y right of the base centre) at the hop's end: a lever is rigid along the leaf, so a
        # hop the arm can't absorb would yank the base (v01 tipped the robot that way). Beyond that the hand slides to
        # the far face BRACE_BEHIND 0.35 m behind the base centre, pressed PRESS 0.01 m in, moving ahead of each hop
        # so the arm is never dragged, until the base centre is CLEAR 0.3 m past the hinge line (the base's rear past
        # the door plane); then lets go. Ends THROUGH 0.75 m past the hinge line (the witness: 0.6 m past the face).
        HOP, BASE_HW, CLEAR_M, REACH, BRACE_BEHIND, PRESS, HINGE_MIN, CLEAR, THROUGH = 0.15, 0.30, 0.05, 0.55, 0.35, 0.01, 0.05, 0.3, 0.75
        leaf_y_max = max(H[1] + d * u2[1] for d in (0.0, LEAF)) + HALF_T
        lane_lo = leaf_y_max + BASE_HW + CLEAR_M
        lane_hi = H[1] + 0.95 - BASE_HW - CLEAR_M
        LANE_Y = min(max(H[1] + 0.45, lane_lo), lane_hi)
        print("lane y", round(LANE_Y, 3), "between", round(lane_lo, 3), "and", round(lane_hi, 3), "door est", round(th2, 3))
        report["lane_ok"] = lane_lo <= lane_hi

        def shoulder_dist(bx, by, p):
            return math.hypot(p[0] - bx, p[1] - (by - SHOULDER_Y))

        def brace_point(bx):
            d = max(HINGE_MIN, min(LEAF - 0.1, (H[0] - (bx - BRACE_BEHIND)) / max(1e-6, -u2[0])))
            return on_leaf(th2, d, HALF_T - PRESS)

        # 7a. first into the lane (sideways, the lever still held: the hand stays, the base shifts within reach)
        bx, by, byaw = behavior.get_base_pose()
        for k in range(1, 4):
            f = k / 3.0
            hop_to("to lane %d" % k, bx, by + f * (LANE_Y - by), 0.0, steps=100)
            reach("hold far lever", gf, qf, tol=0.025, steps=60)
        # 7b. forward with the lever in hand while the arm reaches
        hold = True
        for k in range(12):
            bx, by, byaw = behavior.get_base_pose()
            if shoulder_dist(bx + HOP, by, gf) > REACH:
                break
            hop_to("through %d" % k, bx + HOP, LANE_Y, 0.0, steps=90)
            rr = reach("hold far lever", gf, qf, tol=0.025, steps=60)
            if grip_of(ARM) < MISS:
                hold = False
                break
        report["lever_held_to_x"] = round(behavior.get_base_pose()[0], 3)
        # 7c. hand to the far face near the hinge, then ahead of each hop
        open_gripper(arm=ARM)
        hp = hand_pos()
        reach("off far lever", [hp[0] + CLEAR_N * nf[0], hp[1] + CLEAR_N * nf[1], hp[2]], qf, tol=0.03, steps=60)
        close_gripper(arm=ARM)
        bx = behavior.get_base_pose()[0]
        b0 = brace_point(bx + HOP)
        reach("to the face", [b0[0] + CLEAR_N * nf[0], b0[1] + CLEAR_N * nf[1], b0[2]], qf, tol=0.03, steps=120)
        reach("press", b0, qf, tol=0.02, steps=80)
        report["stage"] = "bracing"
        clear_x = H[0] + CLEAR
        for k in range(14):
            bx, by, byaw = behavior.get_base_pose()
            if bx >= clear_x - 0.03:
                break
            nx = min(bx + HOP, clear_x)
            reach("press ahead", brace_point(nx), qf, tol=0.025, steps=60)
            hop_to("through braced %d" % k, nx, LANE_Y, 0.0, steps=90)
        hp = hand_pos()
        reach("let go", [hp[0] + CLEAR_N * nf[0], hp[1] + CLEAR_N * nf[1], hp[2]], qf, tol=0.03, steps=60)
        r = move_to_joints(home[ARM], tolerance=0.05, max_steps=150, arm=ARM)
        print("right arm home ok:", r.ok)
        goal_x = H[0] + THROUGH
        for k in range(4):
            bx, by, byaw = behavior.get_base_pose()
            if bx >= goal_x - 0.03:
                break
            hop_to("through clear %d" % k, min(bx + 0.3, goal_x), LANE_Y, 0.0, steps=150)
        if yaw_used:
            r = behavior.move_torso(torso0, max_steps=90)
            print("torso straight ok:", r.ok)
        report["base_end"] = fmt(behavior.get_base_pose())
        report["stage"] = "through"
print("report:", report)
