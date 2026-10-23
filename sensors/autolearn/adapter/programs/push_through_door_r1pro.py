# Push the latching, self-closing door open and go through it (pushing_through_door on door_room, run_behavior.py;
# assets/latch_door, door_room.yaml) with the Galaxea R1 Pro, from the entryway: the leaf swings away from the robot.
#  1. find the door and the lever on its near (+x) face with SAM3; fit the door's plane and the hinge;
#  2. the RIGHT hand pinches the lever, presses it past the latch's 30 degrees and pushes: the arm first, then the
#     base driving forward with the arm fixed (the lever moves on the door's arc; the arm re-placed on it after each
#     hop), to ~0.6 rad;
#  3. the LEFT fist goes onto the near face at mid-width, the right lets go and tucks in;
#  4. the base drives through the doorway's lane in short hops, the left fist pushing the leaf open (re-placed on the
#     face after each hop, sliding towards the hinge as the base passes) so the leaf never reaches the base; once the
#     base is 0.65 m past the door plane the fist lets go; the door may close behind.
# Frames: odom (the base at reset, x forward, y left); the hinge frame (x through the door from the corridor to the
# entryway, y along the closed leaf from the hinge to the free edge, origin on the hinge axis); the leaf's angle th
# opens it towards the corridor, the leaf along u(th) = (-sin th, cos th), the near (+x) face's normal n(th) = (cos th,
# sin th). Levers LEVER_R 0.77 m from the hinge, OFF 0.08 m off the leaf's centre plane; leaf 0.9 m
# (assets/latch_door/make_latch_door.py).
import math

report = {"stage": "start"}
ARM = "secondary"     # the right arm opens: the lever is on the robot's right
LEFT = "primary"      # the left fist pushes the leaf open on the way through
GRIP_OPEN = 0.7       # see the grasp
LEVER_R, OFF, LEAF, HALF_T = 0.77, 0.08, 0.9, 0.02


def stage(name):
    """The program's stage, in the report and on stdout (the door witness logs the last `stage:` line per tick)."""
    report["stage"] = name
    print("stage:", name)


def fmt(v, nd=3):
    try:
        return [round(float(x), nd) for x in v]
    except TypeError:
        return round(float(v), nd)


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


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


def frame_quat(za, ya):
    """The hand pointing along za with its finger line along ya (made square to za)."""
    za = unit(za)
    d = sum(a * b for a, b in zip(ya, za))
    ya = unit([a - d * b for a, b in zip(ya, za)])
    return mat_to_quat(cross(ya, za), ya, za)


def approach_vec(yaw, pitch_deg):
    p = math.radians(pitch_deg)
    return [math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), -math.sin(p)]


def pitched_quat(yaw, pitch_deg):
    """The hand approaching along approach_vec, its finger line in the vertical plane of the approach, square to it."""
    za = approach_vec(yaw, pitch_deg)
    p = math.radians(pitch_deg)
    ya = [math.sin(p) * math.cos(yaw), math.sin(p) * math.sin(yaw), math.cos(p)]
    return mat_to_quat(cross(ya, za), ya, za)


def cloud_stats(pts):
    n = len(pts)
    xs = sorted(float(p[0]) for p in pts)
    ys = sorted(float(p[1]) for p in pts)
    zs = sorted(float(p[2]) for p in pts)
    k = max(1, n // 20)
    return {"n": n, "med": [xs[n // 2], ys[n // 2], zs[n // 2]],
            "lo": [xs[k], ys[k], zs[k]], "hi": [xs[-k - 1], ys[-k - 1], zs[-k - 1]]}


def best_points(prompts, min_pts=30):
    """The highest-scoring SAM3 mask over the prompts on the head camera: (score, prompt, points in odom)."""
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
                best = (float(s.score), prompt, [[float(p[0]), float(p[1]), float(p[2])] for p in pts])
    return best


def hand_pos(arm=None):
    return [float(v) for v in get_robot_state().end_effector_poses[arm or ARM].position]


def grip_of(arm=None):
    return round(float(get_robot_state().gripper_positions[arm or ARM]), 3)


def reach(tag, pos, q, tol=0.01, steps=150, arm=None):
    """move_to_pose; if refused, the IK solution seeded at the current joints driven in joint space."""
    arm = arm or ARM
    frame = get_robot_state().base_frame
    pose = Pose(position=pos, quaternion_wxyz=q, frame=frame)
    r = move_to_pose(pose, tolerance=tol, max_steps=steps, arm=arm)
    if not r.ok:
        cur = [float(v) for v in get_robot_state().joint_positions[arm]]
        ik = solve_ik(pose, seed_joints=cur, arm=arm)
        if not ik.ok:
            # the IK is local (seeded at the current joints, a few jittered restarts) and misses poses that a seed
            # at the tucked arm finds (the offline search, which seeds there, reached v09's left edge pinch and the
            # right's waypoints round the edge that the run refused)
            ik = solve_ik(pose, seed_joints=home[arm], arm=arm)
        if ik.ok:
            r = move_to_joints([float(v) for v in ik.joint_positions], tolerance=0.02, max_steps=steps, arm=arm)
    print(tag, "target", fmt(pos), "ok:", r.ok, "status:", str(r.status), "hand:", fmt(hand_pos(arm)))
    return r


def hop_to(tag, x, y, yaw, steps=150):
    """A servo hop; a second one when the first stops HOP_TOL 0.025 m short (with a hand on the door the base stops
    3-5 cm short: v08, which then let the leaf reach the base on 9 of 16 seeds)."""
    r = behavior.navigate_to_pose(x, y, yaw, planner="servo", max_steps=steps)
    b = behavior.get_base_pose()
    # or turns YAW_TOL 0.03 rad short (v13: the servo stopped 0.1 rad short of the brace pose's heading, which put the
    # edge 6 cm off the left hand's reach)
    if math.hypot(b[0] - x, b[1] - y) > 0.025 or abs(math.atan2(math.sin(b[2] - yaw), math.cos(b[2] - yaw))) > 0.03:
        r = behavior.navigate_to_pose(x, y, yaw, planner="servo", max_steps=steps)
        b = behavior.get_base_pose()
    print(tag, "to", fmt([x, y, yaw]), "ok:", r.ok, "base:", fmt(b))
    return r


def set_torso_yaw(yaw):
    tgt = list(torso0)
    tgt[3] = torso0[3] + yaw
    r = behavior.move_torso(tgt, max_steps=90)
    print("torso yaw", yaw, "ok:", r.ok)
    return r


home = {a: [float(v) for v in get_robot_state().joint_positions[a]] for a in (ARM, LEFT)}   # the arms at reset: tucked
torso0 = [float(v) for v in behavior.get_controller_metadata().get("torso_positions", [0.0, 0.0, 0.0, 0.0])]

lever = best_points(["door handle", "lever handle"])
door = best_points(["door"], min_pts=500)
if lever is None or door is None:
    print("lever or door not seen:", lever is not None, door is not None)
    stage("perception")
else:
    L = cloud_stats(lever[2])
    # the door's near face: a line x = a + b y through the door's points (the robot faces it within +-8 degrees)
    mx = cloud_stats(door[2])["med"][0]
    dp = [p for p in door[2] if 0.3 < p[2] < 1.9 and abs(p[0] - mx) < 0.08]
    my_ = sum(p[1] for p in dp) / len(dp)
    mx_ = sum(p[0] for p in dp) / len(dp)
    b = sum((p[1] - my_) * (p[0] - mx_) for p in dp) / max(sum((p[1] - my_) ** 2 for p in dp), 1e-6)
    # the lever is at the free edge, the hinge at the door's far end from it: along the face line, from the hinge to
    # the free edge is EY; EX (EY turned -90 degrees: the hinge frame is right-handed like the world's) runs from the
    # corridor to the entryway, back at the robot
    line = unit([b, 1.0])
    ly = L["med"][1]
    EY = line if ly > my_ else [-line[0], -line[1]]
    EX = [EY[1], -EY[0]]
    PHI = math.atan2(EX[1], EX[0])
    # the spindle: the lever's end nearest the free edge; the leaf's centre plane HALF_T behind the near face
    lp = [q for w, q in sorted((-(p[0] * EY[0] + p[1] * EY[1]), p) for p in lever[2])]
    k = max(1, len(lp) // 20)
    sp = lp[k]
    face_at = mx_ + b * (sp[1] - my_)
    cx = face_at - HALF_T * EX[0]
    cy = sp[1] - HALF_T * EX[1]
    H = [cx - (LEAF - 0.065) * EY[0], cy - (LEAF - 0.065) * EY[1]]
    report["hinge"] = fmt(H)
    report["door_yaw_deg"] = round(math.degrees(PHI), 1)
    print("door face fit: slope", round(b, 3), "frame heading", round(math.degrees(PHI), 1), "hinge", fmt(H))

    def h2o(p):
        return [H[0] + p[0] * EX[0] + p[1] * EY[0], H[1] + p[0] * EX[1] + p[1] * EY[1]]

    def v2o(v):
        return [v[0] * EX[0] + v[1] * EY[0], v[0] * EX[1] + v[1] * EY[1], v[2]]

    def o2h(p):
        d = [p[0] - H[0], p[1] - H[1]]
        return [d[0] * EX[0] + d[1] * EX[1], d[0] * EY[0] + d[1] * EY[1]]

    def u_of(t):
        return [-math.sin(t), math.cos(t), 0.0]

    def n_of(t):
        return [math.cos(t), math.sin(t), 0.0]

    def on_leaf(t, d, off, z):
        u, n = u_of(t), n_of(t)
        p = h2o([d * u[0] + off * n[0], d * u[1] + off * n[1]])
        return [p[0], p[1], z]

    def base_at(p, yaw_h):
        o = h2o(p)
        return o[0], o[1], PHI + yaw_h

    def here_h():
        bo = behavior.get_base_pose()
        return o2h([bo[0], bo[1]]) + [bo[2] - PHI]

    def bearing(p_o):
        q = o2h(p_o)
        return math.atan2(-q[0], q[1])

    TOWARD = PHI + math.pi       # odom heading from the robot to the door (the hinge frame's -x)

    # drive in: the lever REACH_X 0.40 m ahead, SHOULDER_Y 0.25 m to the right (the right shoulder): v13's numbers
    spindle = sp
    tip_dir = [-EY[0], -EY[1]]
    bar_len = max(0.06, max((p[0] - sp[0]) * tip_dir[0] + (p[1] - sp[1]) * tip_dir[1] for p in lever[2]))
    GRIP_R = 0.085
    r_g = min(GRIP_R, max(0.04, bar_len - 0.035))
    gx, gy = spindle[0] + r_g * tip_dir[0], spindle[1] + r_g * tip_dir[1]
    REACH_X, SHOULDER_Y = 0.40, 0.25
    fwd = [math.cos(TOWARD), math.sin(TOWARD)]
    rgt = [math.sin(TOWARD), -math.cos(TOWARD)]
    hop_to("drive in", gx - REACH_X * fwd[0] + SHOULDER_Y * rgt[0], gy - REACH_X * fwd[1] + SHOULDER_Y * rgt[1], TOWARD, steps=400)

    # grasp, press, test push (as open_door_r1pro.py v13 / going_through_door v33: diagonal pinch GRIP_R from the
    # spindle, press PRESS_DEG past the 45-degree stop; a test push that leaves the door within FOLLOW_DEG means the
    # latch held: let go, back off, try again; 3 tries)
    MISS, PRESS_DEG, FOLLOW_DEG = 0.1, 55.0, 3.0
    HAND0 = [0.0]

    def hand_door_angle():
        """the door's angle from the right hand on the lever: its bearing about the hinge less its bearing on the
        closed door"""
        return bearing(hand_pos()) - HAND0[0]

    opened = False
    pitch = 45
    pressed = None
    for attempt in range(3):
        stage("grasp lever %d" % attempt)
        lv = best_points(["door handle", "lever handle"])
        if lv is not None:
            lp = [q for w, q in sorted((-(p[0] * EY[0] + p[1] * EY[1]), p) for p in lv[2])]
            sp = lp[max(1, len(lp) // 20)]
            Lc = cloud_stats(lv[2])
            near_x = min(p[0] * fwd[0] + p[1] * fwd[1] for p in lv[2])     # the bar's front, along the approach
        else:
            Lc = L
            near_x = None
        bar_z = Lc["med"][2]
        gpt = [sp[0] + r_g * tip_dir[0], sp[1] + r_g * tip_dir[1]]
        # the bar's centre: its front face plus half its 2 cm section, along the approach
        if near_x is not None:
            cur = gpt[0] * fwd[0] + gpt[1] * fwd[1]
            gpt = [gpt[0] + (near_x + 0.012 - cur) * fwd[0], gpt[1] + (near_x + 0.012 - cur) * fwd[1]]
        pitch = (45, 30, 60)[attempt]
        gq = pitched_quat(TOWARD, pitch)
        a = approach_vec(TOWARD, pitch)
        set_gripper(GRIP_OPEN, arm=ARM)
        reach("pregrasp", [gpt[0] - 0.10 * a[0], gpt[1] - 0.10 * a[1], bar_z - 0.10 * a[2]], gq)
        reach("grasp", [gpt[0] + 0.005 * a[0], gpt[1] + 0.005 * a[1], bar_z + 0.005 * a[2]], gq, tol=0.008)
        close_gripper(arm=ARM)
        print("gripper closed; opening", grip_of(), "hand:", fmt(hand_pos()))
        if grip_of() >= MISS:
            stage("press lever %d" % attempt)
            HAND0[0] = bearing(hand_pos())
            pr = math.radians(PRESS_DEG)
            # the press turns the grip point about the spindle: towards it by r (1 - cos), down by r sin
            pressed = [gpt[0] - r_g * (1 - math.cos(pr)) * tip_dir[0], gpt[1] - r_g * (1 - math.cos(pr)) * tip_dir[1],
                       bar_z - r_g * math.sin(pr)]
            reach("press", pressed, gq, tol=0.01, steps=120)
            stage("push arm %d" % attempt)
            hp = o2h(hand_pos())
            for deg in (3, 6, 9):
                t = math.radians(deg)
                c, s = math.cos(t), math.sin(t)
                tgt = h2o([c * hp[0] - s * hp[1], s * hp[0] + c * hp[1]])
                reach("push %d" % deg, [tgt[0], tgt[1], pressed[2]], pitched_quat(TOWARD + t, pitch), tol=0.02, steps=90)
            moved = math.degrees(hand_door_angle())
            print("test push: door followed the hand by", round(moved, 1), "deg")
            if moved >= FOLLOW_DEG:
                opened = True
                break
        set_gripper(GRIP_OPEN, arm=ARM)
        hp = hand_pos()
        reach("back off", [hp[0] - 0.12 * a[0], hp[1] - 0.12 * a[1], hp[2] - 0.12 * a[2]], gq, tol=0.03, steps=90)
    report["unlatched"] = opened
    report["grasp_attempts"] = attempt + 1

    # ---- geometry and IK helpers for the drive ----
    def rotz(v, a):
        c, s = math.cos(a), math.sin(a)
        return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]

    def frame_m(za, ya):
        za = unit(za)
        d = sum(x * y for x, y in zip(ya, za))
        ya = unit([x - d * y for x, y in zip(ya, za)])
        return [cross(ya, za), ya, za]

    def reachable(arm, p_h, z, za_h, ya_h, cand):
        """Can `arm` reach hinge-frame point p_h at height z, hand axes za_h / ya_h (hinge frame), with the base at
        hinge-frame pose cand (x, y, yaw)? The target is moved by (current base) (cand)^-1 and given to the IK."""
        cur = behavior.get_base_pose()
        co = h2o(cand[:2])
        cyaw = PHI + cand[2]
        po = h2o(p_h)
        rel = rotz([po[0] - co[0], po[1] - co[1], z], -cyaw)
        t = rotz(rel, cur[2])
        m = frame_m(rotz(v2o(za_h), cur[2] - cyaw), rotz(v2o(ya_h), cur[2] - cyaw))
        q = mat_to_quat(m[0], m[1], m[2])
        return solve_ik(Pose(position=[t[0] + cur[0], t[1] + cur[1], z], quaternion_wxyz=q, frame=get_robot_state().base_frame), arm=arm).ok

    def leaf_gap(c, t):
        """the base footprint's clearance from the leaf at angle t (footprint x -0.398..0.238, y +-0.342)"""
        u = u_of(t)
        cy, sy = math.cos(c[2]), math.sin(c[2])
        g = 9.0
        for i in range(19):
            d = LEAF * i / 18
            wx, wy = d * u[0] - c[0], d * u[1] - c[1]
            px, py = cy * wx + sy * wy, -sy * wx + cy * wy
            g = min(g, math.hypot(max(px - 0.238, -0.398 - px, 0.0), max(abs(py) - 0.342, 0.0)))
        return g

    def wall_free(c):
        """no point of the footprint's outline in the wall (within 0.08 m of the door plane outside the opening)"""
        cy, sy = math.cos(c[2]), math.sin(c[2])
        for k in range(12):
            for x, y in ((-0.398 + 0.636 * k / 11, -0.342), (-0.398 + 0.636 * k / 11, 0.342), (-0.398, -0.342 + 0.684 * k / 11), (0.238, -0.342 + 0.684 * k / 11)):
                wx, wy = c[0] + cy * x - sy * y, c[1] + sy * x + cy * y
                if -0.08 < wx < 0.08 and not (0.02 < wy < 0.88):
                    return False
        return True

    PUSH_OFF = 0.035      # the fist's end-effector point off the leaf's centre plane: half the 4 cm leaf, ~1.5 cm of fingers
    PUSH_DS = (0.6, 0.5, 0.7, 0.4, 0.3, 0.8, 0.2)     # where along the leaf the fist pushes, mid-width first (in from the edge)

    def push_axes(t, kind):
        """the fist pointing into the near face (kind 0, fingers vertical) or along the leaf to the hinge (kind 1)"""
        u, n = u_of(t), n_of(t)
        if kind == 0:
            return [-n[0], -n[1], 0.0], [0.0, 0.0, 1.0]
        return [-u[0], -u[1], 0.0], n

    def push_q(t, kind):
        za, ya = push_axes(t, kind)
        return frame_quat(v2o(za), v2o(ya))

    def push_pt(t, d):
        u, n = u_of(t), n_of(t)
        return [d * u[0] + PUSH_OFF * n[0], d * u[1] + PUSH_OFF * n[1]]

    def find_push(cand, t, hands, zs=(1.1, 1.0, 1.25)):
        """(hand, d, kind, z) a fist reaches on the near face at door angle t with the base at cand; None"""
        for d in PUSH_DS:
            for z in zs:
                for kind in (0, 1):
                    for hand in hands:
                        za, ya = push_axes(t, kind)
                        if reachable(hand, push_pt(t, d), z, za, ya, cand):
                            return hand, d, kind, z
        return None

    def fist_angle(hand, d_used):
        """the door's angle from a fist on the near face (the closer keeps the leaf on it): its bearing about the
        hinge, plus the fist's offset off the leaf's plane seen from the hinge"""
        q = o2h(hand_pos(hand))
        r = max(0.1, math.hypot(q[0], q[1]))
        return math.atan2(-q[0], q[1]) + math.atan2(PUSH_OFF, r)

    def push(tag, hand, t, d, kind, z):
        return reach(tag, on_leaf(t, d, PUSH_OFF, z), push_q(t, kind), tol=0.02, steps=90, arm=hand)

    if opened:
        # ---- the right arm pushes the lever on the door's arc as far as it reaches (the base still) ----
        stage("push with the lever")
        PUSH_STEP = 0.08      # rad of door a reach: the hand moved ~6 cm along the lever's arc
        th = hand_door_angle()
        for k in range(10):
            hp = o2h(hand_pos())
            dt = PUSH_STEP
            c, s = math.cos(dt), math.sin(dt)
            tgt = h2o([c * hp[0] - s * hp[1], s * hp[0] + c * hp[1]])
            reach("lever push %d" % k, [tgt[0], tgt[1], hand_pos()[2]], pitched_quat(TOWARD + min(th + dt, math.radians(40.0)), pitch), tol=0.02, steps=90)
            th_new = hand_door_angle()
            print("   door ~", round(math.degrees(th_new), 1), "deg")
            if th_new - th < 0.02:
                break
            th = th_new
        report["lever_pushed_deg"] = round(math.degrees(th), 1)

        # ---- a fist onto the near face, then the right lets go of the lever ----
        stage("fist on the face")
        LANE_Y = 0.48       # the base's centre line through the doorway (0.9 m clear; the base 0.68 m wide), untuned
        got = find_push(here_h(), th, (LEFT,))
        print("left fist push point:", got)
        fist = None
        if got is not None:
            hand, d, kind, z = got
            close_gripper(arm=LEFT)
            push("left fist on", LEFT, th, d, kind, z)
            fist = (LEFT, d, kind, z)
        set_gripper(1.0, arm=ARM)
        hp = hand_pos()
        ap = approach_vec(TOWARD + min(th, math.radians(40.0)), pitch)
        reach("right off the lever", [hp[0] - 0.10 * ap[0], hp[1] - 0.10 * ap[1], hp[2] - 0.10 * ap[2]], pitched_quat(TOWARD + min(th, math.radians(40.0)), pitch), tol=0.03, steps=80)
        move_to_joints(home[ARM], tolerance=0.05, max_steps=150, arm=ARM)
        if fist is None:     # the right fist then
            close_gripper(arm=ARM)
            got = find_push(here_h(), th, (ARM, LEFT))
            if got is not None:
                push("fist on", got[0], th, got[1], got[2], got[3])
                fist = got
        report["fist"] = None if fist is None else [fist[0], fist[1]]

        # ---- through: push the door open ahead of the base and drive in behind it ----
        # Each round: (1) push the door PUSH_AHEAD rad further (to at most TH_MAX, short of the 1.745 stop) with a fist
        # that reaches from where the base is; (2) move the base on by up to ADV toward the corridor along the lane,
        # where its footprint stays GAP_MIN clear of the leaf (at the door's angle less LAG: the door can lag the fist)
        # and a fist still reaches the face; stop when neither moves. The base goes first to the lane's centre line.
        stage("through")
        PUSH_AHEAD, TH_MAX, ADV, GAP_MIN, LAG, END_X = 0.15, 1.65, 0.08, 0.07, 0.05, -0.68
        hands = (LEFT, ARM)
        for k in range(60):
            if fist is None:
                break
            hand, d, kind, z = fist
            th = fist_angle(hand, d)
            c0 = here_h()
            if c0[0] <= END_X:
                break
            moved = False
            if th < TH_MAX - 0.02:
                tt = min(TH_MAX, th + PUSH_AHEAD)
                g = find_push(c0, tt, hands)
                if g is not None:
                    push("push %d" % k, g[0], tt, g[1], g[2], g[3])
                    th2 = fist_angle(g[0], g[1])
                    print("   door ~", round(math.degrees(th2), 1), "deg")
                    moved = th2 > th + 0.03
                    fist = g
                    th = max(th, th2)
            # advance: the furthest of ADV, ADV/2 towards END_X (and onto the lane) that is clear and keeps a fist on
            for f in (1.0, 0.5):
                nx = max(END_X, c0[0] - f * ADV)
                ny = c0[1] + f * max(-ADV, min(ADV, LANE_Y - c0[1]))
                cand = (nx, ny, math.pi)
                if leaf_gap(cand, th - LAG) < GAP_MIN or not wall_free(cand):
                    continue
                g = find_push(cand, th, hands)
                if g is None:
                    continue
                hop_to("advance %d" % k, *base_at([nx, ny], math.pi), steps=100)
                push("re-place %d" % k, g[0], th, g[1], g[2], g[3])
                fist = g
                moved = True
                break
            if not moved:
                print("no push and no advance: stop")
                break
        report["base_end_h"] = fmt(here_h())
        if here_h()[0] <= END_X + 0.03:
            stage("past: let go")
            if fist is not None:
                hp = hand_pos(fist[0])
                n = v2o(n_of(fist_angle(fist[0], fist[1])))
                reach("fist off", [hp[0] + 0.12 * n[0], hp[1] + 0.12 * n[1], hp[2]], push_q(TH_MAX, fist[2]), tol=0.05, steps=60, arm=fist[0])
                move_to_joints(home[fist[0]], tolerance=0.05, max_steps=150, arm=fist[0])
            hop_to("on", *base_at([END_X - 0.1, LANE_Y], math.pi), steps=150)
            stage("through")
        else:
            stage("stopped")
print("report:", report)
