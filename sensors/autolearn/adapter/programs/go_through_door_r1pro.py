# Open the latching, self-closing door and go through it (going_through_door on door_room, run_behavior.py; assets/
# latch_door, door_room.yaml) with the Galaxea R1 Pro, switching hands as a person does, the base kept out of the
# leaf's sweep (its footprint >= RMIN 0.95 m from the hinge from the hand switch on, on the latch side):
#  1. the RIGHT hand pinches the near lever, presses it to free the latch and pulls the door to ~0.5 rad with the arm
#     and short base hops (the base keeps a fixed offset from the hand, clear of the leaf);
#  2. the torso leans forward a little (the arms then reach ~0.2 m further) and the base steps out to the brace pose,
#     beyond the sweep facing the hinge; the LEFT hand pinches the leaf's free edge, the right lets go and tucks in;
#  3. the base TURNS and walks round the outside of the sweep (hops along an arc about the hinge, the left re-placed
#     on the edge after each) swinging the door open to ~1.5 rad; the robot ends on the latch side of the leaf, the
#     far face towards it;
#  4. the base turns about itself to face back into the corridor (the left fist re-placed after each hop) and backs
#     through the doorway's lane, the left fist pushing the door open on its far face the whole way, sliding towards
#     the hinge as the base passes; once the base's front is past the door plane the fist lets go (the door may shut
#     behind). The right hand stays off the door after the brace (the user dropped the far-lever hand-over).
# Every pose of the base from the hand switch on is checked with the run's own IK before the base moves there (a
# hypothetical base pose is tested by moving the target by (current base) (candidate)^-1), the nominal pose first,
# then neighbours; the nominal poses come from the same check run over a grid on the door room (scratch probe/
# handover_probe.py, lane_probe.py; an offline URDF model disagreed with the run by 3-15 cm once the torso moved).
# Geometry: the hinge frame (x through the door from the corridor, y along the closed leaf from the hinge to the free
# edge, origin on the hinge axis) is fitted from the head camera at the start; levers LEVER_R 0.77 m from the hinge,
# OFF 0.08 m off the leaf's centre plane, leaf 0.9 m (assets/latch_door/make_latch_door.py). Poses come from an
# offline reach search with the robot's URDF and the same IK (scratch kin/: search_out.py, search_turn.py, lane2.py),
# with 3 cm of margin (the run's IK reaches ~3 cm less than that model).
import math

report = {"stage": "start"}
ARM = "secondary"     # the right arm opens: the lever is on the robot's right, on the side of the hinge
LEFT = "primary"      # the left arm holds the door while the hands switch
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

# Block 1: find the lever and the door; fit the door's plane; drive in until the lever is in reach.
lever = best_points(["door handle", "lever handle"])
door = best_points(["door"], min_pts=500)
if lever is None or door is None:
    print("lever or door not seen:", lever is not None, door is not None)
    stage("perception")
else:
    L = cloud_stats(lever[2])
    # the closed leaf's corridor face: a line x = a + b y through the door's points (the robot faces it within the
    # start's +-8 degrees, so the face runs along odom y); points more than 8 cm off the median x (the frame, stray
    # depth) are left out
    mx = cloud_stats(door[2])["med"][0]
    dp = [p for p in door[2] if 0.3 < p[2] < 1.9 and abs(p[0] - mx) < 0.08]
    my_ = sum(p[1] for p in dp) / len(dp)
    mx_ = sum(p[0] for p in dp) / len(dp)
    syy = sum((p[1] - my_) ** 2 for p in dp)
    b = sum((p[1] - my_) * (p[0] - mx_) for p in dp) / max(syy, 1e-6)
    EY = unit([b, 1.0])                 # along the closed leaf, hinge -> free edge
    EX = [EY[1], -EY[0]]                # through the door, corridor -> entryway
    PHI = math.atan2(EX[1], EX[0])      # the hinge frame's heading in odom
    spindle_y, tip_y = L["hi"][1], L["lo"][1]
    # the leaf's centre plane at the spindle: the corridor face (HALF_T in front of it) at the spindle's y
    sx = mx_ + b * (spindle_y - my_) + HALF_T * EX[0]
    # the spindle is 0.065 m (backset) in from the free edge: the hinge is LEAF - 0.065 back along the leaf
    H = [sx - (LEAF - 0.065) * EY[0], spindle_y - (LEAF - 0.065) * EY[1]]
    report["hinge"] = fmt(H)
    report["door_yaw_deg"] = round(math.degrees(PHI), 1)
    print("door face fit: slope", round(b, 3), "heading", round(math.degrees(PHI), 1), "deg; hinge", fmt(H),
          "lever lo", fmt(L["lo"]), "hi", fmt(L["hi"]))

    def h2o(p):
        """hinge frame (x, y) -> odom (x, y)"""
        return [H[0] + p[0] * EX[0] + p[1] * EY[0], H[1] + p[0] * EX[1] + p[1] * EY[1]]

    def v2o(v):
        """a hinge-frame vector (x, y, z) in odom"""
        return [v[0] * EX[0] + v[1] * EY[0], v[0] * EX[1] + v[1] * EY[1], v[2]]

    def o2h(p):
        d = [p[0] - H[0], p[1] - H[1]]
        return [d[0] * EX[0] + d[1] * EX[1], d[0] * EY[0] + d[1] * EY[1]]

    def u_of(t):          # along the leaf, hinge to free edge (hinge frame)
        return [-math.sin(t), math.cos(t), 0.0]

    def n_of(t):          # the far face's outward normal (hinge frame)
        return [math.cos(t), math.sin(t), 0.0]

    def on_leaf(t, d, off, z):
        """odom point d along the leaf from the hinge, off along the far normal, at height z"""
        u, n = u_of(t), n_of(t)
        p = h2o([d * u[0] + off * n[0], d * u[1] + off * n[1]])
        return [p[0], p[1], z]

    HAND0 = [0.0]         # the hand's angle about the hinge on the closed door's lever (set at the grasp)

    def hand_angle():
        hp = o2h(hand_pos())
        return math.atan2(-hp[0], hp[1])

    def hand_door_angle():
        """The door's angle from the right hand on the near lever: the hand's angle about the hinge less its angle when
        it took the lever on the closed door (v07 subtracted the lever's nominal atan(OFF / LEVER_R) instead and read
        the door 8 degrees ahead of the witness: the hand sits off the bar)."""
        return hand_angle() - HAND0[0]

    def edge_q(t):        # left hand pinching the free edge: pointing along the leaf to the hinge, fingers on both faces
        u = u_of(t)
        return frame_quat(v2o([-u[0], -u[1], 0.0]), v2o(n_of(t)))

    def top_q(t, tilt):   # right hand from above on a lever bar, fingers either side of it along the leaf's normal,
        u = u_of(t)       # tipped by `tilt` (rad) about the normal towards the free edge
        za = [u[0] * math.sin(tilt), u[1] * math.sin(tilt), -math.cos(tilt)]
        return frame_quat(v2o(za), v2o(n_of(t)))

    def base_at(p, yaw_h):
        """navigate_to_pose args for a hinge-frame base pose"""
        o = h2o(p)
        return o[0], o[1], PHI + yaw_h

    # drive in: REACH_X 0.40, SHOULDER_Y 0.25 (v13: the forward-pointing hand reaches 1.1 m high 0.3-0.5 m ahead,
    # the right shoulder ~0.25 m right of the base centre)
    grip = [L["lo"][0], (spindle_y + tip_y) / 2.0, L["med"][2]]
    REACH_X, SHOULDER_Y = 0.40, 0.25
    r = behavior.navigate_to_pose(grip[0] - REACH_X, grip[1] + SHOULDER_Y, PHI, planner="servo", max_steps=400)
    print("drive in ok:", r.ok, "base:", fmt(behavior.get_base_pose()))

    # Block 2: grasp, press and test-pull, up to 3 tries. The pinch is diagonal (the hand coming in 45 degrees from
    # above, fingers front-top and back-bottom of the bar), GRIP_R 0.085 m from the spindle (towards the 0.13 m bar's
    # tip, so a press of the hand turns the lever further: v06 pinched at mid-bar and its presses stalled at 22-29
    # degrees, under the 30 that frees the latch, on 7 of 16 seeds); the press aims PRESS_DEG 55, past the lever's
    # 45-degree stop, so the arm holds it at the stop. A test pull of TEST_DEG 8 that leaves the hand within
    # FOLLOW_DEG 4 of where it was means the latch held (or the pinch missed): let go, back off and try again. The
    # arm pulls no further than that with the base standing: the free edge would reach the base's front by ~0.2 rad
    # (v07: the leaf touched the base during the arm pull on seed 40).
    MISS, GRIP_R, PRESS_DEG, TEST_DEG, FOLLOW_DEG = 0.1, 0.085, 55.0, 8.0, 4.0
    opened = False
    pitch = 45
    for attempt in range(3):
        stage("grasp lever %d" % attempt)
        lv = best_points(["door handle", "lever handle"])
        if lv is not None:
            L = cloud_stats(lv[2])
            spindle_y, tip_y = L["hi"][1], L["lo"][1]
        bar_x, bar_z = L["lo"][0] + 0.012, L["med"][2]     # the bar's centre: front face plus half its 2 cm section
        r_g = min(GRIP_R, max(0.04, abs(spindle_y - tip_y) - 0.035))
        gy = spindle_y - r_g
        pitch = (45, 30, 60)[attempt]
        gq = pitched_quat(PHI, pitch)
        a = approach_vec(PHI, pitch)
        set_gripper(GRIP_OPEN, arm=ARM)
        reach("pregrasp", [bar_x - 0.10 * a[0], gy - 0.10 * a[1], bar_z - 0.10 * a[2]], gq)
        reach("grasp", [bar_x + 0.005 * a[0], gy, bar_z + 0.005 * a[2]], gq, tol=0.008)
        close_gripper(arm=ARM)
        print("gripper closed; opening", grip_of(), "hand:", fmt(hand_pos()))
        if grip_of() >= MISS:
            stage("press lever %d" % attempt)
            hz0 = hand_pos()[2]
            HAND0[0] = hand_angle()
            pr = math.radians(PRESS_DEG)
            pressed = [bar_x, gy + r_g * (1 - math.cos(pr)), bar_z - r_g * math.sin(pr)]
            reach("press", pressed, gq, tol=0.01, steps=120)
            drop = hz0 - hand_pos()[2]
            print("press: hand dropped", round(drop, 3), "m, lever ~", round(math.degrees(math.asin(max(-1.0, min(1.0, drop / r_g)))), 1), "deg")
            stage("pull arm %d" % attempt)
            th0 = hand_door_angle()
            hp = o2h(hand_pos())
            for deg in (4, 8):
                t = math.radians(deg)
                c, s = math.cos(t), math.sin(t)
                tgt = h2o([c * hp[0] - s * hp[1], s * hp[0] + c * hp[1]])
                rr = reach("pull %d" % deg, [tgt[0], tgt[1], pressed[2]], pitched_quat(PHI + t, pitch), tol=0.02, steps=90)
                if not rr.ok and deg >= TEST_DEG:
                    break
            moved = math.degrees(hand_door_angle() - th0)
            print("test pull: door followed the hand by", round(moved, 1), "deg")
            if moved >= FOLLOW_DEG:
                opened = True
                break
        # let go and back off for another try
        set_gripper(GRIP_OPEN, arm=ARM)
        hp = hand_pos()
        reach("back off", [hp[0] - 0.12 * a[0], hp[1] - 0.12 * a[1], hp[2] - 0.12 * a[2]], gq, tol=0.03, steps=90)
    report["grasp_attempts"] = attempt + 1
    report["unlatched"] = opened

    # Block 3: pull to TH_B 0.5 rad, the base keeping BASE_FROM_N (-0.43, 0.27) from the near lever (3 cm behind the
    # drive-in's offset: the leaf's sweep then stays >= 0.17 m from the base's footprint; v07's (-0.31, 0.21) left 5 cm and the
    # leaf touched the base on 13 of 13 seeds), in STEP 0.07 rad hops, each re-tensioned by the arm; two hops in a
    # row that leave the door within 1.5 degrees end the pull (the latch caught again or the pinch slipped: v04
    # tipped the robot pulling on a latched door).
    TH_B, STEP = 0.5, 0.07

    def near_lever(t):
        u, n = u_of(t), n_of(t)
        return [LEVER_R * u[0] - OFF * n[0], LEVER_R * u[1] - OFF * n[1]]

    BASE_FROM_N = [-0.43, 0.27]
    braced = False
    if opened:
        stage("pull to brace angle")
        th = hand_door_angle()
        stalls = 0
        for k in range(14):
            if th >= TH_B - 0.02:
                break
            tk = min(TH_B, th + STEP)
            nk = near_lever(tk)
            hop_to("pull hop %d" % k, *base_at([nk[0] + BASE_FROM_N[0], nk[1] + BASE_FROM_N[1]], 0.0), steps=100)
            no = h2o(nk)
            reach("re-tension %d" % k, [no[0], no[1], pressed[2]], pitched_quat(PHI + min(tk, math.radians(40.0)), pitch), tol=0.02, steps=80)
            th_new = hand_door_angle()
            print("   door ~", round(math.degrees(th_new), 1), "deg, gripper", grip_of())
            stalls = stalls + 1 if th_new - th < math.radians(1.5) else 0
            th = th_new
            if stalls >= 2 or grip_of() < MISS:
                print("door not following: pull stopped")
                break
        report["door_at_brace_deg"] = round(math.degrees(th), 1)
        THB = th            # the door's angle for the hand switch: where the pull got to

        THB = th            # the door's angle for the hand switch: where the pull got to

        def set_torso(t):
            tgt = [torso0[i] + t[i] for i in range(4)]
            r = behavior.move_torso(tgt, max_steps=90)
            print("torso", fmt(t), "ok:", r.ok)
            return r

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

        RMIN = 0.95     # the base's footprint stays this far from the hinge (the 0.9 m leaf's sweep plus 5 cm)

        def fp_ok(c):
            """the footprint (x -0.398..0.238, y +-0.342 about the base's origin) outside RMIN of the hinge and off the wall"""
            cy, sy = math.cos(c[2]), math.sin(c[2])
            for x in (-0.398, -0.08, 0.238):
                for y in (-0.342, 0.0, 0.342):
                    wx, wy = c[0] + cy * x - sy * y, c[1] + sy * x + cy * y
                    if math.hypot(wx, wy) < RMIN or (wx > -0.08 and not (0.05 < wy < 0.85)):
                        return False
            return True

        def pick(nominal, checks, dxy=0.06, dyaw=math.radians(10.0), rings=1):
            """the nominal hinge-frame base pose, or the nearest neighbour on a grid (dxy, dyaw steps, `rings` out),
            that keeps the footprint out of the sweep and passes every IK check (arm, point, z, za, ya); None if none"""
            offs = sorted((abs(i) + abs(j) + 0.3 * abs(k), i * dxy, j * dxy, k * dyaw) for i in range(-rings, rings + 1)
                          for j in range(-rings, rings + 1) for k in (0, 1, -1, 2, -2))
            for cost, ox, oy, oyaw in offs:
                c = (nominal[0] + ox, nominal[1] + oy, nominal[2] + oyaw)
                if fp_ok(c) and all(robust(a, p, z, za, ya, c) for a, p, z, za, ya in checks):
                    return c
            return None

        def robust(a, p, z, za, ya, c):
            """reachable from c and from c turned ROB_YAW 0.06 rad either way (where the servo leaves the base: v16
            picked poses the arm reached only exactly, and the base stopped 0.05 rad short of them; v17 also required
            +-ROB_XY 0.03 m and found no brace pose on 5 of 15 seeds)"""
            ROB_XY, ROB_YAW = 0.03, 0.06
            for d in ((0, 0, 0), (0, 0, ROB_YAW), (0, 0, -ROB_YAW)):
                if not reachable(a, p, z, za, ya, (c[0] + d[0], c[1] + d[1], c[2] + d[2])):
                    return False
            return True

        def near_hold_axes():
            a = min(THB, math.radians(40.0))
            p = math.radians(pitch)
            return ([math.cos(p) * math.cos(a), math.cos(p) * math.sin(a), -math.sin(p)],
                    [math.sin(p) * math.cos(a), math.sin(p) * math.sin(a), math.cos(p)])

        def hold_near(tag):
            no = h2o(near_lever(THB))
            return reach(tag, [no[0], no[1], pressed[2]], pitched_quat(PHI + min(THB, math.radians(40.0)), pitch), tol=0.02, steps=80)

        def edge_axes(t):
            u = u_of(t)
            return [-u[0], -u[1], 0.0], n_of(t)

        def top_axes(t, tilt):
            u = u_of(t)
            return [u[0] * math.sin(tilt), u[1] * math.sin(tilt), -math.cos(tilt)], n_of(t)

        LB0 = [0.0]      # the left hand's bearing about the hinge less the door's angle, when it took the edge

        def left_bearing():
            lp = o2h(hand_pos(LEFT))
            return math.atan2(-lp[0], lp[1])

        def edge_pt(t):
            """where the left fist pushes: PUSH_D along the leaf, PUSH_OFF off its centre plane on the far face's side"""
            u, n = u_of(t), n_of(t)
            return [PUSH_D * u[0] + PUSH_OFF * n[0], PUSH_D * u[1] + PUSH_OFF * n[1]]

        def far_pt(t):
            u, n = u_of(t), n_of(t)
            return [LEVER_R * u[0] + OFF * n[0], LEVER_R * u[1] + OFF * n[1]]

        # Block 4: lean the torso forward TORSO_LEAN 0.2 rad (torso joint 1: the shoulders come ~0.2 m forward; at 0.1
        # (v20-v28) no brace pose had IK on 6-11 of 16 seeds; v11's 0.35 tipped the robot 14-16 degrees) in two steps, the
        # right re-tensioning on the near lever after each; then step the base out to the brace pose: nominal
        # B_BRACE (-1.0, 0.92) facing YAW_BRACE -30 degrees (towards the hinge), the middle of the probe's feasible
        # cluster, checked for the near lever (right) and the free edge at Z_EDGE (left)
        stage("lean and step out")
        TORSO_LEAN, B_BRACE, YAW_BRACE, Z_EDGE, EDGE_PRE, EDGE_IN = 0.2, [-1.0, 0.92], math.radians(-30.0), 1.2, 0.12, 0.04
        # the push brace: the closed left gripper on the far face (the side that pushes the door open) PUSH_D 0.84 m
        # from the hinge (6 cm in from the free edge), its end-effector point PUSH_OFF 0.035 m off the leaf's centre
        # plane (the 2 cm half-leaf plus the closed fingers' ~1.5 cm); a grasp of the edge is not assumed (real doors
        # are thicker than the gripper's opening or have trims)
        PUSH_D, PUSH_OFF = 0.84, 0.035
        for k in (1, 2):
            set_torso([TORSO_LEAN * k / 2, 0.0, 0.0, 0.0])
            hold_near("re-tension lean %d" % k)
        THB = hand_door_angle()
        nz, ny = near_hold_axes()
        ez, ey = edge_axes(THB)
        cb = pick((B_BRACE[0], B_BRACE[1], YAW_BRACE), [(ARM, near_lever(THB), pressed[2], nz, ny), (LEFT, edge_pt(THB), Z_EDGE, ez, ey)],
                  dxy=0.08, dyaw=math.radians(15.0), rings=3)
        print("brace pose", cb)
        report["brace_pose"] = None if cb is None else fmt(cb)
        if cb is not None:
            bo = behavior.get_base_pose()
            cur = o2h([bo[0], bo[1]]) + [bo[2] - PHI]
            for k in range(1, 4):
                f = k / 3
                hop_to("step out %d" % k, *base_at([cur[0] + f * (cb[0] - cur[0]), cur[1] + f * (cb[1] - cur[1])], cur[2] + f * (cb[2] - cur[2])), steps=100)
                hold_near("re-tension step %d" % k)
            THB = hand_door_angle()
            report["door_at_brace_deg"] = round(math.degrees(THB), 1)

            # Block 5: the left hand pinches the free edge at Z_EDGE 1.2 m (lower level pinches have no IK; the
            # witness counts a brace within 0.4 m of the levers' height), coming in EDGE_PRE 0.12 m beyond the edge
            # and EDGE_IN 0.04 m onto the leaf
            # the left fist comes round the free edge (on the leaf's line EDGE_PRE past it, then off the far face)
            # and presses onto the far face; braced if it got there (within PUSH_TOL 0.03 m of the push point)
            stage("brace left")
            close_gripper(arm=LEFT)
            reach("left past the edge", on_leaf(THB, LEAF + EDGE_PRE, 0.0, Z_EDGE), edge_q(THB), tol=0.02, arm=LEFT)
            reach("left round the edge", on_leaf(THB, LEAF + 0.06, PUSH_OFF + 0.04, Z_EDGE), edge_q(THB), tol=0.02, arm=LEFT)
            reach("left on the far face", on_leaf(THB, PUSH_D, PUSH_OFF, Z_EDGE), edge_q(THB), tol=0.015, arm=LEFT)
            tgt = on_leaf(THB, PUSH_D, PUSH_OFF, Z_EDGE)
            hp = hand_pos(LEFT)
            PUSH_TOL = 0.03
            braced = math.sqrt(sum((hp[i] - tgt[i]) ** 2 for i in range(3))) < PUSH_TOL
            print("left fist on the far face:", braced)
            LB0[0] = left_bearing() - THB
    report["left_braced"] = braced

    handed = False
    if braced:
        # the right lets go, backs out along its approach and tucks in: the left alone holds the door
        stage("switch: right off")
        set_gripper(1.0, arm=ARM)
        hp = hand_pos()
        ap = approach_vec(PHI + min(THB, math.radians(40.0)), pitch)
        reach("right off the lever", [hp[0] - 0.10 * ap[0], hp[1] - 0.10 * ap[1], hp[2] - 0.10 * ap[2]], pitched_quat(PHI + min(THB, math.radians(40.0)), pitch), tol=0.03, steps=80)
        r = move_to_joints(home[ARM], tolerance=0.05, max_steps=150, arm=ARM)
        print("right arm home ok:", r.ok)

        # Block 6: the turn: TURN_N hops along an arc about the hinge (the base centre's radius and bearing and its
        # heading interpolated) from the brace pose to B_END (-1.28, 0.28) facing the doorway (heading 0) with the
        # door at TH_END 1.3 rad; each hop's pose checked (footprint out of the sweep, the left reaching the edge at
        # the door's angle then), the left re-placed on the edge after it; a lost edge stops the turn. B_END: the
        # probe's pose where the left still holds the edge and the right reaches the far lever from above.
        stage("turn")
        TURN_N, B_END, TH_END = 8, [-1.28, 0.28], 1.3
        bo = behavior.get_base_pose()
        b0 = o2h([bo[0], bo[1]])
        y0 = bo[2] - PHI
        r0, a0 = math.hypot(b0[0], b0[1]), math.atan2(-b0[0], b0[1])
        r1, a1 = math.hypot(B_END[0], B_END[1]), math.atan2(-B_END[0], B_END[1])
        t = THB
        for k in range(1, TURN_N + 1):
            lb = left_bearing() - LB0[0]
            if k > 1 and lb < t - 0.15:
                print("door ahead of the left fist (lost contact): turn stopped")
                break
            f = k / TURN_N
            rr, aa = r0 + f * (r1 - r0), a0 + f * (a1 - a0)
            tk = THB + f * (TH_END - THB)
            ez, ey = edge_axes(tk)
            chk = [(LEFT, edge_pt(tk), Z_EDGE, ez, ey)]
            c = pick((-rr * math.sin(aa), rr * math.cos(aa), y0 + f * (0.0 - y0)), chk, rings=2 if k == TURN_N else 1)
            if c is None:
                print("turn %d: no pose" % k)
                break
            hop_to("turn %d" % k, *base_at(c[:2], c[2]), steps=120)
            reach("left on the far face %d" % k, on_leaf(tk, PUSH_D, PUSH_OFF, Z_EDGE), edge_q(tk), tol=0.02, steps=80, arm=LEFT)
            # the door's angle from the left hand on the edge (v13 took the plan's and was 0.2 rad off by the end)
            t = left_bearing() - LB0[0]
            print("   door ~", round(math.degrees(t), 1), "deg (plan", round(math.degrees(tk), 1), ")")
        TT = t
        report["turned_to_deg"] = round(math.degrees(TT), 1)

        # Block 7: drive through backwards, the left fist on the far face. Each step's base pose (hinge frame) is the
        # nominal one or a neighbour that keeps the footprint GAP_MIN from the leaf and off the wall, and from which
        # the left reaches the push point (some distance along the leaf, PUSH_DS, nearest the edge first; the fist
        # level pointing along the leaf, or into the face) also with the base turned +-0.06 rad (push_probe.py: facing
        # back into the corridor, heading pi, the push points exist all along the lane down to 0.25 m from the hinge)
        stage("turn round")
        GAP_LAG = 0.1
        GAP_MIN, PUSH_DS, TH_D, LANE_Y, PUSH_OUT = 0.07, (0.84, 0.77, 0.7, 0.63, 0.55, 0.48, 0.4, 0.33, 0.25, 0.2), 1.57, 0.48, 0.35

        def leaf_gap(c, t):
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
            """no point of the footprint's outline inside the wall (x within 0.08 of the plane, outside the opening)"""
            cy, sy = math.cos(c[2]), math.sin(c[2])
            for k in range(12):
                for x, y in ((-0.398 + 0.636 * k / 11, -0.342), (-0.398 + 0.636 * k / 11, 0.342), (-0.398, -0.342 + 0.684 * k / 11), (0.238, -0.342 + 0.684 * k / 11)):
                    wx, wy = c[0] + cy * x - sy * y, c[1] + sy * x + cy * y
                    if -0.08 < wx < 0.08 and not (0.02 < wy < 0.88):
                        return False
            return True

        def push_axes(t, kind):
            u, n = u_of(t), n_of(t)
            if kind == 0:
                return [-u[0], -u[1], 0.0], n
            return [-n[0], -n[1], 0.0], [0.0, 0.0, 1.0]

        def push_pt(t, d):
            u, n = u_of(t), n_of(t)
            return [d * u[0] + PUSH_OFF * n[0], d * u[1] + PUSH_OFF * n[1]]

        def push_q(t, kind):
            za, ya = push_axes(t, kind)
            return frame_quat(v2o(za), v2o(ya))

        # the fist pushes at Z_PUSH 1.35 m where it reaches (0.35 m over the levers: v25's fist at 1.2 caught on the far
        # lever's spindle as the base backed away and pulled the robot over), else at Z_EDGE; along the leaf it is
        # kept where the base's last move leaves it (the distance nearest d_want first), so the arm does not stretch
        # back to the edge after every hop
        Z_PUSH = 1.35
        D_WANT = [0.84]

        def pick_drive(nom, t, gap_t=None):
            offs = sorted((abs(i) + abs(j) + 0.5 * abs(k), 0.05 * i, 0.05 * j, 0.15 * k) for i in (-2, -1, 0, 1, 2)
                          for j in (-2, -1, 0, 1, 2) for k in (-2, -1, 0, 1, 2))
            ds = [d for e, d in sorted((abs(d - D_WANT[0]), d) for d in PUSH_DS)]
            # robust (the base turned +-0.06 rad too) first; then, if nothing is, reachable from the pose exactly
            for strict in (True, False):
                for cost, dx, dy, dw in offs:
                    c = (nom[0] + dx, nom[1] + dy, nom[2] + dw)
                    if leaf_gap(c, t if gap_t is None else gap_t) < GAP_MIN or not wall_free(c):
                        continue
                    for d in ds[:5]:
                        for z in (Z_PUSH, Z_EDGE):
                            for kind in (0, 1):
                                za, ya = push_axes(t, kind)
                                if (robust if strict else reachable)(LEFT, push_pt(t, d), z, za, ya, c):
                                    return c, d, kind, z
            return None

        def door_seen():
            """the leaf's angle from the head camera: the median bearing about the hinge of the door mask's points
            0.3-0.85 m from the hinge axis at 0.6-1.6 m high (the frame's jambs lie at ~0 and ~0.9 m); None if unseen"""
            got = best_points(["door"], min_pts=300)
            if got is None:
                return None
            bs = []
            for p in got[2]:
                if 0.6 < p[2] < 1.6:
                    q = o2h(p[:2])
                    if 0.3 < math.hypot(q[0], q[1]) < 0.85:
                        bs.append(math.atan2(-q[0], q[1]))
            if len(bs) < 50:
                return None
            bs.sort()
            return bs[len(bs) // 2]

        def fist_door_angle():
            lp = o2h(hand_pos(LEFT))
            return math.atan2(-lp[0], lp[1]) + math.atan2(PUSH_OFF, max(0.1, math.hypot(lp[0], lp[1])))

        def step(tag, nom, t):
            # the leaf's clearance checked for the door GAP_LAG 0.1 rad less open than aimed (it can lag the fist)
            got = pick_drive(nom, t, gap_t=t - GAP_LAG)
            print(tag, "nominal", fmt(nom), "->", None if got is None else (fmt(got[0]), got[1], got[2]))
            if got is None:
                return False
            c, d, kind, z = got
            bo = behavior.get_base_pose()
            prev = o2h([bo[0], bo[1]])
            hop_to(tag, *base_at(c[:2], c[2]), steps=120)
            reach(tag + " fist", on_leaf(t, d, PUSH_OFF, z), push_q(t, kind), tol=0.02, steps=80, arm=LEFT)
            # where the next step should push: as far towards the hinge as this one moved the base towards it
            D_WANT[0] = max(0.25, d - max(0.0, c[0] - prev[0]))
            return True

        ok = True
        th = fist_door_angle()
        bo = behavior.get_base_pose()
        here = o2h([bo[0], bo[1]]) + [bo[2] - PHI]
        # turn round clockwise at TURN_X -1.15 (probe: every heading from 0 round to pi reachable there, door 1.4)
        TURN_X = -1.15
        # TURN_STEP 0.15 rad a hop: the fist, fixed to the arm while the base turns, swings <= ~6 cm before it is
        # re-placed (v21's 0.4 rad hops swung it off the face and the door shut behind it on 3 of 4 seeds)
        TURN_STEP = 0.15
        y0r = here[2]
        nst = max(1, int(math.ceil(abs(-math.pi - y0r) / TURN_STEP)))
        # the fist opens the door on the way round, from where it is to TH_RD 1.55 rad (push_probe.py: facing back
        # into the corridor the fist reaches the face at ~1.4-1.7 rad, not at v22's ~1.25)
        # Each push point is at most PUSH_AHEAD 0.06 rad past the door's angle read from the fist (v23 aimed at the
        # plan's angle; where the door lagged, the fist went round to its near face and the leaf shut onto the base).
        TH_RD, PUSH_AHEAD = 1.55, 0.06
        for k in range(1, nst + 1):
            th = min(TH_RD, fist_door_angle() + PUSH_AHEAD)
            ok = step("turn round %d" % k, (TURN_X, LANE_Y, y0r + k / nst * (-math.pi - y0r)), th)
            if not ok:
                break
        seen = door_seen()
        print("door seen", None if seen is None else round(math.degrees(seen), 1), "deg")
        report["door_seen_after_turn_deg"] = None if seen is None else round(math.degrees(seen), 1)
        # DOOR_LOST 1.0 rad: a leaf seen under this has got past the fist (it is meant to be at ~1.3-1.6)
        DOOR_LOST = 1.0
        if seen is not None and seen < DOOR_LOST:
            ok = False
            print("door got away from the fist: stop")
        report["turned_round"] = ok
        if ok:
            # the fist opens the door to TH_D (square to the wall: the leaf clear of the lane) in two pushes
            stage("open square")
            for k in range(4):
                t = min(TH_D, fist_door_angle() + PUSH_AHEAD)
                got = pick_drive((TURN_X, LANE_Y, -math.pi), t)
                if got is not None:
                    reach("push open %d" % k, on_leaf(t, got[1], PUSH_OFF, got[3]), push_q(t, got[2]), tol=0.02, steps=80, arm=LEFT)
            th = fist_door_angle()
            report["door_square_deg"] = round(math.degrees(th), 1)
            print("   door ~", round(math.degrees(th), 1), "deg")
            stage("through pushing the door")
            x = TURN_X
            while x < PUSH_OUT - 1e-3:
                x = min(PUSH_OUT, x + 0.1)
                ok = step("back %.2f" % x, (x, LANE_Y, -math.pi), min(TH_D, fist_door_angle() + PUSH_AHEAD))
                if not ok:      # half a step, then on
                    x -= 0.05
                    ok = step("back %.2f" % x, (x, LANE_Y, -math.pi), min(TH_D, fist_door_angle() + PUSH_AHEAD))
                if not ok:
                    break
                if x < -0.3:
                    seen = door_seen()
                    print("   door seen", None if seen is None else round(math.degrees(seen), 1), "deg")
                    if seen is not None and seen < DOOR_LOST:
                        ok = False
                        print("door got away from the fist: stop")
                        break
            report["pushed_to_x"] = round(o2h(behavior.get_base_pose()[:2])[0], 3)
        # Block 8: past the door plane (the base's front PUSH_OUT - 0.24 m beyond it): the fist lets go and the base
        # backs on to END_X 0.8 (the witness wants the centre 0.6 m past the plane); the door shuts behind
        if ok and report.get("pushed_to_x", -9) >= PUSH_OUT - 0.05:
            stage("through")
            reach("fist off", on_leaf(TH_D, 0.2, PUSH_OFF + 0.12, Z_EDGE), push_q(TH_D, 0), tol=0.05, steps=60, arm=LEFT)
            move_to_joints(home[LEFT], tolerance=0.05, max_steps=150, arm=LEFT)
            END_X = 0.8
            hop_to("through", *base_at([END_X, LANE_Y], -math.pi), steps=200)
            set_torso([0.0, 0.0, 0.0, 0.0])
        else:
            stage("stopped")
        report["base_end_h"] = fmt(o2h(behavior.get_base_pose()[:2]))
print("report:", report)
