# Pick up the coffee cup on BEHAVIOR-1K `picking_up_cup` (run_behavior.py's task: the kitchen counter of
# set_up_a_coffee_station_in_your_kitchen) with the Galaxea R1 Pro, through cap-harness-behavior.sh. Learned, not
# hand-written: a coding agent built it from the radio skill and the shared library over instances 36-57 in 7 versions;
# held the cup on 8 of 12 fresh instances (58-69). Explores the counters, pinches the rim with one finger inside the cup,
# then lifts 15 cm. Misses: a cup SAM3 scores too weakly to find, a look-alike chosen from weak detections, a cup at the
# back of the counter out of reach.
# policy.py - one R1 Pro policy for picking_up_cup (BEHAVIOR-1K set_up_a_coffee_station_in_your_kitchen reduced to "pick
# up the cup" from a kitchen countertop) on any instance (generalization dev). Grown from the turning_on_radio
# generalization policy (adapter/programs/pick_radio_r1pro.py, v16): search sweep, floor guard, ring candidates ranked by
# the support edge, servo hop to the edge, incremental torso lean, IK-jump-limited Cartesian moves, wrist reads, a rung
# ladder and a stepped lift. The cup is pinched by its rim wall, one finger inside, so it stays upright.
import math

report = {"stage": "start"}
known = {}      # the last accepted cup centre (see CUP_JUMP_M)


# Code block 1: sweep the head camera in place until the cup segments confidently; collect the floor seen on the way.
def fmt(v, nd=3):
    try:
        return [round(float(x), nd) for x in v]
    except TypeError:
        return round(float(v), nd)


def pose_rows(p):
    return {"pos": fmt(p.position), "q": fmt(p.quaternion_wxyz), "frame": p.frame}


def seg_best(cam, prompt):
    """Best SAM3 hit for a prompt: (score, box_xyxy, n) or (0.0, None, 0)."""
    seg = segment_text(cam, prompt)
    best_s, best_b, n = 0.0, None, 0
    if seg.ok:
        for s in seg.segmentations:
            n += 1
            if float(s.score) > best_s:
                best_s = float(s.score)
                best_b = fmt(s.box_xyxy, 0) if s.box_xyxy is not None else None
    return best_s, best_b, n


ctx = get_task_context()
meta = behavior.get_task_metadata()
cmeta = behavior.get_controller_metadata()
target_prompt = str(meta.get("target_prompt") or "coffee cup")
support_prompt = str(meta.get("support_prompt") or "countertop")
print("language:", ctx.language, "| target_prompt:", target_prompt, "| support_prompt:", support_prompt)
print("torso_positions:", fmt(cmeta.get("torso_positions", [])), "lower:", fmt(cmeta.get("torso_lower_bounds", [])),
      "upper:", fmt(cmeta.get("torso_upper_bounds", [])))
bx, by, byaw = behavior.get_base_pose()
rs = get_robot_state()
print("base_pose:", fmt([bx, by, byaw]), "base_frame:", rs.base_frame)
for a0 in ("primary", "secondary"):
    print(a0, "gripper:", round(float(rs.gripper_positions[a0]), 3), "eef:", pose_rows(rs.end_effector_poses[a0]))
report["base0"] = fmt([bx, by, byaw])
torso0 = [float(v) for v in cmeta.get("torso_positions", [0.0, 0.0, 0.0, 0.0])]
act_lo = [float(v) for v in cmeta.get("action_lower_bounds", [])]
act_hi = [float(v) for v in cmeta.get("action_upper_bounds", [])]


def arm_bounds(arm_name):
    off = 10 + (0 if arm_name == "primary" else 1)
    if len(act_lo) < 24:
        return None
    return [(act_lo[off + 2 * k], act_hi[off + 2 * k]) for k in range(7)]


def ik_joints(ik):
    """The joint solution of an IKResult; None when absent."""
    try:
        v = ik.joint_positions
    except Exception:
        return None
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except TypeError:
        return None


def bounds_margin(arm_name, joints):
    """Smallest distance (rad) of a joint vector to its action bounds; negative means outside. None when unknown."""
    b = arm_bounds(arm_name)
    if b is None or joints is None or len(joints) != 7:
        return None
    return round(min(min(j - lo, hi - j) for j, (lo, hi) in zip(joints, b)), 3)


# Observed floor as 0.1 m cells (odom xy): the base fell off the scene through an opening the traversability map called
# free (turning_on_radio seed 27), so every base goal must be on floor the head camera has seen.
FLOOR_CELL_M = 0.1
floor_cells = {}


def collect_floor(tag):
    fl = localize_object("floor", camera_name="head")
    if not fl.ok:
        print(tag, "floor localize failed:", fl.error.message if fl.error else None)
        return 0
    pts = fl.point_cloud.points
    k = 0
    for i in range(0, len(pts), 8):      # every 8th point: a full-image floor mask is ~100k points
        p = pts[i]
        floor_cells[(int(math.floor(float(p[0]) / FLOOR_CELL_M)), int(math.floor(float(p[1]) / FLOOR_CELL_M)))] = 1
        k += 1
    print(tag, "floor points kept:", k, "of", len(pts), "cells:", len(floor_cells))
    return k


def floor_near(x, y, r=0.25):
    """True when a seen floor cell lies within ~r of (x, y); r 0.25 is half the base footprint, untuned."""
    n = int(math.ceil(r / FLOOR_CELL_M))
    cx0 = int(math.floor(x / FLOOR_CELL_M))
    cy0 = int(math.floor(y / FLOOR_CELL_M))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if (cx0 + i, cy0 + j) in floor_cells:
                ccx = (cx0 + i + 0.5) * FLOOR_CELL_M
                ccy = (cy0 + j + 0.5) * FLOOR_CELL_M
                if (ccx - x) ** 2 + (ccy - y) ** 2 <= (r + 0.5 * FLOOR_CELL_M) ** 2:
                    return True
    return False


def floor_along(x0, y0, x1, y1, step=0.25):
    """Fraction of samples along the segment that have observed floor nearby (1.0 = the whole leg is on seen floor)."""
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    hits = 0
    for i in range(n + 1):
        t = i / float(n)
        if floor_near(x0 + t * (x1 - x0), y0 + t * (y1 - y0)):
            hits += 1
    return hits / float(n + 1)


def median_of(vals):
    v = sorted(vals)
    return v[len(v) // 2]


# SEARCH_MIN_SCORE 0.25: the radio search's threshold (real radio 0.40-0.75 from the head, garbage 0.007-0.04); the cup
# scored 0.45-0.85 from ~1 m (dev v01 seeds 40, 49). CUP_MIN_Z / CUP_MAX_Z 0.5 / 1.25: a detection whose cropped median height is outside this
# band is not on a counter or table the arm can reach (shelves and wall cabinets sit higher, the floor lower); untuned.
SEARCH_MIN_SCORE = 0.25
CUP_MIN_Z = 0.5
CUP_MAX_Z = 1.25
# CUP_CROP_M 0.12: half-width of the box kept around the cloud's per-axis median (kills the depth tail); a cup is ~0.1
# across with its handle, the radio used 0.2. Untuned.
CUP_CROP_M = 0.12


def robust_crop(cloud, half_m):
    """Crop a localized cloud to a box of half-width half_m around its per-axis median (kills the depth tail)."""
    pts = cloud.points
    cx = median_of([float(p[0]) for p in pts])
    cy = median_of([float(p[1]) for p in pts])
    cz = median_of([float(p[2]) for p in pts])
    try:
        return crop_point_cloud(cloud, [cx - half_m, cy - half_m, cz - half_m], [cx + half_m, cy + half_m, cz + half_m])
    except ValueError:
        return cloud


def plausible_target(tag):
    """Localize the cup from the head; keep it only when its median-cropped height is in the reachable band."""
    lc = localize_object(target_prompt, camera_name="head")
    print(tag, "localize [" + target_prompt + "] ok:", lc.ok, "err:", lc.error.message if lc.error else None)
    if not lc.ok or len(lc.point_cloud.points) < 5:
        return None
    cc = robust_crop(lc.point_cloud, CUP_CROP_M)
    zs0 = sorted(float(p[2]) for p in cc.points)
    if not zs0:
        return None
    zmed = zs0[len(zs0) // 2]
    if zmed > CUP_MAX_Z or zmed < CUP_MIN_Z:
        print(tag, "rejected: median height", round(zmed, 3), "outside", CUP_MIN_Z, CUP_MAX_Z)
        return None
    return cc


# Countertop seen on the way (odom xy samples of every support mask scoring >= COUNTER_MIN_SCORE, at counter height):
# where to look next when the cup is too far to segment (dev v01 seed 36: the cup 3.5 m away is ~6 px wide and scored
# 0.007, against 0.85 at ~1 m on seed 40). COUNTER_MIN_SCORE 0.3: the counter masks scored 0.48-0.97, untuned.
COUNTER_MIN_SCORE = 0.3
counter_pts = []


def collect_counter(tag):
    seg = segment_text("head", support_prompt)
    if not seg.ok:
        return 0
    k0 = len(counter_pts)
    for sgm in seg.segmentations:
        if float(sgm.score) < COUNTER_MIN_SCORE:
            continue
        try:
            pc = mask_to_point_cloud(sgm, "head", target_frame=rs.base_frame)
        except ValueError:
            continue
        pts = pc.points
        for i in range(0, len(pts), 25):
            zc = float(pts[i][2])
            if CUP_MIN_Z - 0.1 <= zc <= CUP_MAX_Z:
                counter_pts.append((float(pts[i][0]), float(pts[i][1]), zc))
    print(tag, "counter samples added:", len(counter_pts) - k0)
    return len(counter_pts) - k0


s_r, b_r, n_r = seg_best("head", target_prompt)
s_t, b_t, n_t = seg_best("head", support_prompt)
turn_dir = 1.0
if b_t is not None:
    turn_dir = 1.0 if 0.5 * (b_t[0] + b_t[2]) < 256.0 else -1.0     # the support's box left of centre -> turn left (+yaw)
print("start heading: cup", round(s_r, 3), b_r, "counter", round(s_t, 3), b_t, "turn_dir", turn_dir)
collect_floor("floor@start")
collect_counter("counter@start")
# HINT_MIN_SCORE 0.05: a cup mask scoring this much whose cloud is at counter height is worth a closer look (dev v02
# seed 42: the real cup scored 0.156 on an 8 px box from 3.4 m, while the other boxes scored 0.002-0.1). Untuned.
HINT_MIN_SCORE = 0.05
hints = []


# Consensus: HINT_AGREE hints taken from at least two look poses whose centres lie within HINT_AGREE_M of each other are
# the cup (dev v04 seeds 44 and 46: from 0.9-1.1 m the real cup scored only 0.054-0.089, but five / two hints from
# different poses landed within 0.12 / 0.08 m of it, while the one stray hint stood alone). Untuned.
HINT_AGREE = 2
# HINT_ON_COUNTER_M 0.05: how far a hint's base may sit from the counter height next to it; untuned.
HINT_ON_COUNTER_M = 0.05
HINT_AGREE_M = 0.15
look_id = {"n": 0}


def note_hint(tag, score):
    """Record a weak detection as a hint; returns its cropped cloud when the hints now agree on a spot, else None."""
    if not (HINT_MIN_SCORE <= score < SEARCH_MIN_SCORE):
        return None
    cc = plausible_target(tag + " (hint)")
    if cc is None:
        return None
    hx_ = median_of([float(p[0]) for p in cc.points])
    hy_ = median_of([float(p[1]) for p in cc.points])
    zsh = sorted(float(p[2]) for p in cc.points)
    hbase = zsh[len(zsh) // 20]
    # a hint must stand on a countertop seen beside it: its 5 % height within HINT_ON_COUNTER_M of a counter sample
    # within 0.3 m (dev v06 seed 46: two hints agreed on an object on a wall shelf, base 1.15 m, over a 0.9 m counter,
    # and the policy picked it up)
    on_counter = any(abs(c[2] - hbase) <= HINT_ON_COUNTER_M and math.hypot(c[0] - hx_, c[1] - hy_) <= 0.3 for c in counter_pts)
    if not on_counter:
        print(tag, "hint at", round(hx_, 3), round(hy_, 3), "base", round(hbase, 3), "not on a seen countertop: ignored")
        return None
    hints.append((score, hx_, hy_, look_id["n"]))
    near = [h for h in hints if math.hypot(h[1] - hx_, h[2] - hy_) <= HINT_AGREE_M]
    poses = set(h[3] for h in near)
    print(tag, "hint", round(score, 3), "at", round(hx_, 3), round(hy_, 3), "agreeing hints", len(near), "from poses", len(poses))
    if len(near) >= HINT_AGREE and len(poses) >= 2:
        print(tag, "hints agree: taking this as the cup")
        return cc
    return None


cup_cloud = None
best_heading = (s_r, byaw)
if s_r >= SEARCH_MIN_SCORE:
    cup_cloud = plausible_target("heading 0")
note_hint("heading 0", s_r)
yaw_now = byaw
for k in range(1, 8):
    if cup_cloud is not None:
        break
    yaw_now = math.atan2(math.sin(yaw_now + turn_dir * math.radians(45.0)), math.cos(yaw_now + turn_dir * math.radians(45.0)))
    r = behavior.navigate_to_pose(bx, by, yaw_now, planner="servo", max_steps=200)
    s_r, b_r, n_r = seg_best("head", target_prompt)
    print("turn", k, "to yaw", round(math.degrees(yaw_now), 1), "ok:", r.ok, "steps:", r.steps_executed, "| cup", round(s_r, 3), b_r, "n", n_r)
    collect_floor("floor@" + str(k))
    collect_counter("counter@" + str(k))
    if s_r > best_heading[0]:
        best_heading = (s_r, yaw_now)
    if s_r >= SEARCH_MIN_SCORE:
        cup_cloud = plausible_target("heading " + str(k))
    note_hint("heading " + str(k), s_r)
report["search_best"] = [round(best_heading[0], 3), round(math.degrees(best_heading[1]), 1)]
report["floor_cells"] = len(floor_cells)
report["found"] = cup_cloud is not None
report["stage"] = "search"


# Code block 2: cup geometry, the countertop near it, and the ring pose from which the cup is nearest the counter edge.
def cloud_axes(cloud):
    """Principal horizontal axis of a cloud: (angle_rad, long_len, short_len, cx, cy) from the xy covariance and 5-95 % spans."""
    pts = cloud.points
    n = len(pts)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) / n
    syy = sum((y - my) ** 2 for y in ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    ca, sa = math.cos(ang), math.sin(ang)
    lon = sorted((x - mx) * ca + (y - my) * sa for x, y in zip(xs, ys))
    sho = sorted(-(x - mx) * sa + (y - my) * ca for x, y in zip(xs, ys))
    long_len = lon[(19 * n) // 20] - lon[n // 20]
    short_len = sho[(19 * n) // 20] - sho[n // 20]
    cx = mx + 0.5 * (lon[n // 20] + lon[(19 * n) // 20]) * ca - 0.5 * (sho[n // 20] + sho[(19 * n) // 20]) * sa
    cy = my + 0.5 * (lon[n // 20] + lon[(19 * n) // 20]) * sa + 0.5 * (sho[n // 20] + sho[(19 * n) // 20]) * ca
    return ang, long_len, short_len, cx, cy


def describe(tag, cloud):
    pts = cloud.points
    xs = sorted(float(p[0]) for p in pts)
    ys = sorted(float(p[1]) for p in pts)
    zs = sorted(float(p[2]) for p in pts)
    n = len(zs)
    print(tag, "n:", n, "x[5,50,95]:", round(xs[n // 20], 3), round(xs[n // 2], 3), round(xs[(19 * n) // 20], 3),
          "y[5,50,95]:", round(ys[n // 20], 3), round(ys[n // 2], 3), round(ys[(19 * n) // 20], 3),
          "z[5,50,95]:", round(zs[n // 20], 3), round(zs[n // 2], 3), round(zs[(19 * n) // 20], 3))


# CUP_JUMP_M 0.4: a later read further than this from the last accepted centre is another object (the radio's
# RADIO_JUMP_M: collection seed 56 locked onto a red object on a shelf after the drive). Untuned for the cup.
CUP_JUMP_M = 0.4
# CAND_MIN_SCORE 0.02: candidate masks considered when the best one is implausible (only those near the last accepted
# centre count; the cup scored 0.05-0.09 from some 1 m views, dev v04 seeds 44 and 46); untuned.
CAND_MIN_SCORE = 0.02
# RIM_BAND_M 0.025: the top band of the cup cloud taken as its rim (the handle's top sits at or under the rim); untuned.
RIM_BAND_M = 0.025
# HANDLE_MIN_EXCESS 0.02: a rim band longer than wide by more than this has the handle along its long axis (v01: the round
# rim read 0.074-0.086 by 0.074-0.082 from the head, excess up to 0.015 with no handle there). Untuned.
HANDLE_MIN_EXCESS = 0.02


def cloud_plausible(cloud):
    """(ok, cx, cy, zmed) of a median-cropped cloud: near the last accepted centre and in the reachable height band."""
    pts = cloud.points
    if len(pts) < 10:
        return False, 0.0, 0.0, 0.0
    cx = median_of([float(p[0]) for p in pts])
    cy = median_of([float(p[1]) for p in pts])
    cz = median_of([float(p[2]) for p in pts])
    ok = CUP_MIN_Z <= cz <= CUP_MAX_Z
    if "xy" in known and math.hypot(cx - known["xy"][0], cy - known["xy"][1]) > CUP_JUMP_M:
        ok = False
    return ok, cx, cy, cz


def cup_geometry(cloud):
    """Rim geometry from the top band of the cloud: the band's horizontal axes; when it is elongated (long - short >
    1.2 cm) the handle lies along the long axis, on the end with more points beyond the rim radius. The body centre is
    then the non-handle end plus the radius; the radius is half the short span. Base = 5 % height (the counter)."""
    zs = sorted(float(p[2]) for p in cloud.points)
    top = zs[(19 * len(zs)) // 20]
    base = zs[len(zs) // 20]
    band = cloud
    try:
        b2 = crop_point_cloud(cloud, [-50.0, -50.0, top - RIM_BAND_M], [50.0, 50.0, top + 1.0])
        if len(b2.points) >= 15:
            band = b2
    except ValueError:
        pass
    ang, lk, sk, gx, gy = cloud_axes(band)
    rad = 0.5 * sk
    hx, hy = 0.0, 0.0       # unit vector toward the handle, (0, 0) when none is seen
    ca, sa = math.cos(ang), math.sin(ang)
    if lk - sk > HANDLE_MIN_EXCESS:
        pr = sorted((float(p[0]) - gx) * ca + (float(p[1]) - gy) * sa for p in band.points)
        lo5, hi95 = pr[len(pr) // 20], pr[(19 * len(pr)) // 20]
        n_hi = len([v for v in pr if v > hi95 - (lk - sk)])
        n_lo = len([v for v in pr if v < lo5 + (lk - sk)])
        if n_hi < n_lo:          # the thin handle end has fewer points than the round body's end
            hx, hy = ca, sa
            cpos = lo5 + rad
        else:
            hx, hy = -ca, -sa
            cpos = hi95 - rad
        gx, gy = gx + cpos * ca, gy + cpos * sa
    return {"gx": gx, "gy": gy, "top": top, "base": base, "rad": rad, "long": lk, "short": sk, "ang": ang,
            "hx": hx, "hy": hy, "n": len(zs), "n_band": len(band.points)}


def localize_cup(tag, cam="head"):
    """Segment + localize the cup, median-cropped; returns the rim geometry or None. When the best mask is implausible
    (far from the last accepted centre or out of the height band), take the plausible candidate mask nearest it."""
    loc = localize_object(target_prompt, camera_name=cam)
    cloud = None
    if loc.ok and len(loc.point_cloud.points) >= 10:
        cloud = robust_crop(loc.point_cloud, CUP_CROP_M)
        okp, cx0, cy0, cz0 = cloud_plausible(cloud)
        if not okp:
            print(tag, "best mask implausible: centre", round(cx0, 3), round(cy0, 3), round(cz0, 3), "known", fmt(known.get("xy", [])))
            cloud = None
    else:
        print(tag, "localize failed:", loc.error.message if loc.error else None)
    if cloud is None and "xy" in known:
        seg = segment_text(cam, target_prompt)
        best, best_d = None, None
        if seg.ok:
            for s in seg.segmentations:
                if float(s.score) < CAND_MIN_SCORE:
                    continue
                try:
                    pc = robust_crop(mask_to_point_cloud(s, cam, target_frame=rs.base_frame), CUP_CROP_M)
                except ValueError:
                    continue
                okp, cx1, cy1, cz1 = cloud_plausible(pc)
                dd = math.hypot(cx1 - known["xy"][0], cy1 - known["xy"][1])
                print(tag, "candidate score", round(float(s.score), 3), "centre", round(cx1, 3), round(cy1, 3), round(cz1, 3), "dist", round(dd, 3), "ok", okp)
                if okp and (best_d is None or dd < best_d):
                    best, best_d = pc, dd
        cloud = best
    if cloud is None:
        return None
    describe(tag + " cup cropped", cloud)
    g = cup_geometry(cloud)
    print(tag, "cup centre", round(g["gx"], 3), round(g["gy"], 3), "top", round(g["top"], 3), "base", round(g["base"], 3),
          "rad", round(g["rad"], 3), "rim long/short", round(g["long"], 3), round(g["short"], 3), "handle dir", round(g["hx"], 2), round(g["hy"], 2),
          "n_band", g["n_band"])
    if cam == "head":
        known["xy"] = [g["gx"], g["gy"]]
    return g


def support_near(ox, oy, zbase):
    """Countertop points near the cup: the support mask's cloud cropped to 1.5 m around the cup and to the cup's base
    height (-0.06 .. +0.03); when the best mask has too few points there, the candidate mask with the most."""
    seg = segment_text("head", support_prompt)
    best, best_n = None, 0
    if seg.ok:
        for s in seg.segmentations:
            if float(s.score) < CAND_MIN_SCORE:
                continue
            try:
                pc = mask_to_point_cloud(s, "head", target_frame=rs.base_frame)
                pc = crop_point_cloud(pc, [ox - 1.5, oy - 1.5, zbase - 0.06], [ox + 1.5, oy + 1.5, zbase + 0.03])
            except ValueError:
                continue
            nn = len(pc.points)
            print("support candidate score", round(float(s.score), 3), "points near the cup", nn)
            if nn > best_n:
                best, best_n = pc, nn
    return best


def free_poses_around(ox, oy, dists, step_deg=15):
    """Candidate base poses on rings around (ox, oy) that the traversability map accepts, facing the object."""
    out = []
    for d in dists:
        for k in range(0, 360, step_deg):
            a = math.radians(k)
            x = ox + d * math.cos(a)
            y = oy + d * math.sin(a)
            if behavior.base_pose_is_free(x, y):
                out.append((round(d, 2), k, round(x, 3), round(y, 3), round(math.atan2(oy - y, ox - x), 3)))
    return out


def edge_distance(support, ox, oy, ang):
    """How far the support surface extends from (ox, oy) along bearing ang (rad), within a 0.15 m lateral band."""
    dx, dy = math.cos(ang), math.sin(ang)
    best = 0.0
    for p in support.points:
        rx, ry = float(p[0]) - ox, float(p[1]) - oy
        along = rx * dx + ry * dy
        lateral = abs(-rx * dy + ry * dx)
        if lateral < 0.15 and along > best:
            best = along
    return best


def nav(tag, x, y, yaw, planner, max_steps=1500):
    r = behavior.navigate_to_pose(x, y, yaw, planner=planner, max_steps=max_steps)
    d = dict(r.diagnostics)
    print(tag, planner, "->", fmt([x, y, yaw]), "ok:", r.ok, "status:", str(r.status), "steps:", r.steps_executed,
          "err:", r.error.message if r.error else None,
          "diag:", {k: d[k] for k in d if k in ("route", "hops", "servo_hops", "moved_x", "moved_y", "moved_yaw")},
          "base now:", fmt(behavior.get_base_pose()))
    return r


def rel_of(px, py):
    """(forward, left) of an odom point in the current base frame."""
    bxr, byr, byawr = behavior.get_base_pose()
    return ((px - bxr) * math.cos(byawr) + (py - byr) * math.sin(byawr),
            -(px - bxr) * math.sin(byawr) + (py - byr) * math.cos(byawr))


# Exploration when the sweep found nothing: the cup is ~0.09 m across and does not segment from 2.5-4 m (dev v01/v02
# seeds 36, 42: 0.007-0.16 on ~8 px boxes; 0.45-0.85 from ~1 m on seeds 40, 49). Visit viewpoints VIEW_DIST_M in front of
# what has not been looked at from within LOOK_RANGE_M: weak cup hints first (best score first), then countertop regions
# (0.4 m cells of the counter samples) nearest first; at each, segment facing it and turned +-40 deg. A look covers points
# within LOOK_RANGE_M and 90 deg of the facing direction (the start sweep: all around). MAX_VIEWPOINTS 14: a viewpoint
# costs 60-500 ticks; 8 ran out on dev v05 seed 46 before reaching the far end of the kitchen (the cup at the back wall
# 3.7 m away). VIEW_DIST_M 0.9 / LOOK_RANGE_M 1.2 untuned beyond those seeds.
VIEW_DIST_M = 0.9
# HINT_LOOK_LEAN 0.35: torso j0 for the extra look at a hint from above; untuned.
HINT_LOOK_LEAN = 0.35
LOOK_RANGE_M = 1.2
MAX_VIEWPOINTS = 14
looks = [(bx, by, 0.0, 181.0)]


def covered(px, py):
    for lx, ly, lyaw, half in looks:
        if math.hypot(px - lx, py - ly) <= LOOK_RANGE_M:
            dyaw = abs(math.degrees(math.atan2(math.sin(math.atan2(py - ly, px - lx) - lyaw), math.cos(math.atan2(py - ly, px - lx) - lyaw))))
            if dyaw <= half:
                return True
    return False


if cup_cloud is None and (counter_pts or hints):
    cells = {}
    for cxp, cyp, czp in counter_pts:
        key = (int(math.floor(cxp / 0.4)), int(math.floor(cyp / 0.4)))
        cells.setdefault(key, []).append((cxp, cyp))
    regions = []
    for key in cells:
        v = cells[key]
        if len(v) >= 5:
            regions.append((sum(a for a, b in v) / len(v), sum(b for a, b in v) / len(v)))
    n_vp = 0
    tried = []
    report["n_regions"] = len(regions)
    while n_vp < MAX_VIEWPOINTS and cup_cloud is None:
        bxe, bye, byawe = behavior.get_base_pose()
        hs_todo = sorted([h for h in hints if not covered(h[1], h[2]) and all(math.hypot(h[1] - a, h[2] - b) > 0.3 for a, b in tried)],
                         key=lambda h: -h[0])
        if hs_todo:
            tx_, ty_ = hs_todo[0][1], hs_todo[0][2]
            kind = "hint " + str(round(hs_todo[0][0], 3))
        else:
            todo = [rg for rg in regions if not covered(rg[0], rg[1]) and all(math.hypot(rg[0] - a, rg[1] - b) > 0.3 for a, b in tried)]
            if not todo:
                break
            todo.sort(key=lambda rg: math.hypot(rg[0] - bxe, rg[1] - bye))
            tx_, ty_ = todo[0]
            kind = "region"
        tried.append((tx_, ty_))
        vps = [c for c in free_poses_around(tx_, ty_, (VIEW_DIST_M, VIEW_DIST_M + 0.2, VIEW_DIST_M + 0.4), 20) if floor_near(c[2], c[3])]
        if not vps:
            print("explore: no free viewpoint on seen floor for", kind, fmt([tx_, ty_]))
            continue
        vps.sort(key=lambda c: (c[0], math.hypot(c[2] - bxe, c[3] - bye)))
        vp = vps[0]
        n_vp += 1
        print("explore viewpoint", n_vp, "for", kind, fmt([tx_, ty_]), "->", vp)
        nav("explore " + str(n_vp), vp[2], vp[3], vp[4], "curobo")
        collect_floor("floor@vp" + str(n_vp))
        collect_counter("counter@vp" + str(n_vp))
        bxv, byv, byawv = behavior.get_base_pose()
        looks.append((bxv, byv, byawv, 90.0))
        look_id["n"] += 1
        for turn in (0.0, 40.0, -40.0, "lean"):
            if turn == "lean":
                # a hint still unconfirmed: look again from above with the torso leaned (the rim shows from there)
                if not kind.startswith("hint"):
                    break
                behavior.navigate_to_pose(bxv, byv, byawv, planner="servo", max_steps=200)
                tl = list(torso0)
                tl[0] = torso0[0] + HINT_LOOK_LEAN
                behavior.move_torso(tl, max_steps=120)
                s_v, b_v, n_v = seg_best("head", target_prompt)
                print("explore", n_vp, "leaned look: cup", round(s_v, 3), b_v)
                if s_v >= SEARCH_MIN_SCORE:
                    cup_cloud = plausible_target("explore " + str(n_vp) + " leaned")
                if cup_cloud is None:
                    cup_cloud = note_hint("explore " + str(n_vp) + " leaned", s_v)
                behavior.move_torso(list(torso0), max_steps=120)
                break
            if turn != 0.0:
                behavior.navigate_to_pose(bxv, byv, byawv + math.radians(turn), planner="servo", max_steps=200)
            s_v, b_v, n_v = seg_best("head", target_prompt)
            print("explore", n_vp, "turn", turn, "cup", round(s_v, 3), b_v)
            if s_v >= SEARCH_MIN_SCORE:
                cup_cloud = plausible_target("explore " + str(n_vp) + " turn " + str(turn))
                if cup_cloud is not None:
                    break
            if cup_cloud is None:
                cup_cloud = note_hint("explore " + str(n_vp) + " turn " + str(turn), s_v)
            if cup_cloud is not None:
                break
    report["viewpoints"] = n_vp
    report["found"] = cup_cloud is not None

sup = None
g = None
if report["found"]:
    g = cup_geometry(cup_cloud)
    known["xy"] = [g["gx"], g["gy"]]
    describe("cup (far read)", cup_cloud)
    report["cup_far"] = [round(g["gx"], 3), round(g["gy"], 3), round(g["top"], 3), round(g["base"], 3)]
    sup = support_near(g["gx"], g["gy"], g["base"])
    report["support_pts"] = len(sup.points) if sup is not None else 0
if sup is not None and len(sup.points) >= 30:
    ox, oy = g["gx"], g["gy"]
    cands = [c for c in free_poses_around(ox, oy, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)) if floor_near(c[2], c[3])]
    scored = []
    for c in cands:
        ed = edge_distance(sup, ox, oy, math.atan2(c[3] - oy, c[2] - ox))
        scored.append((round(ed, 3), c[0], c))
    scored.sort()
    for row in scored[:6]:
        print("ring cand (edge_m, d, cand):", row)
    report["n_cands"] = len(scored)
    if scored:
        choice = scored[0][2]
        report["nav_choice"] = choice
        report["edge_m"] = scored[0][0]
        bx0, by0, byaw0 = behavior.get_base_pose()
        planner0 = "servo" if floor_along(bx0, by0, choice[2], choice[3]) >= 0.999 else "curobo"
        nav("nav1", choice[2], choice[3], choice[4], planner0)
        bx1, by1, byaw1 = behavior.get_base_pose()
        if math.hypot(bx1 - choice[2], by1 - choice[3]) > 0.1 or abs(math.atan2(math.sin(byaw1 - choice[4]), math.cos(byaw1 - choice[4]))) > 0.15:
            nav("nav1 servo-fix", choice[2], choice[3], choice[4], "servo", max_steps=600)
        bx1, by1, byaw1 = behavior.get_base_pose()
        report["base_after_nav"] = fmt([bx1, by1, byaw1])
        collect_floor("floor@nav")
        g2 = localize_cup("after nav")
        if g2 is not None:
            g = g2
        report["nav_ok"] = True
report["stage"] = "navigated"


# Code block 3: quaternion helpers and IK-guarded moves.
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx_, by_, bz = b
    return [aw * bw - ax * bx_ - ay * by_ - az * bz, aw * bx_ + ax * bw + ay * bz - az * by_,
            aw * by_ - ax * bz + ay * bw + az * bx_, aw * bz + ax * by_ - ay * bx_ + az * bw]


def qrot(q, v):
    qv = [0.0, v[0], v[1], v[2]]
    qc = [q[0], -q[1], -q[2], -q[3]]
    r = qmul(qmul(q, qv), qc)
    return [r[1], r[2], r[3]]


def qnorm(q):
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return [x / n for x in q]


def quat_yz(yv, zv):
    """Quaternion of the hand frame whose +y (the finger travel axis, radio generalization dev) is yv and whose tool
    axis +z is zv (yv, zv orthogonal unit vectors in odom)."""
    xv = [yv[1] * zv[2] - yv[2] * zv[1], yv[2] * zv[0] - yv[0] * zv[2], yv[0] * zv[1] - yv[1] * zv[0]]
    m = [[xv[0], yv[0], zv[0]], [xv[1], yv[1], zv[1]], [xv[2], yv[2], zv[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        sq = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * sq, (m[2][1] - m[1][2]) / sq, (m[0][2] - m[2][0]) / sq, (m[1][0] - m[0][1]) / sq]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        sq = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        q = [(m[2][1] - m[1][2]) / sq, 0.25 * sq, (m[0][1] + m[1][0]) / sq, (m[0][2] + m[2][0]) / sq]
    elif m[1][1] > m[2][2]:
        sq = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        q = [(m[0][2] - m[2][0]) / sq, (m[0][1] + m[1][0]) / sq, 0.25 * sq, (m[1][2] + m[2][1]) / sq]
    else:
        sq = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        q = [(m[1][0] - m[0][1]) / sq, (m[0][2] + m[2][0]) / sq, (m[1][2] + m[2][1]) / sq, 0.25 * sq]
    return qnorm(q)


def pinch_frame(dx, dy, pitch_deg, sgn):
    """Rim-pinch orientation: d = (dx, dy) the horizontal direction from the pinch point into the cup (away from the
    robot). The tool points into the wall, pitched pitch_deg from straight down toward d; the fingers close along the
    pitched d (one finger outside the wall, one inside). Returns (q, tool z vector)."""
    p = math.radians(pitch_deg)
    zv = [dx * math.sin(p), dy * math.sin(p), -math.cos(p)]
    yv = [sgn * dx * math.cos(p), sgn * dy * math.cos(p), sgn * math.sin(p)]
    return quat_yz(yv, zv), zv


def terminated_by(r):
    return r is not None and (bool(r.terminated) or "terminated" in str(r.status).lower())


arm = "primary"
side = 1.0


def hand_pos():
    return [float(v) for v in get_robot_state().end_effector_poses[arm].position]


# MAX_JOINT_JUMP 1.0 rad / SMALL_JUMP 0.6 rad: an IK solution further than this from the current joints is another arm
# family; move_to_pose interpolates straight in joint space, so a far solution swings the hand (turning_on_radio seed 27
# attempt 4: the radio swept 1 m; generalization dev v05 seed 49: a 2 cm move took j5 -0.35 -> 1.77 and knocked it over).
MAX_JOINT_JUMP = 1.0
SMALL_JUMP = 0.6
# STEP_M 0.04: Cartesian step length of every approach leg near the counter (the radio used 3 cm near the object and 5 cm
# away from it); untuned.
STEP_M = 0.04
alt_quats = []      # the pinch family (other pitches and finger signs) reach_pose may fall back to


def reach_pose(tag, x, y, z, q_ref, tol, steps, max_jump=SMALL_JUMP):
    """move_to_pose when its predicted IK solution (seeded at the current joints) is within max_jump of the current
    joints; otherwise, or when its own IK is refused, take the nearest current-seeded IK solution over the alternative
    orientations and drive it with move_to_joints; failing that, hold still. Returns the last result."""
    cur = [float(v) for v in get_robot_state().joint_positions[arm]]
    pose0 = Pose(position=[x, y, z], quaternion_wxyz=q_ref, frame=rs.base_frame)
    ik0 = solve_ik(pose0, seed_joints=cur, arm=arm)
    j0 = ik_joints(ik0) if ik0.ok else None
    jump0 = max(abs(a - b) for a, b in zip(j0, cur)) if j0 is not None else None
    r = None
    if j0 is None or jump0 <= max_jump:
        r = move_to_pose(pose0, tolerance=tol, max_steps=steps, arm=arm)
        if r.ok or terminated_by(r) or "planner" not in str(r.status).lower():
            return r
    else:
        print(tag, "predicted IK jump", round(jump0, 2), "rad >", max_jump, ": nearest alternative instead")
    best, best_d = None, None
    for qq in [q_ref] + alt_quats:
        ik = solve_ik(Pose(position=[x, y, z], quaternion_wxyz=qq, frame=rs.base_frame), seed_joints=cur, arm=arm)
        if not ik.ok:
            continue
        jj = ik_joints(ik)
        if jj is None:
            continue
        dmax = max(abs(a - b) for a, b in zip(jj, cur))
        m = bounds_margin(arm, jj)
        if dmax <= max_jump and (m is None or m > 0.02) and (best_d is None or dmax < best_d):
            best, best_d = jj, dmax
    if best is not None:
        r = move_to_joints(best, tolerance=0.02, max_steps=steps, arm=arm)
        print(tag, "via joints: jump", round(best_d, 2), "ok:", r.ok, "status:", str(r.status), "hand:", fmt(hand_pos()), "target", fmt([x, y, z]))
        return r
    print(tag, "no current-seeded IK within", max_jump, "rad")
    if max_jump < MAX_JOINT_JUMP and z >= hand_pos()[2] - 0.01:
        return reach_pose(tag + " (wider)", x, y, z, q_ref, tol, steps, max_jump=MAX_JOINT_JUMP)
    if r is None:
        r = move_to_joints(cur, tolerance=0.02, max_steps=5, arm=arm)
    return r


def ik_seeds():
    cur = [float(v) for v in get_robot_state().joint_positions[arm]]
    j2s = 1.0 if arm == "primary" else -1.0
    return [cur, [0.0] * 7] + [[s1, j2s * 0.3, 0.0, s4, 0.0, 0.0, 0.0] for s1 in (-1.0, 0.0, 1.0) for s4 in (-0.5, -1.2)]


def best_ik(pose, seeds):
    """IK over seeds: (joints, margin) of the solution furthest inside the bounds, or (None, None)."""
    best_j, best_m = None, None
    for sd in seeds:
        ik = solve_ik(pose, seed_joints=sd, arm=arm)
        if not ik.ok:
            continue
        jj = ik_joints(ik)
        if jj is None:
            continue
        m = bounds_margin(arm, jj)
        if best_m is None or (m is not None and m > best_m):
            best_j, best_m = jj, m
    return best_j, best_m


def line_to(tag, target, q_use, tol=0.02, step=STEP_M, first_jump=SMALL_JUMP):
    """Cartesian straight line from the hand to target in steps of at most `step`; stops on termination."""
    hs = hand_pos()
    dist = math.sqrt(sum((target[i] - hs[i]) ** 2 for i in range(3)))
    n = max(1, int(math.ceil(dist / step)))
    r = None
    for k in range(1, n + 1):
        t = k / float(n)
        r = reach_pose(tag + " " + str(k) + "/" + str(n), hs[0] + t * (target[0] - hs[0]), hs[1] + t * (target[1] - hs[1]),
                       hs[2] + t * (target[2] - hs[2]), q_use, tol, 150, max_jump=(first_jump if k == 1 else SMALL_JUMP))
        if terminated_by(r):
            break
    return r


# Code block 4: hop to the counter edge with the cup out to the grasping arm's side, then pick the lean and the pinch.
# Where the cup should sit in the base frame: LAT_TARGET out to the grasping arm's side. From the offline PyRoki reach
# study on the R1 Pro URDF (cup/dev/tools/reach_study.py): at counter height (0.92-1.05 m) a straight-down tool reaches
# only ~0.3-0.4 m ahead with margins near 0, while a tool pitched 30-45 deg forward reaches 0.3-0.5 m ahead upright and
# 0.5-0.8 m at torso j0 0.3, with joint margins 0.4-0.8 rad 0.4 m out from the centreline (0.1-0.5 at the 0.25 shoulder
# line). Untuned in simulation.
LAT_TARGET = 0.38
# EDGE_STOP_M 0.33: the base stops this far short of the counter edge (the R1 Pro base is ~0.5 m long; the radio's value
# at a coffee table). HOP_MAX_M 0.4: the largest lateral shift. Untuned.
EDGE_STOP_M = 0.33
HOP_MAX_M = 0.4
# LEANS: torso j0 values tried in order, the first whose pinch, pre-pinch and crossing poses all solve with a 0.05 rad
# bounds margin wins (reach study: upright for a cup ~0.4 ahead, 0.3 for 0.5-0.8 ahead); untuned.
LEANS = (0.0, 0.15, 0.3, 0.45)
# Pinch geometry. PITCHES: tool pitch from vertical toward the cup, preferred first (30 keeps the open outer finger's tip
# ~2.2 cm above the pinch height, clear of a saucer; 45-60 reach further). DZ_PINCH 0.022: the pinch point (fingertip
# midpoint) this far under the rim. PRE_BACK_M 0.07: the pre-pinch pose this far back along the tool axis. DZ_CROSS
# CROSS_BACK_M 0.06: the crossing point is this far short of the pre-pinch pose at its height; RISE_BACK_M 0.08 /
# RISE_DZ 0.065: the hand first rises this far behind the counter edge to this height over the counter top (v01 seed 40:
# rise and crossing points at the rim + 0.10 had no IK at the leans the pinch needed, 0.3-0.45; the reach study solves
# z 0.95 at 0.2-0.3 ahead at j0 0.3). All untuned.
PITCHES = (30, 45, 60)
DZ_PINCH = 0.022
PRE_BACK_M = 0.07
CROSS_BACK_M = 0.06
RISE_BACK_M = 0.08
RISE_DZ = 0.065


def pinch_dir(gk):
    """Horizontal unit vector from the pinch point into the cup: from the robot's grasping shoulder toward the cup centre
    (plan_margins turns it, and skips directions at the handle)."""
    bxs, bys, byaws = behavior.get_base_pose()
    sx = bxs - side * 0.25 * math.sin(byaws)
    sy = bys + side * 0.25 * math.cos(byaws)
    dx, dy = gk["gx"] - sx, gk["gy"] - sy
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    return dx, dy


def edge_fwd():
    """Forward distance from the base to the counter edge (nearest support point ahead within a 0.3 m lateral band)."""
    bxe, bye, byawe = behavior.get_base_pose()
    fxe, fye = math.cos(byawe), math.sin(byawe)
    ea = 9.0
    for p in sup.points:
        rx, ry = float(p[0]) - bxe, float(p[1]) - bye
        along = rx * fxe + ry * fye
        if abs(-rx * fye + ry * fxe) < 0.3 and 0.0 < along < ea:
            ea = along
    return ea


def pinch_plan(gk, dx, dy, pitch, sgn, e_fwd):
    """Pinch point, pre-pinch point, crossing point (over the counter, DZ_CROSS above the rim) and rise point (the same
    height, RISE_BACK_M behind the counter edge) for a pinch from direction d at this pitch."""
    q, zv = pinch_frame(dx, dy, pitch, sgn)
    px = gk["gx"] - gk["rad"] * dx
    py = gk["gy"] - gk["rad"] * dy
    pz = gk["top"] - DZ_PINCH
    pre = [px - PRE_BACK_M * zv[0], py - PRE_BACK_M * zv[1], pz - PRE_BACK_M * zv[2]]
    cross = [pre[0] - CROSS_BACK_M * dx, pre[1] - CROSS_BACK_M * dy, pre[2]]
    bxp, byp, byawp = behavior.get_base_pose()
    lat = -(cross[0] - bxp) * math.sin(byawp) + (cross[1] - byp) * math.cos(byawp)
    fwd = min(e_fwd - RISE_BACK_M, (cross[0] - bxp) * math.cos(byawp) + (cross[1] - byp) * math.sin(byawp))
    rise = [bxp + fwd * math.cos(byawp) - lat * math.sin(byawp), byp + fwd * math.sin(byawp) + lat * math.cos(byawp),
            gk["base"] + RISE_DZ]
    return q, zv, [px, py, pz], pre, cross, rise


# DIR_TURNS: pinch directions tried, in degrees from the shoulder-to-cup line (v01 seed 49: the only direction tried was
# turned 70 deg by a false handle read and nothing solved); a direction within HANDLE_AVOID_DEG of the handle is skipped.
DIR_TURNS = (0.0, 35.0, -35.0)
HANDLE_AVOID_DEG = 50.0


def plan_margins(gk, tag, e_fwd):
    """Best (worst margin over pinch/pre/cross/rise, pitch, sign) and its direction over the pinch family."""
    d0x, d0y = pinch_dir(gk)
    seeds = ik_seeds()[:4]
    best = None
    for turn in DIR_TURNS:
        a = math.radians(turn)
        dx, dy = d0x * math.cos(a) - d0y * math.sin(a), d0x * math.sin(a) + d0y * math.cos(a)
        if (gk["hx"] != 0.0 or gk["hy"] != 0.0) and -(dx * gk["hx"] + dy * gk["hy"]) > math.cos(math.radians(HANDLE_AVOID_DEG)):
            print(tag, "direction", turn, "skipped: the handle is there")
            continue
        for pitch in PITCHES:
            for sgn in (1.0, -1.0):
                q, zv, pp, pre, cross, rise = pinch_plan(gk, dx, dy, pitch, sgn, e_fwd)
                ms = []
                for pt in (pp, pre, cross, rise):
                    jj, m = best_ik(Pose(position=pt, quaternion_wxyz=q, frame=rs.base_frame), seeds)
                    ms.append(m)
                    if m is None:
                        break
                mm = min((m if m is not None else -1.0) for m in ms)
                print(tag, "dir", turn, "pitch", pitch, "sign", sgn, "margins pinch/pre/cross/rise:", ms)
                if best is None or mm > best[0][0] + 0.05:      # earlier (straighter, smaller pitch) wins ties
                    best = ((mm, pitch, sgn), (dx, dy))
            if best is not None and best[0][0] >= 0.15:
                break
        if best is not None and best[0][0] >= 0.15:
            break
    if best is None:
        return None, (d0x, d0y)
    return best


lean = 0.0
plan = None
if report.get("nav_ok") and g is not None:
    bx1, by1, byaw1 = behavior.get_base_pose()
    fx, ly = rel_of(g["gx"], g["gy"])
    side = 1.0 if ly >= 0.0 else -1.0
    arm = "primary" if side > 0 else "secondary"
    report["arm"] = arm
    # forward: stop EDGE_STOP_M short of the counter edge, measured in a 0.3 m band around the shifted base line
    lat_shift = ly - side * LAT_TARGET
    if abs(lat_shift) > HOP_MAX_M:
        print("lateral shift", round(lat_shift, 3), "capped")
        lat_shift = HOP_MAX_M if lat_shift > 0 else -HOP_MAX_M
    sx = bx1 - lat_shift * math.sin(byaw1)
    sy = by1 + lat_shift * math.cos(byaw1)
    fxs, fys = math.cos(byaw1), math.sin(byaw1)
    ea = 9.0
    for p in sup.points:
        rx, ry = float(p[0]) - sx, float(p[1]) - sy
        along = rx * fxs + ry * fys
        if abs(-rx * fys + ry * fxs) < 0.3 and 0.0 < along < ea:
            ea = along
    hop = max(-0.3, ea - EDGE_STOP_M) if ea < 2.0 else 0.0
    tx = bx1 + hop * fxs - lat_shift * math.sin(byaw1)
    ty = by1 + hop * fys + lat_shift * math.cos(byaw1)
    print("cup rel", round(fx, 3), round(ly, 3), "arm", arm, "edge ahead at the shifted pose", round(ea, 3), "hop", round(hop, 3),
          "lateral", round(lat_shift, 3), "floor at target:", floor_near(tx, ty))
    # the floor at the counter's foot is often hidden from the far views: a short sideways hop onto a free map cell is
    # allowed without seen floor (dev v01 seed 49 dropped the shift and the cup stayed on the centreline, out of reach)
    if not floor_near(tx, ty) and not (behavior.base_pose_is_free(tx, ty) and abs(lat_shift) <= HOP_MAX_M and abs(hop) <= 0.3):
        lat_shift = 0.0
        tx, ty = bx1 + hop * fxs, by1 + hop * fys
        print("lateral shift dropped (no seen floor there)")
    if abs(hop) > 0.02 or abs(lat_shift) > 0.02:
        r = behavior.navigate_to_pose(tx, ty, byaw1, planner="servo", tolerance_m=0.02, max_steps=400)
        print("hop servo ->", fmt([tx, ty, byaw1]), "ok:", r.ok, "steps:", r.steps_executed, "base now:", fmt(behavior.get_base_pose()))
    report["hop"] = [round(hop, 3), round(lat_shift, 3), round(ea, 3)]
    g2 = localize_cup("after hop")
    if g2 is not None:
        g = g2
    report["cup_rel"] = fmt(rel_of(g["gx"], g["gy"]))
    best_plan = None
    tgt = list(torso0)
    for lk in LEANS:
        if lk > 0.0:
            tgt[0] = torso0[0] + lk
            behavior.move_torso(tgt, max_steps=120)
            behavior.move_torso(tgt, max_steps=60)
            g2 = localize_cup("lean " + str(lk))
            if g2 is not None:
                g = g2
        lean = lk
        bp, dd = plan_margins(g, "lean " + str(lk), edge_fwd())
        print("lean", lk, "best plan", bp)
        if bp is not None and (best_plan is None or bp[0] > best_plan[0][0]):
            best_plan = (bp, lk, dd)
        if bp is not None and bp[0] >= 0.05:
            break
    if best_plan is not None and best_plan[1] != lean:
        tgt[0] = torso0[0] + best_plan[1]
        behavior.move_torso(tgt, max_steps=120)
        behavior.move_torso(tgt, max_steps=60)
        lean = best_plan[1]
        g2 = localize_cup("back to lean " + str(lean))
        if g2 is not None:
            g = g2
    report["lean"] = lean
    if best_plan is not None and best_plan[0][0] > 0.0:
        plan = best_plan
        report["plan"] = [best_plan[0][0], best_plan[0][1], best_plan[0][2], best_plan[1]]
    report["cup_plan"] = [round(g["gx"], 3), round(g["gy"], 3), round(g["top"], 3), round(g["rad"], 3), round(g["hx"], 2), round(g["hy"], 2)]
    report["stage"] = "planned"


# Code block 5: rise behind the edge, cross, read from the wrist, then the pinch ladder, lift and hold.
# JAW_MIN 0.015 / JAW_MAX 0.97: closing on air reads 0.000-0.003 and fully open 1.0 (radio dev); a pinch on a cup wall
# a few mm thick should read a little above the air band. LIFT_TOTAL_M 0.15: the demonstrated lift, in 3 cm IK steps.
# HOLD_TICKS 330: hold still after the lift so the collector's 300-tick post-success grace ends with the cup in hand.
JAW_MIN = 0.015
JAW_MAX = 0.97
LIFT_TOTAL_M = 0.15
HOLD_TICKS = 330
# LIFT_BACK_M 0.01: per 3 cm lift step the hand also retreats this far toward the robot; LIFT_KEEP_M 0.015: a pinch
# lifted at least this far is kept even if the lift stops early (v03 seed 44 opened a pinch lifted 2.5 cm to try the next
# rung: the witness had latched already, and the cup was put down); the lift may also tilt the tool to pitch 15 or 0
# (lift_quats) when the pinch orientation has no near IK. Untuned.
LIFT_BACK_M = 0.01
LIFT_KEEP_M = 0.015
lift_quats = []
# WRIST_MAX_DEV 0.08: a wrist read further than this from the head read is another object; untuned.
WRIST_MAX_DEV = 0.08


def jaw_now():
    return float(get_robot_state().gripper_positions[arm])


def hold_still(ticks):
    cur = [float(v) for v in get_robot_state().joint_positions[arm]]
    done_t = 0
    for i in range(400):
        r = move_to_joints(cur, tolerance=0.0005, max_steps=20, arm=arm)
        done_t += max(1, int(r.steps_executed))
        if terminated_by(r) or done_t >= ticks:
            return r
    return None


def wrist_cup(tag):
    return localize_cup(tag, "left_wrist" if arm == "primary" else "right_wrist")


def probe(tag, pp, pre, q_use, zv):
    """One rung: pre-pinch pose, slide along the tool axis to the pinch point, close; on a mid-range jaw lift in 3 cm IK
    steps to LIFT_TOTAL_M and hold."""
    out = {"tag": tag}
    r = reach_pose("rung pre", pre[0], pre[1], pre[2], q_use, 0.015, 200)
    if terminated_by(r):
        out["terminated"] = "pre"
        return out
    r = line_to("rung slide", pp, q_use, tol=0.012, step=0.02)
    if terminated_by(r):
        out["terminated"] = "slide"
        print("rung", out)
        return out
    hp = hand_pos()
    out["short_of_pinch"] = round(sum((pp[i] - hp[i]) * zv[i] for i in range(3)), 3)    # > 0.015: a finger met the rim
    out["hand"] = fmt(hp)
    close_gripper(arm=arm)
    out["jaw"] = round(jaw_now(), 3)
    out["terminated"] = ""
    if JAW_MIN < out["jaw"] < JAW_MAX:
        hp2 = hand_pos()
        k = 1
        r = None
        lift_d = [zv[0], zv[1]]      # the tool axis' horizontal part points into the cup
        nd = math.hypot(lift_d[0], lift_d[1]) or 1.0
        lift_d = [lift_d[0] / nd, lift_d[1] / nd]
        saved = list(alt_quats)
        alt_quats.extend(lift_quats)
        while 0.03 * k <= LIFT_TOTAL_M + 1e-6:
            # up 3 cm a step and back toward the robot LIFT_BACK_M a step (v03 seed 44: straight up from a pinch at the
            # reach limit, the 6 cm pose needed a > 1 rad joint jump and the lift stopped at 2.5 cm)
            r2 = reach_pose("rung lift " + str(k), hp2[0] - LIFT_BACK_M * k * lift_d[0], hp2[1] - LIFT_BACK_M * k * lift_d[1],
                            hp2[2] + 0.03 * k, q_use, 0.02, 120)
            r = r2
            if terminated_by(r2) or not (JAW_MIN < jaw_now() < JAW_MAX):
                break
            if not r2.ok and hand_pos()[2] < hp2[2] + 0.03 * k - 0.02:
                break                     # the reach limit: keep the height reached
            k += 1
        alt_quats[:] = saved
        out["lift_m"] = round(hand_pos()[2] - hp2[2], 3)
        out["jaw_after_lift"] = round(jaw_now(), 3)
        if terminated_by(r):
            out["terminated"] = "lift"
        elif out["lift_m"] >= LIFT_KEEP_M and JAW_MIN < out["jaw_after_lift"] < JAW_MAX:
            out["holding"] = True
            rh = hold_still(HOLD_TICKS)
            out["jaw_after_hold"] = round(jaw_now(), 3)
            if terminated_by(rh):
                out["terminated"] = "hold"
    if not out["terminated"] and not out.get("holding"):
        open_gripper(arm=arm)
        line_to("rung back", pre, q_use, step=0.03)
    print("rung", out)
    return out


if plan is not None:
    (mm, pitch, sgn), lean_used, (dx, dy) = plan
    for pt in PITCHES:
        for sg in (1.0, -1.0):
            if pt != pitch or sg != sgn:
                alt_quats.append(pinch_frame(dx, dy, pt, sg)[0])
    for pt in (15, 0):
        lift_quats.append(pinch_frame(dx, dy, pt, sgn)[0])
    e_fwd0 = edge_fwd()
    q, zv, pp, pre, cross, rise = pinch_plan(g, dx, dy, pitch, sgn, e_fwd0)
    print("plan: pitch", pitch, "sign", sgn, "d", fmt([dx, dy]), "pinch", fmt(pp), "pre", fmt(pre), "cross", fmt(cross), "rise", fmt(rise),
          "edge ahead", round(e_fwd0, 3))
    # 1. rise behind the counter edge to the crossing height in joint space (the hanging hand is below the counter top
    #    and points down-back; no Cartesian line from it keeps the pinch orientation), re-issued until it converges
    rj, rm = best_ik(Pose(position=rise, quaternion_wxyz=q, frame=rs.base_frame), ik_seeds())
    r = None
    if rj is not None:
        for kk in range(3):
            r = move_to_joints(rj, tolerance=0.02, max_steps=150, arm=arm)
            jerr = max(abs(a - b) for a, b in zip([float(v) for v in get_robot_state().joint_positions[arm]], rj))
            print("rise call", kk, "ok:", r.ok, "steps:", r.steps_executed, "joint err", round(jerr, 3), "hand", fmt(hand_pos()))
            if r.ok or terminated_by(r) or jerr < 0.02:
                break
    hp = hand_pos()
    print("rise reached", fmt(hp), "target", fmt(rise), "margin", rm)
    report["rise_off"] = round(math.sqrt(sum((hp[i] - rise[i]) ** 2 for i in range(3))), 3)
    if not terminated_by(r) and report["rise_off"] > 0.03:
        r = reach_pose("rise fix", rise[0], rise[1], rise[2], q, 0.02, 200, max_jump=MAX_JOINT_JUMP)
    # 2. across at that height to the crossing point, then down the tool axis line to the pre-pinch pose
    if not terminated_by(r):
        r = line_to("cross", cross, q)
    if not terminated_by(r):
        r = line_to("to pre", pre, q, step=0.03)
    hp = hand_pos()
    report["pre_off"] = round(math.sqrt(sum((hp[i] - pre[i]) ** 2 for i in range(3))), 3)
    print("pre reached", fmt(hp), "off", report["pre_off"])
    report["stage"] = "pre"
    # 3. wrist reads from the pre-pinch pose (two, averaged), then re-plan the pinch from them
    reads = []
    for it in range(2):
        w = wrist_cup("w" + str(it))
        if w is not None and w["n"] >= 15 and math.hypot(w["gx"] - g["gx"], w["gy"] - g["gy"]) <= WRIST_MAX_DEV:
            reads.append(w)
        elif w is not None:
            print("wrist read", it, "rejected: off the head read by", round(math.hypot(w["gx"] - g["gx"], w["gy"] - g["gy"]), 3))
    gw = dict(g)
    if reads:
        nr = float(len(reads))
        for key in ("gx", "gy", "top", "rad"):
            gw[key] = sum(v[key] for v in reads) / nr
        if reads[-1]["hx"] != 0.0 or reads[-1]["hy"] != 0.0:
            gw["hx"], gw["hy"] = reads[-1]["hx"], reads[-1]["hy"]
    report["wrist"] = [round(gw["gx"], 3), round(gw["gy"], 3), round(gw["top"], 3), round(gw["rad"], 3), len(reads)]
    print("ladder cup: centre", round(gw["gx"], 3), round(gw["gy"], 3), "top", round(gw["top"], 3), "rad", round(gw["rad"], 3),
          "shift from head", round(math.hypot(gw["gx"] - g["gx"], gw["gy"] - g["gy"]), 3))
    # 4. rungs: radial offsets (along d: + moves the pinch point into the cup) and sideways offsets
    rungs = []
    done = ""
    ex, ey = -dy, dx
    for dr, dsd in ((0.0, 0.0), (-0.012, 0.0), (0.012, 0.0), (0.0, 0.015), (0.0, -0.015), (-0.024, 0.0)):
        gk = dict(gw)
        gk["gx"] = gw["gx"] + dr * dx + dsd * ex
        gk["gy"] = gw["gy"] + dr * dy + dsd * ey
        qk, zk, ppk, prek, crossk, risek = pinch_plan(gk, dx, dy, pitch, sgn, e_fwd0)
        out = probe("dr=" + str(dr) + " ds=" + str(dsd), ppk, prek, qk, zk)
        rungs.append(out)
        if out.get("terminated") or out.get("holding"):
            done = out["tag"]
            break
        if out.get("short_of_pinch", 0.0) > 0.015 or JAW_MIN < out.get("jaw", 0.0) < JAW_MAX:
            w = wrist_cup("after " + out["tag"])
            if w is not None and w["n"] >= 15 and math.hypot(w["gx"] - gw["gx"], w["gy"] - gw["gy"]) <= WRIST_MAX_DEV:
                gw["gx"], gw["gy"] = w["gx"], w["gy"]
    report["rungs"] = rungs
    report["ladder_done_at"] = done
    report["holding_at_end"] = JAW_MIN < jaw_now() < JAW_MAX
    report["stage"] = "ladder"

result = report
