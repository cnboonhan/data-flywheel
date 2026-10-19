# Pick up the red radio on BEHAVIOR-1K `turning_on_radio` with the Galaxea R1 Pro (run through cap-harness-behavior.sh).
# Learned, not hand-written: per-seed ASPIRE campaign policies (seeds 26, 27) generalized by a coding agent over task
# instances 36-45 in 3 versions; held the radio on 8 of 10 of those instances. Collected with adapter/collect_behavior.sh.
# policy.py - one R1 Pro policy for BEHAVIOR-1K turning_on_radio on any instance (generalization dev).
# Grown from seed 27's campaign policy (itself grown from seed 26's); every pose comes from perception at run time.
import math

report = {"stage": "start"}


# Code block 1: sweep the head camera in place until the radio segments confidently; collect the floor seen on the way.
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
target_prompt = str(meta.get("target_prompt") or "red radio")
support_prompt = str(meta.get("support_prompt") or "table")
print("language:", ctx.language, "| target_prompt:", target_prompt, "| support_prompt:", support_prompt)
print("torso_positions:", fmt(cmeta.get("torso_positions", [])), "lower:", fmt(cmeta.get("torso_lower_bounds", [])),
      "upper:", fmt(cmeta.get("torso_upper_bounds", [])))
bx, by, byaw = behavior.get_base_pose()
rs = get_robot_state()
print("base_pose:", fmt([bx, by, byaw]), "base_frame:", rs.base_frame)
for arm in ("primary", "secondary"):
    print(arm, "gripper:", round(float(rs.gripper_positions[arm]), 3), "eef:", pose_rows(rs.end_effector_poses[arm]))
report["torso"] = fmt(cmeta.get("torso_positions", []))
report["base0"] = fmt([bx, by, byaw])
torso0 = [float(v) for v in cmeta.get("torso_positions", [0.0, 0.0, 0.0, 0.0])]
# lateral offset of each hanging hand from the base centre at reset (odom = base at reset): where each arm's shoulder is
shoulder_off = {}
for arm in ("primary", "secondary"):
    hx, hy = float(rs.end_effector_poses[arm].position[0]), float(rs.end_effector_poses[arm].position[1])
    shoulder_off[arm] = -(hx - bx) * math.sin(byaw) + (hy - by) * math.cos(byaw)
print("shoulder lateral offsets:", {k: round(v, 3) for k, v in shoulder_off.items()})
report["shoulder_off"] = {k: round(v, 3) for k, v in shoulder_off.items()}
# per-arm joint bounds from the controller metadata: arm joints are interleaved left/right after the 6 base and 4 torso entries
act_lo = [float(v) for v in cmeta.get("action_lower_bounds", [])]
act_hi = [float(v) for v in cmeta.get("action_upper_bounds", [])]


def arm_bounds(arm):
    off = 10 + (0 if arm == "primary" else 1)
    if len(act_lo) < 24:
        return None
    return [(act_lo[off + 2 * k], act_hi[off + 2 * k]) for k in range(7)]


def ik_joints(ik):
    """The joint solution of an IKResult, whichever attribute carries it; None when absent."""
    v = None
    try:
        v = ik.joint_positions
    except AttributeError:
        try:
            v = ik.solution
        except AttributeError:
            try:
                v = ik.joints
            except AttributeError:
                v = None
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except TypeError:
        return None


def bounds_margin(arm, joints):
    """Smallest distance (rad) of a joint vector to its action bounds; negative means outside. None when unknown."""
    b = arm_bounds(arm)
    if b is None or joints is None or len(joints) != 7:
        return None
    return round(min(min(j - lo, hi - j) for j, (lo, hi) in zip(joints, b)), 3)


print("arm bounds primary:", arm_bounds("primary"), "secondary:", arm_bounds("secondary"))

# Observed floor (odom xy, subsampled) accumulated over the sweep: attempt 1 drove through an opening the traversability
# map calls free and the base fell off the scene, so every navigation goal is checked against floor the camera has seen.
floor_xy = []


def collect_floor(tag):
    fl = localize_object("floor", camera_name="head")
    if not fl.ok:
        print(tag, "floor localize failed:", fl.error.message if fl.error else None)
        return 0
    pts = fl.point_cloud.points
    k = 0
    for i in range(0, len(pts), 8):      # every 8th point: a full-image floor mask is ~100k points, 0.25 m checks need far fewer
        p = pts[i]
        floor_xy.append((float(p[0]), float(p[1])))
        k += 1
    zs = sorted(float(pts[i][2]) for i in range(0, len(pts), 8))
    print(tag, "floor points kept:", k, "of", len(pts), "z[5,50,95]:", round(zs[len(zs) // 20], 3), round(zs[len(zs) // 2], 3), round(zs[(19 * len(zs)) // 20], 3))
    return k


def floor_near(x, y, r=0.25):
    """True when an observed floor point lies within r of (x, y); r 0.25 is half the base footprint, untuned."""
    r2 = r * r
    for fx, fy in floor_xy:
        if (fx - x) ** 2 + (fy - y) ** 2 <= r2:
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


# Sweep: segment at the start heading, then turn in place in 45 deg steps (8 headings cover the room), first toward the
# side where the support prompt scored best. SEARCH_MIN_SCORE 0.25: seed 26's real radio scored 0.70-0.75 and the
# garbage hits of attempt 1 scored 0.007; "radio" alone scored 0.40 on the real one, so 0.25 separates them with margin.
SEARCH_MIN_SCORE = 0.25
# REACH_MAX_Z 1.0: a detection whose median height is above this is not on a surface the leaned arm can grasp top-down
# (the hanging hand is at 0.66 m and a top-down reach at 0.9 m already needs the base within 0.4 m); generalization dev
# v01: pilot seed 36 scored 0.44 on a wall shelf at 1.19 m and the policy spent the episode under it. Untuned beyond that.
REACH_MAX_Z = 1.0


def plausible_target(tag):
    """Localize the target from the head and keep it only when its median-cropped height is reachable; None otherwise."""
    lc = localize_object(target_prompt, camera_name="head")
    print(tag, "localize [" + target_prompt + "] ok:", lc.ok, "err:", lc.error.message if lc.error else None)
    if not lc.ok:
        return None
    zs0 = sorted(float(p[2]) for p in lc.point_cloud.points)
    if not zs0:
        return None
    zmed = zs0[len(zs0) // 2]
    if zmed > REACH_MAX_Z:
        print(tag, "rejected: median height", round(zmed, 3), "above", REACH_MAX_Z)
        return None
    return lc


found_cloud = None
loc = None
sweep = []
s_r, b_r, n_r = seg_best("head", target_prompt)
s_t, b_t, n_t = seg_best("head", support_prompt)
turn_dir = 1.0
if b_t is not None:
    turn_dir = 1.0 if 0.5 * (b_t[0] + b_t[2]) < 256.0 else -1.0     # box left of centre -> the table is to the left -> turn left (+yaw)
print("start heading: radio", round(s_r, 3), b_r, "table", round(s_t, 3), b_t, "turn_dir", turn_dir)
collect_floor("floor@start")
sweep.append((0, round(s_r, 3), round(s_t, 3)))
if s_r >= SEARCH_MIN_SCORE:
    loc = plausible_target("heading 0")
yaw_now = byaw
for k in range(1, 8):
    if loc is not None:
        break
    yaw_now = yaw_now + turn_dir * math.radians(45.0)
    yaw_now = math.atan2(math.sin(yaw_now), math.cos(yaw_now))
    r = behavior.navigate_to_pose(bx, by, yaw_now, planner="servo", max_steps=200)
    bxn, byn, byawn = behavior.get_base_pose()
    s_r, b_r, n_r = seg_best("head", target_prompt)
    s_t, b_t, n_t = seg_best("head", support_prompt)
    print("turn", k, "to yaw", round(math.degrees(yaw_now), 1), "ok:", r.ok, "steps:", r.steps_executed, "yaw now:", round(math.degrees(byawn), 1),
          "| radio", round(s_r, 3), b_r, "table", round(s_t, 3), b_t)
    collect_floor("floor@" + str(k))
    sweep.append((k, round(s_r, 3), round(s_t, 3)))
    if s_r >= SEARCH_MIN_SCORE:
        loc = plausible_target("heading " + str(k))
report["sweep"] = sweep
report["search_score"] = round(s_r, 3)
report["floor_pts"] = len(floor_xy)

if loc is not None:
    if loc.ok:
        g = loc.geometry
        pts = loc.point_cloud.points
        zs = sorted(float(p[2]) for p in pts)
        n = len(zs)
        print("  geometry center:", fmt(g.pose.position), "extents:", fmt(g.extents), "points:", g.point_count,
              "cloud z quantiles 5/50/95:", round(zs[n // 20], 3), round(zs[n // 2], 3), round(zs[(19 * n) // 20], 3))
        found_cloud = loc
        report["found"] = True
        report["radio_center"] = fmt(g.pose.position)
        report["radio_extents"] = fmt(g.extents)
report.setdefault("found", False)

sup = localize_object(support_prompt, camera_name="head")
print("localize support ok:", sup.ok, "err:", sup.error.message if sup.error else None, "score:", round(s_t, 3))
if sup.ok:
    print("  support center:", fmt(sup.geometry.pose.position), "extents:", fmt(sup.geometry.extents), "points:", sup.geometry.point_count)
    report["support_center"] = fmt(sup.geometry.pose.position)
    report["support_extents"] = fmt(sup.geometry.extents)
    if found_cloud is not None:
        so = behavior.plan_standoff_pose(sup.point_cloud, found_cloud.point_cloud, 0.3)
        print("  standoff(0.3):", fmt(so), "free:", behavior.base_pose_is_free(so[0], so[1]), "floor:", floor_near(so[0], so[1]))
        report["standoff"] = fmt(so)
if found_cloud is not None:
    ap = behavior.plan_approach_pose(found_cloud.point_cloud, 0.7)
    print("  approach(0.7):", fmt(ap), "free:", behavior.base_pose_is_free(ap[0], ap[1]), "floor:", floor_near(ap[0], ap[1]))
    report["approach"] = fmt(ap)

report["stage"] = "search"

# Code block 2: zero-cost geometry and reach diagnostics, then torso probes to learn which joint lowers the hands.
def median_of(vals):
    v = sorted(vals)
    return v[len(v) // 2]


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


def describe(tag, cloud):
    g = estimate_geometry(cloud)
    pts = cloud.points
    xs = sorted(float(p[0]) for p in pts); ys = sorted(float(p[1]) for p in pts); zs = sorted(float(p[2]) for p in pts)
    n = len(zs)
    print(tag, "center:", fmt(g.pose.position), "extents:", fmt(g.extents), "n:", n,
          "x[5,95]:", round(xs[n // 20], 3), round(xs[(19 * n) // 20], 3),
          "y[5,95]:", round(ys[n // 20], 3), round(ys[(19 * n) // 20], 3),
          "z[5,95]:", round(zs[n // 20], 3), round(zs[(19 * n) // 20], 3))
    return g


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


def topdown_pose(x, y, z, ref):
    return Pose(position=[x, y, z], quaternion_wxyz=[float(q) for q in ref.quaternion_wxyz], frame=ref.frame)


def reach_map(tag):
    """Which top-down hand positions (base-relative, forward x / lateral y / height z) IK accepts, both arms."""
    st = get_robot_state()
    bx0, by0, byaw0 = behavior.get_base_pose()
    for arm, side in (("primary", 0.25), ("secondary", -0.25)):
        ref = st.end_effector_poses[arm]
        rows = []
        for fx in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            oks = []
            for z in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                wx = bx0 + fx * math.cos(byaw0) - side * math.sin(byaw0)
                wy = by0 + fx * math.sin(byaw0) + side * math.cos(byaw0)
                ik = solve_ik(topdown_pose(wx, wy, z, ref), arm=arm)
                oks.append("o" if ik.ok else ".")
            rows.append("fx=" + str(fx) + " " + "".join(oks))
        print(tag, arm, "eef_z:", round(float(ref.position[2]), 3), "z=0.4..1.0 ->", " | ".join(rows))


if report.get("found"):
    radio_c = robust_crop(found_cloud.point_cloud, 0.2)
    rg = describe("radio cropped(0.2)", radio_c)
    report["radio_center"] = fmt(rg.pose.position)
    report["radio_extents"] = fmt(rg.extents)
    rz_top = max(float(p[2]) for p in radio_c.points)
    report["radio_top_z"] = round(rz_top, 3)
    if sup.ok:
        sg = describe("support", sup.point_cloud)
        so2 = behavior.plan_standoff_pose(sup.point_cloud, radio_c, 0.3)
        print("standoff(0.3) from cropped radio:", fmt(so2), "free:", behavior.base_pose_is_free(so2[0], so2[1]))
    cands = free_poses_around(float(rg.pose.position[0]), float(rg.pose.position[1]), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2))
    print("free ring poses (d, deg, x, y, yaw):", len(cands))
    for c in cands[:60]:
        print("  cand", c)
    report["n_free_cands"] = len(cands)

report["cands"] = cands[:12] if report.get("found") else []
torso0 = [float(v) for v in cmeta.get("torso_positions", [0.0, 0.0, 0.0, 0.0])]
report["stage"] = "probed"

def cloud_axes(cloud):
    """Principal horizontal axis of a cloud: (angle_rad, long_len, short_len, cx, cy) from the xy covariance and 5-95 % spans."""
    pts = cloud.points
    n = len(pts)
    xs = [float(p[0]) for p in pts]; ys = [float(p[1]) for p in pts]
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) / n; syy = sum((y - my) ** 2 for y in ys) / n
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


# TWIST_MAX_DEG 60: the largest angle between the radio's long axis and the base heading we drive to on purpose. The
# fingers close along hand +y, which must lie across the radio, and the arms reach poorly with +y pointing forward:
# generalization dev v01 seed 42 (long axis 88 deg off the heading) found no IK at any lean; seed 40 (68 deg) and 39 (50
# deg) solved. 60 keeps a margin under 68; a preference, not a filter. Untuned.
TWIST_MAX_DEG = 60.0
# EDGE_SLACK_M 0.25: how much deeper inside the table edge a lower-twist candidate may put the radio than the edge-best one
# (the leaned arm reached 0.88 m ahead with the base 0.33 m off the edge in v01 seed 39); untuned.
EDGE_SLACK_M = 0.25


def top_axis(cloud):
    """Long-axis angle (rad) of the top 3 cm of a cropped radio cloud, or None when the top face is not elongated."""
    zs3 = sorted(float(p[2]) for p in cloud.points)
    if len(zs3) < 30:
        return None
    t3 = zs3[(19 * len(zs3)) // 20]
    try:
        tc = crop_point_cloud(cloud, [-50.0, -50.0, t3 - 0.03], [50.0, 50.0, t3 + 1.0])
    except ValueError:
        return None
    if len(tc.points) < 20:
        return None
    a3, l3, s3, x3, y3 = cloud_axes(tc)
    if l3 < 1.4 * max(s3, 0.01):      # 1.4: the handle reads 0.20 x 0.03, a lying face ~0.24 x 0.15; a near-square read has no axis
        return None
    return a3


def twist_deg(long_ang, heading):
    """Angle in [0, 90] between the radio's long axis and a base heading (0: looking along the radio)."""
    d = abs(math.degrees(long_ang - heading)) % 180.0
    return min(d, 180.0 - d)


def pick_candidate(cands, ox, oy, long_ang, tag):
    """(edge_m, cand) of the ring candidate whose bearing puts the radio nearest the table edge, preferring one whose
    heading keeps the twist under TWIST_MAX_DEG when the axis is known."""
    scored = []
    for c in cands:
        bang = math.atan2(c[3] - oy, c[2] - ox)      # bearing from the radio toward the candidate
        ed = edge_distance(sup.point_cloud, ox, oy, bang)
        tw = twist_deg(long_ang, c[4]) if long_ang is not None else 0.0
        scored.append((round(ed, 3), round(tw, 1), c[0], c))
    scored.sort()
    for row in scored[:6]:
        print(tag, "scored cand (edge_m, twist, d, cand):", row)
    best = scored[0]
    if long_ang is not None and best[1] > TWIST_MAX_DEG:
        low = [row for row in scored if row[1] <= TWIST_MAX_DEG and row[0] <= best[0] + EDGE_SLACK_M]
        if low:
            print(tag, "edge-best twist", best[1], "-> low-twist candidate", low[0])
            best = low[0]
    return best[0], best[1], best[3]


# Code block 3: drive to the free base pose from which the radio is nearest the table edge, re-localize, map reach.
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


def localize_radio(tag):
    """Segment + localize the radio from the head camera, median-cropped; returns (geometry, cloud) or (None, None)."""
    loc = localize_object(str(meta.get("target_prompt") or "red radio"), camera_name="head")
    if not loc.ok:
        print(tag, "localize failed:", loc.error.message if loc.error else None)
        return None, None
    cloud = robust_crop(loc.point_cloud, 0.2)
    g = describe(tag + " radio cropped", cloud)
    return g, cloud


def nav(tag, x, y, yaw, planner, max_steps=1500):
    r = behavior.navigate_to_pose(x, y, yaw, planner=planner, max_steps=max_steps)
    d = dict(r.diagnostics)
    print(tag, planner, "->", fmt([x, y, yaw]), "ok:", r.ok, "status:", str(r.status), "steps:", r.steps_executed,
          "err:", r.error.message if r.error else None,
          "diag:", {k: d[k] for k in d if k in ("route", "hops", "servo_hops", "direct_status", "moved_x", "moved_y", "moved_yaw")},
          "base now:", fmt(behavior.get_base_pose()))
    return r


def leg_clear_of(cloud, x0, y0, x1, y1, margin=0.45, step=0.1):
    """Nearest support point to the straight leg (x0,y0)->(x1,y1) sampled every step; True when further than margin
    (0.45 m: half the base diagonal plus a little, untuned)."""
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    best = 9.0
    pts = cloud.points
    for i in range(n + 1):
        t = i / float(n)
        sx, sy = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        for j in range(0, len(pts), 5):
            d = math.hypot(float(pts[j][0]) - sx, float(pts[j][1]) - sy)
            if d < best:
                best = d
    return best > margin, round(best, 3)


if report.get("found") and sup.ok and cands:
    ox, oy = float(rg.pose.position[0]), float(rg.pose.position[1])
    # attempt 1: the map's free cells included an opening with no floor; keep only candidates on floor the camera saw
    on_floor = [c for c in cands if floor_near(c[2], c[3])]
    print("ring candidates on seen floor:", len(on_floor), "of", len(cands))
    report["n_cands_on_floor"] = len(on_floor)
    if on_floor:
        cands = on_floor
    axis0 = top_axis(radio_c)
    print("far-read long axis:", round(math.degrees(axis0), 1) if axis0 is not None else None)
    e0, tw0, choice = pick_candidate(cands, ox, oy, axis0, "nav1")
    report["nav_choice"] = choice
    report["edge_m"] = e0
    report["twist_planned"] = tw0
    cx, cy, cyaw = choice[2], choice[3], choice[4]
    bx0, by0, byaw0 = behavior.get_base_pose()
    leg_floor = floor_along(bx0, by0, cx, cy)
    leg_clear, leg_min = leg_clear_of(sup.point_cloud, bx0, by0, cx, cy)
    print("straight leg to the choice: on seen floor", round(leg_floor, 2), "clear of the support:", leg_clear, "min", leg_min)
    report["leg_floor"] = round(leg_floor, 2)
    report["leg_clear_m"] = leg_min
    # a straight leg that is entirely on seen floor and clear of the table is driven directly (servo); otherwise the map route
    planner0 = "servo" if (leg_floor >= 0.999 and leg_clear) else "curobo"
    report["nav_planner"] = planner0
    r1 = nav("nav1", cx, cy, cyaw, planner0)
    bx1, by1, byaw1 = behavior.get_base_pose()
    if math.hypot(bx1 - cx, by1 - cy) > 0.1 or abs(math.atan2(math.sin(byaw1 - cyaw), math.cos(byaw1 - cyaw))) > 0.15:
        r2 = nav("nav1 servo-fix", cx, cy, cyaw, "servo", max_steps=600)
    bx1, by1, byaw1 = behavior.get_base_pose()
    report["base_after_nav"] = fmt([bx1, by1, byaw1])
    report["nav_err_m"] = round(math.hypot(bx1 - cx, by1 - cy), 3)
    rg2, radio_c2 = localize_radio("after nav")
    if rg2 is not None:
        report["radio_center2"] = fmt(rg2.pose.position)
        report["radio_top_z2"] = round(max(float(p[2]) for p in radio_c2.points), 3)
        rel_x = (float(rg2.pose.position[0]) - bx1) * math.cos(byaw1) + (float(rg2.pose.position[1]) - by1) * math.sin(byaw1)
        rel_y = -(float(rg2.pose.position[0]) - bx1) * math.sin(byaw1) + (float(rg2.pose.position[1]) - by1) * math.cos(byaw1)
        print("after nav radio relative to base: forward", round(rel_x, 3), "left", round(rel_y, 3))
        report["radio_rel"] = [round(rel_x, 3), round(rel_y, 3)]
        report["nav_ok"] = True
        # the near read's axis decides whether the arm can turn its fingers across the radio from here; if not, drive to
        # a lower-twist ring pose around the re-read centre
        axis1 = top_axis(radio_c2)
        tw1 = twist_deg(axis1, byaw1) if axis1 is not None else None
        print("near-read long axis:", round(math.degrees(axis1), 1) if axis1 is not None else None, "twist", tw1)
        report["twist_after_nav"] = round(tw1, 1) if tw1 is not None else None
        if tw1 is not None and tw1 > TWIST_MAX_DEG:
            ox2, oy2 = float(rg2.pose.position[0]), float(rg2.pose.position[1])
            c2 = [c for c in free_poses_around(ox2, oy2, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)) if floor_near(c[2], c[3])]
            if c2:
                e2, tw2, ch2 = pick_candidate(c2, ox2, oy2, axis1, "nav2")
                if tw2 <= TWIST_MAX_DEG:
                    r3 = nav("nav2", ch2[2], ch2[3], ch2[4], "curobo")
                    bx1, by1, byaw1 = behavior.get_base_pose()
                    if math.hypot(bx1 - ch2[2], by1 - ch2[3]) > 0.1 or abs(math.atan2(math.sin(byaw1 - ch2[4]), math.cos(byaw1 - ch2[4]))) > 0.15:
                        r2 = nav("nav2 servo-fix", ch2[2], ch2[3], ch2[4], "servo", max_steps=600)
                    bx1, by1, byaw1 = behavior.get_base_pose()
                    report["renav"] = {"edge_m": e2, "twist": tw2, "cand": ch2}
                    rg3, radio_c3 = localize_radio("after nav2")
                    if rg3 is not None:
                        rg2, radio_c2 = rg3, radio_c3
                        report["radio_top_z2"] = round(max(float(p[2]) for p in radio_c2.points), 3)
                    rel_x = (float(rg2.pose.position[0]) - bx1) * math.cos(byaw1) + (float(rg2.pose.position[1]) - by1) * math.sin(byaw1)
                    rel_y = -(float(rg2.pose.position[0]) - bx1) * math.sin(byaw1) + (float(rg2.pose.position[1]) - by1) * math.cos(byaw1)
                    print("after nav2 radio relative to base: forward", round(rel_x, 3), "left", round(rel_y, 3))
                    report["radio_rel"] = [round(rel_x, 3), round(rel_y, 3)]
    report["stage"] = "navigated"

def same_axis_quats(q_ref):
    """Orientations that keep the fingers (hand +y) on q_ref's horizontal +y line: both signs, tilts 0/+-15/+-30."""
    yv = qrot(q_ref, [0.0, 1.0, 0.0])
    nv = math.hypot(yv[0], yv[1]) or 1.0
    out = [q_ref]
    for sign in (1.0, -1.0):
        for tilt in (0, 15, -15, 30, -30):
            qq, tf = hand_quat(sign * yv[0] / nv, sign * yv[1] / nv, tilt)
            out.append(qq)
    return out


# MAX_JOINT_JUMP 1.0 rad: an IK solution further than this from the current joints (any joint) is another arm family, and
# the straight joint path to it swept the radio 1 m in seed 27 attempt 4; untuned.
MAX_JOINT_JUMP = 1.0


def reach_pose(tag, x, y, z, q_ref, tol, steps):
    """move_to_pose; when its own (current-seeded) IK is refused (generalization dev v01 seed 40: every rung's descent was
    `ik_unreachable`, residual 1.3 cm / 9.3 deg, and the hand never left the pregrasp), take the nearest current-seeded
    IK solution over orientations with the same finger line and drive it with move_to_joints; failing that, stop 2-6 cm
    higher. Returns the last ExecutionResult."""
    r = move_to_pose(Pose(position=[x, y, z], quaternion_wxyz=q_ref, frame=st.base_frame), tolerance=tol, max_steps=steps, arm=arm)
    if r.ok or terminated_by(r) or "planner" not in str(r.status).lower():
        return r
    cur = [float(v) for v in get_robot_state().joint_positions[arm]]
    for dz in (0.0, 0.02, 0.04, 0.06):
        best, best_d = None, None
        for qq in same_axis_quats(q_ref):
            ik = solve_ik(Pose(position=[x, y, z + dz], quaternion_wxyz=qq, frame=st.base_frame), seed_joints=cur, arm=arm)
            if not ik.ok:
                continue
            jj = ik_joints(ik)
            if jj is None:
                continue
            dmax = max(abs(a - b) for a, b in zip(jj, cur))
            m = bounds_margin(arm, jj)
            if dmax <= MAX_JOINT_JUMP and (m is None or m > 0.02) and (best_d is None or dmax < best_d):
                best, best_d = jj, dmax
        if best is not None:
            r = move_to_joints(best, tolerance=0.02, max_steps=steps, arm=arm)
            hp = [float(v) for v in get_robot_state().end_effector_poses[arm].position]
            print(tag, "via joints: dz", dz, "jump", round(best_d, 2), "ok:", r.ok, "status:", str(r.status), "hand:", fmt(hp), "target z", round(z, 3))
            return r
    print(tag, "no current-seeded IK within", MAX_JOINT_JUMP, "rad at any height")
    return r


# Code block 4: close the last stretch with servo, lean, re-localize, pick a reachable grasp orientation, go to its pregrasp.
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def qaxis(ax, ay, az, ang):
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    sh = math.sin(ang / 2.0)
    return [math.cos(ang / 2.0), ax / n * sh, ay / n * sh, az / n * sh]


def qrot(q, v):
    qv = [0.0, v[0], v[1], v[2]]
    qc = [q[0], -q[1], -q[2], -q[3]]
    r = qmul(qmul(q, qv), qc)
    return [r[1], r[2], r[3]]


def qnorm(q):
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return [x / n for x in q]


def edge_ahead(support, bx, by, byaw, band=0.3):
    """Forward distance from the base to the nearest support point within a lateral band (the table edge)."""
    fx, fy = math.cos(byaw), math.sin(byaw)
    best = 9.0
    for p in support.points:
        rx, ry = float(p[0]) - bx, float(p[1]) - by
        along = rx * fx + ry * fy
        lateral = abs(-rx * fy + ry * fx)
        if lateral < band and 0.0 < along < best:
            best = along
    return best


def wrap_deg(a):
    while a > 90.0:
        a -= 180.0
    while a <= -90.0:
        a += 180.0
    return a


# Heights relative to the head-read top (which includes the wire handle standing on an upright radio). DZ_PRE 0.04: the
# fingertips clear the handle by ~1 cm of head-read error (seed 27 attempt 9 tipped the radio passing 1 cm above it);
# DZ_OVER 0.15: the "over" waypoint of seeds 26/27; DZ_GRASP -0.09: fingertips ~5 cm down the sides of the upright body
# (body ~0.11 tall under a ~0.03 handle, generalization dev pilot 39/40 wrist reads), untuned.
DZ_PRE = 0.04
# DZ_APPROACH 0.08: where the joint-space approach legs end (the hover reads and the ladder then descend by verified
# Cartesian moves). Generalization dev v01 seed 39: the over and pregrasp joint targets landed 3.6-3.8 cm below the poses
# they were solved for (joint error 0.006 rad: an IK residual), and the pregrasp at top + 0.04 hit the handle and tipped
# the radio; 0.08 keeps that error clear of the top. Untuned.
DZ_APPROACH = 0.08
DZ_OVER = 0.15
DZ_GRASP = -0.09
# DZ_HANDLE -0.03: fingertips 3 cm under the top read, below the handle's wires and above the body top (v02 seed 45: a
# descent met the body 5.3 cm under the top read). Untuned.
DZ_HANDLE = -0.03


def oriented(ref_q, byaw, yaw_deg, pitch_deg):
    """Hand orientation: the hanging orientation yawed about up then pitched about the base's left axis."""
    left = [-math.sin(byaw), math.cos(byaw), 0.0]
    q = qnorm(qmul(qaxis(0.0, 0.0, 1.0, math.radians(yaw_deg)), ref_q))
    return qnorm(qmul(qaxis(left[0], left[1], left[2], math.radians(pitch_deg)), q))


if report.get("nav_ok"):
    bx1, by1, byaw1 = behavior.get_base_pose()
    ea = edge_ahead(sup.point_cloud, bx1, by1, byaw1)
    report["edge_ahead_m"] = round(ea, 3)
    # Stop 0.33 m short of the edge: the R1 Pro base is ~0.5 m long, so 0.33 leaves ~8 cm; untuned beyond that.
    hop = max(0.0, ea - 0.33)
    print("edge ahead:", round(ea, 3), "servo hop:", round(hop, 3))
    # edge_ahead returns its 9.0 sentinel when no support point lies ahead (attempt 1 then hopped 8.7 m): a hop longer
    # than the ring radius (1.2 m) means the edge was not seen, so skip it rather than drive blind
    if hop > 1.2:
        print("hop skipped: no support edge seen ahead")
        hop = 0.0
    # Lateral alignment: put the radio in front of the grasping arm's shoulder (the hanging hand's lateral offset at
    # reset, ~0.25 m). Attempt 3: with the radio 0.14 m inside the right shoulder's line the right arm stalled 13 cm short
    # of a pose IK accepted, j2 pinned at its 0.175 rad bound. Shift capped at 0.3 m and only onto seen floor; untuned.
    rel_y_nav = report["radio_rel"][1]
    side_arm = "primary" if rel_y_nav >= 0.0 else "secondary"
    lat_shift = rel_y_nav - shoulder_off[side_arm]      # base moves along its left axis by this to make rel_y == shoulder_off
    if abs(lat_shift) > 0.3:
        print("lateral shift", round(lat_shift, 3), "capped to 0")
        lat_shift = 0.0
    # the table edge is not square to the heading: re-measure it at the shifted lateral position and set the forward hop
    # there (attempt 4: the radio tipped over during the lean settle after a 0.15 m shift measured against the old edge)
    sx = bx1 - lat_shift * math.sin(byaw1)
    sy = by1 + lat_shift * math.cos(byaw1)
    ea2 = edge_ahead(sup.point_cloud, sx, sy, byaw1)
    if ea2 < 2.0:
        hop = max(-0.3, ea2 - 0.33)       # may back up when the edge is nearer on that side
    print("edge ahead at the shifted pose:", round(ea2, 3), "-> hop", round(hop, 3))
    report["edge_ahead_shifted_m"] = round(ea2, 3)
    tx = bx1 + hop * math.cos(byaw1) - lat_shift * math.sin(byaw1)
    ty = by1 + hop * math.sin(byaw1) + lat_shift * math.cos(byaw1)
    print("hop forward", round(hop, 3), "lateral", round(lat_shift, 3), "for arm", side_arm, "floor at target:", floor_near(tx, ty))
    report["hop_lat"] = round(lat_shift, 3)
    if not floor_near(tx, ty):
        lat_shift = 0.0
        tx, ty = bx1 + hop * math.cos(byaw1), by1 + hop * math.sin(byaw1)
    if abs(hop) > 0.02 or abs(lat_shift) > 0.02:
        r = behavior.navigate_to_pose(tx, ty, byaw1, planner="servo", tolerance_m=0.02, max_steps=400)
        print("hop servo ->", fmt([tx, ty, byaw1]), "ok:", r.ok, "status:", str(r.status), "steps:", r.steps_executed,
              "err:", r.error.message if r.error else None, "base now:", fmt(behavior.get_base_pose()))
    bx2, by2, byaw2 = behavior.get_base_pose()
    report["base_after_hop"] = fmt([bx2, by2, byaw2])
    def hand_quat(cx, cy, tilt_deg):
        """Hand orientation with +y along (cx, cy) and the tool (+z) pointing down, tilted tilt_deg about +y.
        The fingers travel along hand +y: the wrist camera's image x is the hand's -y (reset camera pose against the hanging
        hand), and in the pilot's wrist keyframes (seeds 37, 39) the two fingertips sit at the image's left and right
        edges and meet in the middle on close. So +y goes ACROSS the radio's thin side (generalization dev v01)."""
        n = math.hypot(cx, cy) or 1.0
        yx, yy = cx / n, cy / n
        t = math.radians(tilt_deg)
        zx, zy, zz = -yy * math.sin(t), yx * math.sin(t), -math.cos(t)
        xx, xy, xz = yy * zz - 0.0 * zy, 0.0 * zx - yx * zz, yx * zy - yy * zx      # x = y cross z
        m = [[xx, yx, zx], [xy, yy, zy], [xz, 0.0, zz]]      # columns x, y, z (the y axis is horizontal)
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
        return qnorm(q), (zx * math.cos(byaw2) + zy * math.sin(byaw2))

    def read_radio_head(tag):
        """Head localization, median-cropped, with the top-face centre and axes; None when the localization fails."""
        rgk, ck = localize_radio(tag)
        if rgk is None:
            return None
        zsk = sorted(float(p[2]) for p in ck.points)
        topk = zsk[(19 * len(zsk)) // 20]
        basek = zsk[len(zsk) // 20]
        angk, lk, sk, gxk, gyk = cloud_axes(ck)
        gx_all, gy_all = gxk, gyk
        try:
            top_c = crop_point_cloud(ck, [-50.0, -50.0, topk - 0.03], [50.0, 50.0, topk + 1.0])
            if len(top_c.points) >= 30:
                angk, lk, sk, gxk, gyk = cloud_axes(top_c)
        except ValueError:
            print(tag, "top-face crop empty; using the all-points centre")
        print(tag, "top-face centre:", round(gxk, 3), round(gyk, 3), "long deg", round(math.degrees(angk), 1), "long", round(lk, 3),
              "short", round(sk, 3), "top", round(topk, 3), "base", round(basek, 3), "shift from all-points centre:", round(math.hypot(gxk - gx_all, gyk - gy_all), 3))
        return {"gx": gxk, "gy": gyk, "top": topk, "base": basek, "ang": angk, "long": lk, "short": sk, "cloud": ck}

    arm = side_arm
    report["arm"] = arm

    def lean_margins(rr):
        """Best bounds margin of an IK solution at the approach end (top+DZ_APPROACH), over (top+DZ_OVER) and grasp (top+DZ_GRASP)
        heights, over +y along the radio's SHORT axis (both signs, tilts 0/+-15/+-30) and 12 seeds; (m_pre, m_over, m_grasp)."""
        cur = [float(v) for v in get_robot_state().joint_positions[arm]]
        j2s = 1.0 if arm == "primary" else -1.0
        seeds = [cur, [0.0] * 7] + [[s1, j2s * 0.3, 0.0, s4, 0.0, 0.0, 0.0] for s1 in (-1.0, -0.5, 0.0, 0.5, 1.0) for s4 in (-0.5, -1.2)]
        out = []
        for dz in (DZ_APPROACH, DZ_OVER, DZ_GRASP):
            best = None
            for sign in (1.0, -1.0):
                for tilt in (0, -15, 15, -30, 30):
                    qq, tf = hand_quat(-sign * math.sin(rr["ang"]), sign * math.cos(rr["ang"]), tilt)
                    pose_c = Pose(position=[rr["gx"], rr["gy"], rr["top"] + dz], quaternion_wxyz=qq, frame=st_base_frame)
                    for sd in seeds:
                        ik = solve_ik(pose_c, seed_joints=sd, arm=arm)
                        if ik.ok:
                            m = bounds_margin(arm, ik_joints(ik))
                            if m is not None and (best is None or m > best):
                                best = m
            out.append(best)
        return out

    st_base_frame = get_robot_state().base_frame
    # Adaptive lean in 0.2 rad increments (a one-shot 0.8 lean swung the hanging hands and the upright radio tipped over in
    # attempts 4-5): from 0.4 on, localize and stop at the first lean where the pregrasp, an over point 15 cm higher and
    # the grasp depth all have an IK solution >= 0.05 rad inside every bound. Attempt 6 at 0.8: the over point's only
    # solution sat on the elbow bound, the hand dipped into the radio on the way to the pregrasp and tipped it.
    lean = 0.0
    tgt = list(torso0)
    lean_choice = None
    best_lean = None
    for lk in (0.2, 0.4, 0.6, 0.8):
        tgt[0] = torso0[0] + lk
        r = behavior.move_torso(tgt, max_steps=120)
        lean = lk
        hz_k = float(get_robot_state().end_effector_poses[arm].position[2])
        print("lean step j0", round(lk, 2), "ok:", r.ok, "steps:", r.steps_executed, "hand z:", round(hz_k, 3))
        if lk >= 0.4:
            r = behavior.move_torso(tgt, max_steps=60)
            rr = read_radio_head("lean " + str(lk))
            if rr is not None:
                ms = lean_margins(rr)
                print("lean", lk, "margins pre/over/grasp:", ms)
                if all(m is not None and m >= 0.05 for m in ms):
                    lean_choice = lk
                    break
                mm = min((m if m is not None else -1.0) for m in ms)
                if best_lean is None or mm > best_lean[0]:
                    best_lean = (mm, lk)
    # no lean passed: go back to the one whose worst margin was best rather than staying at 0.8 (pilot 41: the 0.8 lean
    # gave margins 0.02/-0.0 and a 6 cm-shifted head read, where 0.6 had 0.48/-0.0/0.60)
    if lean_choice is None and best_lean is not None and best_lean[1] != lean:
        tgt[0] = torso0[0] + best_lean[1]
        r = behavior.move_torso(tgt, max_steps=120)
        lean = best_lean[1]
        print("no lean passed; back to", lean, "worst margin", best_lean[0])
    r = behavior.move_torso(tgt, max_steps=60)      # settle: a second servo to the same target returns when the sway is gone
    print("lean used", lean, "(choice", lean_choice, ") settle ok:", r.ok, "steps:", r.steps_executed, "base:", fmt(behavior.get_base_pose()))
    report["lean"] = lean
    report["lean_choice"] = lean_choice
    rr1 = read_radio_head("after lean, first read")
    rr2 = read_radio_head("after lean, second read")
    if rr1 is not None and rr2 is not None:
        print("shift between the two leaned reads:", round(math.hypot(rr2["gx"] - rr1["gx"], rr2["gy"] - rr1["gy"]), 3))
    rr = rr2 if rr2 is not None else rr1
    if rr is None:
        rr = {"gx": float(rg2.pose.position[0]), "gy": float(rg2.pose.position[1]), "top": report["radio_top_z2"], "base": report["radio_top_z2"] - 0.15,
              "ang": cloud_axes(radio_c2)[0], "long": cloud_axes(radio_c2)[1], "short": cloud_axes(radio_c2)[2]}
    gx, gy, top, base_z, ang, long_len, short_len = rr["gx"], rr["gy"], rr["top"], rr["base"], rr["ang"], rr["long"], rr["short"]
    report["radio_base_z"] = round(base_z, 3)
    print("radio axes: long angle deg", round(math.degrees(ang), 1), "long", round(long_len, 3), "short", round(short_len, 3),
          "centre", round(gx, 3), round(gy, 3), "top", round(top, 3))
    report["grasp_xy"] = [round(gx, 3), round(gy, 3)]
    report["radio_top"] = round(top, 3)
    report["radio_axes"] = [round(math.degrees(ang), 1), round(long_len, 3), round(short_len, 3)]
    report["radio_upright"] = bool(top - base_z > 0.12 and long_len < 0.13)     # shadow: standing on end (0.17 tall, top end 0.09 x 0.06) or lying
    rel_x = (gx - bx2) * math.cos(byaw2) + (gy - by2) * math.sin(byaw2)
    rel_y = -(gx - bx2) * math.sin(byaw2) + (gy - by2) * math.cos(byaw2)
    print("radio relative to base now: forward", round(rel_x, 3), "left", round(rel_y, 3), "upright:", report["radio_upright"])
    st = get_robot_state()
    ref_q = [float(q) for q in st.end_effector_poses[arm].quaternion_wxyz]
    hand0 = [float(v) for v in st.end_effector_poses[arm].position]
    print("hand now:", fmt(hand0), "q:", fmt(ref_q))

    short_axis = (-math.sin(ang), math.cos(ang))
    long_axis = (math.cos(ang), math.sin(ang))
    # heights from the head-read top (handle included); the grasp never below 3 cm above the radio's lowest points
    z_pre, z_grasp = top + DZ_APPROACH, max(base_z + 0.03, top + DZ_GRASP)
    print("heights: pre", round(z_pre, 3), "grasp", round(z_grasp, 3), "base_z", round(base_z, 3), "top", round(top, 3))

    def terminated_by(r):
        return bool(r.terminated) or "terminated" in str(r.status).lower()

    st0 = get_robot_state()
    cur_j = [float(v) for v in st0.joint_positions[arm]]
    j2sign = 1.0 if arm == "primary" else -1.0        # j2's open range is positive for the left arm, negative for the right
    # 22 IK seeds (seed 27 attempt 5): the same pose gave bounds margins from 0.0 to 0.375 depending on the seed
    seed_set = [cur_j, [0.0] * 7]
    for s1 in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for s2 in (0.3, 1.0):
            for s4 in (-0.5, -1.2):
                seed_set.append([s1, j2sign * s2, 0.0, s4, 0.0, 0.0, 0.0])

    def any_ik(pose):
        """True when some seed solves the pose (a single default-seeded solve missed reachable ladders on pilot 38 and 40)."""
        for sd in seed_set:
            if solve_ik(pose, seed_joints=sd, arm=arm).ok:
                return True
        return False

    # The fingers close along hand +y, so +y goes along the radio's SHORT axis: the upright radio is ~0.06 thick under a
    # ~0.10 opening, while its 0.17-0.20 long side cannot be straddled (pilot 39: +y along the long axis put the fingers
    # on the handle, jaw 0.5, and the lift dropped it).
    chosen = None
    orients = []
    for sign in (1.0, -1.0):
        for tilt in (0, -15, 15, -30, 30):
            q, tool_fwd = hand_quat(sign * short_axis[0], sign * short_axis[1], tilt)
            oks = [any_ik(Pose(position=[gx, gy, z], quaternion_wxyz=q, frame=st.base_frame)) for z in (z_pre, z_grasp)]
            print("orient short sign", sign, "tilt", tilt, "tool_fwd", round(tool_fwd, 2), "ik pre/grasp:", oks)
            if oks[0]:
                orients.append(("short", sign, tilt, q, oks[1]))
    orients = [o for o in orients if o[4]] + [o for o in orients if not o[4]]     # both heights first
    if orients:
        chosen = orients[0][:4]
    report["n_orients_ok"] = len(orients)
    report["n_orients_both"] = len([o for o in orients if o[4]])
    if chosen is not None:
        which, sign, tilt, q = chosen
        report["orient"] = {"closes": which, "sign": sign, "tilt": tilt, "q": fmt(q, 4)}
        # Approach in three legs, all in ONE arm configuration family, driven in joint space (seed 27 attempt 4: a
        # re-solved family swept the radio 1 m): solve the pregrasp from many seeds, keep the solution furthest inside the
        # joint bounds, seed "over" and "lift" from it, and execute lift -> over -> pregrasp with move_to_joints.
        way_q = [o[3] for o in orients]
        q = way_q[0]

        def best_ik(pose, seeds, tag):
            """IK over seeds: (joints, margin) of the solution furthest inside the bounds, or (None, None)."""
            best_j, best_m = None, None
            for sd in seeds:
                ik = solve_ik(pose, seed_joints=sd, arm=arm)
                if not ik.ok:
                    continue
                jj = ik_joints(ik)
                m = bounds_margin(arm, jj)
                if jj is None:
                    return None, None
                if best_m is None or (m is not None and m > best_m):
                    best_j, best_m = jj, m
            print(tag, "best margin", best_m, "joints", fmt(best_j, 2) if best_j else None)
            return best_j, best_m

        def go_joints(tag, target_j, pose, tol=0.02, steps=150, calls=3):
            """move_to_joints re-issued until it converges; prints the hand against the pose it was solved for."""
            r = None
            tp = [float(v) for v in pose.position]
            for k in range(calls):
                r = move_to_joints(target_j, tolerance=tol, max_steps=steps, arm=arm)
                st_k = get_robot_state()
                jj = [float(v) for v in st_k.joint_positions[arm]]
                hp = [float(v) for v in st_k.end_effector_poses[arm].position]
                jerr = max(abs(a - b) for a, b in zip(jj, target_j))
                print(tag, "call", k, "ok:", r.ok, "status:", str(r.status), "steps:", r.steps_executed, "err:", r.error.message if r.error else None,
                      "joint err max:", round(jerr, 3), "hand:", fmt(hp), "off_xy:", round(math.hypot(hp[0] - tp[0], hp[1] - tp[1]), 3), "off_z:", round(hp[2] - tp[2], 3))
                if r.ok or terminated_by(r) or jerr < tol:
                    break
            hp = [float(v) for v in get_robot_state().end_effector_poses[arm].position]
            return r, hp

        def go(tag, pose, tol, steps, retries=1):
            """move_to_pose, re-issued while the hand moved but is still outside tolerance (servo lag); < 1 cm moved is a block."""
            tp = [float(v) for v in pose.position]
            hp = [float(v) for v in get_robot_state().end_effector_poses[arm].position]
            r = None
            for k in range(retries + 1):
                before = hp
                r = move_to_pose(pose, tolerance=tol, max_steps=steps, arm=arm)
                st_k = get_robot_state()
                hp = [float(v) for v in st_k.end_effector_poses[arm].position]
                moved = math.hypot(math.hypot(hp[0] - before[0], hp[1] - before[1]), hp[2] - before[2])
                print(tag, "call", k, "ok:", r.ok, "status:", str(r.status), "steps:", r.steps_executed, "err:", r.error.message if r.error else None,
                      "hand:", fmt(hp), "off_xy:", round(math.hypot(hp[0] - tp[0], hp[1] - tp[1]), 3), "off_z:", round(hp[2] - tp[2], 3),
                      "moved:", round(moved, 3), "joints:", fmt(st_k.joint_positions[arm], 2))
                if r.ok or terminated_by(r) or r.steps_executed == 0 or moved < 0.01:
                    break
            return r, hp

        # 1. the pregrasp solution: every orientation x every seed, largest bounds margin wins
        pre_j, pre_m, pre_q = None, None, q
        for qq in way_q:
            pose_c = Pose(position=[gx, gy, z_pre], quaternion_wxyz=qq, frame=st.base_frame)
            jj, m = best_ik(pose_c, seed_set, "pregrasp ik orient " + str(way_q.index(qq)))
            if jj is not None and (pre_m is None or (m is not None and m > pre_m)):
                pre_j, pre_m, pre_q = jj, m, qq
        q = pre_q
        report["pre_margin"] = pre_m
        print("pregrasp family: orient index", way_q.index(q), "margin", pre_m, "joints", fmt(pre_j, 2) if pre_j else None)
        legs = {}
        hp = hand0
        r = None
        if pre_j is not None:
            # 2. over: above the radio, seeded from the pregrasp solution; any orientation, highest reachable height
            over_j, over_m, over_pose = None, None, None
            for dz in (DZ_OVER, 0.20, 0.10, 0.25, 0.08):     # 0.08 last: a low "over" still beats a straight lift -> pregrasp sweep
                for qq in way_q:
                    pose_c = Pose(position=[gx, gy, top + dz], quaternion_wxyz=qq, frame=st.base_frame)
                    jj, m = best_ik(pose_c, [pre_j], "over dz " + str(dz) + " orient " + str(way_q.index(qq)))
                    if jj is not None and m is not None and m >= 0.05 and (over_m is None or m > over_m):
                        over_j, over_m, over_pose = jj, m, pose_c
                if over_j is not None:
                    break
            # 3. lift: the hanging hand straight up, seeded from the over (or pregrasp) solution
            lift_j, lift_m, lift_pose = None, None, None
            for dz in (0.25, 0.30, 0.20):
                for qq in way_q + [ref_q]:
                    pose_c = Pose(position=[hand0[0], hand0[1], top + dz], quaternion_wxyz=qq, frame=st.base_frame)
                    jj, m = best_ik(pose_c, [over_j if over_j is not None else pre_j, cur_j], "lift dz " + str(dz) + " orient " + str((way_q + [ref_q]).index(qq)))
                    if jj is not None and m is not None and m >= 0.05 and (lift_m is None or m > lift_m):
                        lift_j, lift_m, lift_pose = jj, m, pose_c
                if lift_j is not None:
                    break
            report["leg_margins"] = {"lift": lift_m, "over": over_m, "pre": pre_m}
            if lift_j is not None:
                r, hp = go_joints("lift", lift_j, lift_pose)
                legs["lift"] = bool(r.ok)
            if over_j is not None:
                r, hp = go_joints("over", over_j, over_pose)
                legs["over"] = bool(r.ok)
            pre = Pose(position=[gx, gy, z_pre], quaternion_wxyz=q, frame=st.base_frame)
            # the last leg is a verified Cartesian move: joint targets landed 3.6-7.8 cm below their solved poses on v01
            # seeds 39 and 45 (IK residuals <= 1.5 cm, joint errors <= 0.017 rad: the leaned arm sags), and a joint-space
            # approach end that low hit the radio
            r = reach_pose("approach end", gx, gy, z_pre, q, 0.02, 200)
            hp = [float(v) for v in get_robot_state().end_effector_poses[arm].position]
            print("approach end ok:", r.ok, "status:", str(r.status), "hand:", fmt(hp), "off_xy:", round(math.hypot(hp[0] - gx, hp[1] - gy), 3), "off_z:", round(hp[2] - z_pre, 3))
            if not r.ok and not terminated_by(r) and math.hypot(hp[0] - gx, hp[1] - gy) > 0.02:
                r, hp = go_joints("pregrasp", pre_j, pre)     # the solved family as the fallback
        else:
            print("no pregrasp IK solution from any seed")
        report["legs"] = legs
        report["waypoint_ok"] = bool(legs.get("over"))
        st = get_robot_state()
        print("pregrasp: orient index", way_q.index(q), "final_errors:", dict(r.final_errors) if r is not None else None,
              "gripper:", round(float(st.gripper_positions[arm]), 3), "joints:", fmt(st.joint_positions[arm], 2),
              "margin:", bounds_margin(arm, [float(v) for v in st.joint_positions[arm]]))
        report["pregrasp_ok"] = bool(r is not None and (r.ok or math.hypot(hp[0] - gx, hp[1] - gy) < 0.02))
        report["pregrasp_off"] = [round(math.hypot(hp[0] - gx, hp[1] - gy), 3), round(hp[2] - z_pre, 3)]
        cam = "left_wrist" if arm == "primary" else "right_wrist"
        wseg = segment_text(cam, str(meta.get("target_prompt") or "red radio"))   # records a wrist overlay to look at; not used
    else:
        print("no orientation reachable at both pregrasp and grasp height")
    report["stage"] = "pregrasp"

# Code block 5: re-localize from the wrist camera (the head cloud sits ~11 cm short), hover, read again, then a short ladder.
def terminated_by(r):
    return bool(r.terminated) or "terminated" in str(r.status).lower()


def wrist_read(tag):
    """Radio top-face centre, axes and heights from the wrist camera; None when it fails. The wrist shares the hand's kinematics."""
    cam = "left_wrist" if arm == "primary" else "right_wrist"
    wloc = localize_object(str(meta.get("target_prompt") or "red radio"), camera_name=cam)
    if not wloc.ok:
        print(tag, "wrist localize failed:", wloc.error.message if wloc.error else None)
        return None
    try:
        wcl = robust_crop(wloc.point_cloud, 0.2)
        wz = sorted(float(p[2]) for p in wcl.points)
        wtop = wz[(19 * len(wz)) // 20]
        wbase = wz[len(wz) // 20]
        wtc = crop_point_cloud(wcl, [-50.0, -50.0, wtop - 0.03], [50.0, 50.0, wtop + 1.0])
        wang, wlong, wshort, wgx, wgy = cloud_axes(wtc)
    except ValueError:
        print(tag, "wrist crop empty")
        return None
    print(tag, "wrist: centre", round(wgx, 3), round(wgy, 3), "top", round(wtop, 3), "base", round(wbase, 3), "long deg", round(math.degrees(wang), 1),
          "long", round(wlong, 3), "short", round(wshort, 3), "n_top", len(wtc.points), "n_all", len(wcl.points),
          "minus head centre:", round(wgx - gx, 3), round(wgy - gy, 3))
    return {"cx": wgx, "cy": wgy, "top": wtop, "base": wbase, "ang": wang, "long": wlong, "short": wshort, "n": len(wtc.points)}


JAW_MIN = 0.015


def probe(tag, px, py, q_use, zp, zg):
    """One rung: pregrasp above (px, py), descend to zg, close, and on a mid-range jaw lift 6 cm by IK, then un-lean."""
    out = {"tag": tag}
    r = reach_pose("rung pre", px, py, zp, q_use, 0.02, 200)
    out["pre_ok"] = bool(r.ok)
    if terminated_by(r):
        out["terminated"] = "pregrasp"
        return out
    r = reach_pose("rung descent", px, py, zg, q_use, 0.015, 200)
    out["descent_via"] = str(r.status)
    if terminated_by(r):
        out["terminated"] = "descent"     # v02 seed 45: the witness latched during a descent that wedged the radio
        print("rung", out)
        return out
    st2 = get_robot_state()
    hz = float(st2.end_effector_poses[arm].position[2])
    out["descent_ok"] = bool(r.ok)
    out["stopped_above"] = round(hz - zg, 3)          # > 0.02 means the fingers met something before the target height
    rc = close_gripper(arm=arm)
    out["jaw"] = round(float(get_robot_state().gripper_positions[arm]), 3)
    out["terminated"] = ""
    # jaw bands (normalized, 0 closed / 1 open): closing on air reads 0.000-0.003 and a handle pinch 0.12-0.28 (seed 26
    # attempts 13/15, seed-26 pilot 41), so JAW_MIN 0.015 sits between them; > 0.92 is jammed open. Untuned beyond that.
    if JAW_MIN < out["jaw"] < 0.92:
        hp2 = [float(v) for v in get_robot_state().end_effector_poses[arm].position]
        # a slow 6 cm IK lift first: the witness latches at 5 mm, and the 0.2 rad un-lean (a 17 cm swing) shook a pinch loose
        r = reach_pose("rung lift", hp2[0], hp2[1], hp2[2] + 0.06, q_use, 0.02, 120)
        st3 = get_robot_state()
        out["ik_lift_hand_z"] = round(float(st3.end_effector_poses[arm].position[2]), 3)
        out["jaw_after_ik_lift"] = round(float(st3.gripper_positions[arm]), 3)
        out["terminated"] = "ik_lift" if terminated_by(r) else ""
        if not out["terminated"]:
            tgt_up = list(torso0)
            tgt_up[0] = torso0[0] + lean - 0.2        # a 0.2 rad un-lean lifted the hand ~17 cm (seed 26 attempt 10)
            r = behavior.move_torso(tgt_up, max_steps=150)
            st3 = get_robot_state()
            out["lift_hand_z"] = round(float(st3.end_effector_poses[arm].position[2]), 3)
            out["jaw_after_lift"] = round(float(st3.gripper_positions[arm]), 3)
            out["terminated"] = "lift" if terminated_by(r) else ""
            if not out["terminated"]:
                tgt_back = list(torso0)
                tgt_back[0] = torso0[0] + lean
                r = behavior.move_torso(tgt_back, max_steps=150)
    if not out["terminated"]:
        ro = open_gripper(arm=arm)
        r = reach_pose("rung back", px, py, zp, q_use, 0.02, 150)
    print("rung", out)
    return out


# WRIST_MAX_DEV 0.10: a wrist centre further than this from the head's leaned read is a runaway, not a correction (pilot
# 37: four hover reads walked 0.08 -> 0.15 -> 0.22 -> 0.24 m away and the ladder closed on the table beside the radio;
# seed 26 attempt 10 measured a true head bias of 0.11 m, so the guard sits just under it). Untuned.
WRIST_MAX_DEV = 0.10

if report.get("pregrasp_ok") and chosen is not None:
    w1 = wrist_read("w1")
    if w1 is not None and math.hypot(w1["cx"] - gx, w1["cy"] - gy) > WRIST_MAX_DEV:
        print("w1 rejected: off the head centre by", round(math.hypot(w1["cx"] - gx, w1["cy"] - gy), 3))
        w1 = None
    w = w1
    if w is not None:
        # hover above the wrist centre and read again until the centre stops moving (seed 26 attempt 12: a single read saw
        # only part of the top face), at most three times, never past WRIST_MAX_DEV from the head centre
        for it in range(3):
            hov = None
            for dz in (0.08, 0.05, 0.03):
                cand = Pose(position=[w["cx"], w["cy"], w["top"] + dz], quaternion_wxyz=q, frame=st.base_frame)
                if any_ik(cand):
                    hov = cand
                    break
            if hov is None:
                print("hover", it, "unreachable")
                break
            r = reach_pose("hover " + str(it), float(hov.position[0]), float(hov.position[1]), float(hov.position[2]), q, 0.02, 200)
            print("hover", it, "ok:", r.ok, "steps:", r.steps_executed, "hand:", fmt(get_robot_state().end_effector_poses[arm].position))
            wn = wrist_read("w" + str(it + 2))
            if wn is None or wn["n"] < 30:
                break
            if math.hypot(wn["cx"] - gx, wn["cy"] - gy) > WRIST_MAX_DEV:
                print("hover read", it, "rejected: off the head centre by", round(math.hypot(wn["cx"] - gx, wn["cy"] - gy), 3))
                break
            shift = math.hypot(wn["cx"] - w["cx"], wn["cy"] - w["cy"])
            print("hover read", it, "shift:", round(shift, 3))
            w = wn
            if shift < 0.01:
                break
        report["wrist_centre"] = [round(w["cx"], 3), round(w["cy"], 3)]
        report["wrist_top"] = round(w["top"], 3)
        report["wrist_base"] = round(w["base"], 3)
        report["wrist_axes"] = [round(math.degrees(w["ang"]), 1), round(w["long"], 3), round(w["short"], 3)]
        cx, cy = w["cx"], w["cy"]
        # table height from the wrist cloud's lowest points: the head's depth sees through the glass table top to the
        # shelf below (seed 27), several cm too low for a descent floor
        wrist_bases = [v["base"] for v in (w1, w) if v is not None]
        table_z = min(wrist_bases) if wrist_bases else base_z
        zp = w["top"] + DZ_PRE
        zg = max(table_z + 0.02, w["top"] + DZ_GRASP)   # fingertips never below 2 cm above the table
        print("table_z", round(table_z, 3), "(head base_z", round(base_z, 3), ") top", round(w["top"], 3), "height over table", round(w["top"] - table_z, 3))
        report["height_over_table"] = round(w["top"] - table_z, 3)
        # the long-axis angle comes from the head's full view when the wrist saw only part of the top face
        use_ang = ang if w["long"] < 0.14 else w["ang"]
        print("axis angle used (deg):", round(math.degrees(use_ang), 1), "head:", round(math.degrees(ang), 1), "wrist:", round(math.degrees(w["ang"]), 1))
        report["axis_deg_used"] = round(math.degrees(use_ang), 1)
        long_dir = (math.cos(use_ang), math.sin(use_ang))
        short_dir = (-math.sin(use_ang), math.cos(use_ang))
    else:
        cx, cy, zp, zg = gx, gy, top + DZ_PRE, z_grasp
        long_dir = (math.cos(ang), math.sin(ang))
        short_dir = (-math.sin(ang), math.cos(ang))
        report["ladder_axis"] = "head"

    def orient_for(axis_vec, tag):
        """Hand orientation with +y along axis_vec (either sign): first tilt whose pregrasp and grasp both solve IK from
        any seed; else the first that solves the pregrasp only (the descent then stalls where it can)."""
        fallback = None
        for sign in (chosen[1], -chosen[1]):
            for tilt in (chosen[2], 0, 15, -15, 30, -30):
                qq, tf = hand_quat(sign * axis_vec[0], sign * axis_vec[1], tilt)
                ok_pre = any_ik(Pose(position=[cx, cy, zp], quaternion_wxyz=qq, frame=st.base_frame))
                if not ok_pre:
                    continue
                if any_ik(Pose(position=[cx, cy, zg], quaternion_wxyz=qq, frame=st.base_frame)):
                    print(tag, "orientation: sign", sign, "tilt", tilt)
                    return qq
                if fallback is None:
                    fallback = qq
        print(tag, "no orientation solves both heights; pregrasp-only fallback:", fallback is not None)
        return fallback

    st_h = get_robot_state()
    rungs = []
    done = ""
    q_use = orient_for(short_dir, "ladder")
    if q_use is not None:
        ca = qrot(q_use, [0.0, 1.0, 0.0])       # the fingers travel along hand +y
        nv = math.hypot(ca[0], ca[1]) or 1.0
        dxv, dyv = ca[0] / nv, ca[1] / nv
        print("ladder heights: pre", round(zp, 3), "grasp", round(zg, 3), "centre", round(cx, 3), round(cy, 3), "finger axis", round(dxv, 2), round(dyv, 2),
              "radio long", round(math.degrees(math.atan2(long_dir[1], long_dir[0])), 1))
        # Rungs along the finger axis. Body rungs first: straddling the upright body (~0.06 thick under a ~0.10 opening)
        # held at jaw 0.596 on v02 seed 40's second rung (d=+0.015); a rung off-centre by ~2 cm reads jaw 1.0 (a finger
        # already on the body) and the next offset fixes it. Then handle rungs, the fingertips 3 cm under the top read and
        # closed across the wire handle: the seed-26 holds (jaw 0.26-0.28) and seed-26 pilot 41 (jaw 0.12) were pinches.
        zh = max(zg, (w["top"] if w is not None else top) + DZ_HANDLE)
        plan = [(0.0, zg, "body"), (0.015, zg, "body"), (-0.015, zg, "body"), (0.03, zg, "body"), (-0.03, zg, "body"),
                (0.0, zh, "handle"), (0.015, zh, "handle"), (-0.015, zh, "handle")]
        for d, zd, kind in plan:
            out = probe(kind + " d=" + str(d), cx + d * dxv, cy + d * dyv, q_use, zp, zd)
            rungs.append(out)
            if out.get("terminated"):
                done = out["tag"]
                break
            # a mid jaw that slipped may have moved the radio: re-read from the wrist at the rung's pregrasp (zero ticks)
            if JAW_MIN < out.get("jaw", 0.0) < 0.92:
                wr = wrist_read("after " + out["tag"])
                if wr is not None and wr["n"] >= 30 and math.hypot(wr["cx"] - cx, wr["cy"] - cy) < WRIST_MAX_DEV:
                    cx, cy = wr["cx"], wr["cy"]
    report["rungs"] = rungs
    report["ladder_done_at"] = done
    report["stage"] = "ladder"

result = report
