# The omnipicker's finger clearance: from the head depth at the home pose, the points around the right gripper
# expressed in the gripper frame (origin = centre link, z = approach). The tip zone is z in [-4, -0.5] cm (the tips
# are ~2.7 cm behind the centre); the spread of those points along x (the finger axis) and y is the open clearance.
report = {}
obs = get_observation()
cam = obs.cameras["head"]
K = cam.intrinsics
cp = cam.camera_pose
w, x, y, z = [float(v) for v in cp.quaternion_wxyz]
R = ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
     (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
     (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
ct = [float(v) for v in cp.position]
depth = cam.depth_m
h, wd = len(depth), len(depth[0])
for arm in ("primary", "secondary"):
    ee = obs.robot_state.end_effector_poses[arm]
    c = [float(v) for v in ee.position]
    gw, gx, gy, gz = [float(v) for v in ee.quaternion_wxyz]
    G = ((1 - 2 * (gy * gy + gz * gz), 2 * (gx * gy - gw * gz), 2 * (gx * gz + gw * gy)),
         (2 * (gx * gy + gw * gz), 1 - 2 * (gx * gx + gz * gz), 2 * (gy * gz - gw * gx)),
         (2 * (gx * gz - gw * gy), 2 * (gy * gz + gw * gx), 1 - 2 * (gx * gx + gy * gy)))
    d = [c[i] - ct[i] for i in range(3)]
    pc = [sum(R[r][k] * d[r] for r in range(3)) for k in range(3)]
    u0 = int(float(K[0][0]) * pc[0] / pc[2] + float(K[0][2]))
    v0 = int(float(K[1][1]) * pc[1] / pc[2] + float(K[1][2]))
    mask = [[(abs(i - v0) < 70 and abs(j - u0) < 90) for j in range(wd)] for i in range(h)]
    cloud = mask_to_point_cloud(mask, "head", target_frame=obs.robot_state.base_frame)
    # into the gripper frame
    zone = []
    for p in cloud.points:
        q = [float(p[i]) - c[i] for i in range(3)]
        g = [sum(G[r][k] * q[r] for r in range(3)) for k in range(3)]
        if -0.045 < g[2] < -0.005 and abs(g[0]) < 0.12 and abs(g[1]) < 0.12:
            zone.append(g)
    if not zone:
        report[arm] = {"pixel": (u0, v0), "zone_points": 0}
        continue
    xs = sorted(p[0] for p in zone)
    ys = sorted(p[1] for p in zone)
    n = len(zone)
    # the finger sides: points left and right of the axis; the clearance is the gap between the innermost ones
    left = [p for p in zone if p[0] < -0.005]
    right = [p for p in zone if p[0] > 0.005]
    inner_gap = (min(p[0] for p in right) - max(p[0] for p in left)) if left and right else None
    report[arm] = {"pixel": (u0, v0), "zone_points": n, "x_span_cm": ((xs[-1] - xs[0]) * 100), "x_p02_p98_cm": (xs[n // 50] * 100, xs[-1 - n // 50] * 100),
                   "y_span_cm": ((ys[-1] - ys[0]) * 100), "inner_gap_cm": None if inner_gap is None else inner_gap * 100,
                   "left_right_counts": (len(left), len(right))}
print(report)
result = report
