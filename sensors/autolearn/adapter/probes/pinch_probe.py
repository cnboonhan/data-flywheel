# Where do the fingers meet? Close the gripper in free space above the table and locate the closed fingers with the
# head camera: their extent along the gripper's approach axis, relative to the gripper-centre link.
import math
report = {}
state = get_robot_state()
base = state.base_frame
arm = "primary" if "left" in (get_observation().task_context.language or "").lower() else "secondary"
report["arm"] = arm
# approach tilted 20 deg, same as the pick programs
t = math.radians(20.0)
z = (math.sin(t), 0.0, -math.cos(t))
y = (0.0, 1.0, 0.0)
x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
m00, m11, m22 = x[0], y[1], z[2]
s = math.sqrt(1.0 + m11 - m00 - m22) * 2
q = ((x[2] - z[0]) / s, (y[0] + x[1]) / s, 0.25 * s, (z[1] + y[2]) / s)
sign = 1.0 if arm == "primary" else -1.0
target = {"position": (0.70, sign * 0.15, 0.95), "quaternion_wxyz": q, "frame": base}
moved = move_to_pose(target, tolerance=0.02, max_steps=300, arm=arm)
report["moved"] = (bool(moved.ok), None if moved.error is None else str(moved.error.message)[:60])
ee = get_robot_state().end_effector_poses[arm]
report["ee"] = [float(v) for v in ee.position]
for label, fn in (("open", open_gripper), ("closed", close_gripper)):
    fn(arm=arm)
    seg = segment_text("head", "robot gripper")
    best = None
    if seg.ok:
        cand = [s for s in seg.segmentations if s.score is not None]
        if cand:
            best = max(cand, key=lambda s: float(s.score))
    if best is None:
        report[label] = "no gripper segmentation"
        continue
    cloud = mask_to_point_cloud(best.mask, "head", target_frame=base)
    pts = [p for p in cloud.points if 0.3 < float(p[2]) < 1.6]
    # project each point onto the approach axis z (unit, in base) through the centre link
    along = [(float(p[0]) - ee.position[0]) * z[0] + (float(p[1]) - ee.position[1]) * z[1] + (float(p[2]) - ee.position[2]) * z[2] for p in pts]
    lateral = [math.sqrt((float(p[0]) - ee.position[0]) ** 2 + (float(p[1]) - ee.position[1]) ** 2 + (float(p[2]) - ee.position[2]) ** 2 - a * a) for p, a in zip(pts, along)]
    along_s = sorted(along)
    n = len(along_s)
    report[label] = {"points": n, "score": float(best.score), "along_min": along_s[0] if n else None,
                     "along_p05": along_s[n // 20] if n else None, "along_p95": along_s[19 * n // 20] if n else None,
                     "along_max": along_s[-1] if n else None, "gripper": float(get_robot_state().gripper_positions[arm])}
print(report)
result = report
