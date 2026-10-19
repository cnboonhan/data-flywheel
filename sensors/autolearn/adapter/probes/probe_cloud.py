# Diagnostic: depth validity and the localized point cloud for the named block.
obs = get_observation()
report = {}
for name, cam in obs.cameras.items():
    d = cam.depth_m
    vals = [float(v) for row in [d[i] for i in range(0, len(d), max(1, len(d) // 40))] for v in [row[j] for j in range(0, len(row), max(1, len(row) // 40))]]
    zeros = sum(1 for v in vals if v <= 0)
    nans = sum(1 for v in vals if v != v)
    finite = [v for v in vals if v == v and v > 0]
    report[name] = {"shape": (len(d), len(d[0])), "sampled": len(vals), "zeros": zeros, "nans": nans,
                    "min": min(finite) if finite else None, "max": max(finite) if finite else None}
language = ""
try:
    language = obs.task_context.language.lower()
except Exception:
    pass
color = "red"
for word in ["red", "yellow", "blue", "green", "purple"]:
    if word in language:
        color = word
        break
t = localize_object(color + " block", camera_name="head", target_frame=get_robot_state().base_frame)
report["localized"] = bool(t.ok)
if t.ok:
    pts = t.point_cloud.points
    n = len(pts)
    xs = [float(p[0]) for p in pts]
    zs = [float(p[2]) for p in pts]
    report["cloud"] = {"n": n, "nan_x": sum(1 for v in xs if v != v), "z_min": min(zs) if zs else None, "z_max": max(zs) if zs else None,
                       "first": [float(v) for v in pts[0]] if n else None}
else:
    report["localize_error"] = str(t.error.message) if t.error else None
print(report)
result = report
