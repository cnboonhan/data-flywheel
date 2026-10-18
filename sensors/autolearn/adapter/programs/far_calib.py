# Head-camera check away from the home pose: move the right gripper to a far, low pose, project its FK centre into
# the head image, and look for the gripper in the depth around that pixel. Reports the pixel and the depth error at
# the projection, and where (pixel offset, 3D point) the nearest gripper surface really is.
import math

report = {}
state = get_robot_state()
base = state.base_frame
arm = "secondary"
home = state.end_effector_poses[arm]
t = math.radians(20.0)
z = (math.sin(t), 0.0, -math.cos(t))
y = (0.0, 1.0, 0.0)
x = (y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0])
m00, m01, m02 = x[0], y[0], z[0]
m10, m11, m12 = x[1], y[1], z[1]
m20, m21, m22 = x[2], y[2], z[2]
s = math.sqrt(m00 + m11 + m22 + 1.0) * 2
q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
up = move_to_pose({"position": (float(home.position[0]), float(home.position[1]), float(home.position[2]) + 0.15),
                   "quaternion_wxyz": [float(v) for v in home.quaternion_wxyz], "frame": base}, tolerance=0.03, max_steps=200, arm=arm)


def check(label):
    obs = get_observation()
    cam = obs.cameras["head"]
    K = cam.intrinsics
    cw, cx, cy, cz = [float(v) for v in cam.camera_pose.quaternion_wxyz]
    R = ((1 - 2 * (cy * cy + cz * cz), 2 * (cx * cy - cw * cz), 2 * (cx * cz + cw * cy)),
         (2 * (cx * cy + cw * cz), 1 - 2 * (cx * cx + cz * cz), 2 * (cy * cz - cw * cx)),
         (2 * (cx * cz - cw * cy), 2 * (cy * cz + cw * cx), 1 - 2 * (cx * cx + cy * cy)))
    ct = [float(v) for v in cam.camera_pose.position]
    ee = obs.robot_state.end_effector_poses[arm]
    c = [float(v) for v in ee.position]
    d = [c[i] - ct[i] for i in range(3)]
    pc = [sum(R[r][k] * d[r] for r in range(3)) for k in range(3)]      # R^T d: into the camera frame
    fx, fy, u0, v0 = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    u, v = fx * pc[0] / pc[2] + u0, fy * pc[1] / pc[2] + v0
    depth = cam.depth_m
    h, w = len(depth), len(depth[0])
    ui, vi = int(round(u)), int(round(v))
    seen = float(depth[vi][ui]) if 0 <= vi < h and 0 <= ui < w else None
    # the nearest surface in a window: where the gripper really is in the image
    best = None
    for dv in range(-40, 41, 2):
        for du in range(-40, 41, 2):
            uu, vv = ui + du, vi + dv
            if 0 <= vv < h and 0 <= uu < w:
                dz = float(depth[vv][uu])
                if dz > 0.05 and (best is None or dz < best[0]):
                    best = (dz, du, dv)
    out = {"ee": [round(a, 4) for a in c], "pixel": (round(u, 1), round(v, 1)), "fk_depth": round(pc[2], 4), "seen_depth": seen}
    if seen:
        out["depth_error_cm"] = round((seen - pc[2]) * 100, 1)
    if best is not None:
        dz, du, dv = best
        # back-project the nearest pixel into base
        pcn = ((ui + du - u0) * dz / fx, (vi + dv - v0) * dz / fy, dz)
        pb = [ct[i] + sum(R[i][k] * pcn[k] for k in range(3)) for i in range(3)]
        out["nearest"] = {"pixel_offset": (du, dv), "depth": round(dz, 4), "point": [round(a, 4) for a in pb],
                          "offset_from_fk_cm": [round((pb[i] - c[i]) * 100, 1) for i in range(3)]}
    report[label] = out


check("home_up")
far = move_to_pose({"position": (0.80, -0.10, 0.80), "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
report["far"] = (bool(far.ok), None if far.error is None else str(far.error.message)[:60])
open_gripper(arm=arm)      # a snapshot with --debug, and a settled observation
check("far")
mid = move_to_pose({"position": (0.75, -0.20, 0.90), "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
open_gripper(arm=arm)
check("mid")
print(report)
result = report
