# How accurate is "head pixel + depth -> base-frame point" at the table's range? Put a fingertip (a known 3D point from
# FK: 2.7 cm behind the gripper-centre link along its approach axis) where the blocks are, project it into the head
# image, read the depth there, back-project, and compare with the known point. Done at the home pose and over the table.
import math
report = {}
state = get_robot_state()
base = state.base_frame
arm = "secondary"
TIP_BACK = 0.027


def rotate(q, v):
    w, x, y, z = q
    # rotate v by the unit quaternion q (wxyz)
    t = (2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0]))
    return (v[0] + w * t[0] + (y * t[2] - z * t[1]), v[1] + w * t[1] + (z * t[0] - x * t[2]), v[2] + w * t[2] + (x * t[1] - y * t[0]))


def check(label):
    obs = get_observation()
    cam = obs.cameras["head"]
    K = cam.intrinsics
    cp = cam.camera_pose
    cq = [float(v) for v in cp.quaternion_wxyz]
    ct = [float(v) for v in cp.position]
    # camera rotation matrix columns from the quaternion
    w, x, y, z = cq
    R = ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
         (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
         (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
    out = {}
    for a in ("primary", "secondary"):
        ee = obs.robot_state.end_effector_poses[a]
        q = [float(v) for v in ee.quaternion_wxyz]
        axis = rotate(q, (0.0, 0.0, 1.0))
        tip = tuple(float(ee.position[i]) - TIP_BACK * axis[i] for i in range(3))
        # into the camera frame: p_c = R^T (p - t)
        d = tuple(tip[i] - ct[i] for i in range(3))
        pc = tuple(sum(R[r][c] * d[r] for r in range(3)) for c in range(3))
        if pc[2] <= 0.05:
            out[a] = "behind camera"
            continue
        u = float(K[0][0]) * pc[0] / pc[2] + float(K[0][2])
        v = float(K[1][1]) * pc[1] / pc[2] + float(K[1][2])
        ui, vi = int(round(u)), int(round(v))
        depth = cam.depth_m
        h, wd = len(depth), len(depth[0])
        if not (2 <= ui < wd - 2 and 2 <= vi < h - 2):
            out[a] = ("outside image", u, v)
            continue
        vals = sorted(float(depth[vi + i][ui + j]) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2) if float(depth[vi + i][ui + j]) > 0)
        zc = vals[len(vals) // 2]
        pcb = ((u - float(K[0][2])) * zc / float(K[0][0]), (v - float(K[1][2])) * zc / float(K[1][1]), zc)
        back = tuple(ct[r] + sum(R[r][c] * pcb[c] for c in range(3)) for r in range(3))
        out[a] = {"pixel": (ui, vi), "fk_depth_m": pc[2], "measured_depth_m": zc, "tip_fk": tip, "tip_from_depth": back,
                  "error_cm": tuple((back[i] - tip[i]) * 100 for i in range(3))}
    report[label] = out


check("home")
# the right gripper over the table where the blocks are, pointing down, 10 cm above the table
t = math.radians(20.0)
zax = (math.sin(t), 0.0, -math.cos(t))
yax = (0.0, 1.0, 0.0)
xax = (yax[1] * zax[2] - yax[2] * zax[1], yax[2] * zax[0] - yax[0] * zax[2], yax[0] * zax[1] - yax[1] * zax[0])
m00, m11, m22 = xax[0], yax[1], zax[2]
s = math.sqrt(1.0 + m11 - m00 - m22) * 2
q = ((zax[0] - xax[2]) / s, (yax[0] + xax[1]) / s, 0.25 * s, (zax[1] + yax[2]) / s)
for label, pos in (("over_table_near", (0.65, -0.15, 0.90)), ("over_table_far", (0.80, -0.10, 0.90))):
    r = move_to_pose({"position": pos, "quaternion_wxyz": q, "frame": base}, tolerance=0.02, max_steps=300, arm=arm)
    report[label + "_move"] = (bool(r.ok), None if r.error is None else str(r.error.message)[:50])
    check(label)
print(report)
result = report
