# /// script
# requires-python = ">=3.10"
# dependencies = ["pycolmap-cuda12>=4.2"]
# ///
"""Refine a posed COLMAP model's camera poses against its images; the lenses stay fixed.

    uv run sim/splat/refine_poses.py <colmap> <colmap_refined>

For captures posed by a robot's localisation (sensors/real2sim): a few cm and about a degree of pose error blur a splat
far more than anything in training. GPU SIFT features, matches between each image and its 30 nearest views facing the
same way (pairs from the input poses), triangulation from the input poses, then bundle adjustment of poses and points
(Cauchy loss, intrinsics fixed), twice. The result is mapped back onto the input camera centres with a similarity, so it
stays in the input (map) frame at metric scale; the input's points3D (lidar seed) are kept alongside the triangulated
ones. Writes <colmap_refined>/sparse/0 (text), an images/ symlink, and work/ (feature database, reused on reruns).
x86_64 only: pycolmap-cuda12 has no aarch64 wheels.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

PAIRS_PER_IMAGE, MAX_AXIS_ANGLE = 30, 75       # matching pairs: nearest views whose optical axes differ by < 75 deg
MAX_REPROJ_ERROR, MAX_DISTANCE = 4.0, 50.0     # px; m from the cameras' centroid

src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work = dst / "work"
work.mkdir(parents=True, exist_ok=True)
rec = pycolmap.Reconstruction(src / "sparse/0")
seed = [(np.array(p.xyz), np.array(p.color)) for p in rec.points3D.values()]   # copies: triangulation frees these
C0 = {im.name: im.projection_center() for im in rec.images.values()}
R0 = {im.name: im.cam_from_world().rotation.matrix() for im in rec.images.values()}

# Features: the database gets the model's own cameras, rigs, frames and images (same ids, lenses marked as known, or
# COLMAP rejects fisheye pairs as degenerate); extraction then adds keypoints to those images.
db = work / "database.db"
if not (db.exists() and pycolmap.Database.open(db).num_inlier_matches() > 0):
    db.unlink(missing_ok=True)
    d = pycolmap.Database.open(db)
    for cam in rec.cameras.values():
        cam.has_prior_focal_length = True
        d.write_camera(cam, use_camera_id=True)
    for rig in rec.rigs.values():
        d.write_rig(rig, use_rig_id=True)
    for frame in rec.frames.values():
        d.write_frame(frame, use_frame_id=True)
    for im in rec.images.values():
        d.write_image(pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id, frame_id=im.frame_id),
                      use_image_id=True)
    d.close()
    pycolmap.extract_features(db, src / "images", device=pycolmap.Device.cuda)

    ims = sorted(rec.images.values(), key=lambda im: im.name)
    C = np.array([im.projection_center() for im in ims])
    Z = np.array([im.cam_from_world().rotation.matrix()[2] for im in ims])   # optical axes in the world frame
    pairs = set()
    for i in range(len(ims)):
        ok = (Z @ Z[i] > np.cos(np.radians(MAX_AXIS_ANGLE))) & (np.arange(len(ims)) != i)
        near = np.argsort(np.where(ok, np.linalg.norm(C - C[i], axis=1), np.inf))[:PAIRS_PER_IMAGE]
        pairs |= {tuple(sorted((ims[i].name, ims[j].name))) for j in near if ok[j]}
    (work / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    pairing = pycolmap.ImportedPairingOptions(match_list_path=str(work / "pairs.txt"))
    pycolmap.match_image_pairs(db, pairing_options=pairing, device=pycolmap.Device.cuda)
    print(f"features and matches: {len(rec.images)} images, {len(pairs)} pairs", flush=True)


def clean(rec, tag):
    """Drop points with a large reprojection error or far from the cameras; print the error spread."""
    rec.update_point_3d_errors()
    centre = np.mean([im.projection_center() for im in rec.images.values()], 0)
    for pid in [pid for pid, p in rec.points3D.items()
                if not p.error < MAX_REPROJ_ERROR or np.linalg.norm(p.xyz - centre) > MAX_DISTANCE]:
        rec.delete_point3D(pid)
    err = np.array([p.error for p in rec.points3D.values()])
    print(f"{tag}: {len(err)} points, reprojection error median {np.median(err):.2f} px, "
          f"90th pct {np.percentile(err, 90):.2f} px", flush=True)


ba = pycolmap.BundleAdjustmentOptions(refine_focal_length=False, refine_principal_point=False,
                                      refine_extra_params=False)
ba.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
for it in range(2):
    rec = pycolmap.triangulate_points(rec, db, src / "images", work / "triangulated", clear_points=True)
    clean(rec, f"triangulated ({it + 1}/2)")
    pycolmap.bundle_adjustment(rec, ba)
    clean(rec, f"bundle adjusted ({it + 1}/2)")
shutil.rmtree(work / "triangulated", ignore_errors=True)

# Back onto the input frame: the similarity mapping refined camera centres onto the input ones (Umeyama).
names = sorted(C0)
A, B = np.array([rec.find_image_with_name(n).projection_center() for n in names]), np.array([C0[n] for n in names])
ma, mb = A.mean(0), B.mean(0)
U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma))
D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R, s = U @ D @ Vt, np.trace(np.diag(S) @ D) / ((A - ma) ** 2).sum()
rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), mb - s * R @ ma))
moved = np.linalg.norm([rec.find_image_with_name(n).projection_center() - C0[n] for n in names], axis=1)
R1 = {n: rec.find_image_with_name(n).cam_from_world().rotation.matrix() for n in names}
turned = np.degrees([np.arccos(np.clip((np.trace(R1[n] @ R0[n].T) - 1) / 2, -1, 1)) for n in names])
print(f"poses changed: position median {np.median(moved) * 100:.1f} cm "
      f"(90th pct {np.percentile(moved, 90) * 100:.1f}), rotation median {np.median(turned):.2f} deg "
      f"(90th pct {np.percentile(turned, 90):.2f}); scale {s:.4f}", flush=True)

for xyz, color in seed:
    rec.add_point3D(xyz, pycolmap.Track(), color)
out = dst / "sparse/0"
out.mkdir(parents=True, exist_ok=True)
rec.write_text(out)
if not (dst / "images").exists():
    (dst / "images").symlink_to(src / "images")
print(f"wrote {out}: {len(rec.images)} images, {len(rec.points3D)} points", flush=True)
