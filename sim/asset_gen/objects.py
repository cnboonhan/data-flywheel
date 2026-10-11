# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "trimesh", "scipy", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""The objects of a pano_ideate session as simulation assets, generated freely rather than reconstructed: the VLM says
what each object most likely is and how a typical one looks, from where it was last seen; the image editor draws a
clean, new one (the session's view as a loose reference) and its back (camera-angle LoRA); TripoSG makes the shape
from the drawing, coloured from both drawings; the session sets its real size and place (scaled to its measured height,
turned about the vertical axis until its silhouette from the capture point matches its last-seen 3D points, standing
where it stood).

    uv run sim/asset_gen/objects.py <scene>_session.zip [--asset <name>] [--seed 0] [--out datasets/asset_gen]

Models (services/triton/triton_endpoint.py; each a model name on the flywheel Triton or a full URL):
    VLM_ENDPOINT      Qwen3.5-122B-A10B      what the object is (sim/pano_ideate/models.py)
    EDIT_ENDPOINT     Qwen-Image-Edit-2511   the drawings, front and back (its camera-angle LoRA)
    SEGMENT_ENDPOINT  sam2.1-hiera-large     cutting the drawings out
    MESH_ENDPOINT     TripoSG                the shape
Writes <out>/<scene>/objects/<asset>/: model.glb (glTF, y up, vertex colours), model.usda (z up, metres; a rigid body
with convex-decomposition collision, a mass from its volume, the colours as displayColor), pose.json (where it stands
in the session's scene frame, how it is turned, what it was generated as), views/ (the session's view, the drawings).
Both meshes are in the object's own frame: origin at the centre of its base.
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "triton"))
from triton_endpoint import Endpoint  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pano_ideate"))
from models import Triton  # noqa: E402

DENSITY = 400.0          # kg/m^3 of the convex hull: household objects are mostly hollow or light
IDENTIFY = ("This is one object, cut out of a photo of a room (it may be small, blurry or partly hidden); it was "
            "labelled {label}. Say what it most likely is, as a shopper would name it, and describe a clean, new, typical "
            "one that fits: overall shape, proportions (height to width), parts (lid, handle, ...), material, colours. No "
            "labels, dirt or residue. Answer JSON: {{\"name\": ..., \"description\": one sentence}}.")
DRAW = ("Picture 1 shows a {name}. Draw a new, complete, clean {name}: {description} Product photo, the whole object "
        "centred and filling about two thirds of the frame, three-quarter view from slightly above, plain pure white "
        "background, soft even studio light, no shadow, no other objects.")
BACK = "<sks> back view elevated shot medium shot"   # the camera-angle LoRA's prompt (services/triton/README.md)


def crop(png: bytes, size: int = 512) -> bytes:
    """An RGBA cutout cropped square around the object (alpha), with a margin: the editor redraws an object framed
    small in a large view poorly."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    ys, xs = np.nonzero(img[..., 3] > 128)
    if not len(xs):
        return png
    half = int(0.65 * max(np.ptp(xs), np.ptp(ys))) + 1
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    square = padded[cy:cy + 2 * half, cx:cx + 2 * half]
    return cv2.imencode(".png", cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA))[1].tobytes()


def on_white(png: bytes) -> np.ndarray:
    """An RGBA cutout (cropped around the object) composited on white, BGR."""
    img = cv2.imdecode(np.frombuffer(crop(png), np.uint8), cv2.IMREAD_UNCHANGED)
    a = img[..., 3:] / 255.0
    return (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def edit(editor: Endpoint, images: list[np.ndarray], prompt: str, seed: int = 0, lora: str = "",
         lora_scale: float = 1.4) -> np.ndarray:
    """The image editor's result (BGR) for BGR pictures; lora "angles" turns on its camera-angle LoRA (at 1.0 it barely
    moves the camera off a studio shot's own view)."""
    out = editor.infer({"images": [cv2.imencode(".png", i)[1].tobytes() for i in images], "prompt": [prompt.encode()],
                        "seed": np.array([seed], np.int32), "lora": [lora.encode()],
                        "lora_scale": np.array([lora_scale], np.float32)}, ["image"])["image"]
    return cv2.imdecode(np.frombuffer(np.ravel(out)[0], np.uint8), cv2.IMREAD_COLOR)


def identify(vlm: Triton, cutout: np.ndarray, label: str) -> dict:
    """What the object most likely is and a typical one's look ({"name", "description"}), from a BGR cutout."""
    answer = vlm.vlm(IDENTIFY.format(label=label), cutout[..., ::-1], 300)
    try:
        found = json.loads(answer[answer.index("{"):answer.rindex("}") + 1])
        return {"name": str(found["name"]), "description": str(found["description"])}
    except (ValueError, KeyError):
        return {"name": label, "description": ""}


def generate(generator: Endpoint, rgba: np.ndarray, seed: int = 0) -> trimesh.Trimesh:
    """TripoSG's mesh (shape only) of an RGBA cutout, z up."""
    out = generator.infer({"image": [cv2.imencode(".png", rgba)[1].tobytes()], "seed": np.array([seed], np.int32),
                           "faces": np.array([60000], np.int32)}, ["mesh"])["mesh"]
    mesh = trimesh.load(io.BytesIO(np.ravel(out)[0]), file_type="glb", force="mesh")
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))   # glTF y up -> z up
    return mesh


def model(vlm, editor, segmenter, generator, cutouts: list[bytes], label: str, seed: int = 0, log=print):
    """The object's mesh (z up, TripoSG's units) with vertex colours, and the images made on the way (name -> BGR)."""
    cutout = on_white(cutouts[-1])                                    # the latest view: where it was last seen
    what = identify(vlm, cutout, label)
    log(f"  a {what['name']}: {what['description']}")
    front = edit(editor, [cutout], DRAW.format(**what), seed)
    back = edit(editor, [front], BACK, seed, lora="angles")
    rgba = matte(segmenter, front)
    mesh = generate(generator, rgba, seed)
    log(f"  generated: {len(mesh.faces)} faces")
    paint(mesh, [(front, rgba[..., 3] > 128, None), (back, matte(segmenter, back)[..., 3] > 128, 180.0)], log)
    return mesh, {"cutout": cutout, "front": front, "back": back}, what


def matte(segmenter: Endpoint, img: np.ndarray) -> np.ndarray:
    """The object in a studio view as RGBA: SAM 2.1 on the box around what is not white (no ground shadow)."""
    ys, xs = np.nonzero(np.abs(img.astype(int) - 255).max(-1) > 18)
    box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32)
    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    mask = segmenter.infer({"image": [jpg], "boxes": box}, ["masks"])["masks"].reshape(-1, *img.shape[:2])[0]
    return np.dstack([img, mask.astype(np.uint8) * 255])


def look(azimuth: float, elevation: float) -> np.ndarray:
    """Rows: image right, image down, view direction, of an orthographic camera circling z (azimuth 0 looks along +y)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    f = np.array([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), -np.sin(e)])
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def silhouette(xy: np.ndarray, faces: np.ndarray, shape) -> np.ndarray:
    sil = np.zeros(shape, np.uint8)
    cv2.fillPoly(sil, np.round(xy[faces]).astype(np.int32), 1)
    return sil


def pose(mesh: trimesh.Trimesh, mask: np.ndarray, azimuths, elevations):
    """The orthographic view of the mesh (look() rows, scale, offset) whose silhouette best overlaps a mask, and the
    overlap: the views are drawn, so their camera is found rather than known."""
    S = 128
    small = cv2.resize(mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(small)
    faces = mesh.faces[np.random.default_rng(0).choice(len(mesh.faces), min(len(mesh.faces), 20000), replace=False)]
    best = (-1.0, None)
    for a in azimuths:
        for e in elevations:
            R = look(a, e)
            xy = mesh.vertices @ R[:2].T
            lo, hi = xy.min(0), xy.max(0)
            scale = np.mean([np.ptp(xs) / max(hi[0] - lo[0], 1e-9), np.ptp(ys) / max(hi[1] - lo[1], 1e-9)])
            offset = np.array([xs.min() + xs.max(), ys.min() + ys.max()]) / 2 - scale * (lo + hi) / 2
            sil = silhouette(xy * scale + offset, faces, (S, S))
            best = max(best, ((sil & small).sum() / max((sil | small).sum(), 1), (a, e, R, scale, offset)),
                       key=lambda b: b[0])
    iou, (a, e, R, scale, offset) = best
    k = mask.shape[1] / S
    return R, scale * k, offset * k, iou, a, e


def paint(mesh: trimesh.Trimesh, views: list, log=print):
    """Vertex colours from drawn views (BGR image, mask, azimuth relative to the first view or None) where each sees
    the mesh; the rest from the nearest coloured vertex."""
    from scipy.spatial import cKDTree
    colour, weight = np.zeros((len(mesh.vertices), 3)), np.zeros(len(mesh.vertices))
    first = None
    for img, mask, turn in views:
        if first is None:
            R, scale, offset, iou, a, e = pose(mesh, mask, np.arange(0, 360, 10), np.arange(0, 61, 10))
            first = (a, e)
        else:                                                         # the LoRA turns the camera roughly by turn
            R, scale, offset, iou, a, e = pose(mesh, mask, first[0] + turn + np.arange(-30, 31, 10), [first[1]])
        log(f"  view at azimuth {a:.0f}, elevation {e:.0f}: silhouette overlap {iou:.2f}")
        xy = mesh.vertices @ R[:2].T * scale + offset
        depth = mesh.vertices @ R[2]
        h, w = mask.shape
        ids = np.full((h, w), -1, np.int32)                           # nearest face per pixel: far faces drawn first
        tri = np.round(xy[mesh.faces]).astype(np.int32)
        for f in np.argsort(-depth[mesh.faces].mean(1)):
            cv2.fillConvexPoly(ids, tri[f], int(f))
        seen = np.zeros(len(mesh.faces), bool)
        seen[np.unique(ids[ids >= 0])] = True
        v = np.unique(mesh.faces[seen])
        x, y = np.clip(np.round(xy[v]).astype(int), 0, [w - 1, h - 1]).T
        inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)   # edges blend with white
        v, x, y = v[inner[y, x]], x[inner[y, x]], y[inner[y, x]]
        wgt = np.clip(-(mesh.vertex_normals[v] @ R[2]), 0.05, 1) * iou
        colour[v] += img[y, x, ::-1] * wgt[:, None]
        weight[v] += wgt
    have = weight > 0
    colour[have] /= weight[have, None]
    if have.any() and not have.all():
        colour[~have] = colour[have][cKDTree(mesh.vertices[have]).query(mesh.vertices[~have])[1]]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[colour.clip(0, 255).astype(np.uint8),
                                                                        np.full(len(colour), 255, np.uint8)])


def project(rays: np.ndarray, forward: np.ndarray, fov: float, size: int):
    """Pixel coordinates of rays (from the capture point, scene frame) in a pinhole view looking along forward."""
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    z = np.maximum(rays @ forward, 1e-6)
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    return (rays @ right) / z * f + size / 2, (rays @ down) / z * f + size / 2, rays @ forward > 1e-6


def fit(mesh: trimesh.Trimesh, shape: dict, camera_height: float, size: list[float] | None, log=print):
    """mesh (z up) in the object's own frame (z up, metres, base centre at the origin), its yaw about z (degrees), its
    base in the scene frame and the silhouette overlap, from the object's last-seen 3D points (shape)."""
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    pts, base = shape["xyz"], np.asarray(shape["base"], float)
    plo, phi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    height = max(phi[2] - plo[2], size[2] if size else 0)      # the points miss the far side, not the height
    mesh.apply_scale(height / max(mesh.extents[2], 1e-6))

    # Yaw: the turn whose silhouette, seen from the capture point, best overlaps the points' (both in one view).
    eye = np.array([0, 0, camera_height])
    rays = pts - eye
    d = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    forward = d.mean(0) / np.linalg.norm(d.mean(0))
    fov = float(np.clip(3 * np.degrees(np.percentile(np.arccos(np.clip(d @ forward, -1, 1)), 99)) + 5, 10, 120))
    S = 256
    x, y, ok = project(rays, forward, fov, S)
    seen = np.zeros((S, S), np.uint8)
    keep = ok & (x >= 0) & (x < S) & (y >= 0) & (y < S)
    seen[y[keep].astype(int), x[keep].astype(int)] = 1
    seen = cv2.morphologyEx(cv2.dilate(seen, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    faces = mesh.faces[np.random.default_rng(0).choice(len(mesh.faces), min(len(mesh.faces), 30000), replace=False)]
    best = (-1.0, 0.0)
    for a in np.arange(0, 360, 5):
        R = trimesh.transformations.rotation_matrix(np.radians(a), [0, 0, 1])[:3, :3]
        x, y, ok = project(mesh.vertices @ R.T + base - eye, forward, fov, S)
        sil = np.zeros((S, S), np.uint8)
        tri = np.stack([x, y], -1)[faces][ok[faces].all(1)]
        cv2.fillPoly(sil, np.round(tri).astype(np.int32), 1)
        best = max(best, ((sil & seen).sum() / max((sil | seen).sum(), 1), a))
    log(f"  fitted: {height:.2f} m tall, turned {best[1]:.0f} deg, silhouette overlap {best[0]:.2f}")
    return mesh, float(best[1]), base, float(best[0])


def usda(mesh: trimesh.Trimesh, label: str, mass: float) -> str:
    """The mesh as a USD stage (z up, metres): a rigid body with convex-decomposition collision, its vertex colours as
    displayColor."""
    name = "".join(c if c.isalnum() else "_" for c in label).strip("_").capitalize() or "Object"
    fmt = lambda a: ", ".join(f"({', '.join(f'{v:.6g}' for v in row)})" for row in a)
    colours = getattr(mesh.visual, "vertex_colors", None)
    colour = (f'        color3f[] primvars:displayColor = [{fmt(np.asarray(colours)[:, :3] / 255.0)}] (\n'
              f'            interpolation = "vertex"\n        )\n' if colours is not None and len(colours) else "")
    return f'''#usda 1.0
(
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {mass:.4g}

    def Mesh "mesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "convexDecomposition"
        int[] faceVertexCounts = [{", ".join(["3"] * len(mesh.faces))}]
        int[] faceVertexIndices = [{", ".join(str(i) for i in mesh.faces.ravel())}]
        point3f[] points = [{fmt(mesh.vertices)}]
        normal3f[] normals = [{fmt(mesh.vertex_normals)}] (
            interpolation = "vertex"
        )
{colour}        uniform token subdivisionScheme = "none"
    }}
}}
'''


def write(out: Path, local: trimesh.Trimesh, info: dict, what: dict, yaw: float, base: np.ndarray, overlap: float,
          images: dict):
    (out / "views").mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        cv2.imwrite(str(out / "views" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    mass = float(np.clip(DENSITY * local.convex_hull.volume, 0.02, 50.0))
    gltf = local.copy()
    gltf.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))         # z up -> y up
    (out / "model.glb").write_bytes(gltf.export(file_type="glb"))
    (out / "model.usda").write_text(usda(local, info["label"], mass))
    (out / "pose.json").write_text(json.dumps({
        "label": info["label"], "position": [round(float(c), 4) for c in base], "yaw_deg": round(yaw, 1),
        "frame": "the session's scene frame (scene.json): z up, metres, origin on the floor below the capture point",
        "in_room": info.get("in_room"), "size_m": [round(float(c), 4) for c in local.extents],
        "mass_kg": round(mass, 3), "silhouette_overlap": round(overlap, 3), "generated_as": what,
        "generated_by": "Qwen3.5 (what it is), Qwen-Image-Edit-2511 (drawings), SAM 2.1, TripoSG (shape)"},
        indent=1))


def run(session: Path, out_root: Path, assets: list[str] | None = None, seed: int = 0, log=print) -> list[Path]:
    vlm, editor = Triton(), Endpoint("EDIT", "Qwen-Image-Edit-2511")
    segmenter, generator = Endpoint("SEGMENT", "sam2.1-hiera-large"), Endpoint("MESH", "TripoSG")
    log(f"{vlm.vlm_}\n{editor}\n{segmenter}\n{generator}")
    with zipfile.ZipFile(session) as z:
        scene = json.loads(z.read("scene.json"))
        h = scene["frame"]["camera_height"]
        written = []
        for name in scene["assets"]["objects"]:
            if assets and name not in assets:
                continue
            info = json.loads(z.read(f"objects/{name}/info.json"))
            views = sorted(n for n in z.namelist() if n.startswith(f"objects/{name}/views/"))
            if not views or f"objects/{name}/shape.npz" not in z.namelist():
                log(f"{name}: no views recorded, skipped")
                continue
            log(f"{name}: {len(views)} views")
            shape = dict(np.load(io.BytesIO(z.read(f"objects/{name}/shape.npz"))))
            mesh, images, what = model(vlm, editor, segmenter, generator, [z.read(v) for v in views], info["label"],
                                       seed, log)
            local, yaw, base, overlap = fit(mesh, shape, h, info.get("size_m"), log)
            out = out_root / scene["scene"] / "objects" / name
            write(out, local, info, what, yaw, base, overlap, images)
            written.append(out)
    log(f"{len(written)} objects -> {out_root / scene['scene'] / 'objects'}")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("session", type=Path, help="a session zip from pano_ideate's viewer (Download session)")
    p.add_argument("--asset", action="append", help="only this asset (repeat for more)")
    p.add_argument("--out", type=Path, default=Path("datasets/asset_gen"))
    p.add_argument("--seed", type=int, default=0, help="another seed draws another object")
    a = p.parse_args()
    run(a.session, a.out, a.asset, a.seed)
