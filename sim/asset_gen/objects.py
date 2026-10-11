# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "trimesh", "scipy", "tritonclient[http]", "transformers", "jinja2",
#                 "questionary", "xatlas", "pillow"]
# ///
"""The objects of a pano_ideate session as simulation assets, generated freely rather than reconstructed: the VLM says
what each object most likely is and how a typical one looks, from where it was last seen; the image editor draws a
clean, new one (the session's view as a loose reference); TripoSG makes the shape from the drawing; the editor paints
grey renders of that shape from five sides, told the drawing's colours in words, and the paintings are baked into a
UV texture (texture.py); the session sets its real size and place (scaled to its measured height, turned about the
vertical axis until its silhouette from the capture point matches its last-seen 3D points, standing where it stood).

    uv run sim/asset_gen/objects.py <scene>_session.zip [--asset <name>] [--seed 0] [--out datasets/asset_gen]

Without --asset, a checklist in the terminal asks which objects to generate.

Models (services/triton/triton_endpoint.py; each a model name on the flywheel Triton or a full URL):
    VLM_ENDPOINT      Qwen3.5-122B-A10B      what the object is, its colours in words (sim/pano_ideate/models.py)
    EDIT_ENDPOINT     Qwen-Image-Edit-2511   the drawing, the painted renders
    SEGMENT_ENDPOINT  sam2.1-hiera-large     cutting the drawing out
    MESH_ENDPOINT     TripoSG                the shape
Writes <out>/<scene>/objects/<asset>/: model.glb (glTF, y up, textured), model.usda and texture.png (z up, metres; a
rigid body with convex-decomposition collision, a mass from its volume, a UsdPreviewSurface reading the texture),
pose.json (where it stands in the session's scene frame, how it is turned, what it was generated as), views/ (the
session's view, the drawing, each side's render and painting). Both meshes are in the object's own frame: origin at
the centre of its base.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pano_ideate"))
from models import Triton  # noqa: E402  (VLM, EDIT, SEGMENT; it puts services/triton on the path)
from triton_endpoint import Endpoint  # noqa: E402

import texture  # noqa: E402

DENSITY = 400.0          # kg/m^3 of the convex hull: household objects are mostly hollow or light
# The label comes from seeing the whole room; the cutout may show only part of the object (a chair behind a table was
# once named a fence panel), so the VLM keeps the label's kind and only refines it. The example names no material: one
# that did ("a wooden dining chair") made a wooden blender and a wooden cup.
IDENTIFY = ("This is one object, cut out of a photo of a room; it may be small, blurry or partly hidden, so the "
            "picture may show only part of it. It is a {label} (named from the whole room): keep that kind of object. "
            "Say what kind of {label} it most likely is, as a shopper would name it, and describe a clean, new, "
            "typical, complete one that fits: overall shape, proportions (height to width), parts (lid, handle, legs, "
            "...), material, colours. No labels, dirt or residue. The name ends with the word {label}, naming a kind "
            "of {label} (for a chair: a dining chair, not a bench). Answer JSON: {{\"name\": ..., \"description\": "
            "one sentence}}.")
DRAW = ("Picture 1 shows a {name}. Draw a new, complete, clean {name}: {description} Product photo, the whole object "
        "centred and filling about two thirds of the frame, three-quarter view from slightly above, plain pure white "
        "background, soft even studio light, no shadow, no other objects.")
# Painting renders of the mesh. The editor is given only the grey render and the colours in words (COLOURS, the VLM's
# reading of the drawing): given the drawing as well, it redraws the drawing's own view (a fresh pose, outline overlap
# 0.40); given the render alone it keeps the outline (0.84-0.98). Flat light: Isaac Sim lights the asset itself, and
# baked highlights showed as pale streaks.
COLOURS = ("Describe the colours and materials of every part of this {name}, part by part, in one sentence (for "
           "example: brushed stainless steel body, black plastic handles and feet, clear glass lid with a black knob)."
           " Colours and materials only, no shape.")
PAINT = ("Colour this plain grey 3D render of a {name}, {side}: {colours} Keep its exact outline, shape, viewpoint, "
         "size and position; change only the colours and materials. Even, flat, shadowless light, no highlights or "
         "reflections, plain pure white background.")
# Views painted, from the drawing's own (azimuth, elevation in degrees): it, its sides and back, and the top for lids
# (75: a top that is still a view of the object, not a plan; untuned). Each says which side the render shows: told
# nothing, the editor painted a slow cooker's control panel on its back too.
SIDES = {"front": (0, None, "seen from the front"),
         "right": (90, None, "seen from its side: its front, with any controls, display or label, faces left"),
         "back": (180, None, "seen from the back: its front, with any controls, display or label, faces away and is "
                             "not visible"),
         "left": (270, None, "seen from its side: its front, with any controls, display or label, faces right"),
         "top": (0, 75, "seen from above, its front at the bottom of the picture")}

def on_white(png: bytes, size: int = 512) -> np.ndarray:
    """An RGBA cutout cropped square around the object with a margin (the editor redraws an object framed small in a
    large view poorly), on white, RGB. size: the editor's input side in pixels, untuned."""
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    ys, xs = np.nonzero(img[..., 3] > 128)
    if len(xs):
        half = int(0.65 * max(np.ptp(xs), np.ptp(ys))) + 1           # object fills ~77% of the side; untuned
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        img = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        img = cv2.resize(img[cy:cy + 2 * half, cx:cx + 2 * half], (size, size), interpolation=cv2.INTER_AREA)
    a = img[..., 3:] / 255.0
    return (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def identify(triton: Triton, cutout: np.ndarray, label: str) -> dict:
    """What the object most likely is and a typical one's look ({"name", "description"}), from an RGB cutout."""
    answer = triton.vlm(IDENTIFY.format(label=label), cutout, 300)   # max tokens: fits the one-line JSON; untuned
    try:
        found = json.loads(answer[answer.index("{"):answer.rindex("}") + 1])
        name, description = str(found["name"]), str(found["description"])
    except (ValueError, KeyError):
        name, description = label, ""
    if label.lower().split()[-1] not in name.lower():                 # another kind of object: keep the label
        name, description = label, ""
    return {"name": name, "description": description}


def generate(generator: Endpoint, rgba: np.ndarray, seed: int = 0) -> trimesh.Trimesh:
    """TripoSG's mesh (shape only) of an RGBA cutout, z up."""
    png = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))[1].tobytes()
    out = generator.infer({"image": [png], "seed": np.array([seed], np.int32),
                           "faces": np.array([60000], np.int32)}, ["mesh"])["mesh"]   # GLB < ~1 MB; else ~1.6M faces
    mesh = trimesh.load(io.BytesIO(np.ravel(out)[0]), file_type="glb", force="mesh")
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))   # glTF y up -> z up
    return mesh


def model(triton: Triton, generator: Endpoint, cutouts: list[bytes], label: str, seed: int = 0, log=print):
    """The object's mesh (z up, TripoSG's units) with a texture, and the images made on the way (name -> RGB)."""
    cutout = on_white(cutouts[-1])                                    # the latest view: where it was last seen
    what = identify(triton, cutout, label)
    log(f"  a {what['name']}: {what['description']}")
    images = {"cutout": cutout, "drawing": triton.edit([cutout], DRAW.format(**what), seed=seed)}
    colours = triton.vlm(COLOURS.format(name=what["name"]), images["drawing"], 150)   # one sentence; untuned
    log(f"  colours: {colours}")
    rgba = matte(triton, images["drawing"])
    mesh = generate(generator, rgba, seed)
    log(f"  generated: {len(mesh.faces)} faces")
    # The drawing's own view of the mesh, so the front painting faces the way the drawing does.
    _, _, _, iou, a0, e0 = pose(mesh, rgba[..., 3] > 128, np.arange(0, 360, 10), np.arange(0, 61, 10))
    log(f"  drawn from azimuth {a0:.0f}, elevation {e0:.0f} (silhouette overlap {iou:.2f})")
    scale, offset = texture.framing(mesh)
    views = []
    for side, (turn, elevation, seen) in SIDES.items():
        R = texture.look(a0 + turn, e0 if elevation is None else elevation)
        ids, z = texture.raster(mesh, R, scale, offset(R))
        images[f"render_{side}"] = render = texture.grey(mesh, ids, R)
        painted = triton.edit([render], PAINT.format(name=what["name"], side=seen, colours=colours), seed=seed)
        images[f"paint_{side}"] = painted = cv2.resize(painted, (texture.SIZE, texture.SIZE))
        # The painting's outline from SAM, boxed to the render's: "not white" misses a white mug or a clear jar.
        ys, xs = np.nonzero(ids >= 0)
        outline = triton.segment(painted, [[xs.min(), ys.min(), xs.max(), ys.max()]]).reshape(-1, *ids.shape)[0]
        fit = texture.silhouette_fit(outline, ids)
        log(f"  painted {side}: outline overlap {fit:.2f}" + ("" if fit >= texture.MIN_FIT else ", not used"))
        if fit >= texture.MIN_FIT:
            views.append((painted, R, scale, offset(R), ids, z))
    if not views:
        raise RuntimeError("no painted view fits the mesh: see views/paint_*.jpg")
    return texture.bake(mesh, views, log), images, {**what, "colours": colours}


def matte(triton: Triton, img: np.ndarray) -> np.ndarray:
    """The object in an RGB studio view as RGBA: SAM 2.1 on the box around what is not white (no ground shadow)."""
    ys, xs = np.nonzero(np.abs(img.astype(int) - 255).max(-1) > 40)   # ground shadows clear 18 but not 40
    mask = triton.segment(img, [[xs.min(), ys.min(), xs.max(), ys.max()]]).reshape(-1, *img.shape[:2])[0]
    return np.dstack([img, mask.astype(np.uint8) * 255])


def silhouette(xy: np.ndarray, faces: np.ndarray, shape) -> np.ndarray:
    sil = np.zeros(shape, np.uint8)
    cv2.fillPoly(sil, np.round(xy[faces]).astype(np.int32), 1)
    return sil


def pose(mesh: trimesh.Trimesh, mask: np.ndarray, azimuths, elevations):
    """The orthographic view of the mesh (look() rows, scale, offset) whose silhouette best overlaps a mask, and the
    overlap: the views are drawn, so their camera is found rather than known."""
    S = 128                                                           # matching resolution (px): speed vs. fit; untuned
    small = cv2.resize(mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(small)
    # 20000 faces drawn per candidate view, for speed; untuned
    faces = mesh.faces[np.random.default_rng(0).choice(len(mesh.faces), min(len(mesh.faces), 20000), replace=False)]
    best = (-1.0, None)
    for a in azimuths:
        for e in elevations:
            R = texture.look(a, e)
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
    plo, phi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)  # 2/98: drops stray points; untuned
    height = max(phi[2] - plo[2], size[2] if size else 0)      # the points miss the far side, not the height
    mesh.apply_scale(height / max(mesh.extents[2], 1e-6))

    # Yaw: the turn whose silhouette, seen from the capture point, best overlaps the points' (both in one view).
    eye = np.array([0, 0, camera_height])
    rays = pts - eye
    d = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    forward = d.mean(0) / np.linalg.norm(d.mean(0))
    # View angle: 3x the points' angular radius + 5 deg (room for any turn's silhouette), 10-120 deg; untuned.
    fov = float(np.clip(3 * np.degrees(np.percentile(np.arccos(np.clip(d @ forward, -1, 1)), 99)) + 5, 10, 120))
    S = 256                                                   # matching resolution (px); untuned
    x, y, ok = project(rays, forward, fov, S)
    seen = np.zeros((S, S), np.uint8)
    keep = ok & (x >= 0) & (x < S) & (y >= 0) & (y < S)
    seen[y[keep].astype(int), x[keep].astype(int)] = 1
    # The points into a solid silhouette: 3 px dilate, 7 px close fill the gaps between them; untuned.
    seen = cv2.morphologyEx(cv2.dilate(seen, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # 30000 faces drawn per yaw, for speed; untuned
    faces = mesh.faces[np.random.default_rng(0).choice(len(mesh.faces), min(len(mesh.faces), 30000), replace=False)]
    best = (-1.0, 0.0)
    for a in np.arange(0, 360, 5):                            # 5 deg yaw steps; untuned
        R = trimesh.transformations.rotation_matrix(np.radians(a), [0, 0, 1])[:3, :3]
        x, y, ok = project(mesh.vertices @ R.T + base - eye, forward, fov, S)
        sil = np.zeros((S, S), np.uint8)
        tri = np.stack([x, y], -1)[faces][ok[faces].all(1)]
        cv2.fillPoly(sil, np.round(tri).astype(np.int32), 1)
        best = max(best, ((sil & seen).sum() / max((sil | seen).sum(), 1), a))
    log(f"  fitted: {height:.2f} m tall, turned {best[1]:.0f} deg, silhouette overlap {best[0]:.2f}")
    return mesh, float(best[1]), base, float(best[0])


def usda(mesh: trimesh.Trimesh, label: str, mass: float) -> str:
    """The mesh as a USD stage (z up, metres): a rigid body with convex-decomposition collision, textured with
    texture.png (next to it) through a UsdPreviewSurface."""
    name = "".join(c if c.isalnum() else "_" for c in label).strip("_").capitalize() or "Object"
    fmt = lambda a: ", ".join(f"({', '.join(f'{v:.6g}' for v in row)})" for row in a)
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
        texCoord2f[] primvars:st = [{fmt(mesh.visual.uv)}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </{name}/Looks/Material>
    }}

    def Scope "Looks"
    {{
        def Material "Material"
        {{
            token outputs:surface.connect = </{name}/Looks/Material/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </{name}/Looks/Material/Texture.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 0.8
                token outputs:surface
            }}

            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @texture.png@
                float2 inputs:st.connect = </{name}/Looks/Material/St.outputs:result>
                token inputs:sourceColorSpace = "sRGB"
                float3 outputs:rgb
            }}

            def Shader "St"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
'''


def write(out: Path, local: trimesh.Trimesh, info: dict, what: dict, yaw: float, base: np.ndarray, overlap: float,
          images: dict):
    (out / "views").mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        cv2.imwrite(str(out / "views" / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])                       # for inspection only; untuned
    mass = float(np.clip(DENSITY * local.convex_hull.volume, 0.02, 50.0))   # kg; clamp bounds untuned
    gltf = local.copy()
    gltf.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))         # z up -> y up
    (out / "model.glb").write_bytes(gltf.export(file_type="glb"))
    local.visual.material.baseColorTexture.save(out / "texture.png")
    (out / "model.usda").write_text(usda(local, info["label"], mass))
    (out / "pose.json").write_text(json.dumps({
        "label": info["label"], "position": [round(float(c), 4) for c in base], "yaw_deg": round(yaw, 1),
        "frame": "the session's scene frame (scene.json): z up, metres, origin on the floor below the capture point",
        "in_room": info.get("in_room"), "size_m": [round(float(c), 4) for c in local.extents],
        "mass_kg": round(mass, 3), "silhouette_overlap": round(overlap, 3), "generated_as": what,
        "generated_by": "Qwen3.5 (what it is, its colours), Qwen-Image-Edit-2511 (drawing, painted views), SAM 2.1, "
                        "TripoSG (shape)"},
        indent=1))


def choose(z: zipfile.ZipFile, scene: dict, out_root: Path) -> list[str]:
    """A checklist of the session's objects in the terminal: which to generate. Each starts ticked unless it was
    generated before."""
    import questionary
    choices = []
    for name in scene["assets"]["objects"]:
        info = json.loads(z.read(f"objects/{name}/info.json"))
        done = (out_root / scene["scene"] / "objects" / name / "model.glb").exists()
        how = "interacted" if info.get("interacted", True) else "added"
        title = f"{name:<16} {info['label']:<14} {how}{', generated before' if done else ''}"
        choices.append(questionary.Choice(title, name, checked=not done))
    picked = questionary.checkbox("Objects to generate (space ticks, enter starts)", choices=choices).ask()
    if picked is None:                                                # Ctrl-C
        raise SystemExit(1)
    return picked


def run(session: Path, out_root: Path, assets: list[str] | None = None, seed: int = 0, log=print) -> list[Path]:
    """Generate the session's objects; assets: only these (each an asset name such as 02_chair, or a label such as
    chair), else every one."""
    triton, generator = Triton(), Endpoint("MESH", "TripoSG")
    log(f"{triton.vlm_}\n{triton.edit_}\n{triton.segment_}\n{generator}")
    with zipfile.ZipFile(session) as z:
        scene = json.loads(z.read("scene.json"))
        h = scene["frame"]["camera_height"]
        written = []
        for name in scene["assets"]["objects"]:
            info = json.loads(z.read(f"objects/{name}/info.json"))
            if assets and not {name, info["label"]} & set(assets):
                continue
            views = sorted(n for n in z.namelist() if n.startswith(f"objects/{name}/views/"))
            if not views or f"objects/{name}/shape.npz" not in z.namelist():
                log(f"{name}: no views recorded, skipped")
                continue
            log(f"{name}: {len(views)} views")
            shape = dict(np.load(io.BytesIO(z.read(f"objects/{name}/shape.npz"))))
            mesh, images, what = model(triton, generator, [z.read(v) for v in views], info["label"], seed, log)
            local, yaw, base, overlap = fit(mesh, shape, h, info.get("size_m"), log)
            out = out_root / scene["scene"] / "objects" / name
            write(out, local, info, what, yaw, base, overlap, images)
            written.append(out)
    log(f"{len(written)} objects -> {out_root / scene['scene'] / 'objects'}")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("session", type=Path, help="a session zip from pano_ideate's viewer (Download session)")
    p.add_argument("--asset", action="append", help="only this asset: its name (02_chair) or label (chair); repeat. "
                   "Without it a checklist asks which to generate (every one when not run in a terminal)")
    p.add_argument("--out", type=Path, default=Path("datasets/asset_gen"))
    p.add_argument("--seed", type=int, default=0, help="another seed draws another object")
    a = p.parse_args()
    if not a.asset and sys.stdin.isatty():
        with zipfile.ZipFile(a.session) as z:
            a.asset = choose(z, json.loads(z.read("scene.json")), a.out)
        if not a.asset:
            raise SystemExit("Nothing ticked.")
    run(a.session, a.out, a.asset, a.seed)
