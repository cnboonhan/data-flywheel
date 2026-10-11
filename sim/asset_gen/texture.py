"""A texture for a generated mesh, painted onto renders of the mesh itself so every painting fits its shape: the mesh is
rendered grey from a few views around the drawing's own, the image editor paints each render as the drawn object (flat
light, no highlights), and the paintings are baked into a UV texture (xatlas unwrap), weighted by how squarely each
view sees the surface; what no view sees is inpainted in the texture.

Cameras are orthographic: rows of R are image right, image down and the view direction; a point X lands at
X @ R[:2].T * scale + offset (pixels).
"""

import cv2
import numpy as np
import trimesh

SIZE = 1024             # px, the renders and paintings: the editor's own output size
TEXTURE = 1024          # px, the texture's side (untuned; 2048 shows finer print at 4x the file size)
MIN_FIT = 0.8           # a painting whose outline (SAM) overlaps the render's less redrew the object: not used.
                        # Good ones overlap 0.94-0.99; one that ignored the render (a fresh pose) about 0.4


def look(azimuth: float, elevation: float) -> np.ndarray:
    """Rows: image right, image down, view direction, of a camera circling z (azimuth 0 looks along +y)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    f = np.array([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), -np.sin(e)])
    right = np.cross(f, [0, 0, 1.0]) if abs(f[2]) < 0.999 else np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def framing(mesh: trimesh.Trimesh, size: int = SIZE, fill: float = 0.66):
    """scale and offset that put the mesh, seen from any side, in the middle `fill` of a size-px view (fill matches
    DRAW's "two thirds of the frame")."""
    radius = np.linalg.norm(mesh.vertices - mesh.bounds.mean(0), axis=1).max()
    scale = fill * size / (2 * radius)
    return scale, lambda R: size / 2 - scale * (mesh.bounds.mean(0) @ R[:2].T)


def raster(mesh: trimesh.Trimesh, R: np.ndarray, scale: float, offset: np.ndarray, size: int = SIZE):
    """The nearest face per pixel (-1: none) and the depth per pixel, far faces drawn first."""
    xy = mesh.vertices @ R[:2].T * scale + offset
    depth = mesh.vertices @ R[2]
    ids = np.full((size, size), -1, np.int32)
    tri = np.round(xy[mesh.faces] * 4).astype(np.int32)              # 2 bits of subpixel precision
    for f in np.argsort(-depth[mesh.faces].mean(1)):
        cv2.fillConvexPoly(ids, tri[f], int(f), shift=2)
    z = np.where(ids >= 0, depth[mesh.faces].mean(1)[ids.clip(0)], np.inf)
    return ids, z


def grey(mesh: trimesh.Trimesh, ids: np.ndarray, R: np.ndarray) -> np.ndarray:
    """A shaded grey render (RGB on white) from a face-id raster: what the editor paints."""
    shade = 0.3 + 0.6 * np.clip(-(mesh.face_normals @ R[2]), 0, 1)   # light from the camera; untuned
    img = np.full(ids.shape + (3,), 255, np.uint8)
    img[ids >= 0] = (200 * shade[ids[ids >= 0]])[:, None].astype(np.uint8)
    return img


def unwrap(mesh: trimesh.Trimesh):
    """The mesh split along xatlas's seams, and its uv per vertex."""
    import xatlas
    vmap, faces, uv = xatlas.parametrize(mesh.vertices, mesh.faces)
    return trimesh.Trimesh(mesh.vertices[vmap], faces, process=False), uv


def texels(mesh: trimesh.Trimesh, uv: np.ndarray, size: int = TEXTURE):
    """Each covered texel's 3D point and normal (and the texel's row, column)."""
    ids = np.full((size, size), -1, np.int32)
    tuv = uv[mesh.faces] * [size, size]
    tuv[..., 1] = size - tuv[..., 1]                                  # image rows run down, v up
    for f, t in enumerate(np.round(tuv * 4).astype(np.int32)):
        cv2.fillConvexPoly(ids, t, f, shift=2)
    rows, cols = np.nonzero(ids >= 0)
    f = ids[rows, cols]
    p = np.stack([cols + 0.5, rows + 0.5], -1)
    a, b, c = tuv[f, 0], tuv[f, 1], tuv[f, 2]
    v0, v1, v2 = b - a, c - a, p - a                                  # barycentric weights of each texel centre
    d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / d
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / d
    w = np.clip(np.stack([1 - w1 - w2, w1, w2], -1), 0, 1)
    w /= w.sum(1, keepdims=True)
    tri = mesh.vertices[mesh.faces[f]]
    return rows, cols, np.einsum("nk,nkd->nd", w, tri), mesh.face_normals[f]


def bake(mesh: trimesh.Trimesh, views: list, log=print) -> trimesh.Trimesh:
    """The mesh unwrapped and textured from painted views (RGB painting, R, scale, offset, face-id raster, depth)."""
    out, uv = unwrap(mesh)
    rows, cols, X, n = texels(out, uv)
    colour, weight = np.zeros((len(X), 3)), np.zeros(len(X))
    extent = np.ptp(mesh.vertices, 0).max()
    for img, R, scale, offset, ids, z in views:
        xy = X @ R[:2].T * scale + offset
        x, y = np.clip(np.round(xy).astype(int), 0, SIZE - 1).T
        seen = X @ R[2] <= z[y, x] + 0.01 * extent                     # nearest surface there: not hidden (1%, untuned)
        inner = cv2.erode((ids >= 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # edges blend
        cos = np.clip(-(n @ R[2]), 0, 1)
        w = np.where(seen & inner[y, x], cos ** 2, 0)                 # squarely seen counts most (cos^2, untuned)
        colour += img[y, x] * w[:, None]
        weight += w
    from scipy.spatial import cKDTree
    have = weight > 1e-3
    colour[have] /= weight[have, None]
    log(f"  texture: {have.mean():.0%} of the surface painted, the rest from the nearest painted point")
    # Unseen texels (an underside) take the nearest painted point in 3D: their texture neighbours can be anywhere.
    if have.any() and not have.all():
        colour[~have] = colour[have][cKDTree(X[have]).query(X[~have])[1]]
    tex = np.zeros((TEXTURE, TEXTURE, 3), np.uint8)
    tex[rows, cols] = colour.clip(0, 255).astype(np.uint8)
    gutter = np.ones((TEXTURE, TEXTURE), np.uint8)
    gutter[rows, cols] = 0
    tex = cv2.inpaint(tex, gutter, 3, cv2.INPAINT_TELEA)              # between charts, so filtering doesn't bleed black
    from PIL import Image
    out.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(tex), metallicFactor=0.0, roughnessFactor=0.8))   # plain, matte: untuned
    return out


def silhouette_fit(painted: np.ndarray, ids: np.ndarray) -> float:
    """How well a painting's object outline (a mask, e.g. SAM's) overlaps the render's (IoU)."""
    sil = ids >= 0
    return (painted & sil).sum() / max((painted | sil).sum(), 1)
