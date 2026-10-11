# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""The background of a pano_ideate session as a textured mesh room: the erased panorama on its own depth, for renderers
that can't draw Gaussian splats (Isaac Sim on GPUs without RT cores, such as the GB300 nodes). Sharper up close than
nothing, coarser than colmap_splat's splat; it shows the photo as it is (emissive), whatever the sim's lights.

    uv run sim/asset_gen/background.py <scene>_session.zip [--out datasets/asset_gen] [--grid 512] [--jump 0.1]

Writes <out>/<scene>/background/: scene.usda (z up, metres, the session's scene frame) and texture.jpg. scene.usda has
the same layout as colmap_splat's: the room, and an invisible collision floor `floor` under the capture area, so
sim/isaaclab_arena loads it as a background the same way (ARENA_SPLAT_SCENE). The room is also a static triangle-mesh
collider, so objects rest on its counters and tables as the panorama saw them.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

FAR = 30.0          # m; farther depth is sky or noise (colmap_splat crops at 30 m too)
FLOOR_BAND = 0.03   # m; points this close to z = 0 are floor, for the collision floor's extent (untuned)


def mesh(distance: np.ndarray, R: np.ndarray, height: float, grid: int, jump: float | None = None):
    """Vertices (scene frame), uvs and triangles of the panorama on its depth, on a grid x grid/2 lattice of pixel
    corners (the seam column repeated so the texture wraps). jump: leave out triangles whose corner distances differ by
    more than this fraction (they bridge a depth edge, an object in front of a wall); kept by default, since a sim
    camera near the capture point sees them edge-on and their gaps show as lines of sky."""
    h, w = distance.shape
    u, v = np.linspace(0, 1, grid + 1), np.linspace(0, 1, grid // 2 + 1)
    U, V = np.meshgrid(u, v)
    # Depth at each corner: the nearest pixel's (averaging across an edge makes a surface that isn't there).
    d = distance[np.clip((V * h).astype(int), 0, h - 1), np.clip((U * w).astype(int) % w, 0, w - 1)]
    d = np.minimum(d, FAR)
    theta, phi = (1 - U) * 2 * np.pi, V * np.pi                      # sim/pano_ideate/common.py's directions()
    dirs = np.stack([-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)], -1)
    xyz = (d[..., None] * dirs) @ R.T + [0, 0, height]
    cols = grid + 1
    i = np.arange(grid // 2)[:, None] * cols + np.arange(grid)[None, :]
    tris = np.concatenate([np.stack([i, i + cols, i + 1], -1).reshape(-1, 3),
                           np.stack([i + 1, i + cols, i + cols + 1], -1).reshape(-1, 3)])
    dd = d.reshape(-1)[tris]
    keep = dd.max(1) < FAR
    if jump is not None:
        keep &= dd.max(1) / np.maximum(dd.min(1), 1e-6) - 1 < jump
    return xyz.reshape(-1, 3), np.stack([U, 1 - V], -1).reshape(-1, 2), tris[keep]


def usda(xyz: np.ndarray, uv: np.ndarray, tris: np.ndarray, floor: tuple) -> str:
    fmt = lambda a: ", ".join(f"({', '.join(f'{v:.4g}' for v in row)})" for row in a)
    (cx, cy), (sx, sy) = floor
    return f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Mesh "room" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "none"
        uniform bool doubleSided = 1
        int[] faceVertexCounts = [{", ".join(["3"] * len(tris))}]
        int[] faceVertexIndices = [{", ".join(str(i) for i in tris.ravel())}]
        point3f[] points = [{fmt(xyz)}]
        texCoord2f[] primvars:st = [{fmt(uv)}] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </World/Looks/Photo>
    }}

    def Cube "floor" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        double size = 1
        token visibility = "invisible"
        double3 xformOp:translate = ({cx:.3f}, {cy:.3f}, -0.05)
        float3 xformOp:scale = ({sx:.3f}, {sy:.3f}, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}

    def Scope "Looks"
    {{
        def Material "Photo"
        {{
            token outputs:surface.connect = </World/Looks/Photo/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0, 0, 0)
                color3f inputs:emissiveColor.connect = </World/Looks/Photo/Texture.outputs:rgb>
                float inputs:roughness = 1
                token outputs:surface
            }}

            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @texture.jpg@
                float2 inputs:st.connect = </World/Looks/Photo/St.outputs:result>
                token inputs:sourceColorSpace = "sRGB"
                token inputs:wrapS = "repeat"
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


def run(session: Path, out_root: Path, grid: int = 512, jump: float | None = None, log=print) -> Path:
    with zipfile.ZipFile(session) as z:
        scene = json.loads(z.read("scene.json"))
        pano = cv2.imdecode(np.frombuffer(z.read("background/pano.jpg"), np.uint8), cv2.IMREAD_COLOR)
        distance = np.load(io.BytesIO(z.read("background/distance.npy")))
    frame = scene["frame"]
    xyz, uv, tris = mesh(distance, np.array(frame["R_moge_to_level"]), frame["camera_height"], grid, jump)
    flat = xyz[np.abs(xyz[:, 2]) < FLOOR_BAND]
    lo, hi = np.percentile(flat[:, :2], 2, 0), np.percentile(flat[:, :2], 98, 0)   # the floor, without stray points
    out = out_root / scene["scene"] / "background"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "texture.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "scene.usda").write_text(usda(xyz, uv, tris, ((lo + hi) / 2, hi - lo)))
    log(f"{len(tris)} triangles, floor {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} m -> {out / 'scene.usda'}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("session", type=Path, help="a session zip from pano_ideate's viewer (Download session)")
    p.add_argument("--out", type=Path, default=Path("datasets/asset_gen"))
    p.add_argument("--grid", type=int, default=512, help="vertices around the panorama (half as many top to bottom); "
                   "512: ~260k triangles, a few cm at room distances, the texture carries the detail")
    p.add_argument("--jump", type=float, help="leave out triangles across depth edges whose corner distances differ by "
                   "more than this fraction (0.1 as the viewer), for cameras far from the capture point")
    a = p.parse_args()
    run(a.session, a.out, a.grid, a.jump)
