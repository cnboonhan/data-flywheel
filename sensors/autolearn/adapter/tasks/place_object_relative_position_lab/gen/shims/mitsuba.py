"""Stand-in for the `mitsuba` package on the generator's CPU layout path.

geniesim_generator imports Mitsuba for rendering, but `app.py` only needs it for bounding boxes:
`engine/utils/mitsuba_utils.compute_bboxes` builds a Mitsuba scene from the shape dicts and reads
`shape.bbox()`. Neither the geniesim3 container nor any host env ships `mitsuba` and nothing may be
downloaded, so this module provides exactly the surface the layout path touches:
`set_variant`, `scalar_rgb.Transform4f`, `ScalarPoint3f`, `load_dict(...).shapes()[i].bbox()`, plus
attribute access for type hints (`mi.Sensor`, `mi.Scene`). Rendering functions are not provided.

Box conventions follow Mitsuba's shape plugins: `cube` is [-1, 1]^3 under `to_world`, `sphere` is the
unit sphere under `to_world` (its box is the transformed [-1, 1]^3, exact for uniform scale), `cylinder`
is `p0`-`p1` with `radius`, both points taken through `to_world`.
"""

import numpy as np


def set_variant(*_args, **_kwargs):
    pass


def variant():
    return "scalar_rgb"


class Transform4f:
    def __init__(self, m=None):
        if m is None:
            self.matrix = np.eye(4)
        else:
            self.matrix = np.asarray(getattr(m, "matrix", m), dtype=float).reshape(4, 4)

    def __matmul__(self, other):
        o = other.matrix if isinstance(other, Transform4f) else np.asarray(other, dtype=float)
        return Transform4f(self.matrix @ o)

    def __array__(self, dtype=None, copy=None):
        return self.matrix if dtype is None else self.matrix.astype(dtype)

    def __repr__(self):
        return f"Transform4f({self.matrix.tolist()})"


class scalar_rgb:  # noqa: N801  (mirrors `mitsuba.scalar_rgb`)
    Transform4f = Transform4f


ScalarTransform4f = Transform4f


def ScalarPoint3f(*p):  # noqa: N802
    if len(p) == 1:
        p = p[0]
    return np.asarray(p, dtype=float)


class _BBox:
    def __init__(self, lo, hi):
        self.min = np.asarray(lo, dtype=float)
        self.max = np.asarray(hi, dtype=float)


class _Shape:
    def __init__(self, lo, hi):
        self._bbox = _BBox(lo, hi)

    def bbox(self):
        return self._bbox


def _apply(m, pts):
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ m.T
    return h[:, :3] / h[:, 3:4]


_UNIT_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def _shape_box(d):
    m = np.asarray(d.get("to_world", np.eye(4)), dtype=float).reshape(4, 4)
    t = d["type"]
    if t in ("cube", "sphere"):
        pts = _apply(m, _UNIT_CORNERS)
        return pts.min(axis=0), pts.max(axis=0)
    if t == "cylinder":
        p = _apply(m, [d["p0"], d["p1"]])
        # radius scales with the largest axis scale of to_world; exact for axis-aligned cylinders
        r = float(d.get("radius", 1.0)) * float(np.max(np.linalg.norm(m[:3, :3], axis=0)))
        return p.min(axis=0) - r, p.max(axis=0) + r
    return None


class _Scene:
    def __init__(self, shapes):
        self._shapes = shapes

    def shapes(self):
        return self._shapes


def load_dict(scene_dict):
    shapes = []
    for k, v in scene_dict.items():
        if k == "type" or not isinstance(v, dict):
            continue
        box = _shape_box(v) if "type" in v else None
        if box is not None:
            shapes.append(_Shape(*box))
    return _Scene(shapes)


def __getattr__(name):
    # Type-hint placeholders (mi.Sensor, mi.Scene, mi.Bitmap, ...). Calling one is an error on purpose.
    if name.startswith("__"):
        raise AttributeError(name)

    class _Unavailable:
        def __init__(self, *a, **k):
            raise RuntimeError(f"mitsuba.{name} is not available in the stand-in module (no rendering)")

    _Unavailable.__name__ = name
    return _Unavailable
