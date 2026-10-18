"""The four Gohlke `transformations.py` functions the DSL's math_utils imports, same signatures and
conventions (4x4 homogeneous, row-major, column vectors)."""

import math

import numpy as np


def identity_matrix():
    return np.identity(4)


def translation_matrix(direction):
    m = np.identity(4)
    m[:3, 3] = np.asarray(direction, dtype=float)[:3]
    return m


def _unit(v):
    v = np.asarray(v, dtype=float)[:3]
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return v / n


def rotation_matrix(angle, direction, point=None):
    sa, ca = math.sin(angle), math.cos(angle)
    d = _unit(direction)
    r = np.diag([ca, ca, ca]) + np.outer(d, d) * (1.0 - ca)
    d *= sa
    r += np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    m = np.identity(4)
    m[:3, :3] = r
    if point is not None:
        p = np.asarray(point, dtype=float)[:3]
        m[:3, 3] = p - r @ p
    return m


def reflection_matrix(point, normal):
    n = _unit(normal)
    m = np.identity(4)
    m[:3, :3] -= 2.0 * np.outer(n, n)
    m[:3, 3] = (2.0 * np.dot(np.asarray(point, dtype=float)[:3], n)) * n
    return m
