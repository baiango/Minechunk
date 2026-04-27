from __future__ import annotations

import numpy as np

from .numba_compat import njit
from .surface_mesher import _emit_quad


@njit(cache=True, fastmath=True)
def _emit_voxel_face(
    vertices: np.ndarray,
    vertex_index: int,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    normal: tuple[float, float, float],
    color: tuple[float, float, float],
) -> int:
    return _emit_quad(vertices, vertex_index, p0, p1, p2, p3, normal, color)


@njit(cache=True, fastmath=True)
def _emit_quad_components_ao(
    vertices: np.ndarray,
    vertex_index: int,
    p0x: float, p0y: float, p0z: float,
    p1x: float, p1y: float, p1z: float,
    p2x: float, p2y: float, p2z: float,
    p3x: float, p3y: float, p3z: float,
    nx: float, ny: float, nz: float,
    c0r: float, c0g: float, c0b: float,
    c1r: float, c1g: float, c1b: float,
    c2r: float, c2g: float, c2b: float,
    c3r: float, c3g: float, c3b: float,
) -> int:
    row = vertices[vertex_index]
    row[0] = p0x
    row[1] = p0y
    row[2] = p0z
    row[3] = 1.0
    row[4] = nx
    row[5] = ny
    row[6] = nz
    row[7] = 0.0
    row[8] = c0r
    row[9] = c0g
    row[10] = c0b
    row[11] = 1.0
    vertex_index += 1

    row = vertices[vertex_index]
    row[0] = p1x
    row[1] = p1y
    row[2] = p1z
    row[3] = 1.0
    row[4] = nx
    row[5] = ny
    row[6] = nz
    row[7] = 0.0
    row[8] = c1r
    row[9] = c1g
    row[10] = c1b
    row[11] = 1.0
    vertex_index += 1

    row = vertices[vertex_index]
    row[0] = p2x
    row[1] = p2y
    row[2] = p2z
    row[3] = 1.0
    row[4] = nx
    row[5] = ny
    row[6] = nz
    row[7] = 0.0
    row[8] = c2r
    row[9] = c2g
    row[10] = c2b
    row[11] = 1.0
    vertex_index += 1

    row = vertices[vertex_index]
    row[0] = p0x
    row[1] = p0y
    row[2] = p0z
    row[3] = 1.0
    row[4] = nx
    row[5] = ny
    row[6] = nz
    row[7] = 0.0
    row[8] = c0r
    row[9] = c0g
    row[10] = c0b
    row[11] = 1.0
    vertex_index += 1

    row = vertices[vertex_index]
    row[0] = p2x
    row[1] = p2y
    row[2] = p2z
    row[3] = 1.0
    row[4] = nx
    row[5] = ny
    row[6] = nz
    row[7] = 0.0
    row[8] = c2r
    row[9] = c2g
    row[10] = c2b
    row[11] = 1.0
    vertex_index += 1

    row = vertices[vertex_index]
    row[0] = p3x
    row[1] = p3y
    row[2] = p3z
    row[3] = 1.0
    row[4] = nx
    row[5] = ny
    row[6] = nz
    row[7] = 0.0
    row[8] = c3r
    row[9] = c3g
    row[10] = c3b
    row[11] = 1.0
    vertex_index += 1

    return vertex_index
