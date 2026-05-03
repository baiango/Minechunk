from __future__ import annotations

import numpy as np

from .numba_compat import njit
from .surface_mesher import _emit_quad


@njit(cache=True, fastmath=True, inline="always")
def _write_vertex_components(vertices: np.ndarray, vertex_index: int, px: float, py: float, pz: float, nx: float, ny: float, nz: float, cr: float, cg: float, cb: float) -> int:
    row = vertices[vertex_index]
    row[0] = px
    row[1] = py
    row[2] = pz
    row[3] = nx
    row[4] = ny
    row[5] = nz
    row[6] = cr
    row[7] = cg
    row[8] = cb
    return vertex_index + 1


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
    vertex_index = _write_vertex_components(vertices, vertex_index, p0x, p0y, p0z, nx, ny, nz, c0r, c0g, c0b)
    vertex_index = _write_vertex_components(vertices, vertex_index, p1x, p1y, p1z, nx, ny, nz, c1r, c1g, c1b)
    vertex_index = _write_vertex_components(vertices, vertex_index, p2x, p2y, p2z, nx, ny, nz, c2r, c2g, c2b)
    vertex_index = _write_vertex_components(vertices, vertex_index, p0x, p0y, p0z, nx, ny, nz, c0r, c0g, c0b)
    vertex_index = _write_vertex_components(vertices, vertex_index, p2x, p2y, p2z, nx, ny, nz, c2r, c2g, c2b)
    vertex_index = _write_vertex_components(vertices, vertex_index, p3x, p3y, p3z, nx, ny, nz, c3r, c3g, c3b)
    return vertex_index


@njit(cache=True, fastmath=True, inline="always")
def _emit_quad_components_uniform_color(
    vertices: np.ndarray,
    vertex_index: int,
    p0x: float, p0y: float, p0z: float,
    p1x: float, p1y: float, p1z: float,
    p2x: float, p2y: float, p2z: float,
    p3x: float, p3y: float, p3z: float,
    nx: float, ny: float, nz: float,
    cr: float, cg: float, cb: float,
) -> int:
    vertex_index = _write_vertex_components(vertices, vertex_index, p0x, p0y, p0z, nx, ny, nz, cr, cg, cb)
    vertex_index = _write_vertex_components(vertices, vertex_index, p1x, p1y, p1z, nx, ny, nz, cr, cg, cb)
    vertex_index = _write_vertex_components(vertices, vertex_index, p2x, p2y, p2z, nx, ny, nz, cr, cg, cb)
    vertex_index = _write_vertex_components(vertices, vertex_index, p0x, p0y, p0z, nx, ny, nz, cr, cg, cb)
    vertex_index = _write_vertex_components(vertices, vertex_index, p2x, p2y, p2z, nx, ny, nz, cr, cg, cb)
    vertex_index = _write_vertex_components(vertices, vertex_index, p3x, p3y, p3z, nx, ny, nz, cr, cg, cb)
    return vertex_index
