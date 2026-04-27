from __future__ import annotations

import numpy as np

from .numba_compat import njit
from .materials import (
    BEDROCK,
    DIRT,
    GRASS,
    MAX_FACES_PER_CELL,
    SAND,
    SNOW,
    STONE,
    VERTEX_COMPONENTS,
    VERTICES_PER_FACE,
    _scale_color,
    _terrain_color,
)

@njit(cache=True, fastmath=True)
def _write_vertex(
    vertices: np.ndarray,
    vertex_index: int,
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    color: tuple[float, float, float],
    alpha: float,
) -> int:
    row = vertices[vertex_index]
    row[0] = position[0]
    row[1] = position[1]
    row[2] = position[2]
    row[3] = 1.0
    row[4] = normal[0]
    row[5] = normal[1]
    row[6] = normal[2]
    row[7] = 0.0
    row[8] = color[0]
    row[9] = color[1]
    row[10] = color[2]
    row[11] = alpha
    return vertex_index + 1


@njit(cache=True, fastmath=True)
def _emit_triangle(
    vertices: np.ndarray,
    vertex_index: int,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    normal: tuple[float, float, float],
    color: tuple[float, float, float],
) -> int:
    vertex_index = _write_vertex(vertices, vertex_index, a, normal, color, 1.0)
    vertex_index = _write_vertex(vertices, vertex_index, b, normal, color, 1.0)
    vertex_index = _write_vertex(vertices, vertex_index, c, normal, color, 1.0)
    return vertex_index


@njit(cache=True, fastmath=True)
def _emit_quad(
    vertices: np.ndarray,
    vertex_index: int,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    normal: tuple[float, float, float],
    color: tuple[float, float, float],
) -> int:
    vertex_index = _emit_triangle(vertices, vertex_index, p0, p1, p2, normal, color)
    vertex_index = _emit_triangle(vertices, vertex_index, p0, p2, p3, normal, color)
    return vertex_index


@njit(cache=True, fastmath=True)
def build_chunk_vertex_array(
    height_grid: np.ndarray,
    material_grid: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    block_size: float = 1.0,
) -> tuple[np.ndarray, int]:
    stride = chunk_size + 2
    max_vertices = chunk_size * chunk_size * MAX_FACES_PER_CELL * VERTICES_PER_FACE
    vertices = np.empty((max_vertices, VERTEX_COMPONENTS), dtype=np.float32)
    vertex_index = 0
    step = float(block_size)
    origin_x = float(chunk_x * chunk_size) * step
    origin_z = float(chunk_z * chunk_size) * step

    for local_z in range(chunk_size):
        for local_x in range(chunk_size):
            cell_index = (local_z + 1) * stride + (local_x + 1)
            height = int(height_grid[cell_index])
            if height == 0:
                continue

            west_height = int(height_grid[cell_index - 1])
            east_height = int(height_grid[cell_index + 1])
            north_height = int(height_grid[cell_index - stride])
            south_height = int(height_grid[cell_index + stride])
            material = int(material_grid[cell_index])

            x0 = origin_x + float(local_x) * step
            x1 = x0 + step
            z0 = origin_z + float(local_z) * step
            z1 = z0 + step
            y0 = float(height) * step
            west_y = float(west_height) * step
            east_y = float(east_height) * step
            north_y = float(north_height) * step
            south_y = float(south_height) * step

            if material == BEDROCK:
                base = (0.24, 0.22, 0.20)
            elif material == STONE:
                base = (0.42, 0.40, 0.38)
            elif material == DIRT:
                base = (0.47, 0.31, 0.18)
            elif material == GRASS:
                base = (0.31, 0.68, 0.24)
            elif material == SAND:
                base = (0.78, 0.71, 0.49)
            elif material == SNOW:
                base = (0.95, 0.97, 0.98)
            else:
                base = _terrain_color(height)

            top = base
            east = _scale_color(base, 0.80)
            south = _scale_color(base, 0.72)
            west = _scale_color(base, 0.64)
            north = _scale_color(base, 0.60)

            vertex_index = _emit_quad(
                vertices,
                vertex_index,
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y0, z1),
                (x0, y0, z1),
                (0.0, 1.0, 0.0),
                top,
            )

            if height > east_height:
                vertex_index = _emit_quad(
                    vertices,
                    vertex_index,
                    (x1, east_y, z0),
                    (x1, y0, z0),
                    (x1, y0, z1),
                    (x1, east_y, z1),
                    (1.0, 0.0, 0.0),
                    east,
                )

            if height > west_height:
                vertex_index = _emit_quad(
                    vertices,
                    vertex_index,
                    (x0, west_y, z0),
                    (x0, y0, z0),
                    (x0, y0, z1),
                    (x0, west_y, z1),
                    (-1.0, 0.0, 0.0),
                    west,
                )

            if height > south_height:
                vertex_index = _emit_quad(
                    vertices,
                    vertex_index,
                    (x0, south_y, z1),
                    (x0, y0, z1),
                    (x1, y0, z1),
                    (x1, south_y, z1),
                    (0.0, 0.0, 1.0),
                    south,
                )

            if height > north_height:
                vertex_index = _emit_quad(
                    vertices,
                    vertex_index,
                    (x0, north_y, z0),
                    (x0, y0, z0),
                    (x1, y0, z0),
                    (x1, north_y, z0),
                    (0.0, 0.0, -1.0),
                    north,
                )

    return vertices, vertex_index

