from __future__ import annotations

import numpy as np

from .numba_compat import njit
from .materials import (
    BEDROCK,
    DIRT,
    GRASS,
    SAND,
    SNOW,
    STONE,
    VERTEX_COMPONENTS,
    VERTICES_PER_FACE,
    _MATERIAL_COLOR_B,
    _MATERIAL_COLOR_G,
    _MATERIAL_COLOR_R,
    _terrain_color,
    _voxel_material_color,
)
from .surface_mesher import _emit_quad
from .voxel_ao import (
    _ambient_occlusion_factor,
    _ao_x_from_planes,
    _ao_y_from_plane,
    _ao_z_from_planes,
    _solid_at_with_boundaries,
)
from .voxel_emit import (
    _emit_quad_components_ao,
    _emit_voxel_face,
)
from .voxel_faces import (
    FACE_BOTTOM,
    FACE_EAST,
    FACE_NORTH,
    FACE_SOUTH,
    FACE_TOP,
    FACE_WEST,
    _build_chunk_face_masks_with_boundaries,
)


@njit(cache=True, fastmath=True)
def build_chunk_vertex_array_from_voxels_with_boundaries(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    height_limit: int,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
    block_size: float = 1.0,
    chunk_y: int = 0,
) -> tuple[np.ndarray, int]:
    face_masks, vertex_count = _build_chunk_face_masks_with_boundaries(blocks, chunk_size, height_limit, top_boundary, bottom_boundary)
    vertices = np.empty((vertex_count, VERTEX_COMPONENTS), dtype=np.float32)
    if vertex_count == 0:
        return vertices, 0

    sample_size = chunk_size + 2
    end = sample_size - 1
    last_y = height_limit - 1
    step = float(block_size)
    origin_x = float(chunk_x * chunk_size) * step
    origin_y = float(chunk_y * chunk_size) * step
    origin_z = float(chunk_z * chunk_size) * step
    vertex_index = 0

    for y in range(height_limit):
        plane = blocks[y]
        mat_plane = materials[y]
        mask_plane = face_masks[y]
        plane_above = blocks[y + 1] if y < last_y else top_boundary
        plane_below = blocks[y - 1] if y > 0 else bottom_boundary
        y0 = origin_y + float(y) * step
        y1 = y0 + step

        z0 = origin_z
        for local_z in range(1, end):
            row = plane[local_z]
            row_mat = mat_plane[local_z]
            row_mask = mask_plane[local_z]
            row_north = plane[local_z - 1]
            row_south = plane[local_z + 1]
            row_above = plane_above[local_z]
            row_below = plane_below[local_z]
            z1 = z0 + step

            x0 = origin_x
            for local_x in range(1, end):
                if row[local_x] == 0:
                    x0 += step
                    continue

                x1 = x0 + step
                mask = row_mask[local_x]
                if mask == 0:
                    x0 = x1
                    continue

                material = int(row_mat[local_x])

                if 0 <= material <= SNOW:
                    cr = _MATERIAL_COLOR_R[material]
                    cg = _MATERIAL_COLOR_G[material]
                    cb = _MATERIAL_COLOR_B[material]
                else:
                    if y <= 14:
                        cr = 0.78
                        cg = 0.71
                        cb = 0.49
                    elif y >= 90:
                        cr = 0.95
                        cg = 0.97
                        cb = 0.98
                    elif y >= 70:
                        cr = 0.60
                        cg = 0.72
                        cb = 0.49
                    elif y >= 40:
                        cr = 0.38
                        cg = 0.64
                        cb = 0.31
                    else:
                        cr = 0.28
                        cg = 0.54
                        cb = 0.22

                if mask & FACE_TOP:
                    top_r = cr
                    top_g = cg
                    top_b = cb
                    ao0 = _ao_y_from_plane(plane_above, local_x, local_z, -1, -1)
                    ao1 = _ao_y_from_plane(plane_above, local_x, local_z, 1, -1)
                    ao2 = _ao_y_from_plane(plane_above, local_x, local_z, 1, 1)
                    ao3 = _ao_y_from_plane(plane_above, local_x, local_z, -1, 1)
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y1, z0,
                        x1, y1, z0,
                        x1, y1, z1,
                        x0, y1, z1,
                        0.0, 1.0, 0.0,
                        top_r * ao0, top_g * ao0, top_b * ao0,
                        top_r * ao1, top_g * ao1, top_b * ao1,
                        top_r * ao2, top_g * ao2, top_b * ao2,
                        top_r * ao3, top_g * ao3, top_b * ao3,
                    )

                if mask & FACE_BOTTOM:
                    bottom_r = cr * 0.50
                    bottom_g = cg * 0.50
                    bottom_b = cb * 0.50
                    ao0 = _ao_y_from_plane(plane_below, local_x, local_z, -1, -1)
                    ao1 = _ao_y_from_plane(plane_below, local_x, local_z, -1, 1)
                    ao2 = _ao_y_from_plane(plane_below, local_x, local_z, 1, 1)
                    ao3 = _ao_y_from_plane(plane_below, local_x, local_z, 1, -1)
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y0, z1,
                        x1, y0, z1,
                        x1, y0, z0,
                        0.0, -1.0, 0.0,
                        bottom_r * ao0, bottom_g * ao0, bottom_b * ao0,
                        bottom_r * ao1, bottom_g * ao1, bottom_b * ao1,
                        bottom_r * ao2, bottom_g * ao2, bottom_b * ao2,
                        bottom_r * ao3, bottom_g * ao3, bottom_b * ao3,
                    )

                if mask & FACE_EAST:
                    east_r = cr * 0.80
                    east_g = cg * 0.80
                    east_b = cb * 0.80
                    sample_x = local_x + 1
                    ao0 = _ao_x_from_planes(plane, plane_below, sample_x, local_z, -1)
                    ao1 = _ao_x_from_planes(plane, plane_above, sample_x, local_z, -1)
                    ao2 = _ao_x_from_planes(plane, plane_above, sample_x, local_z, 1)
                    ao3 = _ao_x_from_planes(plane, plane_below, sample_x, local_z, 1)
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x1, y0, z0,
                        x1, y1, z0,
                        x1, y1, z1,
                        x1, y0, z1,
                        1.0, 0.0, 0.0,
                        east_r * ao0, east_g * ao0, east_b * ao0,
                        east_r * ao1, east_g * ao1, east_b * ao1,
                        east_r * ao2, east_g * ao2, east_b * ao2,
                        east_r * ao3, east_g * ao3, east_b * ao3,
                    )

                if mask & FACE_WEST:
                    west_r = cr * 0.64
                    west_g = cg * 0.64
                    west_b = cb * 0.64
                    sample_x = local_x - 1
                    ao0 = _ao_x_from_planes(plane, plane_below, sample_x, local_z, -1)
                    ao1 = _ao_x_from_planes(plane, plane_below, sample_x, local_z, 1)
                    ao2 = _ao_x_from_planes(plane, plane_above, sample_x, local_z, 1)
                    ao3 = _ao_x_from_planes(plane, plane_above, sample_x, local_z, -1)
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y0, z1,
                        x0, y1, z1,
                        x0, y1, z0,
                        -1.0, 0.0, 0.0,
                        west_r * ao0, west_g * ao0, west_b * ao0,
                        west_r * ao1, west_g * ao1, west_b * ao1,
                        west_r * ao2, west_g * ao2, west_b * ao2,
                        west_r * ao3, west_g * ao3, west_b * ao3,
                    )

                if mask & FACE_SOUTH:
                    south_r = cr * 0.72
                    south_g = cg * 0.72
                    south_b = cb * 0.72
                    sample_z = local_z + 1
                    ao0 = _ao_z_from_planes(plane, plane_below, local_x, sample_z, -1)
                    ao1 = _ao_z_from_planes(plane, plane_below, local_x, sample_z, 1)
                    ao2 = _ao_z_from_planes(plane, plane_above, local_x, sample_z, 1)
                    ao3 = _ao_z_from_planes(plane, plane_above, local_x, sample_z, -1)
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z1,
                        x1, y0, z1,
                        x1, y1, z1,
                        x0, y1, z1,
                        0.0, 0.0, 1.0,
                        south_r * ao0, south_g * ao0, south_b * ao0,
                        south_r * ao1, south_g * ao1, south_b * ao1,
                        south_r * ao2, south_g * ao2, south_b * ao2,
                        south_r * ao3, south_g * ao3, south_b * ao3,
                    )

                if mask & FACE_NORTH:
                    north_r = cr * 0.60
                    north_g = cg * 0.60
                    north_b = cb * 0.60
                    sample_z = local_z - 1
                    ao0 = _ao_z_from_planes(plane, plane_below, local_x, sample_z, -1)
                    ao1 = _ao_z_from_planes(plane, plane_above, local_x, sample_z, -1)
                    ao2 = _ao_z_from_planes(plane, plane_above, local_x, sample_z, 1)
                    ao3 = _ao_z_from_planes(plane, plane_below, local_x, sample_z, 1)
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y1, z0,
                        x1, y1, z0,
                        x1, y0, z0,
                        0.0, 0.0, -1.0,
                        north_r * ao0, north_g * ao0, north_b * ao0,
                        north_r * ao1, north_g * ao1, north_b * ao1,
                        north_r * ao2, north_g * ao2, north_b * ao2,
                        north_r * ao3, north_g * ao3, north_b * ao3,
                    )

                x0 = x1

            z0 = z1

    return vertices, vertex_index


@njit(cache=True, fastmath=True)
def build_chunk_vertex_array_from_voxels(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    height_limit: int,
    block_size: float = 1.0,
    chunk_y: int = 0,
) -> tuple[np.ndarray, int]:
    sample_size = chunk_size + 2
    empty_plane = np.zeros((sample_size, sample_size), dtype=blocks.dtype)
    return build_chunk_vertex_array_from_voxels_with_boundaries(
        blocks,
        materials,
        chunk_x,
        chunk_z,
        chunk_size,
        height_limit,
        empty_plane,
        empty_plane,
        block_size,
        chunk_y,
    )


@njit(cache=True, fastmath=True)
def count_chunk_voxel_vertices_with_boundaries(
    blocks: np.ndarray,
    chunk_size: int,
    height_limit: int,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
) -> int:
    sample_size = chunk_size + 2
    end = sample_size - 1
    last_y = height_limit - 1
    vertex_count = 0

    for y in range(height_limit):
        plane = blocks[y]
        for local_z in range(1, end):
            row = plane[local_z]
            row_north = plane[local_z - 1]
            row_south = plane[local_z + 1]
            if y < last_y:
                row_above = blocks[y + 1][local_z]
            else:
                row_above = top_boundary[local_z]
            if y > 0:
                row_below = blocks[y - 1][local_z]
            else:
                row_below = bottom_boundary[local_z]

            for local_x in range(1, end):
                if row[local_x] == 0:
                    continue
                if row_above[local_x] == 0:
                    vertex_count += VERTICES_PER_FACE
                if row_below[local_x] == 0:
                    vertex_count += VERTICES_PER_FACE
                if row[local_x + 1] == 0:
                    vertex_count += VERTICES_PER_FACE
                if row[local_x - 1] == 0:
                    vertex_count += VERTICES_PER_FACE
                if row_south[local_x] == 0:
                    vertex_count += VERTICES_PER_FACE
                if row_north[local_x] == 0:
                    vertex_count += VERTICES_PER_FACE

    return vertex_count


@njit(cache=True, fastmath=True)
def count_chunk_voxel_vertices(blocks: np.ndarray, chunk_size: int, height_limit: int) -> int:
    sample_size = chunk_size + 2
    empty_plane = np.zeros((sample_size, sample_size), dtype=blocks.dtype)
    return count_chunk_voxel_vertices_with_boundaries(blocks, chunk_size, height_limit, empty_plane, empty_plane)
