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
    _emit_quad_components_uniform_color,
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


_U64_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=True, inline="always")
def _low_u64_bits(bit_count: int) -> np.uint64:
    if bit_count <= 0:
        return np.uint64(0)
    if bit_count >= 64:
        return _U64_ALL
    return (np.uint64(1) << bit_count) - np.uint64(1)


@njit(cache=True, inline="always")
def _valid_chunk_bit_mask(chunk_size: int) -> np.uint64:
    if chunk_size >= 64:
        return _U64_ALL
    return _low_u64_bits(chunk_size)


@njit(cache=True, inline="always")
def _ctz_u64(value: np.uint64) -> int:
    bits = np.uint64(value)
    index = 0
    if (bits & np.uint64(0xFFFFFFFF)) == 0:
        bits >>= 32
        index += 32
    if (bits & np.uint64(0xFFFF)) == 0:
        bits >>= 16
        index += 16
    if (bits & np.uint64(0xFF)) == 0:
        bits >>= 8
        index += 8
    if (bits & np.uint64(0xF)) == 0:
        bits >>= 4
        index += 4
    if (bits & np.uint64(0x3)) == 0:
        bits >>= 2
        index += 2
    if (bits & np.uint64(0x1)) == 0:
        index += 1
    return index


@njit(cache=True, inline="always")
def _popcount_u64(value: np.uint64) -> int:
    bits = np.uint64(value)
    count = 0
    while bits != 0:
        bits &= bits - np.uint64(1)
        count += 1
    return count


@njit(cache=True, inline="always")
def _row_occupancy_bits(row: np.ndarray, chunk_size: int) -> np.uint64:
    bits = np.uint64(0)
    for bit_index in range(chunk_size):
        if row[bit_index + 1] != 0:
            bits |= np.uint64(1) << bit_index
    return bits


@njit(cache=True, inline="always")
def _row_material_break_bits(row_materials: np.ndarray, chunk_size: int) -> np.uint64:
    bits = np.uint64(0)
    previous = int(row_materials[1])
    for bit_index in range(1, chunk_size):
        material = int(row_materials[bit_index + 1])
        if material != previous:
            bits |= np.uint64(1) << bit_index
        previous = material
    return bits


@njit(cache=True, inline="always")
def _east_neighbor_bits(row: np.ndarray, row_bits: np.uint64, chunk_size: int) -> np.uint64:
    bits = row_bits >> 1
    if row[chunk_size + 1] != 0:
        bits |= np.uint64(1) << (chunk_size - 1)
    return bits


@njit(cache=True, inline="always")
def _west_neighbor_bits(row: np.ndarray, row_bits: np.uint64, valid_mask: np.uint64) -> np.uint64:
    bits = (row_bits << 1) & valid_mask
    if row[0] != 0:
        bits |= np.uint64(1)
    return bits


@njit(cache=True, fastmath=True)
def _chunk_unit_face_vertex_count_from_bits(
    blocks: np.ndarray,
    chunk_size: int,
    height_limit: int,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
) -> int:
    end = chunk_size + 1
    last_y = height_limit - 1
    valid_mask = _valid_chunk_bit_mask(chunk_size)
    face_count = 0

    for y in range(height_limit):
        plane = blocks[y]
        plane_above = blocks[y + 1] if y < last_y else top_boundary
        plane_below = blocks[y - 1] if y > 0 else bottom_boundary
        for local_z in range(1, end):
            row_bits = _row_occupancy_bits(plane[local_z], chunk_size)
            if row_bits == 0:
                continue
            above_bits = _row_occupancy_bits(plane_above[local_z], chunk_size)
            below_bits = _row_occupancy_bits(plane_below[local_z], chunk_size)
            south_bits = _row_occupancy_bits(plane[local_z + 1], chunk_size)
            north_bits = _row_occupancy_bits(plane[local_z - 1], chunk_size)

            face_count += _popcount_u64(row_bits & (~above_bits & valid_mask))
            face_count += _popcount_u64(row_bits & (~below_bits & valid_mask))
            face_count += _popcount_u64(row_bits & (~south_bits & valid_mask))
            face_count += _popcount_u64(row_bits & (~north_bits & valid_mask))
            face_count += _popcount_u64(row_bits & (~_east_neighbor_bits(plane[local_z], row_bits, chunk_size) & valid_mask))
            face_count += _popcount_u64(row_bits & (~_west_neighbor_bits(plane[local_z], row_bits, valid_mask) & valid_mask))

    return face_count * VERTICES_PER_FACE


@njit(cache=True, fastmath=True, inline="always")
def _material_rgb(material: int, y: int) -> tuple[float, float, float]:
    if 0 <= material <= SNOW:
        return _MATERIAL_COLOR_R[material], _MATERIAL_COLOR_G[material], _MATERIAL_COLOR_B[material]
    if y <= 14:
        return 0.78, 0.71, 0.49
    if y >= 90:
        return 0.95, 0.97, 0.98
    if y >= 70:
        return 0.60, 0.72, 0.49
    if y >= 40:
        return 0.38, 0.64, 0.31
    return 0.28, 0.54, 0.22


@njit(cache=True, fastmath=True, inline="always")
def _emit_top_run(vertices, vertex_index, plane_above, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step):
    cr, cg, cb = _material_rgb(material, y)
    x0 = origin_x + float(start_x - 1) * step
    x1 = origin_x + float(end_x - 1) * step
    y1 = origin_y + float(y + 1) * step
    z0 = origin_z + float(local_z - 1) * step
    z1 = origin_z + float(local_z) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1,
        0.0, 1.0, 0.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_bottom_run(vertices, vertex_index, plane_below, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step):
    cr, cg, cb = _material_rgb(material, y)
    x0 = origin_x + float(start_x - 1) * step
    x1 = origin_x + float(end_x - 1) * step
    y0 = origin_y + float(y) * step
    z0 = origin_z + float(local_z - 1) * step
    z1 = origin_z + float(local_z) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y0, z0, x0, y0, z1, x1, y0, z1, x1, y0, z0,
        0.0, -1.0, 0.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_south_run(vertices, vertex_index, plane, plane_above, plane_below, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step):
    cr, cg, cb = _material_rgb(material, y)
    x0 = origin_x + float(start_x - 1) * step
    x1 = origin_x + float(end_x - 1) * step
    y0 = origin_y + float(y) * step
    y1 = y0 + step
    z1 = origin_z + float(local_z) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
        0.0, 0.0, 1.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_north_run(vertices, vertex_index, plane, plane_above, plane_below, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step):
    cr, cg, cb = _material_rgb(material, y)
    x0 = origin_x + float(start_x - 1) * step
    x1 = origin_x + float(end_x - 1) * step
    y0 = origin_y + float(y) * step
    y1 = y0 + step
    z0 = origin_z + float(local_z - 1) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y0, z0, x0, y1, z0, x1, y1, z0, x1, y0, z0,
        0.0, 0.0, -1.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_east_run(vertices, vertex_index, plane, plane_above, plane_below, material, y, local_x, start_z, end_z, origin_x, origin_y, origin_z, step):
    cr, cg, cb = _material_rgb(material, y)
    x1 = origin_x + float(local_x) * step
    y0 = origin_y + float(y) * step
    y1 = y0 + step
    z0 = origin_z + float(start_z - 1) * step
    z1 = origin_z + float(end_z - 1) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1,
        1.0, 0.0, 0.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_west_run(vertices, vertex_index, plane, plane_above, plane_below, material, y, local_x, start_z, end_z, origin_x, origin_y, origin_z, step):
    cr, cg, cb = _material_rgb(material, y)
    x0 = origin_x + float(local_x - 1) * step
    y0 = origin_y + float(y) * step
    y1 = y0 + step
    z0 = origin_z + float(start_z - 1) * step
    z1 = origin_z + float(end_z - 1) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0,
        -1.0, 0.0, 0.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_top_surface_run(vertices, vertex_index, material, surface_height, local_z, start_x, end_x, origin_x, origin_z, step):
    cr, cg, cb = _material_rgb(material, surface_height - 1)
    x0 = origin_x + float(start_x - 1) * step
    x1 = origin_x + float(end_x - 1) * step
    y1 = float(surface_height) * step
    z0 = origin_z + float(local_z - 1) * step
    z1 = origin_z + float(local_z) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1,
        0.0, 1.0, 0.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_top_surface_rect(vertices, vertex_index, material, surface_height, start_x, end_x, start_z, end_z, origin_x, origin_z, step):
    cr, cg, cb = _material_rgb(material, surface_height - 1)
    x0 = origin_x + float(start_x - 1) * step
    x1 = origin_x + float(end_x - 1) * step
    y1 = float(surface_height) * step
    z0 = origin_z + float(start_z - 1) * step
    z1 = origin_z + float(end_z - 1) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1,
        0.0, 1.0, 0.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_bottom_surface_run(vertices, vertex_index, local_z, start_x, end_x, origin_x, origin_z, step):
    cr, cg, cb = _material_rgb(BEDROCK, 0)
    x0 = origin_x + float(start_x - 1) * step
    x1 = origin_x + float(end_x - 1) * step
    z0 = origin_z + float(local_z - 1) * step
    z1 = origin_z + float(local_z) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, 0.0, z0, x0, 0.0, z1, x1, 0.0, z1, x1, 0.0, z0,
        0.0, -1.0, 0.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_surface_side_span(vertices, vertex_index, face_kind, material, y_start, y_end, local_x, local_z, origin_x, origin_z, step):
    cr, cg, cb = _material_rgb(material, y_end - 1)
    x0 = origin_x + float(local_x - 1) * step
    x1 = origin_x + float(local_x) * step
    y0 = float(y_start) * step
    y1 = float(y_end) * step
    z0 = origin_z + float(local_z - 1) * step
    z1 = origin_z + float(local_z) * step
    if face_kind == 0:
        return _emit_quad_components_uniform_color(
            vertices, vertex_index,
            x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1,
            1.0, 0.0, 0.0,
            cr, cg, cb,
        )
    if face_kind == 1:
        return _emit_quad_components_uniform_color(
            vertices, vertex_index,
            x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0,
            -1.0, 0.0, 0.0,
            cr, cg, cb,
        )
    if face_kind == 2:
        return _emit_quad_components_uniform_color(
            vertices, vertex_index,
            x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
            0.0, 0.0, 1.0,
            cr, cg, cb,
        )
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y0, z0, x0, y1, z0, x1, y1, z0, x1, y0, z0,
        0.0, 0.0, -1.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _emit_surface_side_run(vertices, vertex_index, face_kind, material, y_start, y_end, fixed_local, start_axis, end_axis, origin_x, origin_z, step):
    cr, cg, cb = _material_rgb(material, y_end - 1)
    y0 = float(y_start) * step
    y1 = float(y_end) * step
    if face_kind == 0:
        x1 = origin_x + float(fixed_local) * step
        z0 = origin_z + float(start_axis - 1) * step
        z1 = origin_z + float(end_axis - 1) * step
        return _emit_quad_components_uniform_color(
            vertices, vertex_index,
            x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1,
            1.0, 0.0, 0.0,
            cr, cg, cb,
        )
    if face_kind == 1:
        x0 = origin_x + float(fixed_local - 1) * step
        z0 = origin_z + float(start_axis - 1) * step
        z1 = origin_z + float(end_axis - 1) * step
        return _emit_quad_components_uniform_color(
            vertices, vertex_index,
            x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0,
            -1.0, 0.0, 0.0,
            cr, cg, cb,
        )
    if face_kind == 2:
        x0 = origin_x + float(start_axis - 1) * step
        x1 = origin_x + float(end_axis - 1) * step
        z1 = origin_z + float(fixed_local) * step
        return _emit_quad_components_uniform_color(
            vertices, vertex_index,
            x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
            0.0, 0.0, 1.0,
            cr, cg, cb,
        )
    x0 = origin_x + float(start_axis - 1) * step
    x1 = origin_x + float(end_axis - 1) * step
    z0 = origin_z + float(fixed_local - 1) * step
    return _emit_quad_components_uniform_color(
        vertices, vertex_index,
        x0, y0, z0, x0, y1, z0, x1, y1, z0, x1, y0, z0,
        0.0, 0.0, -1.0,
        cr, cg, cb,
    )


@njit(cache=True, fastmath=True, inline="always")
def _count_surface_material_spans(surface_height: int, y_start: int, y_end: int) -> int:
    count = 0
    start = y_start
    stop = min(y_end, 1)
    if start < stop:
        count += 1
    start = max(y_start, 1)
    stop = min(y_end, surface_height - 4)
    if start < stop:
        count += 1
    start = max(y_start, surface_height - 4)
    stop = min(y_end, surface_height - 1)
    if start < stop:
        count += 1
    start = max(y_start, surface_height - 1)
    if start < y_end:
        count += 1
    return count


@njit(cache=True, fastmath=True, inline="always")
def _emit_surface_material_spans(vertices, vertex_index, face_kind, surface_height, surface_material, y_start, y_end, local_x, local_z, origin_x, origin_z, step):
    start = y_start
    stop = min(y_end, 1)
    if start < stop:
        vertex_index = _emit_surface_side_span(vertices, vertex_index, face_kind, BEDROCK, start, stop, local_x, local_z, origin_x, origin_z, step)
    start = max(y_start, 1)
    stop = min(y_end, surface_height - 4)
    if start < stop:
        vertex_index = _emit_surface_side_span(vertices, vertex_index, face_kind, STONE, start, stop, local_x, local_z, origin_x, origin_z, step)
    start = max(y_start, surface_height - 4)
    stop = min(y_end, surface_height - 1)
    if start < stop:
        vertex_index = _emit_surface_side_span(vertices, vertex_index, face_kind, DIRT, start, stop, local_x, local_z, origin_x, origin_z, step)
    start = max(y_start, surface_height - 1)
    if start < y_end:
        vertex_index = _emit_surface_side_span(vertices, vertex_index, face_kind, surface_material, start, y_end, local_x, local_z, origin_x, origin_z, step)
    return vertex_index


@njit(cache=True, fastmath=True, inline="always")
def _surface_material_span_at(surface_height: int, surface_material: int, y_start: int, y_end: int, span_index: int) -> tuple[int, int, int, int]:
    if span_index == 0:
        start = y_start
        stop = min(y_end, 1)
        material = BEDROCK
    elif span_index == 1:
        start = max(y_start, 1)
        stop = min(y_end, surface_height - 4)
        material = STONE
    elif span_index == 2:
        start = max(y_start, surface_height - 4)
        stop = min(y_end, surface_height - 1)
        material = DIRT
    else:
        start = max(y_start, surface_height - 1)
        stop = y_end
        material = surface_material
    if start < stop:
        return 1, int(material), int(start), int(stop)
    return 0, 0, 0, 0


@njit(cache=True, fastmath=True, inline="always")
def _surface_side_interval_material(surface_height: int, surface_material: int, y_end: int) -> int:
    sample_y = y_end - 1
    if sample_y <= 0:
        return BEDROCK
    if sample_y < surface_height - 4:
        return STONE
    if sample_y < surface_height - 1:
        return DIRT
    return surface_material


@njit(cache=True, fastmath=True, inline="always")
def _surface_face_material_span(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    origin_y: int,
    top_y: int,
    local_x: int,
    local_z: int,
    face_kind: int,
    span_index: int,
) -> tuple[int, int, int, int]:
    index = local_z * sample_size + local_x
    surface_height = int(surface_heights[index])
    if surface_height <= origin_y:
        return 0, 0, 0, 0
    y_end = min(top_y, surface_height)
    if y_end <= origin_y:
        return 0, 0, 0, 0
    if face_kind == 0:
        neighbor_height = int(surface_heights[index + 1])
    elif face_kind == 1:
        neighbor_height = int(surface_heights[index - 1])
    elif face_kind == 2:
        neighbor_height = int(surface_heights[index + sample_size])
    else:
        neighbor_height = int(surface_heights[index - sample_size])
    y_start = max(origin_y, neighbor_height)
    if y_start >= y_end:
        return 0, 0, 0, 0
    return _surface_material_span_at(surface_height, int(surface_materials[index]), y_start, y_end, span_index)


@njit(cache=True, fastmath=True, inline="always")
def _surface_face_interval(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    origin_y: int,
    top_y: int,
    local_x: int,
    local_z: int,
    face_kind: int,
) -> tuple[int, int, int, int, int]:
    index = local_z * sample_size + local_x
    surface_height = int(surface_heights[index])
    if surface_height <= origin_y:
        return 0, 0, 0, 0, 0
    y_end = min(top_y, surface_height)
    if y_end <= origin_y:
        return 0, 0, 0, 0, 0
    if face_kind == 0:
        neighbor_height = int(surface_heights[index + 1])
    elif face_kind == 1:
        neighbor_height = int(surface_heights[index - 1])
    elif face_kind == 2:
        neighbor_height = int(surface_heights[index + sample_size])
    else:
        neighbor_height = int(surface_heights[index - sample_size])
    y_start = max(origin_y, neighbor_height)
    if y_start >= y_end:
        return 0, 0, 0, 0, 0
    return 1, surface_height, int(surface_materials[index]), y_start, y_end


@njit(cache=True, fastmath=True)
def _count_surface_side_runs_z(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
    face_kind: int,
) -> int:
    run_count = 0
    for local_x in range(1, chunk_size + 1):
        active_valid = 0
        active_material = 0
        active_y_start = 0
        active_y_end = 0
        for local_z in range(1, chunk_size + 1):
            interval_valid, surface_height, surface_material, y_start, y_end = _surface_face_interval(
                surface_heights,
                surface_materials,
                sample_size,
                origin_y,
                top_y,
                local_x,
                local_z,
                face_kind,
            )
            if interval_valid != 0:
                material = _surface_side_interval_material(surface_height, surface_material, y_end)
                if active_valid != 0 and active_material == material and active_y_start == y_start and active_y_end == y_end:
                    continue
                if active_valid != 0:
                    run_count += 1
                active_valid = 1
                active_material = material
                active_y_start = y_start
                active_y_end = y_end
            elif active_valid != 0:
                run_count += 1
                active_valid = 0
        if active_valid != 0:
            run_count += 1
    return run_count * VERTICES_PER_FACE


@njit(cache=True, fastmath=True)
def _count_surface_side_runs_x(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
    face_kind: int,
) -> int:
    run_count = 0
    for local_z in range(1, chunk_size + 1):
        active_valid = 0
        active_material = 0
        active_y_start = 0
        active_y_end = 0
        for local_x in range(1, chunk_size + 1):
            interval_valid, surface_height, surface_material, y_start, y_end = _surface_face_interval(
                surface_heights,
                surface_materials,
                sample_size,
                origin_y,
                top_y,
                local_x,
                local_z,
                face_kind,
            )
            if interval_valid != 0:
                material = _surface_side_interval_material(surface_height, surface_material, y_end)
                if active_valid != 0 and active_material == material and active_y_start == y_start and active_y_end == y_end:
                    continue
                if active_valid != 0:
                    run_count += 1
                active_valid = 1
                active_material = material
                active_y_start = y_start
                active_y_end = y_end
            elif active_valid != 0:
                run_count += 1
                active_valid = 0
        if active_valid != 0:
            run_count += 1
    return run_count * VERTICES_PER_FACE


@njit(cache=True, fastmath=True)
def _emit_surface_side_runs_z(
    vertices: np.ndarray,
    vertex_index: int,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
    face_kind: int,
    origin_x: float,
    origin_z: float,
    step: float,
) -> int:
    for local_x in range(1, chunk_size + 1):
        active_valid = 0
        active_material = 0
        active_y_start = 0
        active_y_end = 0
        active_start_axis = 0
        for local_z in range(1, chunk_size + 1):
            interval_valid, surface_height, surface_material, y_start, y_end = _surface_face_interval(
                surface_heights,
                surface_materials,
                sample_size,
                origin_y,
                top_y,
                local_x,
                local_z,
                face_kind,
            )
            if interval_valid != 0:
                material = _surface_side_interval_material(surface_height, surface_material, y_end)
                if active_valid != 0 and active_material == material and active_y_start == y_start and active_y_end == y_end:
                    continue
                if active_valid != 0:
                    vertex_index = _emit_surface_side_run(
                        vertices,
                        vertex_index,
                        face_kind,
                        active_material,
                        active_y_start,
                        active_y_end,
                        local_x,
                        active_start_axis,
                        local_z,
                        origin_x,
                        origin_z,
                        step,
                    )
                active_valid = 1
                active_material = material
                active_y_start = y_start
                active_y_end = y_end
                active_start_axis = local_z
            elif active_valid != 0:
                vertex_index = _emit_surface_side_run(
                    vertices,
                    vertex_index,
                    face_kind,
                    active_material,
                    active_y_start,
                    active_y_end,
                    local_x,
                    active_start_axis,
                    local_z,
                    origin_x,
                    origin_z,
                    step,
                )
                active_valid = 0
        if active_valid != 0:
            vertex_index = _emit_surface_side_run(
                vertices,
                vertex_index,
                face_kind,
                active_material,
                active_y_start,
                active_y_end,
                local_x,
                active_start_axis,
                chunk_size + 1,
                origin_x,
                origin_z,
                step,
            )
    return vertex_index


@njit(cache=True, fastmath=True)
def _emit_surface_side_runs_x(
    vertices: np.ndarray,
    vertex_index: int,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
    face_kind: int,
    origin_x: float,
    origin_z: float,
    step: float,
) -> int:
    for local_z in range(1, chunk_size + 1):
        active_valid = 0
        active_material = 0
        active_y_start = 0
        active_y_end = 0
        active_start_axis = 0
        for local_x in range(1, chunk_size + 1):
            interval_valid, surface_height, surface_material, y_start, y_end = _surface_face_interval(
                surface_heights,
                surface_materials,
                sample_size,
                origin_y,
                top_y,
                local_x,
                local_z,
                face_kind,
            )
            if interval_valid != 0:
                material = _surface_side_interval_material(surface_height, surface_material, y_end)
                if active_valid != 0 and active_material == material and active_y_start == y_start and active_y_end == y_end:
                    continue
                if active_valid != 0:
                    vertex_index = _emit_surface_side_run(
                        vertices,
                        vertex_index,
                        face_kind,
                        active_material,
                        active_y_start,
                        active_y_end,
                        local_z,
                        active_start_axis,
                        local_x,
                        origin_x,
                        origin_z,
                        step,
                    )
                active_valid = 1
                active_material = material
                active_y_start = y_start
                active_y_end = y_end
                active_start_axis = local_x
            elif active_valid != 0:
                vertex_index = _emit_surface_side_run(
                    vertices,
                    vertex_index,
                    face_kind,
                    active_material,
                    active_y_start,
                    active_y_end,
                    local_z,
                    active_start_axis,
                    local_x,
                    origin_x,
                    origin_z,
                    step,
                )
                active_valid = 0
        if active_valid != 0:
            vertex_index = _emit_surface_side_run(
                vertices,
                vertex_index,
                face_kind,
                active_material,
                active_y_start,
                active_y_end,
                local_z,
                active_start_axis,
                chunk_size + 1,
                origin_x,
                origin_z,
                step,
            )
    return vertex_index


@njit(cache=True, fastmath=True)
def _count_top_surface_rect_vertices(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
) -> int:
    consumed = np.zeros((sample_size, sample_size), dtype=np.uint8)
    rect_count = 0
    for local_z in range(1, chunk_size + 1):
        for local_x in range(1, chunk_size + 1):
            if consumed[local_z, local_x] != 0:
                continue
            index = local_z * sample_size + local_x
            surface_height = int(surface_heights[index])
            if surface_height <= origin_y or surface_height > top_y:
                continue
            material = int(surface_materials[index])
            end_x = local_x + 1
            while end_x <= chunk_size:
                next_index = local_z * sample_size + end_x
                if consumed[local_z, end_x] != 0 or int(surface_heights[next_index]) != surface_height or int(surface_materials[next_index]) != material:
                    break
                end_x += 1
            end_z = local_z + 1
            while end_z <= chunk_size:
                row_matches = True
                for scan_x in range(local_x, end_x):
                    next_index = end_z * sample_size + scan_x
                    if consumed[end_z, scan_x] != 0 or int(surface_heights[next_index]) != surface_height or int(surface_materials[next_index]) != material:
                        row_matches = False
                        break
                if not row_matches:
                    break
                end_z += 1
            for mark_z in range(local_z, end_z):
                for mark_x in range(local_x, end_x):
                    consumed[mark_z, mark_x] = 1
            rect_count += 1
    return rect_count * VERTICES_PER_FACE


@njit(cache=True, fastmath=True)
def _emit_top_surface_rects(
    vertices: np.ndarray,
    vertex_index: int,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
    origin_x: float,
    origin_z: float,
    step: float,
) -> int:
    consumed = np.zeros((sample_size, sample_size), dtype=np.uint8)
    for local_z in range(1, chunk_size + 1):
        for local_x in range(1, chunk_size + 1):
            if consumed[local_z, local_x] != 0:
                continue
            index = local_z * sample_size + local_x
            surface_height = int(surface_heights[index])
            if surface_height <= origin_y or surface_height > top_y:
                continue
            material = int(surface_materials[index])
            end_x = local_x + 1
            while end_x <= chunk_size:
                next_index = local_z * sample_size + end_x
                if consumed[local_z, end_x] != 0 or int(surface_heights[next_index]) != surface_height or int(surface_materials[next_index]) != material:
                    break
                end_x += 1
            end_z = local_z + 1
            while end_z <= chunk_size:
                row_matches = True
                for scan_x in range(local_x, end_x):
                    next_index = end_z * sample_size + scan_x
                    if consumed[end_z, scan_x] != 0 or int(surface_heights[next_index]) != surface_height or int(surface_materials[next_index]) != material:
                        row_matches = False
                        break
                if not row_matches:
                    break
                end_z += 1
            for mark_z in range(local_z, end_z):
                for mark_x in range(local_x, end_x):
                    consumed[mark_z, mark_x] = 1
            vertex_index = _emit_top_surface_rect(vertices, vertex_index, material, surface_height, local_x, end_x, local_z, end_z, origin_x, origin_z, step)
    return vertex_index


@njit(cache=True, fastmath=True)
def _append_top_surface_rect_runs(
    runs: np.ndarray,
    run_index: int,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
) -> int:
    consumed = np.zeros((sample_size, sample_size), dtype=np.uint8)
    for local_z in range(1, chunk_size + 1):
        for local_x in range(1, chunk_size + 1):
            if consumed[local_z, local_x] != 0:
                continue
            index = local_z * sample_size + local_x
            surface_height = int(surface_heights[index])
            if surface_height <= origin_y or surface_height > top_y:
                continue
            material = int(surface_materials[index])
            end_x = local_x + 1
            while end_x <= chunk_size:
                next_index = local_z * sample_size + end_x
                if consumed[local_z, end_x] != 0 or int(surface_heights[next_index]) != surface_height or int(surface_materials[next_index]) != material:
                    break
                end_x += 1
            end_z = local_z + 1
            while end_z <= chunk_size:
                row_matches = True
                for scan_x in range(local_x, end_x):
                    next_index = end_z * sample_size + scan_x
                    if consumed[end_z, scan_x] != 0 or int(surface_heights[next_index]) != surface_height or int(surface_materials[next_index]) != material:
                        row_matches = False
                        break
                if not row_matches:
                    break
                end_z += 1
            for mark_z in range(local_z, end_z):
                for mark_x in range(local_x, end_x):
                    consumed[mark_z, mark_x] = 1
            runs[run_index, 0] = 4
            runs[run_index, 1] = material
            runs[run_index, 2] = surface_height
            runs[run_index, 3] = 0
            runs[run_index, 4] = local_x
            runs[run_index, 5] = end_x
            runs[run_index, 6] = local_z
            runs[run_index, 7] = end_z
            run_index += 1
    return run_index


@njit(cache=True, fastmath=True)
def _append_surface_side_runs_z(
    runs: np.ndarray,
    run_index: int,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
    face_kind: int,
) -> int:
    for local_x in range(1, chunk_size + 1):
        active_valid = 0
        active_material = 0
        active_y_start = 0
        active_y_end = 0
        active_start_axis = 0
        for local_z in range(1, chunk_size + 1):
            interval_valid, surface_height, surface_material, y_start, y_end = _surface_face_interval(
                surface_heights,
                surface_materials,
                sample_size,
                origin_y,
                top_y,
                local_x,
                local_z,
                face_kind,
            )
            if interval_valid != 0:
                material = _surface_side_interval_material(surface_height, surface_material, y_end)
                if active_valid != 0 and active_material == material and active_y_start == y_start and active_y_end == y_end:
                    continue
                if active_valid != 0:
                    runs[run_index, 0] = face_kind
                    runs[run_index, 1] = active_material
                    runs[run_index, 2] = active_y_start
                    runs[run_index, 3] = active_y_end
                    runs[run_index, 4] = local_x
                    runs[run_index, 5] = active_start_axis
                    runs[run_index, 6] = local_z
                    runs[run_index, 7] = 0
                    run_index += 1
                active_valid = 1
                active_material = material
                active_y_start = y_start
                active_y_end = y_end
                active_start_axis = local_z
            elif active_valid != 0:
                runs[run_index, 0] = face_kind
                runs[run_index, 1] = active_material
                runs[run_index, 2] = active_y_start
                runs[run_index, 3] = active_y_end
                runs[run_index, 4] = local_x
                runs[run_index, 5] = active_start_axis
                runs[run_index, 6] = local_z
                runs[run_index, 7] = 0
                run_index += 1
                active_valid = 0
        if active_valid != 0:
            runs[run_index, 0] = face_kind
            runs[run_index, 1] = active_material
            runs[run_index, 2] = active_y_start
            runs[run_index, 3] = active_y_end
            runs[run_index, 4] = local_x
            runs[run_index, 5] = active_start_axis
            runs[run_index, 6] = chunk_size + 1
            runs[run_index, 7] = 0
            run_index += 1
    return run_index


@njit(cache=True, fastmath=True)
def _append_surface_side_runs_x(
    runs: np.ndarray,
    run_index: int,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    sample_size: int,
    chunk_size: int,
    origin_y: int,
    top_y: int,
    face_kind: int,
) -> int:
    for local_z in range(1, chunk_size + 1):
        active_valid = 0
        active_material = 0
        active_y_start = 0
        active_y_end = 0
        active_start_axis = 0
        for local_x in range(1, chunk_size + 1):
            interval_valid, surface_height, surface_material, y_start, y_end = _surface_face_interval(
                surface_heights,
                surface_materials,
                sample_size,
                origin_y,
                top_y,
                local_x,
                local_z,
                face_kind,
            )
            if interval_valid != 0:
                material = _surface_side_interval_material(surface_height, surface_material, y_end)
                if active_valid != 0 and active_material == material and active_y_start == y_start and active_y_end == y_end:
                    continue
                if active_valid != 0:
                    runs[run_index, 0] = face_kind
                    runs[run_index, 1] = active_material
                    runs[run_index, 2] = active_y_start
                    runs[run_index, 3] = active_y_end
                    runs[run_index, 4] = local_z
                    runs[run_index, 5] = active_start_axis
                    runs[run_index, 6] = local_x
                    runs[run_index, 7] = 0
                    run_index += 1
                active_valid = 1
                active_material = material
                active_y_start = y_start
                active_y_end = y_end
                active_start_axis = local_x
            elif active_valid != 0:
                runs[run_index, 0] = face_kind
                runs[run_index, 1] = active_material
                runs[run_index, 2] = active_y_start
                runs[run_index, 3] = active_y_end
                runs[run_index, 4] = local_z
                runs[run_index, 5] = active_start_axis
                runs[run_index, 6] = local_x
                runs[run_index, 7] = 0
                run_index += 1
                active_valid = 0
        if active_valid != 0:
            runs[run_index, 0] = face_kind
            runs[run_index, 1] = active_material
            runs[run_index, 2] = active_y_start
            runs[run_index, 3] = active_y_end
            runs[run_index, 4] = local_z
            runs[run_index, 5] = active_start_axis
            runs[run_index, 6] = chunk_size + 1
            runs[run_index, 7] = 0
            run_index += 1
    return run_index


@njit(cache=True, fastmath=True)
def build_chunk_surface_run_table_from_heightmap_clipped(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_size: int,
    height_limit: int,
    chunk_y: int = 0,
) -> tuple[np.ndarray, int, int]:
    sample_size = chunk_size + 2
    origin_y = int(chunk_y) * int(chunk_size)
    top_y = origin_y + int(height_limit)
    max_runs = chunk_size * chunk_size * 5
    if origin_y == 0:
        max_runs += chunk_size
    runs = np.empty((max_runs, 8), dtype=np.int32)
    run_index = 0

    run_index = _append_top_surface_rect_runs(runs, run_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y)

    if origin_y == 0:
        for local_z in range(1, chunk_size + 1):
            local_x = 1
            while local_x <= chunk_size:
                index = local_z * sample_size + local_x
                if int(surface_heights[index]) > 0:
                    run_end = local_x + 1
                    while run_end <= chunk_size and int(surface_heights[local_z * sample_size + run_end]) > 0:
                        run_end += 1
                    runs[run_index, 0] = 5
                    runs[run_index, 1] = BEDROCK
                    runs[run_index, 2] = 0
                    runs[run_index, 3] = 0
                    runs[run_index, 4] = local_z
                    runs[run_index, 5] = local_x
                    runs[run_index, 6] = run_end
                    runs[run_index, 7] = 0
                    run_index += 1
                    local_x = run_end
                else:
                    local_x += 1

    run_index = _append_surface_side_runs_z(runs, run_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 0)
    run_index = _append_surface_side_runs_z(runs, run_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 1)
    run_index = _append_surface_side_runs_x(runs, run_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 2)
    run_index = _append_surface_side_runs_x(runs, run_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 3)
    return runs[:run_index], run_index, run_index * VERTICES_PER_FACE


@njit(cache=True, fastmath=True)
def emit_chunk_surface_run_table_vertices(
    vertices: np.ndarray,
    vertex_offset: int,
    runs: np.ndarray,
    run_count: int,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    block_size: float = 1.0,
) -> int:
    step = float(block_size)
    origin_x = float(chunk_x * chunk_size) * step
    origin_z = float(chunk_z * chunk_size) * step
    start_vertex_index = int(vertex_offset)
    vertex_index = start_vertex_index
    for run_index in range(run_count):
        face_kind = int(runs[run_index, 0])
        material = int(runs[run_index, 1])
        if face_kind == 4:
            vertex_index = _emit_top_surface_rect(
                vertices,
                vertex_index,
                material,
                int(runs[run_index, 2]),
                int(runs[run_index, 4]),
                int(runs[run_index, 5]),
                int(runs[run_index, 6]),
                int(runs[run_index, 7]),
                origin_x,
                origin_z,
                step,
            )
        elif face_kind == 5:
            vertex_index = _emit_bottom_surface_run(
                vertices,
                vertex_index,
                int(runs[run_index, 4]),
                int(runs[run_index, 5]),
                int(runs[run_index, 6]),
                origin_x,
                origin_z,
                step,
            )
        else:
            vertex_index = _emit_surface_side_run(
                vertices,
                vertex_index,
                face_kind,
                material,
                int(runs[run_index, 2]),
                int(runs[run_index, 3]),
                int(runs[run_index, 4]),
                int(runs[run_index, 5]),
                int(runs[run_index, 6]),
                origin_x,
                origin_z,
                step,
            )
    return vertex_index - start_vertex_index


@njit(cache=True, fastmath=True)
def _count_clipped_surface_vertices(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_size: int,
    height_limit: int,
    chunk_y: int,
) -> int:
    sample_size = chunk_size + 2
    origin_y = int(chunk_y) * int(chunk_size)
    top_y = origin_y + int(height_limit)
    vertex_count = 0
    vertex_count += _count_top_surface_rect_vertices(surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y)

    if origin_y == 0:
        for local_z in range(1, chunk_size + 1):
            local_x = 1
            while local_x <= chunk_size:
                index = local_z * sample_size + local_x
                if int(surface_heights[index]) > 0:
                    run_end = local_x + 1
                    while run_end <= chunk_size and int(surface_heights[local_z * sample_size + run_end]) > 0:
                        run_end += 1
                    vertex_count += VERTICES_PER_FACE
                    local_x = run_end
                else:
                    local_x += 1

    vertex_count += _count_surface_side_runs_z(surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 0)
    vertex_count += _count_surface_side_runs_z(surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 1)
    vertex_count += _count_surface_side_runs_x(surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 2)
    vertex_count += _count_surface_side_runs_x(surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 3)
    return vertex_count


@njit(cache=True, fastmath=True)
def count_chunk_surface_vertices_from_heightmap_clipped(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_size: int,
    height_limit: int,
    chunk_y: int = 0,
) -> int:
    return _count_clipped_surface_vertices(surface_heights, surface_materials, chunk_size, height_limit, chunk_y)


@njit(cache=True, fastmath=True)
def emit_chunk_surface_vertices_from_heightmap_clipped(
    vertices: np.ndarray,
    vertex_offset: int,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    height_limit: int,
    block_size: float = 1.0,
    chunk_y: int = 0,
) -> int:
    sample_size = chunk_size + 2
    origin_y = int(chunk_y) * int(chunk_size)
    top_y = origin_y + int(height_limit)
    step = float(block_size)
    origin_x = float(chunk_x * chunk_size) * step
    origin_z = float(chunk_z * chunk_size) * step
    start_vertex_index = int(vertex_offset)
    vertex_index = start_vertex_index

    vertex_index = _emit_top_surface_rects(vertices, vertex_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, origin_x, origin_z, step)

    if origin_y == 0:
        for local_z in range(1, chunk_size + 1):
            local_x = 1
            while local_x <= chunk_size:
                index = local_z * sample_size + local_x
                if int(surface_heights[index]) > 0:
                    run_end = local_x + 1
                    while run_end <= chunk_size and int(surface_heights[local_z * sample_size + run_end]) > 0:
                        run_end += 1
                    vertex_index = _emit_bottom_surface_run(vertices, vertex_index, local_z, local_x, run_end, origin_x, origin_z, step)
                    local_x = run_end
                else:
                    local_x += 1

    vertex_index = _emit_surface_side_runs_z(vertices, vertex_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 0, origin_x, origin_z, step)
    vertex_index = _emit_surface_side_runs_z(vertices, vertex_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 1, origin_x, origin_z, step)
    vertex_index = _emit_surface_side_runs_x(vertices, vertex_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 2, origin_x, origin_z, step)
    vertex_index = _emit_surface_side_runs_x(vertices, vertex_index, surface_heights, surface_materials, sample_size, chunk_size, origin_y, top_y, 3, origin_x, origin_z, step)
    return vertex_index - start_vertex_index


@njit(cache=True, fastmath=True)
def build_chunk_surface_vertex_array_from_heightmap_clipped(
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    height_limit: int,
    block_size: float = 1.0,
    chunk_y: int = 0,
) -> tuple[np.ndarray, int]:
    vertex_count = count_chunk_surface_vertices_from_heightmap_clipped(surface_heights, surface_materials, chunk_size, height_limit, chunk_y)
    vertices = np.empty((vertex_count, VERTEX_COMPONENTS), dtype=np.float32)
    if vertex_count == 0:
        return vertices, 0
    emitted_count = emit_chunk_surface_vertices_from_heightmap_clipped(
        vertices,
        0,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_z,
        chunk_size,
        height_limit,
        block_size,
        chunk_y,
    )
    return vertices[:emitted_count], emitted_count


@njit(cache=True, fastmath=True)
def _emit_x_face_runs_from_bits(
    vertices: np.ndarray,
    vertex_index: int,
    face_kind: int,
    face_bits: np.uint64,
    material_break_bits: np.uint64,
    valid_mask: np.uint64,
    row_materials: np.ndarray,
    plane: np.ndarray,
    plane_above: np.ndarray,
    plane_below: np.ndarray,
    y: int,
    local_z: int,
    chunk_size: int,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    step: float,
) -> int:
    bits = np.uint64(face_bits)
    stop_source = ((~bits) & valid_mask) | (material_break_bits & valid_mask)
    while bits != 0:
        start_bit = _ctz_u64(bits)
        start_x = start_bit + 1
        stop_candidates = stop_source & (valid_mask ^ _low_u64_bits(start_bit + 1))
        if stop_candidates != 0:
            stop_bit = _ctz_u64(stop_candidates)
            end_x = stop_bit + 1
            bits &= valid_mask ^ _low_u64_bits(stop_bit)
        else:
            end_x = chunk_size + 1
            bits = np.uint64(0)

        material = int(row_materials[start_x])
        if face_kind == 0:
            vertex_index = _emit_top_run(vertices, vertex_index, plane_above, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step)
        elif face_kind == 1:
            vertex_index = _emit_bottom_run(vertices, vertex_index, plane_below, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step)
        elif face_kind == 2:
            vertex_index = _emit_south_run(vertices, vertex_index, plane, plane_above, plane_below, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step)
        else:
            vertex_index = _emit_north_run(vertices, vertex_index, plane, plane_above, plane_below, material, y, local_z, start_x, end_x, origin_x, origin_y, origin_z, step)
    return vertex_index


@njit(cache=True, fastmath=True)
def _emit_chunk_voxel_bit_run_vertices_with_boundaries(
    vertices: np.ndarray,
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    height_limit: int,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
    block_size: float,
    chunk_y: int,
) -> int:
    end = chunk_size + 1
    last_y = height_limit - 1
    valid_mask = _valid_chunk_bit_mask(chunk_size)
    step = float(block_size)
    origin_x = float(chunk_x * chunk_size) * step
    origin_y = float(chunk_y * chunk_size) * step
    origin_z = float(chunk_z * chunk_size) * step
    vertex_index = 0

    east_open_bits = np.uint64(0)
    west_open_bits = np.uint64(0)
    east_start = np.zeros(chunk_size, dtype=np.int64)
    west_start = np.zeros(chunk_size, dtype=np.int64)
    east_material = np.zeros(chunk_size, dtype=np.int64)
    west_material = np.zeros(chunk_size, dtype=np.int64)

    for y in range(height_limit):
        plane = blocks[y]
        mat_plane = materials[y]
        plane_above = blocks[y + 1] if y < last_y else top_boundary
        plane_below = blocks[y - 1] if y > 0 else bottom_boundary
        east_open_bits = np.uint64(0)
        west_open_bits = np.uint64(0)

        for local_z in range(1, end):
            row = plane[local_z]
            row_materials = mat_plane[local_z]
            row_bits = _row_occupancy_bits(row, chunk_size)
            if row_bits == 0:
                close_bits = east_open_bits
                while close_bits != 0:
                    bit_index = _ctz_u64(close_bits)
                    local_x = bit_index + 1
                    vertex_index = _emit_east_run(vertices, vertex_index, plane, plane_above, plane_below, int(east_material[bit_index]), y, local_x, int(east_start[bit_index]), local_z, origin_x, origin_y, origin_z, step)
                    east_open_bits &= ~(np.uint64(1) << bit_index)
                    close_bits &= close_bits - np.uint64(1)
                close_bits = west_open_bits
                while close_bits != 0:
                    bit_index = _ctz_u64(close_bits)
                    local_x = bit_index + 1
                    vertex_index = _emit_west_run(vertices, vertex_index, plane, plane_above, plane_below, int(west_material[bit_index]), y, local_x, int(west_start[bit_index]), local_z, origin_x, origin_y, origin_z, step)
                    west_open_bits &= ~(np.uint64(1) << bit_index)
                    close_bits &= close_bits - np.uint64(1)
                continue

            above_bits = _row_occupancy_bits(plane_above[local_z], chunk_size)
            below_bits = _row_occupancy_bits(plane_below[local_z], chunk_size)
            south_bits = _row_occupancy_bits(plane[local_z + 1], chunk_size)
            north_bits = _row_occupancy_bits(plane[local_z - 1], chunk_size)
            material_break_bits = _row_material_break_bits(row_materials, chunk_size)

            top_bits = row_bits & (~above_bits & valid_mask)
            bottom_bits = row_bits & (~below_bits & valid_mask)
            south_face_bits = row_bits & (~south_bits & valid_mask)
            north_face_bits = row_bits & (~north_bits & valid_mask)
            east_bits = row_bits & (~_east_neighbor_bits(row, row_bits, chunk_size) & valid_mask)
            west_bits = row_bits & (~_west_neighbor_bits(row, row_bits, valid_mask) & valid_mask)

            if top_bits != 0:
                vertex_index = _emit_x_face_runs_from_bits(
                    vertices, vertex_index, 0, top_bits, material_break_bits, valid_mask,
                    row_materials, plane, plane_above, plane_below, y, local_z, chunk_size,
                    origin_x, origin_y, origin_z, step,
                )
            if bottom_bits != 0:
                vertex_index = _emit_x_face_runs_from_bits(
                    vertices, vertex_index, 1, bottom_bits, material_break_bits, valid_mask,
                    row_materials, plane, plane_above, plane_below, y, local_z, chunk_size,
                    origin_x, origin_y, origin_z, step,
                )
            if south_face_bits != 0:
                vertex_index = _emit_x_face_runs_from_bits(
                    vertices, vertex_index, 2, south_face_bits, material_break_bits, valid_mask,
                    row_materials, plane, plane_above, plane_below, y, local_z, chunk_size,
                    origin_x, origin_y, origin_z, step,
                )
            if north_face_bits != 0:
                vertex_index = _emit_x_face_runs_from_bits(
                    vertices, vertex_index, 3, north_face_bits, material_break_bits, valid_mask,
                    row_materials, plane, plane_above, plane_below, y, local_z, chunk_size,
                    origin_x, origin_y, origin_z, step,
                )

            close_bits = east_open_bits & ~east_bits
            while close_bits != 0:
                bit_index = _ctz_u64(close_bits)
                local_x = bit_index + 1
                vertex_index = _emit_east_run(vertices, vertex_index, plane, plane_above, plane_below, int(east_material[bit_index]), y, local_x, int(east_start[bit_index]), local_z, origin_x, origin_y, origin_z, step)
                east_open_bits &= ~(np.uint64(1) << bit_index)
                close_bits &= close_bits - np.uint64(1)

            close_bits = west_open_bits & ~west_bits
            while close_bits != 0:
                bit_index = _ctz_u64(close_bits)
                local_x = bit_index + 1
                vertex_index = _emit_west_run(vertices, vertex_index, plane, plane_above, plane_below, int(west_material[bit_index]), y, local_x, int(west_start[bit_index]), local_z, origin_x, origin_y, origin_z, step)
                west_open_bits &= ~(np.uint64(1) << bit_index)
                close_bits &= close_bits - np.uint64(1)

            scan_bits = east_bits
            while scan_bits != 0:
                bit_index = _ctz_u64(scan_bits)
                bit = np.uint64(1) << bit_index
                local_x = bit_index + 1
                material = int(row_materials[local_x])
                if east_open_bits & bit:
                    if material != int(east_material[bit_index]):
                        vertex_index = _emit_east_run(vertices, vertex_index, plane, plane_above, plane_below, int(east_material[bit_index]), y, local_x, int(east_start[bit_index]), local_z, origin_x, origin_y, origin_z, step)
                        east_start[bit_index] = local_z
                        east_material[bit_index] = material
                else:
                    east_open_bits |= bit
                    east_start[bit_index] = local_z
                    east_material[bit_index] = material
                scan_bits &= scan_bits - np.uint64(1)

            scan_bits = west_bits
            while scan_bits != 0:
                bit_index = _ctz_u64(scan_bits)
                bit = np.uint64(1) << bit_index
                local_x = bit_index + 1
                material = int(row_materials[local_x])
                if west_open_bits & bit:
                    if material != int(west_material[bit_index]):
                        vertex_index = _emit_west_run(vertices, vertex_index, plane, plane_above, plane_below, int(west_material[bit_index]), y, local_x, int(west_start[bit_index]), local_z, origin_x, origin_y, origin_z, step)
                        west_start[bit_index] = local_z
                        west_material[bit_index] = material
                else:
                    west_open_bits |= bit
                    west_start[bit_index] = local_z
                    west_material[bit_index] = material
                scan_bits &= scan_bits - np.uint64(1)

        close_bits = east_open_bits
        while close_bits != 0:
            bit_index = _ctz_u64(close_bits)
            local_x = bit_index + 1
            vertex_index = _emit_east_run(vertices, vertex_index, plane, plane_above, plane_below, int(east_material[bit_index]), y, local_x, int(east_start[bit_index]), end, origin_x, origin_y, origin_z, step)
            close_bits &= close_bits - np.uint64(1)
        close_bits = west_open_bits
        while close_bits != 0:
            bit_index = _ctz_u64(close_bits)
            local_x = bit_index + 1
            vertex_index = _emit_west_run(vertices, vertex_index, plane, plane_above, plane_below, int(west_material[bit_index]), y, local_x, int(west_start[bit_index]), end, origin_x, origin_y, origin_z, step)
            close_bits &= close_bits - np.uint64(1)

    return vertex_index


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
    unit_vertex_count = _chunk_unit_face_vertex_count_from_bits(blocks, chunk_size, height_limit, top_boundary, bottom_boundary)
    vertices = np.empty((unit_vertex_count, VERTEX_COMPONENTS), dtype=np.float32)
    if unit_vertex_count == 0:
        return vertices, 0
    vertex_index = _emit_chunk_voxel_bit_run_vertices_with_boundaries(
        vertices,
        blocks,
        materials,
        chunk_x,
        chunk_z,
        chunk_size,
        height_limit,
        top_boundary,
        bottom_boundary,
        float(block_size),
        int(chunk_y),
    )
    return vertices[:vertex_index], vertex_index


@njit(cache=True, fastmath=True)
def _build_chunk_vertex_array_unit_faces_from_voxels_with_boundaries(
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
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y1, z0,
                        x1, y1, z0,
                        x1, y1, z1,
                        x0, y1, z1,
                        0.0, 1.0, 0.0,
                        top_r, top_g, top_b,
                        top_r, top_g, top_b,
                        top_r, top_g, top_b,
                        top_r, top_g, top_b,
                    )

                if mask & FACE_BOTTOM:
                    bottom_r = cr
                    bottom_g = cg
                    bottom_b = cb
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y0, z1,
                        x1, y0, z1,
                        x1, y0, z0,
                        0.0, -1.0, 0.0,
                        bottom_r, bottom_g, bottom_b,
                        bottom_r, bottom_g, bottom_b,
                        bottom_r, bottom_g, bottom_b,
                        bottom_r, bottom_g, bottom_b,
                    )

                if mask & FACE_EAST:
                    east_r = cr
                    east_g = cg
                    east_b = cb
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x1, y0, z0,
                        x1, y1, z0,
                        x1, y1, z1,
                        x1, y0, z1,
                        1.0, 0.0, 0.0,
                        east_r, east_g, east_b,
                        east_r, east_g, east_b,
                        east_r, east_g, east_b,
                        east_r, east_g, east_b,
                    )

                if mask & FACE_WEST:
                    west_r = cr
                    west_g = cg
                    west_b = cb
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y0, z1,
                        x0, y1, z1,
                        x0, y1, z0,
                        -1.0, 0.0, 0.0,
                        west_r, west_g, west_b,
                        west_r, west_g, west_b,
                        west_r, west_g, west_b,
                        west_r, west_g, west_b,
                    )

                if mask & FACE_SOUTH:
                    south_r = cr
                    south_g = cg
                    south_b = cb
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z1,
                        x1, y0, z1,
                        x1, y1, z1,
                        x0, y1, z1,
                        0.0, 0.0, 1.0,
                        south_r, south_g, south_b,
                        south_r, south_g, south_b,
                        south_r, south_g, south_b,
                        south_r, south_g, south_b,
                    )

                if mask & FACE_NORTH:
                    north_r = cr
                    north_g = cg
                    north_b = cb
                    vertex_index = _emit_quad_components_ao(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y1, z0,
                        x1, y1, z0,
                        x1, y0, z0,
                        0.0, 0.0, -1.0,
                        north_r, north_g, north_b,
                        north_r, north_g, north_b,
                        north_r, north_g, north_b,
                        north_r, north_g, north_b,
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
    return _chunk_unit_face_vertex_count_from_bits(blocks, chunk_size, height_limit, top_boundary, bottom_boundary)


@njit(cache=True, fastmath=True)
def count_chunk_voxel_vertices(blocks: np.ndarray, chunk_size: int, height_limit: int) -> int:
    sample_size = chunk_size + 2
    empty_plane = np.zeros((sample_size, sample_size), dtype=blocks.dtype)
    return count_chunk_voxel_vertices_with_boundaries(blocks, chunk_size, height_limit, empty_plane, empty_plane)
