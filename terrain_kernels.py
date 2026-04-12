from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback for environments without numba
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


AIR = 0
BEDROCK = 1
STONE = 2
DIRT = 3
GRASS = 4
SAND = 5
SNOW = 6

MAX_FACES_PER_CELL = 5
VERTICES_PER_FACE = 6
VERTEX_COMPONENTS = 12


@njit(cache=True, fastmath=True)
def _hash2(ix: int, iy: int, seed: int) -> float:
    value = math.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453123
    return value - math.floor(value)


@njit(cache=True, fastmath=True)
def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, fastmath=True)
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@njit(cache=True, fastmath=True)
def _value_noise_2d(x: float, y: float, seed: int, frequency: float) -> float:
    x *= frequency
    y *= frequency

    x0 = math.floor(x)
    y0 = math.floor(y)
    xf = x - x0
    yf = y - y0

    ix0 = int(x0)
    iy0 = int(y0)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    v00 = _hash2(ix0, iy0, seed)
    v10 = _hash2(ix1, iy0, seed)
    v01 = _hash2(ix0, iy1, seed)
    v11 = _hash2(ix1, iy1, seed)

    u = _fade(xf)
    v = _fade(yf)
    nx0 = _lerp(v00, v10, u)
    nx1 = _lerp(v01, v11, u)
    return _lerp(nx0, nx1, v) * 2.0 - 1.0


@njit(cache=True, fastmath=True)
def _terrain_color(height: int) -> tuple[float, float, float]:
    if height <= 14:
        return 0.78, 0.71, 0.49
    if height >= 90:
        return 0.95, 0.97, 0.98
    if height >= 70:
        return 0.60, 0.72, 0.49
    if height >= 40:
        return 0.38, 0.64, 0.31
    return 0.28, 0.54, 0.22


@njit(cache=True, fastmath=True)
def _scale_color(color: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return color[0] * scale, color[1] * scale, color[2] * scale


@njit(cache=True, fastmath=True)
def surface_profile_at(
    x: float,
    z: float,
    seed: int,
    height_limit: int,
) -> tuple[int, int]:
    broad = _value_noise_2d(x, z, seed + 11, 0.0009765625)
    ridge = _value_noise_2d(x, z, seed + 23, 0.00390625)
    detail = _value_noise_2d(x, z, seed + 47, 0.010416667)

    height = 26.0 + broad * 18.0 + ridge * 14.0 + detail * 8.0
    height_i = int(height)
    if height_i < 4:
        height_i = 4

    upper_bound = height_limit - 1
    if height_i > upper_bound:
        height_i = upper_bound

    if height_i >= 90:
        return height_i, SNOW
    if height_i <= 14:
        return height_i, SAND
    if height_i >= 70 and detail > 0.12:
        return height_i, STONE
    return height_i, GRASS


@njit(cache=True, fastmath=True)
def fill_chunk_surface_grids(
    heights: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    height_limit: int,
) -> None:
    sample_size = chunk_size + 2
    origin_x = chunk_x * chunk_size - 1
    origin_z = chunk_z * chunk_size - 1

    for local_z in range(sample_size):
        world_z = origin_z + local_z
        row_offset = local_z * sample_size
        for local_x in range(sample_size):
            world_x = origin_x + local_x
            height, material = surface_profile_at(float(world_x), float(world_z), seed, height_limit)
            cell_index = row_offset + local_x
            heights[cell_index] = height
            materials[cell_index] = material


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
) -> tuple[np.ndarray, int]:
    stride = chunk_size + 2
    max_vertices = chunk_size * chunk_size * MAX_FACES_PER_CELL * VERTICES_PER_FACE
    vertices = np.empty((max_vertices, VERTEX_COMPONENTS), dtype=np.float32)
    vertex_index = 0
    origin_x = chunk_x * chunk_size
    origin_z = chunk_z * chunk_size

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

            x0 = float(origin_x + local_x)
            x1 = x0 + 1.0
            z0 = float(origin_z + local_z)
            z1 = z0 + 1.0
            y0 = float(height)
            west_y = float(west_height)
            east_y = float(east_height)
            north_y = float(north_height)
            south_y = float(south_height)

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


@njit(cache=True, fastmath=True)
def fill_chunk_voxel_grid(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    height_limit: int,
) -> None:
    sample_size = chunk_size + 2
    origin_x = chunk_x * chunk_size - 1
    origin_z = chunk_z * chunk_size - 1

    for local_z in range(sample_size):
        world_z = origin_z + local_z
        for local_x in range(sample_size):
            world_x = origin_x + local_x
            surface_height, surface_material = surface_profile_at(float(world_x), float(world_z), seed, height_limit)
            for y in range(height_limit):
                if y >= surface_height:
                    continue
                if y == 0:
                    material = BEDROCK
                elif y < surface_height - 4:
                    material = STONE
                elif y < surface_height - 1:
                    material = DIRT
                else:
                    material = surface_material
                blocks[y, local_z, local_x] = 1
                materials[y, local_z, local_x] = material


@njit(cache=True, fastmath=True)
def expand_chunk_surface_to_voxel_grid(
    blocks: np.ndarray,
    materials: np.ndarray,
    height_grid: np.ndarray,
    material_grid: np.ndarray,
    chunk_size: int,
    height_limit: int,
) -> None:
    sample_size = chunk_size + 2
    for local_z in range(sample_size):
        row_offset = local_z * sample_size
        for local_x in range(sample_size):
            cell_index = row_offset + local_x
            surface_height = int(height_grid[cell_index])
            surface_material = int(material_grid[cell_index])
            for y in range(height_limit):
                if y >= surface_height:
                    continue
                if y == 0:
                    material = BEDROCK
                elif y < surface_height - 4:
                    material = STONE
                elif y < surface_height - 1:
                    material = DIRT
                else:
                    material = surface_material
                blocks[y, local_z, local_x] = 1
                materials[y, local_z, local_x] = material


@njit(cache=True, fastmath=True)
def _voxel_material_color(material: int, height: int) -> tuple[float, float, float]:
    if material == BEDROCK:
        return 0.24, 0.22, 0.20
    if material == STONE:
        return 0.42, 0.40, 0.38
    if material == DIRT:
        return 0.47, 0.31, 0.18
    if material == GRASS:
        return 0.31, 0.68, 0.24
    if material == SAND:
        return 0.78, 0.71, 0.49
    if material == SNOW:
        return 0.95, 0.97, 0.98
    return _terrain_color(height)


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
def _emit_quad_components(
    vertices: np.ndarray,
    vertex_index: int,
    p0x: float, p0y: float, p0z: float,
    p1x: float, p1y: float, p1z: float,
    p2x: float, p2y: float, p2z: float,
    p3x: float, p3y: float, p3z: float,
    nx: float, ny: float, nz: float,
    cr: float, cg: float, cb: float,
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
    row[8] = cr
    row[9] = cg
    row[10] = cb
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
    row[8] = cr
    row[9] = cg
    row[10] = cb
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
    row[8] = cr
    row[9] = cg
    row[10] = cb
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
    row[8] = cr
    row[9] = cg
    row[10] = cb
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
    row[8] = cr
    row[9] = cg
    row[10] = cb
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
    row[8] = cr
    row[9] = cg
    row[10] = cb
    row[11] = 1.0
    vertex_index += 1

    return vertex_index


@njit(cache=True, fastmath=True)
def build_chunk_vertex_array_from_voxels(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    height_limit: int,
) -> tuple[np.ndarray, int]:
    vertex_count = count_chunk_voxel_vertices(blocks, chunk_size, height_limit)
    vertices = np.empty((vertex_count, VERTEX_COMPONENTS), dtype=np.float32)
    if vertex_count == 0:
        return vertices, 0

    sample_size = chunk_size + 2
    end = sample_size - 1
    last_y = height_limit - 1
    origin_x = float(chunk_x * chunk_size)
    origin_z = float(chunk_z * chunk_size)
    vertex_index = 0

    for y in range(height_limit):
        plane = blocks[y]
        mat_plane = materials[y]
        above = blocks[y + 1] if y < last_y else plane
        below = blocks[y - 1] if y > 0 else plane
        y0 = float(y)
        y1 = y0 + 1.0

        z0 = origin_z
        for local_z in range(1, end):
            row = plane[local_z]
            row_mat = mat_plane[local_z]
            row_north = plane[local_z - 1]
            row_south = plane[local_z + 1]
            row_above = above[local_z]
            row_below = below[local_z]
            z1 = z0 + 1.0

            x0 = origin_x
            for local_x in range(1, end):
                if row[local_x] == 0:
                    x0 += 1.0
                    continue

                x1 = x0 + 1.0
                material = int(row_mat[local_x])

                if material == BEDROCK:
                    cr = 0.24
                    cg = 0.22
                    cb = 0.20
                elif material == STONE:
                    cr = 0.42
                    cg = 0.40
                    cb = 0.38
                elif material == DIRT:
                    cr = 0.47
                    cg = 0.31
                    cb = 0.18
                elif material == GRASS:
                    cr = 0.31
                    cg = 0.68
                    cb = 0.24
                elif material == SAND:
                    cr = 0.78
                    cg = 0.71
                    cb = 0.49
                elif material == SNOW:
                    cr = 0.95
                    cg = 0.97
                    cb = 0.98
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

                top_r = cr
                top_g = cg
                top_b = cb

                east_r = cr * 0.80
                east_g = cg * 0.80
                east_b = cb * 0.80

                west_r = cr * 0.64
                west_g = cg * 0.64
                west_b = cb * 0.64

                south_r = cr * 0.72
                south_g = cg * 0.72
                south_b = cb * 0.72

                north_r = cr * 0.60
                north_g = cg * 0.60
                north_b = cb * 0.60

                bottom_r = cr * 0.50
                bottom_g = cg * 0.50
                bottom_b = cb * 0.50

                if y == last_y or row_above[local_x] == 0:
                    vertex_index = _emit_quad_components(
                        vertices,
                        vertex_index,
                        x0, y1, z0,
                        x1, y1, z0,
                        x1, y1, z1,
                        x0, y1, z1,
                        0.0, 1.0, 0.0,
                        top_r, top_g, top_b,
                    )

                if y == 0 or row_below[local_x] == 0:
                    vertex_index = _emit_quad_components(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y0, z1,
                        x1, y0, z1,
                        x1, y0, z0,
                        0.0, -1.0, 0.0,
                        bottom_r, bottom_g, bottom_b,
                    )

                if row[local_x + 1] == 0:
                    vertex_index = _emit_quad_components(
                        vertices,
                        vertex_index,
                        x1, y0, z0,
                        x1, y1, z0,
                        x1, y1, z1,
                        x1, y0, z1,
                        1.0, 0.0, 0.0,
                        east_r, east_g, east_b,
                    )

                if row[local_x - 1] == 0:
                    vertex_index = _emit_quad_components(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y0, z1,
                        x0, y1, z1,
                        x0, y1, z0,
                        -1.0, 0.0, 0.0,
                        west_r, west_g, west_b,
                    )

                if row_south[local_x] == 0:
                    vertex_index = _emit_quad_components(
                        vertices,
                        vertex_index,
                        x0, y0, z1,
                        x1, y0, z1,
                        x1, y1, z1,
                        x0, y1, z1,
                        0.0, 0.0, 1.0,
                        south_r, south_g, south_b,
                    )

                if row_north[local_x] == 0:
                    vertex_index = _emit_quad_components(
                        vertices,
                        vertex_index,
                        x0, y0, z0,
                        x0, y1, z0,
                        x1, y1, z0,
                        x1, y0, z0,
                        0.0, 0.0, -1.0,
                        north_r, north_g, north_b,
                    )

                x0 = x1

            z0 = z1

    return vertices, vertex_index


@njit(cache=True, fastmath=True)
def count_chunk_voxel_vertices(blocks: np.ndarray, chunk_size: int, height_limit: int) -> int:
    sample_size = chunk_size + 2
    end = sample_size - 1
    last_y = height_limit - 1
    vertex_count = 0

    for y in range(height_limit):
        plane = blocks[y]
        above = blocks[y + 1] if y < last_y else plane
        below = blocks[y - 1] if y > 0 else plane

        for local_z in range(1, end):
            row = plane[local_z]
            row_north = plane[local_z - 1]
            row_south = plane[local_z + 1]
            row_above = above[local_z]
            row_below = below[local_z]

            for local_x in range(1, end):
                if row[local_x] == 0:
                    continue
                if y == last_y or row_above[local_x] == 0:
                    vertex_count += VERTICES_PER_FACE
                if y == 0 or row_below[local_x] == 0:
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
