from __future__ import annotations

import math

import numpy as np

from .world_constants import BLOCK_SIZE


try:
    from numba import njit, prange
except Exception:  # pragma: no cover - fallback for environments without numba
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range


AIR = 0
BEDROCK = 1
STONE = 2
DIRT = 3
GRASS = 4
SAND = 5
SNOW = 6

_MATERIAL_COLOR_R = (0.0, 0.24, 0.42, 0.47, 0.31, 0.78, 0.95)
_MATERIAL_COLOR_G = (0.0, 0.22, 0.40, 0.31, 0.68, 0.71, 0.97)
_MATERIAL_COLOR_B = (0.0, 0.20, 0.38, 0.18, 0.24, 0.49, 0.98)

MAX_FACES_PER_CELL = 5
VERTICES_PER_FACE = 6
VERTEX_COMPONENTS = 12


@njit(cache=True, fastmath=True)
def _hash2(ix: int, iy: int, seed: int) -> float:
    value = math.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453123
    return value - math.floor(value)


@njit(cache=True, fastmath=True)
def _hash3(ix: int, iy: int, iz: int, seed: int) -> float:
    value = math.sin(ix * 127.1 + iy * 311.7 + iz * 74.7 + seed * 19.19) * 43758.5453123
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
def _value_noise_3d(x: float, y: float, z: float, seed: int, frequency: float) -> float:
    x *= frequency
    y *= frequency
    z *= frequency

    x0 = math.floor(x)
    y0 = math.floor(y)
    z0 = math.floor(z)
    xf = x - x0
    yf = y - y0
    zf = z - z0

    ix0 = int(x0)
    iy0 = int(y0)
    iz0 = int(z0)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    c000 = _hash3(ix0, iy0, iz0, seed)
    c100 = _hash3(ix1, iy0, iz0, seed)
    c010 = _hash3(ix0, iy1, iz0, seed)
    c110 = _hash3(ix1, iy1, iz0, seed)
    c001 = _hash3(ix0, iy0, iz1, seed)
    c101 = _hash3(ix1, iy0, iz1, seed)
    c011 = _hash3(ix0, iy1, iz1, seed)
    c111 = _hash3(ix1, iy1, iz1, seed)

    x00 = _lerp(c000, c100, u)
    x10 = _lerp(c010, c110, u)
    x01 = _lerp(c001, c101, u)
    x11 = _lerp(c011, c111, u)
    y0v = _lerp(x00, x10, v)
    y1v = _lerp(x01, x11, v)
    return _lerp(y0v, y1v, w) * 2.0 - 1.0


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


TERRAIN_FREQUENCY_SCALE = 0.3
CAVE_FREQUENCY_SCALE = 0.5
CAVE_BEDROCK_CLEARANCE = 3
SURFACE_BREACH_FREQUENCY_SCALE = 1.0


@njit(cache=True, fastmath=True)
def surface_profile_at(
    x: float,
    z: float,
    seed: int,
    height_limit: int,
) -> tuple[int, int]:
    sample_x = x
    sample_z = z

    broad = _value_noise_2d(sample_x, sample_z, seed + 11, 0.0009765625 * TERRAIN_FREQUENCY_SCALE)
    ridge = _value_noise_2d(sample_x, sample_z, seed + 23, 0.00390625 * TERRAIN_FREQUENCY_SCALE)
    detail = _value_noise_2d(sample_x, sample_z, seed + 47, 0.010416667 * TERRAIN_FREQUENCY_SCALE)
    micro = _value_noise_2d(sample_x, sample_z, seed + 71, 0.020833334 * TERRAIN_FREQUENCY_SCALE)
    nano = _value_noise_2d(sample_x, sample_z, seed + 97, 0.041666668 * TERRAIN_FREQUENCY_SCALE)

    upper_bound = height_limit - 1
    normalized_height = 24.0 + broad * 11.0 + ridge * 8.0 + detail * 4.5 + micro * 1.75 + nano * 0.75
    height_scale = float(upper_bound) / 50.0 if upper_bound > 0 else 1.0
    height_i = int(normalized_height * height_scale)
    if height_i < 4:
        height_i = 4
    if height_i > upper_bound:
        height_i = upper_bound

    sand_threshold = max(4, int(height_limit * 0.18))
    stone_threshold = max(sand_threshold + 6, int(height_limit * 0.58))
    snow_threshold = max(stone_threshold + 6, int(height_limit * 0.82))

    if height_i >= snow_threshold:
        return height_i, SNOW
    if height_i <= sand_threshold:
        return height_i, SAND
    if height_i >= stone_threshold and (detail + micro * 0.5 + nano * 0.35) > 0.10:
        return height_i, STONE
    return height_i, GRASS


@njit(cache=True, fastmath=True, inline="always")
def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(cache=True, fastmath=True)
def _should_carve_cave(
    world_x: int,
    world_y: int,
    world_z: int,
    surface_height: int,
    seed: int,
    world_height_limit: int,
) -> bool:
    if world_y <= CAVE_BEDROCK_CLEARANCE:
        return False
    depth_below_surface = surface_height - world_y
    if world_y >= world_height_limit - 2:
        return False

    normalized_y = float(world_y) / float(max(1, world_height_limit - 1))
    vertical_band = 1.0 - abs(normalized_y - 0.45) * 1.6
    vertical_band = _clamp01(vertical_band)
    if vertical_band <= 0.0:
        return False

    xf = float(world_x)
    yf = float(world_y)
    zf = float(world_z)
    cave_primary = _value_noise_3d(xf, yf * 0.85, zf, seed + 101, 0.018 * CAVE_FREQUENCY_SCALE)
    cave_detail = _value_noise_3d(xf, yf * 1.15, zf, seed + 149, 0.041666668 * CAVE_FREQUENCY_SCALE)
    cave_shape = _value_noise_3d(xf, yf * 0.35, zf, seed + 173, 0.009765625 * CAVE_FREQUENCY_SCALE)
    density = cave_primary * 0.70 + cave_detail * 0.25 - cave_shape * 0.10

    depth_bonus = float(depth_below_surface) * 0.004
    if depth_bonus > 0.12:
        depth_bonus = 0.12

    shallow_bonus = 0.0
    if depth_below_surface <= 6:
        shallow_bonus = (6.0 - float(depth_below_surface)) * (0.12 / 6.0)

    threshold = 0.62 - vertical_band * 0.08 - depth_bonus - shallow_bonus
    if density > threshold:
        return True

    if depth_below_surface <= 2:
        breach_primary = _value_noise_2d(xf, zf, seed + 211, 0.020833334 * SURFACE_BREACH_FREQUENCY_SCALE)
        breach_detail = _value_noise_3d(xf, yf, zf, seed + 233, 0.03125 * CAVE_FREQUENCY_SCALE)
        breach_density = breach_primary * 0.65 + breach_detail * 0.35
        breach_threshold = 0.78 - vertical_band * 0.06
        return breach_density > breach_threshold

    return False


@njit(cache=True, fastmath=True, inline="always")
def _terrain_material_from_surface_profile(
    world_x: int,
    world_y: int,
    world_z: int,
    surface_height: int,
    surface_material: int,
    seed: int,
    world_height_limit: int,
) -> int:
    if world_y < 0 or world_y >= world_height_limit:
        return AIR
    if world_y >= surface_height:
        return AIR
    if _should_carve_cave(world_x, world_y, world_z, surface_height, seed, world_height_limit):
        return AIR
    if world_y == 0:
        return BEDROCK
    if world_y < surface_height - 4:
        return STONE
    if world_y < surface_height - 1:
        return DIRT
    return surface_material


@njit(cache=True, fastmath=True)
def terrain_block_material_at(
    world_x: int,
    world_y: int,
    world_z: int,
    seed: int,
    world_height_limit: int,
) -> int:
    if world_y < 0 or world_y >= world_height_limit:
        return AIR

    surface_height, surface_material = surface_profile_at(float(world_x), float(world_z), seed, world_height_limit)
    return _terrain_material_from_surface_profile(
        world_x,
        world_y,
        world_z,
        int(surface_height),
        int(surface_material),
        seed,
        world_height_limit,
    )


@njit(cache=True, fastmath=True, parallel=True)
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
    total_columns = sample_size * sample_size

    for column_index in prange(total_columns):
        local_z = column_index // sample_size
        local_x = column_index - local_z * sample_size
        world_z = origin_z + local_z
        world_x = origin_x + local_x
        height, material = surface_profile_at(float(world_x), float(world_z), seed, height_limit)
        heights[column_index] = height
        materials[column_index] = material


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


@njit(cache=True, fastmath=True, parallel=True)
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
    total_columns = sample_size * sample_size

    for column_index in prange(total_columns):
        local_z = column_index // sample_size
        local_x = column_index - local_z * sample_size
        world_z = origin_z + local_z
        world_x = origin_x + local_x
        surface_height, surface_material = surface_profile_at(float(world_x), float(world_z), seed, height_limit)
        solid_limit = min(height_limit, int(surface_height))
        for y in range(solid_limit):
            material = _terrain_material_from_surface_profile(
                world_x,
                y,
                world_z,
                int(surface_height),
                int(surface_material),
                seed,
                height_limit,
            )
            if material == AIR:
                continue
            blocks[y, local_z, local_x] = 1
            materials[y, local_z, local_x] = material


@njit(cache=True, fastmath=True, parallel=True)
def fill_stacked_chunk_voxel_grid(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
) -> None:
    sample_size = chunk_size + 2
    origin_x = chunk_x * chunk_size - 1
    origin_z = chunk_z * chunk_size - 1
    origin_y = chunk_y * chunk_size
    local_height = blocks.shape[0]
    fill_start_y = max(origin_y, 0)
    fill_top_y = min(origin_y + local_height, world_height_limit)
    total_columns = sample_size * sample_size

    for column_index in prange(total_columns):
        local_z = column_index // sample_size
        local_x = column_index - local_z * sample_size
        world_z = origin_z + local_z
        world_x = origin_x + local_x
        surface_height, surface_material = surface_profile_at(float(world_x), float(world_z), seed, world_height_limit)
        fill_end_y = min(fill_top_y, int(surface_height))
        if fill_end_y <= fill_start_y:
            continue
        for world_y in range(fill_start_y, fill_end_y):
            local_y = world_y - origin_y
            material = _terrain_material_from_surface_profile(
                world_x,
                world_y,
                world_z,
                int(surface_height),
                int(surface_material),
                seed,
                world_height_limit,
            )
            if material == AIR:
                continue
            blocks[local_y, local_z, local_x] = 1
            materials[local_y, local_z, local_x] = material


@njit(cache=True, fastmath=True, parallel=True)
def fill_stacked_chunk_vertical_neighbor_planes(
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
) -> None:
    sample_size = chunk_size + 2
    origin_x = chunk_x * chunk_size - 1
    origin_z = chunk_z * chunk_size - 1
    top_world_y = chunk_y * chunk_size + chunk_size
    bottom_world_y = chunk_y * chunk_size - 1
    top_in_bounds = 0 <= top_world_y < world_height_limit
    bottom_in_bounds = 0 <= bottom_world_y < world_height_limit
    total_columns = sample_size * sample_size

    for column_index in prange(total_columns):
        local_z = column_index // sample_size
        local_x = column_index - local_z * sample_size
        world_z = origin_z + local_z
        world_x = origin_x + local_x
        surface_height, surface_material = surface_profile_at(float(world_x), float(world_z), seed, world_height_limit)
        solid_surface_height = int(surface_height)
        solid_surface_material = int(surface_material)
        if top_in_bounds and top_world_y < solid_surface_height:
            if _terrain_material_from_surface_profile(
                world_x,
                top_world_y,
                world_z,
                solid_surface_height,
                solid_surface_material,
                seed,
                world_height_limit,
            ) != AIR:
                top_plane[local_z, local_x] = 1
        if bottom_in_bounds and bottom_world_y < solid_surface_height:
            if _terrain_material_from_surface_profile(
                world_x,
                bottom_world_y,
                world_z,
                solid_surface_height,
                solid_surface_material,
                seed,
                world_height_limit,
            ) != AIR:
                bottom_plane[local_z, local_x] = 1


@njit(cache=True, fastmath=True, parallel=True)
def fill_stacked_chunk_voxel_grid_with_neighbor_planes(
    blocks: np.ndarray,
    materials: np.ndarray,
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
) -> None:
    sample_size = chunk_size + 2
    probe_heights = np.empty(sample_size * sample_size, dtype=np.uint32)
    probe_materials = np.empty(sample_size * sample_size, dtype=np.uint32)
    fill_chunk_surface_grids(
        probe_heights,
        probe_materials,
        chunk_x,
        chunk_z,
        chunk_size,
        seed,
        world_height_limit,
    )
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        blocks,
        materials,
        top_plane,
        bottom_plane,
        probe_heights,
        probe_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height_limit,
    )


@njit(cache=True, fastmath=True, parallel=True)
def fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
    blocks: np.ndarray,
    materials: np.ndarray,
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
) -> None:
    sample_size = chunk_size + 2
    origin_x = chunk_x * chunk_size - 1
    origin_z = chunk_z * chunk_size - 1
    origin_y = chunk_y * chunk_size
    local_height = blocks.shape[0]
    fill_start_y = max(origin_y, 0)
    fill_top_y = min(origin_y + local_height, world_height_limit)
    top_world_y = origin_y + local_height
    bottom_world_y = origin_y - 1
    top_in_bounds = 0 <= top_world_y < world_height_limit
    bottom_in_bounds = 0 <= bottom_world_y < world_height_limit
    total_columns = sample_size * sample_size

    for column_index in prange(total_columns):
        local_z = column_index // sample_size
        local_x = column_index - local_z * sample_size
        world_z = origin_z + local_z
        world_x = origin_x + local_x
        solid_surface_height = int(surface_heights[column_index])
        solid_surface_material = int(surface_materials[column_index])

        fill_end_y = min(fill_top_y, solid_surface_height)
        stone_limit = solid_surface_height - 4
        dirt_limit = solid_surface_height - 1
        if fill_end_y > fill_start_y:
            block_column = blocks[:, local_z, local_x]
            material_column = materials[:, local_z, local_x]
            for world_y in range(fill_start_y, fill_end_y):
                if _should_carve_cave(world_x, world_y, world_z, solid_surface_height, seed, world_height_limit):
                    continue
                local_y = world_y - origin_y
                if world_y == 0:
                    material = BEDROCK
                elif world_y < stone_limit:
                    material = STONE
                elif world_y < dirt_limit:
                    material = DIRT
                else:
                    material = solid_surface_material
                block_column[local_y] = 1
                material_column[local_y] = material

        if top_in_bounds and top_world_y < solid_surface_height:
            if not _should_carve_cave(world_x, top_world_y, world_z, solid_surface_height, seed, world_height_limit):
                top_plane[local_z, local_x] = 1

        if bottom_in_bounds and bottom_world_y < solid_surface_height:
            if not _should_carve_cave(world_x, bottom_world_y, world_z, solid_surface_height, seed, world_height_limit):
                bottom_plane[local_z, local_x] = 1


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


@njit(cache=True, fastmath=True, inline="always")
def _solid_at_with_boundaries(
    blocks: np.ndarray,
    local_x: int,
    local_z: int,
    sample_y: int,
    height_limit: int,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
) -> int:
    if sample_y < 0:
        return int(bottom_boundary[local_z, local_x] != 0)
    if sample_y >= height_limit:
        return int(top_boundary[local_z, local_x] != 0)
    return int(blocks[sample_y, local_z, local_x] != 0)


@njit(cache=True, fastmath=True, inline="always")
def _ambient_occlusion_factor(side1: int, side2: int, corner: int) -> float:
    if side1 != 0 and side2 != 0:
        occlusion = 3
    else:
        occlusion = side1 + side2 + corner
    if occlusion <= 0:
        return 1.0
    if occlusion == 1:
        return 0.82
    if occlusion == 2:
        return 0.68
    return 0.54


@njit(cache=True, fastmath=True, inline="always")
def _ao_y_from_plane(
    plane: np.ndarray,
    local_x: int,
    local_z: int,
    dx: int,
    dz: int,
) -> float:
    side1 = int(plane[local_z, local_x + dx] != 0)
    side2 = int(plane[local_z + dz, local_x] != 0)
    corner = int(plane[local_z + dz, local_x + dx] != 0)
    return _ambient_occlusion_factor(side1, side2, corner)


@njit(cache=True, fastmath=True, inline="always")
def _ao_x_from_planes(
    current_plane: np.ndarray,
    neighbor_y_plane: np.ndarray,
    sample_x: int,
    local_z: int,
    dz: int,
) -> float:
    side1 = int(neighbor_y_plane[local_z, sample_x] != 0)
    side2 = int(current_plane[local_z + dz, sample_x] != 0)
    corner = int(neighbor_y_plane[local_z + dz, sample_x] != 0)
    return _ambient_occlusion_factor(side1, side2, corner)


@njit(cache=True, fastmath=True, inline="always")
def _ao_z_from_planes(
    current_plane: np.ndarray,
    neighbor_y_plane: np.ndarray,
    local_x: int,
    sample_z: int,
    dx: int,
) -> float:
    side1 = int(current_plane[sample_z, local_x + dx] != 0)
    side2 = int(neighbor_y_plane[sample_z, local_x] != 0)
    corner = int(neighbor_y_plane[sample_z, local_x + dx] != 0)
    return _ambient_occlusion_factor(side1, side2, corner)


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
    vertex_count = count_chunk_voxel_vertices_with_boundaries(blocks, chunk_size, height_limit, top_boundary, bottom_boundary)
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
        plane_above = blocks[y + 1] if y < last_y else top_boundary
        plane_below = blocks[y - 1] if y > 0 else bottom_boundary
        y0 = origin_y + float(y) * step
        y1 = y0 + step

        z0 = origin_z
        for local_z in range(1, end):
            row = plane[local_z]
            row_mat = mat_plane[local_z]
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
                top_empty = row_above[local_x] == 0
                bottom_empty = row_below[local_x] == 0
                east_empty = row[local_x + 1] == 0
                west_empty = row[local_x - 1] == 0
                south_empty = row_south[local_x] == 0
                north_empty = row_north[local_x] == 0
                if not (top_empty or bottom_empty or east_empty or west_empty or south_empty or north_empty):
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

                if top_empty:
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

                if bottom_empty:
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

                if east_empty:
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

                if west_empty:
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

                if south_empty:
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

                if north_empty:
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
