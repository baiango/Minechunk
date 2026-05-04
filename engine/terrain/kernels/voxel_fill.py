from __future__ import annotations

import numpy as np

from .numba_compat import njit, prange
from .materials import AIR, BEDROCK, DIRT, STONE
from .terrain_profile import (
    CAVE_ACTIVE_BAND_MIN,
    CAVE_BEDROCK_CLEARANCE,
    CAVE_MODEL_VERSION,
    _should_carve_cave,
    _terrain_material_from_surface_profile,
    surface_profile_at,
)


@njit(cache=True, fastmath=True, inline="always")
def _cave_vertical_band_start_y(world_height_limit: int) -> int:
    cave_model_version = CAVE_MODEL_VERSION
    if cave_model_version >= 14:
        return 0
    active_min = np.float32(CAVE_ACTIVE_BAND_MIN)
    if cave_model_version < 13:
        active_min = np.float32(CAVE_ACTIVE_BAND_MIN)
    denominator = np.float32(max(1, world_height_limit - 1))
    active_half_width = (np.float32(1.0) - active_min) / np.float32(1.6)
    lower = (np.float32(0.45) - active_half_width) * denominator
    start_y = int(np.floor(lower)) - 1
    if start_y < 0:
        return 0
    return start_y


@njit(cache=True, fastmath=True, inline="always")
def _cave_vertical_band_end_y(world_height_limit: int) -> int:
    cave_model_version = CAVE_MODEL_VERSION
    active_min = np.float32(CAVE_ACTIVE_BAND_MIN)
    if cave_model_version < 13:
        active_min = np.float32(CAVE_ACTIVE_BAND_MIN)
    denominator = np.float32(max(1, world_height_limit - 1))
    active_half_width = (np.float32(1.0) - active_min) / np.float32(1.6)
    upper = (np.float32(0.45) + active_half_width) * denominator
    end_y = int(np.ceil(upper)) + 2
    if end_y > world_height_limit:
        return world_height_limit
    return end_y


@njit(cache=True, fastmath=True, inline="always")
def _fill_material_segment(
    block_column: np.ndarray,
    material_column: np.ndarray,
    origin_y: int,
    start_y: int,
    end_y: int,
    material: int,
) -> None:
    if end_y <= start_y:
        return
    for world_y in range(start_y, end_y):
        local_y = world_y - origin_y
        block_column[local_y] = 1
        material_column[local_y] = material


@njit(cache=True, fastmath=True, inline="always")
def _fill_solid_column_no_caves(
    block_column: np.ndarray,
    material_column: np.ndarray,
    origin_y: int,
    fill_start_y: int,
    fill_end_y: int,
    stone_limit: int,
    dirt_limit: int,
    surface_material: int,
) -> None:
    if fill_start_y == 0 and fill_end_y > 0:
        _fill_material_segment(block_column, material_column, origin_y, 0, 1, BEDROCK)
    _fill_material_segment(block_column, material_column, origin_y, max(fill_start_y, 1), min(fill_end_y, stone_limit), STONE)
    _fill_material_segment(block_column, material_column, origin_y, max(fill_start_y, stone_limit), min(fill_end_y, dirt_limit), DIRT)
    _fill_material_segment(block_column, material_column, origin_y, max(fill_start_y, dirt_limit), fill_end_y, surface_material)


@njit(cache=True, fastmath=True, inline="always")
def _fill_cave_checked_range(
    block_column: np.ndarray,
    material_column: np.ndarray,
    origin_y: int,
    start_y: int,
    end_y: int,
    stone_limit: int,
    dirt_limit: int,
    surface_height: int,
    surface_material: int,
    world_x: int,
    world_z: int,
    seed: int,
    world_height_limit: int,
) -> None:
    for world_y in range(start_y, end_y):
        if _should_carve_cave(world_x, world_y, world_z, surface_height, seed, world_height_limit):
            continue
        local_y = world_y - origin_y
        if world_y == 0:
            material = BEDROCK
        elif world_y < stone_limit:
            material = STONE
        elif world_y < dirt_limit:
            material = DIRT
        else:
            material = surface_material
        block_column[local_y] = 1
        material_column[local_y] = material


@njit(cache=True, fastmath=True, inline="always")
def _fill_column_with_optional_caves(
    block_column: np.ndarray,
    material_column: np.ndarray,
    origin_y: int,
    fill_start_y: int,
    fill_end_y: int,
    stone_limit: int,
    dirt_limit: int,
    surface_height: int,
    surface_material: int,
    world_x: int,
    world_z: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool,
) -> None:
    if fill_end_y <= fill_start_y:
        return
    if not carve_caves:
        _fill_solid_column_no_caves(block_column, material_column, origin_y, fill_start_y, fill_end_y, stone_limit, dirt_limit, surface_material)
        return

    cave_model_version = CAVE_MODEL_VERSION
    if cave_model_version < 13:
        _fill_solid_column_no_caves(block_column, material_column, origin_y, fill_start_y, fill_end_y, stone_limit, dirt_limit, surface_material)
        return

    cave_start_y = max(fill_start_y, CAVE_BEDROCK_CLEARANCE + 1)
    cave_start_y = max(cave_start_y, _cave_vertical_band_start_y(world_height_limit))
    cave_end_y = min(fill_end_y, surface_height)
    cave_end_y = min(cave_end_y, world_height_limit - 2)
    cave_end_y = min(cave_end_y, _cave_vertical_band_end_y(world_height_limit))

    current_y = fill_start_y
    if cave_end_y > cave_start_y:
        _fill_solid_column_no_caves(block_column, material_column, origin_y, current_y, cave_start_y, stone_limit, dirt_limit, surface_material)
        _fill_cave_checked_range(
            block_column,
            material_column,
            origin_y,
            max(current_y, cave_start_y),
            cave_end_y,
            stone_limit,
            dirt_limit,
            surface_height,
            surface_material,
            world_x,
            world_z,
            seed,
            world_height_limit,
        )
        current_y = cave_end_y

    _fill_solid_column_no_caves(block_column, material_column, origin_y, current_y, fill_end_y, stone_limit, dirt_limit, surface_material)

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



@njit(cache=True, fastmath=True, parallel=True)
def fill_chunk_voxel_grid(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    height_limit: int,
    carve_caves: bool = True,
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
        _fill_column_with_optional_caves(
            blocks[:, local_z, local_x],
            materials[:, local_z, local_x],
            0,
            0,
            solid_limit,
            int(surface_height) - 4,
            int(surface_height) - 1,
            int(surface_height),
            int(surface_material),
            world_x,
            world_z,
            seed,
            height_limit,
            carve_caves,
        )


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
    carve_caves: bool = True,
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
        _fill_column_with_optional_caves(
            blocks[:, local_z, local_x],
            materials[:, local_z, local_x],
            origin_y,
            fill_start_y,
            fill_end_y,
            int(surface_height) - 4,
            int(surface_height) - 1,
            int(surface_height),
            int(surface_material),
            world_x,
            world_z,
            seed,
            world_height_limit,
            carve_caves,
        )


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
    carve_caves: bool = True,
) -> None:
    sample_size = chunk_size + 2
    origin_x = chunk_x * chunk_size - 1
    origin_z = chunk_z * chunk_size - 1
    top_world_y = chunk_y * chunk_size + chunk_size
    bottom_world_y = chunk_y * chunk_size - 1
    top_in_bounds = 0 <= top_world_y < world_height_limit
    bottom_in_bounds = 0 <= bottom_world_y < world_height_limit
    total_columns = sample_size * sample_size
    cave_model_version = CAVE_MODEL_VERSION

    for column_index in prange(total_columns):
        local_z = column_index // sample_size
        local_x = column_index - local_z * sample_size
        world_z = origin_z + local_z
        world_x = origin_x + local_x
        surface_height, surface_material = surface_profile_at(float(world_x), float(world_z), seed, world_height_limit)
        solid_surface_height = int(surface_height)
        solid_surface_material = int(surface_material)
        if top_in_bounds and top_world_y < solid_surface_height:
            if cave_model_version >= 13 and _terrain_material_from_surface_profile(
                world_x,
                top_world_y,
                world_z,
                solid_surface_height,
                solid_surface_material,
                seed,
                world_height_limit,
                carve_caves,
            ) != AIR:
                top_plane[local_z, local_x] = 1
        if bottom_in_bounds and bottom_world_y < solid_surface_height:
            if cave_model_version >= 13 and _terrain_material_from_surface_profile(
                world_x,
                bottom_world_y,
                world_z,
                solid_surface_height,
                solid_surface_material,
                seed,
                world_height_limit,
                carve_caves,
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
    carve_caves: bool = True,
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
        carve_caves,
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
    carve_caves: bool = True,
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
    cave_model_version = CAVE_MODEL_VERSION

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
            _fill_column_with_optional_caves(
                blocks[:, local_z, local_x],
                materials[:, local_z, local_x],
                origin_y,
                fill_start_y,
                fill_end_y,
                stone_limit,
                dirt_limit,
                solid_surface_height,
                solid_surface_material,
                world_x,
                world_z,
                seed,
                world_height_limit,
                carve_caves,
            )

        if top_in_bounds and top_world_y < solid_surface_height:
            if not carve_caves or (cave_model_version >= 13 and not _should_carve_cave(world_x, top_world_y, world_z, solid_surface_height, seed, world_height_limit)):
                top_plane[local_z, local_x] = 1

        if bottom_in_bounds and bottom_world_y < solid_surface_height:
            if not carve_caves or (cave_model_version >= 13 and not _should_carve_cave(world_x, bottom_world_y, world_z, solid_surface_height, seed, world_height_limit)):
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
