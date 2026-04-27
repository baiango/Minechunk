from __future__ import annotations

import numpy as np

from .numba_compat import njit, prange
from .materials import AIR, BEDROCK, DIRT, STONE
from .terrain_profile import (
    _should_carve_cave,
    _terrain_material_from_surface_profile,
    surface_profile_at,
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


