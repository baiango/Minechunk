from __future__ import annotations

import numpy as np

from engine.terrain.kernels import materials as terrain_materials
from engine.terrain.kernels import terrain_profile, voxel_fill, zig_kernel


SURFACE_MATERIAL_IDS = {
    terrain_materials.STONE,
    terrain_materials.GRASS,
    terrain_materials.SAND,
    terrain_materials.SNOW,
}


def _assert_surface_drift_bounded(
    actual_heights: np.ndarray,
    actual_materials: np.ndarray,
    expected_heights: np.ndarray,
) -> None:
    assert actual_heights.dtype == np.uint32
    assert actual_materials.dtype == np.uint32
    assert np.max(np.abs(actual_heights.astype(np.int64) - expected_heights.astype(np.int64))) <= 1
    assert set(np.unique(actual_materials).tolist()) <= SURFACE_MATERIAL_IDS


def test_numba_f32_surface_grid_snapshot() -> None:
    chunk_size = 64
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size
    heights = np.empty(cell_count, dtype=np.uint32)
    materials = np.empty(cell_count, dtype=np.uint32)

    voxel_fill.fill_chunk_surface_grids(heights, materials, -2, 3, chunk_size, 1337, 2000)

    assert heights[:16].tolist() == [
        1283,
        1284,
        1285,
        1286,
        1287,
        1288,
        1289,
        1290,
        1291,
        1292,
        1294,
        1295,
        1296,
        1297,
        1298,
        1299,
    ]
    assert materials[:16].tolist() == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2]
    assert (int(heights.sum()), int(heights.min()), int(heights.max()), int(materials.sum())) == (
        5657235,
        1241,
        1325,
        9196,
    )


def test_zig_surface_profile_wrapper_tracks_numba_f32_reference() -> None:
    samples = [
        (-17, 5, 1337, 128),
        (0, 0, 1, 64),
        (123, -42, 98765, 256),
    ]

    for x, z, seed, height_limit in samples:
        expected = terrain_profile.surface_profile_at(float(x), float(z), seed, height_limit)
        actual = zig_kernel.surface_profile_at(float(x), float(z), seed, height_limit)
        assert abs(actual[0] - int(expected[0])) <= 1
        assert actual[1] in SURFACE_MATERIAL_IDS


def test_zig_surface_grid_wrapper_tracks_numba_f32_reference() -> None:
    chunk_size = 64
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size

    expected_heights = np.empty(cell_count, dtype=np.uint32)
    expected_materials = np.empty(cell_count, dtype=np.uint32)
    actual_heights = np.empty(cell_count, dtype=np.uint32)
    actual_materials = np.empty(cell_count, dtype=np.uint32)

    voxel_fill.fill_chunk_surface_grids(expected_heights, expected_materials, -2, 3, chunk_size, 1337, 2000)
    zig_kernel.fill_chunk_surface_grids(actual_heights, actual_materials, -2, 3, chunk_size, 1337, 2000)

    _assert_surface_drift_bounded(actual_heights, actual_materials, expected_heights)


def test_zig_surface_grid_batch_wrapper_matches_zig_single_chunk_path() -> None:
    chunk_size = 6
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size
    chunks = np.array([[-2, 3], [0, 0], [5, -4]], dtype=np.int32)

    expected_heights = np.empty((len(chunks), cell_count), dtype=np.uint32)
    expected_materials = np.empty((len(chunks), cell_count), dtype=np.uint32)
    actual_heights = np.empty_like(expected_heights)
    actual_materials = np.empty_like(expected_materials)

    for index, (chunk_x, chunk_z) in enumerate(chunks):
        zig_kernel.fill_chunk_surface_grids(
            expected_heights[index],
            expected_materials[index],
            int(chunk_x),
            int(chunk_z),
            chunk_size,
            1337,
            128,
        )

    zig_kernel.fill_chunk_surface_grids_batch(
        actual_heights,
        actual_materials,
        chunks[:, 0],
        chunks[:, 1],
        chunk_size,
        1337,
        128,
    )

    np.testing.assert_array_equal(actual_heights, expected_heights)
    np.testing.assert_array_equal(actual_materials, expected_materials)


def test_zig_stacked_voxel_from_surface_wrapper_matches_numba_cave_model() -> None:
    chunk_size = 16
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size
    local_height = 16
    chunk_x = -32
    chunk_y = 1
    chunk_z = 22
    seed = 1337
    world_height = 128

    surface_heights = np.empty(cell_count, dtype=np.uint32)
    surface_materials = np.empty(cell_count, dtype=np.uint32)
    voxel_fill.fill_chunk_surface_grids(
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_z,
        chunk_size,
        seed,
        world_height,
    )

    expected_blocks = np.zeros((local_height, sample_size, sample_size), dtype=np.uint8)
    expected_materials = np.zeros((local_height, sample_size, sample_size), dtype=np.uint32)
    expected_top = np.zeros((sample_size, sample_size), dtype=np.uint8)
    expected_bottom = np.zeros((sample_size, sample_size), dtype=np.uint8)

    actual_blocks = np.zeros_like(expected_blocks)
    actual_materials = np.zeros_like(expected_materials)
    actual_top = np.zeros_like(expected_top)
    actual_bottom = np.zeros_like(expected_bottom)

    voxel_fill.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        expected_blocks,
        expected_materials,
        expected_top,
        expected_bottom,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height,
        True,
    )
    zig_kernel.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        actual_blocks,
        actual_materials,
        actual_top,
        actual_bottom,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height,
        True,
    )

    np.testing.assert_array_equal(actual_blocks, expected_blocks)
    np.testing.assert_array_equal(actual_materials, expected_materials)
    np.testing.assert_array_equal(actual_top, expected_top)
    np.testing.assert_array_equal(actual_bottom, expected_bottom)
    assert 0 < int(actual_blocks.sum()) < actual_blocks.size


def test_3d_cave_punches_surface_and_matches_zig() -> None:
    chunk_size = 64
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size
    chunk_x = -180
    chunk_y = 21
    chunk_z = -180
    local_x = 30
    local_z = 1
    seed = 1337
    world_height = 2000
    world_x = chunk_x * chunk_size - 1 + local_x
    world_z = chunk_z * chunk_size - 1 + local_z
    surface_height, _ = terrain_profile.surface_profile_at(float(world_x), float(world_z), seed, world_height)
    world_y = int(surface_height) - 1
    local_y = world_y - chunk_y * chunk_size

    assert 0 <= local_y < chunk_size
    assert terrain_profile._should_carve_cave(world_x, world_y, world_z, int(surface_height), seed, world_height)
    assert terrain_profile.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, True) == terrain_materials.AIR
    assert terrain_profile.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, False) != terrain_materials.AIR
    assert zig_kernel.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, True) == terrain_materials.AIR
    assert zig_kernel.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, False) != terrain_materials.AIR
    assert terrain_profile.terrain_block_material_at(world_x, 0, world_z, seed, world_height, True) == terrain_materials.BEDROCK

    surface_heights = np.empty(cell_count, dtype=np.uint32)
    surface_materials = np.empty(cell_count, dtype=np.uint32)
    voxel_fill.fill_chunk_surface_grids(
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_z,
        chunk_size,
        seed,
        world_height,
    )

    expected_blocks = np.zeros((chunk_size, sample_size, sample_size), dtype=np.uint8)
    expected_materials = np.zeros((chunk_size, sample_size, sample_size), dtype=np.uint32)
    expected_top = np.zeros((sample_size, sample_size), dtype=np.uint8)
    expected_bottom = np.zeros((sample_size, sample_size), dtype=np.uint8)
    actual_blocks = np.zeros_like(expected_blocks)
    actual_materials = np.zeros_like(expected_materials)
    actual_top = np.zeros_like(expected_top)
    actual_bottom = np.zeros_like(expected_bottom)
    voxel_fill.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        expected_blocks,
        expected_materials,
        expected_top,
        expected_bottom,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height,
        True,
    )
    zig_kernel.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        actual_blocks,
        actual_materials,
        actual_top,
        actual_bottom,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height,
        True,
    )
    np.testing.assert_array_equal(actual_blocks, expected_blocks)
    np.testing.assert_array_equal(actual_materials, expected_materials)
    np.testing.assert_array_equal(actual_top, expected_top)
    np.testing.assert_array_equal(actual_bottom, expected_bottom)
    assert actual_blocks[local_y, local_z, local_x] == 0
    assert actual_materials[local_y, local_z, local_x] == terrain_materials.AIR


def test_3d_caves_continue_to_low_world_layers_and_match_zig() -> None:
    chunk_size = 64
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size
    chunk_x = -8
    chunk_y = 0
    chunk_z = -5
    local_x = 1
    local_y = 8
    local_z = 40
    seed = 1337
    world_height = 2000
    world_x = chunk_x * chunk_size - 1 + local_x
    world_y = chunk_y * chunk_size + local_y
    world_z = chunk_z * chunk_size - 1 + local_z
    surface_height, _ = terrain_profile.surface_profile_at(float(world_x), float(world_z), seed, world_height)

    assert world_y < int(world_height * 0.18)
    assert terrain_profile._should_carve_cave(world_x, world_y, world_z, int(surface_height), seed, world_height)
    assert terrain_profile.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, True) == terrain_materials.AIR
    assert terrain_profile.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, False) != terrain_materials.AIR
    assert zig_kernel.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, True) == terrain_materials.AIR
    assert zig_kernel.terrain_block_material_at(world_x, world_y, world_z, seed, world_height, False) != terrain_materials.AIR
    assert terrain_profile.terrain_block_material_at(world_x, 0, world_z, seed, world_height, True) == terrain_materials.BEDROCK

    surface_heights = np.empty(cell_count, dtype=np.uint32)
    surface_materials = np.empty(cell_count, dtype=np.uint32)
    voxel_fill.fill_chunk_surface_grids(
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_z,
        chunk_size,
        seed,
        world_height,
    )

    expected_blocks = np.zeros((chunk_size, sample_size, sample_size), dtype=np.uint8)
    expected_materials = np.zeros((chunk_size, sample_size, sample_size), dtype=np.uint32)
    expected_top = np.zeros((sample_size, sample_size), dtype=np.uint8)
    expected_bottom = np.zeros((sample_size, sample_size), dtype=np.uint8)
    actual_blocks = np.zeros_like(expected_blocks)
    actual_materials = np.zeros_like(expected_materials)
    actual_top = np.zeros_like(expected_top)
    actual_bottom = np.zeros_like(expected_bottom)
    voxel_fill.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        expected_blocks,
        expected_materials,
        expected_top,
        expected_bottom,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height,
        True,
    )
    zig_kernel.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        actual_blocks,
        actual_materials,
        actual_top,
        actual_bottom,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height,
        True,
    )

    np.testing.assert_array_equal(actual_blocks, expected_blocks)
    np.testing.assert_array_equal(actual_materials, expected_materials)
    np.testing.assert_array_equal(actual_top, expected_top)
    np.testing.assert_array_equal(actual_bottom, expected_bottom)
    assert actual_blocks[local_y, local_z, local_x] == 0
    assert actual_materials[local_y, local_z, local_x] == terrain_materials.AIR
    assert actual_blocks[0, local_z, local_x] == 1
    assert actual_materials[0, local_z, local_x] == terrain_materials.BEDROCK


def test_zig_stacked_voxel_from_surface_batch_wrapper_matches_zig_single_chunk_path() -> None:
    chunk_size = 6
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size
    local_height = 6
    chunks = np.array([[-1, 2, 4], [2, 0, -3], [0, 4, 1]], dtype=np.int32)
    seed = 1337
    world_height = 96

    surface_heights = np.empty((len(chunks), cell_count), dtype=np.uint32)
    surface_materials = np.empty((len(chunks), cell_count), dtype=np.uint32)
    expected_blocks = np.zeros((len(chunks), local_height, sample_size, sample_size), dtype=np.uint8)
    expected_materials = np.zeros((len(chunks), local_height, sample_size, sample_size), dtype=np.uint32)
    expected_top = np.zeros((len(chunks), sample_size, sample_size), dtype=np.uint8)
    expected_bottom = np.zeros((len(chunks), sample_size, sample_size), dtype=np.uint8)

    actual_blocks = np.zeros_like(expected_blocks)
    actual_materials = np.zeros_like(expected_materials)
    actual_top = np.zeros_like(expected_top)
    actual_bottom = np.zeros_like(expected_bottom)

    for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunks):
        voxel_fill.fill_chunk_surface_grids(
            surface_heights[index],
            surface_materials[index],
            int(chunk_x),
            int(chunk_z),
            chunk_size,
            seed,
            world_height,
        )
        zig_kernel.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
            expected_blocks[index],
            expected_materials[index],
            expected_top[index],
            expected_bottom[index],
            surface_heights[index],
            surface_materials[index],
            int(chunk_x),
            int(chunk_y),
            int(chunk_z),
            chunk_size,
            seed,
            world_height,
            True,
        )

    zig_kernel.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch(
        actual_blocks,
        actual_materials,
        actual_top,
        actual_bottom,
        surface_heights,
        surface_materials,
        chunks,
        chunk_size,
        seed,
        world_height,
        True,
    )

    np.testing.assert_array_equal(actual_blocks, expected_blocks)
    np.testing.assert_array_equal(actual_materials, expected_materials)
    np.testing.assert_array_equal(actual_top, expected_top)
    np.testing.assert_array_equal(actual_bottom, expected_bottom)
