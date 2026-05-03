from __future__ import annotations

import numpy as np

from engine.terrain.kernels.materials import DIRT, GRASS
from engine.terrain.kernels.voxel_mesher import (
    build_chunk_surface_run_table_from_heightmap_clipped,
    build_chunk_surface_vertex_array_from_heightmap_clipped,
    build_chunk_vertex_array_from_voxels_with_boundaries,
    count_chunk_surface_vertices_from_heightmap_clipped,
    emit_chunk_surface_run_table_vertices,
    emit_chunk_surface_vertices_from_heightmap_clipped,
)


def _empty_planes(sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((sample_size, sample_size), dtype=np.uint8),
        np.zeros((sample_size, sample_size), dtype=np.uint8),
    )


def test_flat_voxel_slab_uses_run_length_quads() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    blocks = np.zeros((1, sample_size, sample_size), dtype=np.uint8)
    materials = np.zeros((1, sample_size, sample_size), dtype=np.uint32)
    blocks[0, 1:-1, 1:-1] = 1
    materials[0, 1:-1, 1:-1] = GRASS
    top, bottom = _empty_planes(sample_size)

    vertices, vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        blocks,
        materials,
        0,
        0,
        chunk_size,
        1,
        top,
        bottom,
        1.0,
        0,
    )

    assert vertex_count == 72
    assert vertices.shape == (72, 9)
    assert float(vertices[:, 0].min()) == 0.0
    assert float(vertices[:, 0].max()) == 4.0
    assert float(vertices[:, 2].min()) == 0.0
    assert float(vertices[:, 2].max()) == 4.0
    np.testing.assert_allclose(
        np.unique(vertices[:, 6:9], axis=0),
        np.array([[0.31, 0.68, 0.24]], dtype=np.float32),
    )


def test_surface_heightmap_mesher_builds_clipped_no_cave_chunk() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    heights = np.full(sample_size * sample_size, 3, dtype=np.uint32)
    materials = np.full(sample_size * sample_size, GRASS, dtype=np.uint32)

    vertices, vertex_count = build_chunk_surface_vertex_array_from_heightmap_clipped(
        heights,
        materials,
        0,
        0,
        chunk_size,
        4,
        1.0,
        0,
    )

    assert vertex_count > 0
    assert vertices.shape == (vertex_count, 9)
    assert float(vertices[:, 0].min()) == 0.0
    assert float(vertices[:, 0].max()) == 4.0
    assert float(vertices[:, 1].min()) == 0.0
    assert float(vertices[:, 1].max()) == 3.0
    assert float(vertices[:, 2].min()) == 0.0
    assert float(vertices[:, 2].max()) == 4.0


def test_surface_heightmap_mesher_can_emit_into_existing_batch_array() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    heights = np.zeros(sample_size * sample_size, dtype=np.uint32)
    materials = np.full(sample_size * sample_size, GRASS, dtype=np.uint32)
    for local_z in range(1, chunk_size + 1):
        for local_x in range(1, chunk_size + 1):
            heights[local_z * sample_size + local_x] = 5 + (local_x & 1)

    vertices, vertex_count = build_chunk_surface_vertex_array_from_heightmap_clipped(
        heights,
        materials,
        2,
        -3,
        chunk_size,
        8,
        1.0,
        0,
    )
    counted_vertex_count = count_chunk_surface_vertices_from_heightmap_clipped(
        heights,
        materials,
        chunk_size,
        8,
        0,
    )
    batched_vertices = np.full((vertex_count + 4, 9), -1.0, dtype=np.float32)
    emitted_vertex_count = emit_chunk_surface_vertices_from_heightmap_clipped(
        batched_vertices,
        2,
        heights,
        materials,
        2,
        -3,
        chunk_size,
        8,
        1.0,
        0,
    )

    assert counted_vertex_count == vertex_count
    assert emitted_vertex_count == vertex_count
    np.testing.assert_allclose(batched_vertices[2 : 2 + vertex_count], vertices)
    np.testing.assert_allclose(batched_vertices[:2], -1.0)
    np.testing.assert_allclose(batched_vertices[2 + vertex_count :], -1.0)


def test_surface_heightmap_run_table_matches_direct_emission() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    heights = np.zeros(sample_size * sample_size, dtype=np.uint32)
    materials = np.full(sample_size * sample_size, GRASS, dtype=np.uint32)
    for local_z in range(1, chunk_size + 1):
        for local_x in range(1, chunk_size + 1):
            heights[local_z * sample_size + local_x] = 6 + ((local_x + local_z) & 1)

    vertices, vertex_count = build_chunk_surface_vertex_array_from_heightmap_clipped(
        heights,
        materials,
        -2,
        3,
        chunk_size,
        8,
        1.0,
        0,
    )
    run_table, run_count, run_vertex_count = build_chunk_surface_run_table_from_heightmap_clipped(
        heights,
        materials,
        chunk_size,
        8,
        0,
    )
    run_vertices = np.empty((run_vertex_count, 9), dtype=np.float32)
    emitted_vertex_count = emit_chunk_surface_run_table_vertices(
        run_vertices,
        0,
        run_table,
        run_count,
        -2,
        3,
        chunk_size,
        1.0,
    )

    assert run_count * 6 == run_vertex_count == vertex_count
    assert emitted_vertex_count == vertex_count
    np.testing.assert_allclose(run_vertices, vertices)


def test_surface_heightmap_mesher_merges_top_rectangles_and_side_runs() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    heights = np.zeros(sample_size * sample_size, dtype=np.uint32)
    materials = np.full(sample_size * sample_size, GRASS, dtype=np.uint32)
    for local_z in range(1, chunk_size + 1):
        for local_x in range(1, chunk_size + 1):
            heights[local_z * sample_size + local_x] = 8

    vertices, vertex_count = build_chunk_surface_vertex_array_from_heightmap_clipped(
        heights,
        materials,
        0,
        0,
        chunk_size,
        8,
        1.0,
        0,
    )

    assert vertex_count == 54
    assert vertices.shape == (54, 9)
    top_vertices = vertices[np.isclose(vertices[:, 4], 1.0)]
    assert top_vertices.shape[0] == 6
    bottom_vertices = vertices[np.isclose(vertices[:, 4], -1.0)]
    assert bottom_vertices.shape[0] == 24
    side_vertices = vertices[np.isclose(np.abs(vertices[:, 3]) + np.abs(vertices[:, 5]), 1.0)]
    assert side_vertices.shape[0] == 24


def test_run_length_mesher_splits_runs_on_material_changes() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    top, bottom = _empty_planes(sample_size)

    uniform_blocks = np.zeros((1, sample_size, sample_size), dtype=np.uint8)
    uniform_materials = np.zeros((1, sample_size, sample_size), dtype=np.uint32)
    uniform_blocks[0, 1, 1:-1] = 1
    uniform_materials[0, 1, 1:-1] = GRASS

    mixed_blocks = uniform_blocks.copy()
    mixed_materials = uniform_materials.copy()
    mixed_materials[0, 1, 2] = DIRT
    mixed_materials[0, 1, 4] = DIRT

    _uniform_vertices, uniform_vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        uniform_blocks,
        uniform_materials,
        0,
        0,
        chunk_size,
        1,
        top,
        bottom,
        1.0,
        0,
    )
    _mixed_vertices, mixed_vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        mixed_blocks,
        mixed_materials,
        0,
        0,
        chunk_size,
        1,
        top,
        bottom,
        1.0,
        0,
    )

    assert mixed_vertex_count > uniform_vertex_count


def test_run_length_mesher_respects_x_axis_boundary_samples() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    top, bottom = _empty_planes(sample_size)

    open_blocks = np.zeros((1, sample_size, sample_size), dtype=np.uint8)
    open_materials = np.zeros((1, sample_size, sample_size), dtype=np.uint32)
    open_blocks[0, 1, chunk_size] = 1
    open_materials[0, 1, chunk_size] = GRASS

    closed_blocks = open_blocks.copy()
    closed_materials = open_materials.copy()
    closed_blocks[0, 1, chunk_size + 1] = 1
    closed_materials[0, 1, chunk_size + 1] = GRASS

    _open_vertices, open_vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        open_blocks,
        open_materials,
        0,
        0,
        chunk_size,
        1,
        top,
        bottom,
        1.0,
        0,
    )
    _closed_vertices, closed_vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        closed_blocks,
        closed_materials,
        0,
        0,
        chunk_size,
        1,
        top,
        bottom,
        1.0,
        0,
    )

    assert open_vertex_count == 36
    assert closed_vertex_count == 30


def test_run_length_mesher_uses_z_runs_for_east_west_faces() -> None:
    chunk_size = 4
    sample_size = chunk_size + 2
    top, bottom = _empty_planes(sample_size)

    uniform_blocks = np.zeros((1, sample_size, sample_size), dtype=np.uint8)
    uniform_materials = np.zeros((1, sample_size, sample_size), dtype=np.uint32)
    uniform_blocks[0, 1:-1, 1] = 1
    uniform_materials[0, 1:-1, 1] = GRASS

    mixed_blocks = uniform_blocks.copy()
    mixed_materials = uniform_materials.copy()
    mixed_materials[0, 2, 1] = DIRT
    mixed_materials[0, 4, 1] = DIRT

    _uniform_vertices, uniform_vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        uniform_blocks,
        uniform_materials,
        0,
        0,
        chunk_size,
        1,
        top,
        bottom,
        1.0,
        0,
    )
    _mixed_vertices, mixed_vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        mixed_blocks,
        mixed_materials,
        0,
        0,
        chunk_size,
        1,
        top,
        bottom,
        1.0,
        0,
    )

    assert uniform_vertex_count == 72
    assert mixed_vertex_count == 108
