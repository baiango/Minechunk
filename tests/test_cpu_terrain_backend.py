from __future__ import annotations

import numpy as np


def test_cpu_backend_label_reports_selected_numba_kernel(monkeypatch) -> None:
    from engine.terrain.backends import cpu_terrain_backend

    monkeypatch.setenv("MINECHUNK_TERRAIN_KERNEL", "numba")

    backend = cpu_terrain_backend.CpuTerrainBackend(seed=1, height=32, chunk_size=4)

    assert backend.terrain_backend_label() == "CPU/Numba"


def test_cpu_backend_label_reports_selected_zig_kernel(monkeypatch) -> None:
    from engine.terrain.backends import cpu_terrain_backend

    monkeypatch.setenv("MINECHUNK_TERRAIN_KERNEL", "zig")

    backend = cpu_terrain_backend.CpuTerrainBackend(seed=1, height=32, chunk_size=4)

    assert backend.terrain_backend_label() == "CPU/Zig"


def test_cpu_surface_poll_uses_batch_fill(monkeypatch) -> None:
    from engine.terrain.backends import cpu_terrain_backend

    calls = []

    def fake_fill_surface_batch(heights, materials, chunk_xs, chunk_zs, chunk_size, seed, height) -> None:
        calls.append((tuple(chunk_xs.tolist()), tuple(chunk_zs.tolist()), heights.shape))
        for index in range(len(chunk_xs)):
            heights[index].fill(index + 10)
            materials[index].fill(index + 20)

    monkeypatch.setattr(cpu_terrain_backend, "fill_chunk_surface_grids_batch", fake_fill_surface_batch)

    backend = cpu_terrain_backend.CpuTerrainBackend(seed=1, height=32, chunk_size=4, chunks_per_poll=2)
    backend.request_chunk_surface_batch([(3, 0, 5), (4, 0, 6), (7, 0, 8)])

    ready = backend.poll_ready_chunk_surface_batches()

    assert len(ready) == 2
    assert calls == [((3, 4), (5, 6), (2, 36))]
    np.testing.assert_array_equal(ready[0].heights, np.full(36, 10, dtype=np.uint32))
    np.testing.assert_array_equal(ready[1].materials, np.full(36, 21, dtype=np.uint32))


def test_cpu_voxel_poll_reuses_surface_grid_for_vertical_stack(monkeypatch) -> None:
    from engine.terrain.backends import cpu_terrain_backend

    calls: list[tuple[int, int]] = []

    def fake_fill_surface(heights, materials, chunk_x, chunk_z, chunk_size, seed, height) -> None:
        calls.append((int(chunk_x), int(chunk_z)))
        heights.fill(80)
        materials.fill(1)

    monkeypatch.setattr(cpu_terrain_backend, "fill_chunk_surface_grids", fake_fill_surface)

    backend = cpu_terrain_backend.CpuTerrainBackend(seed=1, height=256, chunk_size=64, chunks_per_poll=2)
    backend.request_chunk_voxel_batch([(3, 2, 5), (3, 3, 5)])

    ready = backend.poll_ready_chunk_voxel_batches()

    assert len(ready) == 2
    assert all(result.is_empty for result in ready)
    assert calls == [(3, 5)]


def test_cpu_voxel_poll_passes_cave_toggle_to_fill(monkeypatch) -> None:
    from engine.terrain.backends import cpu_terrain_backend

    calls: list[bool] = []

    def fake_fill_surface(heights, materials, chunk_x, chunk_z, chunk_size, seed, height) -> None:
        heights.fill(80)
        materials.fill(1)

    def fake_fill_voxels(*args) -> None:
        calls.append(bool(args[-1]))

    monkeypatch.setattr(cpu_terrain_backend, "fill_chunk_surface_grids", fake_fill_surface)
    monkeypatch.setattr(
        cpu_terrain_backend,
        "fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface",
        fake_fill_voxels,
    )

    backend = cpu_terrain_backend.CpuTerrainBackend(
        seed=1,
        height=256,
        chunk_size=64,
        chunks_per_poll=1,
        terrain_caves_enabled=True,
    )
    backend.request_chunk_voxel_batch([(3, 0, 5)])

    ready = backend.poll_ready_chunk_voxel_batches()

    assert len(ready) == 1
    assert calls == [True]


def test_cpu_voxel_poll_uses_surface_mesher_for_no_cave_surface_chunk(monkeypatch) -> None:
    from engine.terrain.backends import cpu_terrain_backend

    fill_calls = []

    def fake_fill_surface(heights, materials, chunk_x, chunk_z, chunk_size, seed, height) -> None:
        heights.fill(80)
        materials.fill(1)

    def fake_fill_voxels(*args) -> None:
        fill_calls.append(args)

    monkeypatch.setattr(cpu_terrain_backend, "fill_chunk_surface_grids", fake_fill_surface)
    monkeypatch.setattr(
        cpu_terrain_backend,
        "fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface",
        fake_fill_voxels,
    )

    backend = cpu_terrain_backend.CpuTerrainBackend(
        seed=1,
        height=256,
        chunk_size=64,
        chunks_per_poll=1,
        terrain_caves_enabled=False,
    )
    backend.request_chunk_voxel_batch([(3, 1, 5)])

    ready = backend.poll_ready_chunk_voxel_batches()

    assert len(ready) == 1
    assert ready[0].is_empty is False
    assert ready[0].is_fully_occluded is False
    assert ready[0].use_surface_mesher is True
    assert ready[0].surface_heights is not None
    assert ready[0].surface_materials is not None
    assert fill_calls == []


def test_cpu_voxel_poll_skips_no_cave_fully_occluded_chunk(monkeypatch) -> None:
    from engine.terrain.backends import cpu_terrain_backend

    fill_calls = []

    def fake_fill_surface(heights, materials, chunk_x, chunk_z, chunk_size, seed, height) -> None:
        heights.fill(200)
        materials.fill(1)

    def fake_fill_voxels(*args) -> None:
        fill_calls.append(args)

    monkeypatch.setattr(cpu_terrain_backend, "fill_chunk_surface_grids", fake_fill_surface)
    monkeypatch.setattr(
        cpu_terrain_backend,
        "fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface",
        fake_fill_voxels,
    )

    backend = cpu_terrain_backend.CpuTerrainBackend(
        seed=1,
        height=256,
        chunk_size=64,
        chunks_per_poll=1,
        terrain_caves_enabled=False,
    )
    backend.request_chunk_voxel_batch([(3, 1, 5)])

    ready = backend.poll_ready_chunk_voxel_batches()

    assert len(ready) == 1
    assert ready[0].is_empty is False
    assert ready[0].is_fully_occluded is True
    assert fill_calls == []
