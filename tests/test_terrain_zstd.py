from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from engine.terrain.types import ChunkVoxelResult


def _sample_voxel_result(*, empty: bool = False, boundaries: bool = True, fully_occluded: bool = False) -> ChunkVoxelResult:
    blocks = np.zeros((4, 6, 6), dtype=np.uint8)
    materials = np.zeros((4, 6, 6), dtype=np.uint32)
    if not empty:
        blocks[1, 2, 3] = 1
        blocks[2, 3, 4] = 1
        materials[1, 2, 3] = 2
        materials[2, 3, 4] = 3
    top_boundary = np.ones((6, 6), dtype=np.uint8) if boundaries else None
    bottom_boundary = np.tri(6, 6, dtype=np.uint8) if boundaries else None
    return ChunkVoxelResult(
        chunk_x=4,
        chunk_y=2,
        chunk_z=-3,
        blocks=blocks,
        materials=materials,
        source="test",
        top_boundary=top_boundary,
        bottom_boundary=bottom_boundary,
        is_empty=empty,
        is_fully_occluded=fully_occluded,
    )


@pytest.mark.parametrize(
    ("empty", "boundaries"),
    [
        (False, True),
        (True, False),
        (False, False),
    ],
)
def test_chunk_voxel_zstd_round_trip_preserves_payload(empty: bool, boundaries: bool) -> None:
    pytest.importorskip("zstandard")
    from engine.terrain.compression import compress_chunk_voxel_result, decompress_chunk_voxel_result

    result = _sample_voxel_result(empty=empty, boundaries=boundaries, fully_occluded=(empty and not boundaries))
    compressed = compress_chunk_voxel_result(result)
    assert compressed.compressed_nbytes == len(compressed.payload)
    assert compressed.blocks.offset == 0
    assert compressed.materials.offset == result.blocks.nbytes
    if compressed.top_boundary is not None:
        assert compressed.top_boundary.offset == result.blocks.nbytes + result.materials.nbytes
    restored = decompress_chunk_voxel_result(compressed)

    assert (restored.chunk_x, restored.chunk_y, restored.chunk_z) == (4, 2, -3)
    assert restored.source == "test"
    assert restored.is_empty is empty
    assert restored.is_fully_occluded is (empty and not boundaries)
    assert restored.blocks.dtype == result.blocks.dtype
    assert restored.materials.dtype == result.materials.dtype
    assert restored.blocks.flags.writeable
    assert restored.materials.flags.writeable
    np.testing.assert_array_equal(restored.blocks, result.blocks)
    np.testing.assert_array_equal(restored.materials, result.materials)
    if boundaries:
        np.testing.assert_array_equal(restored.top_boundary, result.top_boundary)
        np.testing.assert_array_equal(restored.bottom_boundary, result.bottom_boundary)
    else:
        assert restored.top_boundary is None
        assert restored.bottom_boundary is None

    restored_views = decompress_chunk_voxel_result(compressed, copy=False)
    assert restored_views.blocks.flags.c_contiguous
    assert restored_views.materials.flags.c_contiguous
    assert not restored_views.blocks.flags.writeable
    assert not restored_views.materials.flags.writeable
    np.testing.assert_array_equal(restored_views.blocks, result.blocks)
    np.testing.assert_array_equal(restored_views.materials, result.materials)


def test_chunk_voxel_zstd_round_trip_preserves_surface_mesher_payload() -> None:
    pytest.importorskip("zstandard")
    from engine.terrain.compression import compress_chunk_voxel_result, decompress_chunk_voxel_result

    surface_heights = np.arange(36, dtype=np.uint32)
    surface_materials = np.full(36, 1, dtype=np.uint32)
    result = replace(
        _sample_voxel_result(empty=True, boundaries=False),
        surface_heights=surface_heights,
        surface_materials=surface_materials,
        use_surface_mesher=True,
    )

    compressed = compress_chunk_voxel_result(result)
    restored = decompress_chunk_voxel_result(compressed)

    assert compressed.use_surface_mesher is True
    assert restored.use_surface_mesher is True
    np.testing.assert_array_equal(restored.surface_heights, surface_heights)
    np.testing.assert_array_equal(restored.surface_materials, surface_materials)


def test_voxel_world_zstd_cache_eviction_and_clear() -> None:
    pytest.importorskip("zstandard")
    from engine.terrain.world import VoxelWorld

    world = VoxelWorld(seed=1, terrain_zstd_enabled=True, terrain_zstd_cache_limit=1)

    world._store_terrain_zstd_result(replace(_sample_voxel_result(), chunk_x=1, chunk_y=0, chunk_z=1))
    assert world.terrain_zstd_cache_stats()["entries"] == 1
    assert (1, 0, 1) in world._terrain_zstd_cache

    world._store_terrain_zstd_result(replace(_sample_voxel_result(), chunk_x=2, chunk_y=0, chunk_z=2))
    assert world.terrain_zstd_cache_stats()["entries"] == 1
    assert (1, 0, 1) not in world._terrain_zstd_cache
    assert (2, 0, 2) in world._terrain_zstd_cache

    world.clear_terrain_zstd_cache()
    stats = world.terrain_zstd_cache_stats()
    assert stats["entries"] == 0
    assert stats["raw_bytes"] == 0
    assert stats["compressed_bytes"] == 0


def test_store_chunk_meshes_keeps_zstd_chunk_cache_after_meshing() -> None:
    pytest.importorskip("zstandard")
    from engine.cache.tile_mesh_cache import store_chunk_meshes
    from engine.meshing_types import ChunkMesh
    from engine.terrain.world import VoxelWorld

    world = VoxelWorld(seed=1, terrain_zstd_enabled=True, terrain_zstd_cache_limit=4)
    result = _sample_voxel_result(boundaries=True)
    world._store_terrain_zstd_result(result)
    key = (result.chunk_x, result.chunk_y, result.chunk_z)
    cached = world._terrain_zstd_cache[key]
    retained_buffers = []
    released_buffers = []
    renderer = SimpleNamespace(
        world=world,
        chunk_cache=OrderedDict(),
        _visible_chunk_coord_set=set(),
        _visible_tile_mesh_slots={},
        _visible_chunk_origin=None,
        _visible_rel_coord_to_tile_slot={},
        _visible_tile_base=(0, 0, 0),
        _visible_active_tile_key_set=set(),
        _visible_active_tile_keys=[],
        _visible_tile_active_meshes={},
        _tile_dirty_keys=set(),
        _tile_versions={},
        _tile_mutation_version=0,
        _visible_tile_key_set=set(),
        _visible_tile_dirty_keys=set(),
        _visible_tile_mutation_version=0,
        _cached_tile_draw_batches=[],
        _cached_visible_render_batches=[],
        _pending_chunk_coords={key},
        _visible_displayed_coords=set(),
        _visible_missing_coords=set(),
        _visible_display_state_dirty=False,
        _chunk_request_queue_dirty=False,
        max_cached_chunks=8,
        _retain_mesh_buffer=retained_buffers.append,
        _release_mesh_buffer=released_buffers.append,
    )
    mesh = ChunkMesh(
        chunk_x=result.chunk_x,
        chunk_y=result.chunk_y,
        chunk_z=result.chunk_z,
        vertex_count=0,
        vertex_buffer=object(),
        max_height=0,
    )

    store_chunk_meshes(renderer, [mesh])

    assert key not in renderer._pending_chunk_coords
    assert world._terrain_zstd_cache[key] is cached
    assert world.terrain_zstd_cache_stats()["entries"] == 1
    assert retained_buffers == [mesh.vertex_buffer]
    assert released_buffers == []


def test_voxel_world_uses_only_meshing_ready_cached_results_for_requests() -> None:
    pytest.importorskip("zstandard")
    from engine.terrain.world import VoxelWorld

    class DummyBackend:
        def __init__(self):
            self.requests: list[list[tuple[int, int, int]]] = []

        def chunk_voxel_grid(self, chunk_x, chunk_y, chunk_z):
            return np.ones((4, 6, 6), dtype=np.uint8), np.ones((4, 6, 6), dtype=np.uint32)

        def request_chunk_voxel_batch(self, chunks):
            self.requests.append(list(chunks))
            return len(self.requests)

        def poll_ready_chunk_voxel_batches(self):
            return []

        def has_pending_chunk_voxel_batches(self):
            return False

    backend = DummyBackend()
    world = VoxelWorld(seed=1, terrain_zstd_enabled=True, terrain_zstd_cache_limit=4)
    world._backend = backend

    world.chunk_voxel_grid(1, 0, 1)
    assert (1, 0, 1) not in world._terrain_zstd_cache
    world.request_chunk_voxel_batch([(1, 0, 1)])
    assert backend.requests == [[(1, 0, 1)]]

    ready_result = _sample_voxel_result(boundaries=True)
    world._store_terrain_zstd_result(ready_result)
    world.request_chunk_voxel_batch([(ready_result.chunk_x, ready_result.chunk_y, ready_result.chunk_z)])
    assert backend.requests == [[(1, 0, 1)]]
    assert len(world._ready_cached_voxel_results) == 1
    world.drop_terrain_zstd_cache_entries([(ready_result.chunk_x, ready_result.chunk_y, ready_result.chunk_z)])
    assert world.terrain_zstd_cache_stats()["entries"] == 0
    assert len(world._ready_cached_voxel_results) == 0

    world._store_terrain_zstd_result(ready_result)
    world.request_chunk_voxel_batch([(ready_result.chunk_x, ready_result.chunk_y, ready_result.chunk_z)])
    ready = world.poll_ready_chunk_voxel_batches()
    assert len(ready) == 1
    np.testing.assert_array_equal(ready[0].top_boundary, ready_result.top_boundary)


def test_voxel_world_payload_poll_reuses_cached_compressed_result_and_public_api_returns_arrays() -> None:
    pytest.importorskip("zstandard")
    from engine.terrain.compression import CompressedChunkVoxelResult
    from engine.terrain.world import VoxelWorld

    ready_result = _sample_voxel_result(boundaries=True)

    class DummyBackend:
        def __init__(self):
            self.ready = [ready_result]

        def poll_ready_chunk_voxel_batches(self):
            ready = self.ready
            self.ready = []
            return ready

        def flush_chunk_voxel_batches(self):
            return self.poll_ready_chunk_voxel_batches()

    world = VoxelWorld(seed=1, terrain_zstd_enabled=True, terrain_zstd_cache_limit=4)
    world._backend = DummyBackend()

    payloads = world.poll_ready_chunk_voxel_payloads()
    assert len(payloads) == 1
    assert isinstance(payloads[0], CompressedChunkVoxelResult)
    assert payloads[0] is world._terrain_zstd_cache[(ready_result.chunk_x, ready_result.chunk_y, ready_result.chunk_z)]

    from engine.pipelines.profiling_summary import terrain_zstd_runtime_stats
    from engine.pipelines.profiling_stats import record_frame_breakdown_sample

    renderer = SimpleNamespace(
        world=world,
        _pending_voxel_mesh_results=deque(payloads),
        frame_breakdown_samples={
            "terrain_zstd_stream_entries": deque(maxlen=4),
            "terrain_zstd_stream_raw_bytes": deque(maxlen=4),
            "terrain_zstd_stream_compressed_bytes": deque(maxlen=4),
        },
        frame_breakdown_sample_sums={
            "terrain_zstd_stream_entries": 0.0,
            "terrain_zstd_stream_raw_bytes": 0.0,
            "terrain_zstd_stream_compressed_bytes": 0.0,
        },
        _terrain_zstd_total_entries=3,
        _terrain_zstd_total_raw_bytes=payloads[0].raw_nbytes * 3,
        _terrain_zstd_total_compressed_bytes=payloads[0].compressed_nbytes * 3,
    )
    record_frame_breakdown_sample(renderer, "terrain_zstd_stream_entries", 1.0)
    record_frame_breakdown_sample(renderer, "terrain_zstd_stream_raw_bytes", float(payloads[0].raw_nbytes))
    record_frame_breakdown_sample(renderer, "terrain_zstd_stream_compressed_bytes", float(payloads[0].compressed_nbytes))
    stats = terrain_zstd_runtime_stats(renderer)
    assert stats["bypassed"] is False
    assert stats["cache_entries"] == 1
    assert stats["queue_entries"] == 1
    assert stats["live_entries"] == 1
    assert stats["cache_compressed_bytes"] == payloads[0].compressed_nbytes
    assert stats["queue_compressed_bytes"] == 0
    assert stats["raw_bytes"] == payloads[0].raw_nbytes
    assert stats["compressed_bytes"] == payloads[0].compressed_nbytes
    assert stats["stream_entries"] == 1.0
    assert stats["stream_raw_bytes"] == payloads[0].raw_nbytes
    assert stats["stream_compressed_bytes"] == payloads[0].compressed_nbytes
    assert stats["total_entries"] == 3
    assert stats["total_raw_bytes"] == payloads[0].raw_nbytes * 3
    assert stats["total_compressed_bytes"] == payloads[0].compressed_nbytes * 3

    world._backend.ready = [ready_result]
    raw_results = world.poll_ready_chunk_voxel_batches()
    assert len(raw_results) == 1
    assert not isinstance(raw_results[0], CompressedChunkVoxelResult)
    np.testing.assert_array_equal(raw_results[0].blocks, ready_result.blocks)
    np.testing.assert_array_equal(raw_results[0].materials, ready_result.materials)


def test_terrain_zstd_profile_marks_native_surface_bypass() -> None:
    from engine.pipelines.profiling_summary import terrain_zstd_runtime_stats

    world = SimpleNamespace(
        terrain_backend_label=lambda: "Wgpu",
        terrain_zstd_cache_stats=lambda: {
            "enabled": True,
            "entries": 0,
            "raw_bytes": 0,
            "compressed_bytes": 0,
        },
    )
    renderer = SimpleNamespace(
        world=world,
        use_gpu_meshing=True,
        mesh_backend_label="Wgpu",
        _pending_voxel_mesh_results=deque(),
        frame_breakdown_samples={},
    )
    stats = terrain_zstd_runtime_stats(renderer)
    assert stats["enabled"] is True
    assert stats["bypassed"] is True


def test_pipeline_pending_voxel_results_are_compressed_until_meshing() -> None:
    pytest.importorskip("zstandard")
    from engine.pipelines import chunk_pipeline
    from engine.terrain.compression import CompressedChunkVoxelResult, chunk_voxel_result_stream_nbytes

    renderer = SimpleNamespace(
        terrain_zstd_enabled=True,
        world=SimpleNamespace(terrain_zstd_enabled=True),
        _pending_voxel_mesh_results=deque(),
    )
    result = _sample_voxel_result(boundaries=True)

    chunk_pipeline._queue_voxel_result_for_meshing(renderer, result)
    pending = renderer._pending_voxel_mesh_results[0]
    assert isinstance(pending, CompressedChunkVoxelResult)
    assert chunk_voxel_result_stream_nbytes(pending) == result.blocks.nbytes + result.materials.nbytes

    renderer._pending_voxel_mesh_results.clear()
    chunk_pipeline._queue_voxel_result_for_meshing(renderer, pending)
    assert renderer._pending_voxel_mesh_results[0] is pending

    restored = chunk_pipeline._decompress_voxel_mesh_batch_for_meshing([pending])[0]
    assert not restored.blocks.flags.writeable
    assert not restored.materials.flags.writeable
    np.testing.assert_array_equal(restored.blocks, result.blocks)
    np.testing.assert_array_equal(restored.materials, result.materials)
    np.testing.assert_array_equal(restored.top_boundary, result.top_boundary)


def test_terrain_zstd_cli_defaults_and_launcher_flags() -> None:
    from engine import renderer_config as cfg
    from main import _build_arg_parser
    from benchmark_launcher import LauncherConfig, build_entrypoint_command

    assert cfg.TERRAIN_ZSTD_ENABLED is False
    parser = _build_arg_parser()
    assert parser.parse_args([]).terrain_zstd is False
    assert parser.parse_args(["--terrain-zstd"]).terrain_zstd is True
    assert parser.parse_args(["--no-terrain-zstd"]).terrain_zstd is False

    enabled_command = build_entrypoint_command(
        LauncherConfig(name="test", mode="interactive", terrain_zstd_enabled=True)
    )
    command = build_entrypoint_command(
        LauncherConfig(name="test", mode="interactive")
    )
    assert "--terrain-zstd" in enabled_command
    assert "--no-terrain-zstd" in command
