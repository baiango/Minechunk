from __future__ import annotations

from collections import OrderedDict, deque
from types import SimpleNamespace

import pytest

from engine import render_contract as render_consts
from engine.meshing_types import ChunkDrawBatch, ChunkRenderBatch


class FakeBuffer:
    def __init__(self, size: int = 0, data: bytes | bytearray | None = None) -> None:
        self.data = bytearray(data if data is not None else b"\0" * int(size))
        self.map_state = "unmapped"
        self.destroyed = False

    def map_async(self, _mode, _offset: int, _size: int):
        self.map_state = "mapped"
        return object()

    def read_mapped(self, offset: int, size: int, copy: bool = False):
        view = memoryview(self.data)[int(offset) : int(offset) + int(size)]
        return bytes(view) if copy else view

    def unmap(self) -> None:
        self.map_state = "unmapped"

    def destroy(self) -> None:
        self.destroyed = True


class FakeEncoder:
    def __init__(self) -> None:
        self.commands: list[tuple[FakeBuffer, int, FakeBuffer, int, int]] = []

    def copy_buffer_to_buffer(self, src, src_offset: int, dst, dst_offset: int, size: int) -> None:
        self.commands.append((src, int(src_offset), dst, int(dst_offset), int(size)))

    def finish(self):
        return list(self.commands)


class FakeQueue:
    def submit(self, command_buffers) -> None:
        for commands in command_buffers:
            for src, src_offset, dst, dst_offset, size in commands:
                dst.data[dst_offset : dst_offset + size] = src.data[src_offset : src_offset + size]

    def write_buffer(self, buffer, offset: int, data) -> None:
        payload = bytes(data)
        end = int(offset) + len(payload)
        if end > len(buffer.data):
            buffer.data.extend(b"\0" * (end - len(buffer.data)))
        buffer.data[int(offset) : end] = payload


class FakeDevice:
    def __init__(self) -> None:
        self.queue = FakeQueue()

    def create_buffer(self, *, size: int, usage) -> FakeBuffer:
        return FakeBuffer(size=size)

    def create_command_encoder(self) -> FakeEncoder:
        return FakeEncoder()


def _fake_renderer() -> SimpleNamespace:
    return SimpleNamespace(
        device=FakeDevice(),
        max_cached_chunks=8,
        mesh_zstd_enabled=True,
        tile_zstd_enabled=True,
        _tile_zstd_cache=OrderedDict(),
        _pending_tile_zstd_readbacks=deque(),
        _pending_tile_zstd_readback_keys=set(),
        _tile_render_batches={},
        _tile_dirty_keys=set(),
        _tile_versions={},
        _visible_tile_dirty_keys=set(),
        _visible_tile_key_set=set(),
        _visible_active_tile_keys=[],
        _visible_active_tile_key_set=set(),
        _visible_tile_active_meshes={},
        _visible_tile_masks={},
        _visible_chunk_coords=[(99, 0, 99)],
        _visible_layout_version=1,
        _visible_tile_mutation_version=1,
        _tile_render_batch_cleanup_layout_version=0,
        _cached_tile_draw_batches={},
        _transient_render_buffers=[],
        _gpu_mesh_deferred_buffer_cleanup=deque(),
    )


def _tile_batch(tile_key=(1, 0, 2), raw: bytes | None = None) -> tuple[ChunkRenderBatch, bytes]:
    stride = int(render_consts.VERTEX_STRIDE)
    raw = raw if raw is not None else bytes([31]) * stride * 2
    buffer = FakeBuffer(size=len(raw), data=raw)
    draw_batch = ChunkDrawBatch(
        vertex_buffer=buffer,
        binding_offset=0,
        vertex_count=len(raw) // stride,
        first_vertex=0,
        bounds=(1.0, 2.0, 3.0, 4.0),
        chunk_count=2,
    )
    render_batches = ((buffer, 0, draw_batch.vertex_count, 0),)
    batch = ChunkRenderBatch(
        signature=((4, 0, 8), (5, 0, 8)),
        vertex_count=draw_batch.vertex_count,
        vertex_buffer=buffer,
        bounds=draw_batch.bounds,
        chunk_count=2,
        complete_tile=False,
        all_mature=True,
        visible_mask=3,
        source_version=7,
        cached_draw_batches=(draw_batch,),
        cached_render_batches=render_batches,
        cached_grouped_render_batches=(),
        next_refresh_at=0.0,
        visible_chunk_count=2,
        merged_chunk_count=2,
        visible_vertex_count=draw_batch.vertex_count,
        owns_vertex_buffer=True,
        owned_vertex_buffer_capacity_bytes=len(raw),
    )
    return batch, raw


def test_tile_zstd_readback_compresses_and_restore_reuploads_batch() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import tile_zstd

    renderer = _fake_renderer()
    tile_key = (1, 0, 2)
    batch, raw = _tile_batch(tile_key)

    assert tile_zstd.schedule_tile_zstd_readback(renderer, tile_key, batch) is True
    assert tile_zstd.finalize_tile_zstd_readbacks(renderer) == 1

    assert batch.vertex_buffer.destroyed is True
    assert tile_key in renderer._tile_zstd_cache
    stats = tile_zstd.tile_zstd_runtime_stats(renderer)
    assert stats["cache_entries"] == 1
    assert stats["cache_raw_bytes"] == len(raw)

    restored = tile_zstd.restore_tile_zstd_batch(renderer, tile_key, visible_mask=3, source_version=7)

    assert restored is renderer._tile_render_batches[tile_key]
    assert tile_key not in renderer._tile_zstd_cache
    assert restored.vertex_count == batch.vertex_count
    assert bytes(restored.vertex_buffer.data[: len(raw)]) == raw


def test_tile_zstd_restore_drops_stale_source_version() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import tile_zstd

    renderer = _fake_renderer()
    tile_key = (1, 0, 2)
    batch, _raw = _tile_batch(tile_key)
    assert tile_zstd.schedule_tile_zstd_readback(renderer, tile_key, batch) is True
    assert tile_zstd.finalize_tile_zstd_readbacks(renderer) == 1

    assert tile_zstd.restore_tile_zstd_batch(renderer, tile_key, visible_mask=3, source_version=8) is None
    assert tile_key not in renderer._tile_zstd_cache


def test_stale_visible_tile_batch_schedules_tile_zstd_readback() -> None:
    pytest.importorskip("zstandard")
    from engine.cache.tile_draw_batches import build_tile_draw_batches

    renderer = _fake_renderer()
    tile_key = (1, 0, 2)
    batch, _raw = _tile_batch(tile_key)
    renderer._tile_render_batches[tile_key] = batch
    renderer._visible_tile_key_set = set()
    renderer._visible_active_tile_keys = []

    draw_batches, merged, visible, vertices, next_refresh = build_tile_draw_batches(
        renderer,
        None,
        FakeEncoder(),
        age_gate=False,
    )

    assert draw_batches == []
    assert (merged, visible, vertices, next_refresh) == (0, 0, 0, 0.0)
    assert tile_key not in renderer._tile_render_batches
    assert len(renderer._pending_tile_zstd_readbacks) == 1
    assert batch.vertex_buffer.destroyed is False


def test_dirty_visible_empty_tile_clears_dirty_state_and_allows_cache() -> None:
    from engine.cache.tile_draw_batches import build_tile_draw_batches

    renderer = _fake_renderer()
    renderer.tile_zstd_enabled = False
    tile_key = (1, 0, 2)
    batch, _raw = _tile_batch(tile_key)
    renderer._tile_render_batches[tile_key] = batch
    renderer._tile_dirty_keys = {tile_key}
    renderer._visible_tile_dirty_keys = {tile_key}
    renderer._tile_versions[tile_key] = 8
    renderer._visible_tile_key_set = {tile_key}
    renderer._visible_tile_masks = {tile_key: 3}
    renderer._visible_active_tile_keys = []
    renderer._visible_active_tile_key_set = set()
    renderer._visible_tile_active_meshes = {}

    draw_batches, merged, visible, vertices, next_refresh = build_tile_draw_batches(
        renderer,
        None,
        FakeEncoder(),
        age_gate=False,
    )

    assert draw_batches == []
    assert (merged, visible, vertices, next_refresh) == (0, 0, 0, 0.0)
    assert tile_key not in renderer._tile_render_batches
    assert tile_key not in renderer._tile_dirty_keys
    assert tile_key not in renderer._visible_tile_dirty_keys
    assert renderer._cached_tile_draw_batches[(1, 1, 0)] == (0.0, [], 0, 0, 0)


def test_tile_merging_disabled_keeps_visible_tile_batches_direct(monkeypatch) -> None:
    from engine.cache import tile_mesh_cache
    from engine.cache.tile_draw_batches import build_tile_draw_batches
    from engine.meshing_types import ChunkMesh

    def _unexpected_merge(*_args, **_kwargs):
        raise AssertionError("tile merge should not run when tile_merging_enabled is false")

    monkeypatch.setattr(tile_mesh_cache, "merge_tile_meshes", _unexpected_merge)

    renderer = _fake_renderer()
    renderer.tile_merging_enabled = False
    renderer.tile_zstd_enabled = False
    stride = int(render_consts.VERTEX_STRIDE)
    first_buffer = FakeBuffer(size=stride * 2)
    second_buffer = FakeBuffer(size=stride * 3)
    meshes = [
        ChunkMesh(chunk_x=0, chunk_z=0, chunk_y=0, vertex_count=2, vertex_buffer=first_buffer, max_height=1),
        ChunkMesh(chunk_x=1, chunk_z=0, chunk_y=0, vertex_count=3, vertex_buffer=second_buffer, max_height=1),
    ]

    draw_batches, merged, visible, vertices, next_refresh = build_tile_draw_batches(
        renderer,
        meshes,
        FakeEncoder(),
        age_gate=False,
    )

    assert len(draw_batches) == 2
    assert (merged, visible, vertices, next_refresh) == (0, 2, 5, 0.0)
    batch = renderer._tile_render_batches[(0, 0, 0)]
    assert batch.owns_vertex_buffer is False
    assert batch.vertex_buffer is first_buffer
    assert batch.merged_chunk_count == 0
    assert batch.visible_chunk_count == 2
    assert batch.visible_vertex_count == 5
