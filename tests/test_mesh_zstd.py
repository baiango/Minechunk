from __future__ import annotations

from collections import OrderedDict, deque
from types import SimpleNamespace

import pytest

from engine import render_contract as render_consts
from engine.meshing_types import ChunkMesh, MeshBufferAllocation, MeshOutputSlab


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
    retained_buffers: list[object] = []
    released_buffers: list[object] = []
    return SimpleNamespace(
        device=FakeDevice(),
        chunk_cache=OrderedDict(),
        max_cached_chunks=16,
        mesh_zstd_enabled=True,
        _mesh_zstd_cache=OrderedDict(),
        _pending_mesh_zstd_readbacks=deque(),
        _pending_mesh_zstd_readback_keys=set(),
        _mesh_compaction_retired_cleanup_bytes=deque(),
        _mesh_compaction_last_stats={},
        _visible_chunk_coord_set=set(),
        _visible_tile_mesh_slots={},
        _visible_chunk_origin=None,
        _visible_rel_coord_to_tile_slot={},
        _visible_tile_base=(0, 0, 0),
        _visible_active_tile_key_set=set(),
        _visible_active_tile_keys=[],
        _visible_tile_active_meshes={},
        _visible_tile_key_set=set(),
        _visible_tile_dirty_keys=set(),
        _visible_tile_mutation_version=0,
        _tile_dirty_keys=set(),
        _tile_versions={},
        _tile_mutation_version=0,
        _cached_tile_draw_batches={},
        _cached_visible_render_batches={},
        _pending_chunk_coords=set(),
        _visible_displayed_coords=set(),
        _visible_missing_coords=set(),
        _visible_display_state_dirty=False,
        _chunk_request_queue_dirty=False,
        _mesh_allocations={},
        _mesh_output_slabs=OrderedDict(),
        _mesh_output_slabs_by_size_class={},
        _deferred_mesh_output_frees=deque(),
        _gpu_mesh_deferred_buffer_cleanup=deque(),
        _next_mesh_output_slab_id=1,
        _mesh_output_append_slab_id=None,
        _next_mesh_allocation_id=1,
        _mesh_output_binding_alignment=int(render_consts.VERTEX_STRIDE),
        _mesh_output_min_slab_bytes=int(render_consts.VERTEX_STRIDE) * 4,
        _shared_empty_chunk_vertex_buffer=None,
        _retain_mesh_buffer=retained_buffers.append,
        _release_mesh_buffer=released_buffers.append,
        retained_buffers=retained_buffers,
        released_buffers=released_buffers,
    )


def _mesh(
    *,
    coord: tuple[int, int, int],
    raw: bytes,
    allocation_id: int | None = None,
    buffer: FakeBuffer | None = None,
) -> ChunkMesh:
    if buffer is None:
        buffer = FakeBuffer(data=raw)
    mesh = ChunkMesh(
        chunk_x=coord[0],
        chunk_y=coord[1],
        chunk_z=coord[2],
        vertex_count=len(raw) // int(render_consts.VERTEX_STRIDE),
        vertex_buffer=buffer,
        vertex_offset=0,
        max_height=17,
        allocation_id=allocation_id,
    )
    mesh.bounds = (1.0, 2.0, 3.0, 4.0)
    return mesh


def _install_mesh_allocation(renderer, mesh: ChunkMesh, raw_nbytes: int) -> None:
    slab = MeshOutputSlab(
        slab_id=1,
        buffer=mesh.vertex_buffer,
        size_bytes=max(1, raw_nbytes),
        free_ranges=[],
        append_offset=max(1, raw_nbytes),
        size_class_bytes=max(1, raw_nbytes),
    )
    renderer._mesh_output_slabs[1] = slab
    renderer._mesh_allocations[mesh.allocation_id] = MeshBufferAllocation(
        allocation_id=mesh.allocation_id,
        buffer=mesh.vertex_buffer,
        offset_bytes=0,
        size_bytes=max(1, raw_nbytes),
        slab_id=1,
        refcount=1,
    )


def _set_mesh_offset(mesh: ChunkMesh, offset: int) -> ChunkMesh:
    stride = int(render_consts.VERTEX_STRIDE)
    mesh.vertex_offset = int(offset)
    mesh.binding_offset = int(mesh.vertex_offset % stride)
    mesh.first_vertex = int((mesh.vertex_offset - mesh.binding_offset) // stride)
    return mesh


def _install_compaction_slab(
    renderer,
    *,
    slab_id: int,
    allocation_id: int,
    raw: bytes,
    meshes: list[ChunkMesh],
    slab_size: int | None = None,
) -> MeshOutputSlab:
    stride = int(render_consts.VERTEX_STRIDE)
    allocation_size = ((len(raw) + stride - 1) // stride) * stride
    slab_size = int(slab_size if slab_size is not None else max(allocation_size, stride * 4))
    buffer = FakeBuffer(size=slab_size)
    buffer.data[: len(raw)] = raw
    slab = MeshOutputSlab(
        slab_id=int(slab_id),
        buffer=buffer,
        size_bytes=slab_size,
        free_ranges=[],
        append_offset=allocation_size,
        size_class_bytes=allocation_size,
    )
    renderer._mesh_output_slabs[slab.slab_id] = slab
    renderer._mesh_output_slabs_by_size_class.setdefault(allocation_size, OrderedDict())[slab.slab_id] = slab
    renderer._next_mesh_output_slab_id = max(renderer._next_mesh_output_slab_id, slab.slab_id + 1)
    renderer._mesh_allocations[int(allocation_id)] = MeshBufferAllocation(
        allocation_id=int(allocation_id),
        buffer=buffer,
        offset_bytes=0,
        size_bytes=allocation_size,
        slab_id=slab.slab_id,
        refcount=len(meshes),
    )
    for mesh in meshes:
        mesh.vertex_buffer = buffer
        mesh.allocation_id = int(allocation_id)
    return slab


def _compaction_renderer_with_two_sparse_slabs():
    renderer = _fake_renderer()
    stride = int(render_consts.VERTEX_STRIDE)
    renderer._mesh_output_min_slab_bytes = stride * 4
    raw_a = bytes([11]) * (stride * 2)
    raw_b = bytes([22]) * (stride * 2)
    mesh_a0 = _set_mesh_offset(_mesh(coord=(0, 0, 0), raw=raw_a[:stride], allocation_id=10), 0)
    mesh_a1 = _set_mesh_offset(_mesh(coord=(1, 0, 0), raw=raw_a[stride:], allocation_id=10), stride)
    mesh_b = _set_mesh_offset(_mesh(coord=(2, 0, 0), raw=raw_b, allocation_id=20), 0)
    slab_a = _install_compaction_slab(
        renderer,
        slab_id=1,
        allocation_id=10,
        raw=raw_a,
        meshes=[mesh_a0, mesh_a1],
        slab_size=stride * 4,
    )
    slab_b = _install_compaction_slab(
        renderer,
        slab_id=2,
        allocation_id=20,
        raw=raw_b,
        meshes=[mesh_b],
        slab_size=stride * 4,
    )
    renderer.chunk_cache[(0, 0, 0)] = mesh_a0
    renderer.chunk_cache[(1, 0, 0)] = mesh_a1
    renderer.chunk_cache[(2, 0, 0)] = mesh_b
    return renderer, (slab_a, slab_b), (mesh_a0, mesh_a1, mesh_b), (raw_a, raw_b)


def test_mesh_zstd_round_trip_non_empty_and_zero_vertex() -> None:
    pytest.importorskip("zstandard")
    from engine.cache.mesh_zstd import compress_chunk_mesh_bytes, decompress_chunk_mesh_bytes

    raw = bytes(range(int(render_consts.VERTEX_STRIDE) * 2))
    mesh = _mesh(coord=(1, 0, 2), raw=raw)
    compressed = compress_chunk_mesh_bytes(mesh, raw)
    assert compressed.coord == (1, 0, 2)
    assert compressed.vertex_count == 2
    assert compressed.max_height == mesh.max_height
    assert compressed.bounds == mesh.bounds
    assert compressed.raw_nbytes == len(raw)
    assert decompress_chunk_mesh_bytes(compressed) == raw

    zero_mesh = _mesh(coord=(3, 0, 4), raw=b"")
    zero = compress_chunk_mesh_bytes(zero_mesh, b"")
    assert zero.raw_nbytes == 0
    assert zero.payload == b""
    assert decompress_chunk_mesh_bytes(zero) == b""


def test_mesh_zstd_scheduling_skips_visible_pending_and_compressed_chunks() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import mesh_zstd

    renderer = _fake_renderer()
    visible_key = (0, 0, 0)
    offscreen_key = (1, 0, 0)
    pending_key = (2, 0, 0)
    cached_key = (3, 0, 0)
    raw = b"\x01" * int(render_consts.VERTEX_STRIDE)
    renderer._visible_chunk_coord_set = {visible_key}
    renderer.chunk_cache[visible_key] = _mesh(coord=visible_key, raw=raw)
    renderer.chunk_cache[offscreen_key] = _mesh(coord=offscreen_key, raw=raw)
    renderer.chunk_cache[pending_key] = _mesh(coord=pending_key, raw=raw)
    renderer.chunk_cache[cached_key] = _mesh(coord=cached_key, raw=raw)
    renderer._pending_mesh_zstd_readback_keys.add(pending_key)
    renderer._mesh_zstd_cache[cached_key] = mesh_zstd.compress_chunk_mesh_bytes(renderer.chunk_cache[cached_key], raw)

    scheduled = mesh_zstd.schedule_mesh_zstd_readbacks(renderer, max_readbacks=8, max_raw_bytes=10_000)

    assert scheduled == 1
    assert [pending.coord for pending in renderer._pending_mesh_zstd_readbacks] == [offscreen_key]
    assert visible_key not in renderer._pending_mesh_zstd_readback_keys
    assert pending_key in renderer._pending_mesh_zstd_readback_keys
    assert cached_key not in [pending.coord for pending in renderer._pending_mesh_zstd_readbacks]


def test_mesh_zstd_scheduling_respects_count_and_byte_budgets() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import mesh_zstd

    raw = b"\x02" * int(render_consts.VERTEX_STRIDE)
    renderer = _fake_renderer()
    renderer._visible_chunk_coord_set = {(99, 0, 99)}
    renderer.chunk_cache[(1, 0, 0)] = _mesh(coord=(1, 0, 0), raw=raw)
    renderer.chunk_cache[(2, 0, 0)] = _mesh(coord=(2, 0, 0), raw=raw)

    assert mesh_zstd.schedule_mesh_zstd_readbacks(renderer, max_readbacks=1, max_raw_bytes=10_000) == 1
    assert len(renderer._pending_mesh_zstd_readbacks) == 1

    renderer = _fake_renderer()
    renderer._visible_chunk_coord_set = {(99, 0, 99)}
    renderer.chunk_cache[(1, 0, 0)] = _mesh(coord=(1, 0, 0), raw=raw)
    assert mesh_zstd.schedule_mesh_zstd_readbacks(renderer, max_readbacks=8, max_raw_bytes=len(raw) - 1) == 0
    assert len(renderer._pending_mesh_zstd_readbacks) == 0


def test_mesh_zstd_compressed_cache_is_limited_by_renderer_cache_limit() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import mesh_zstd

    renderer = _fake_renderer()
    renderer.max_cached_chunks = 2
    renderer._visible_chunk_coord_set = {(99, 0, 99)}
    for index in range(4):
        coord = (index, 0, 0)
        renderer.chunk_cache[coord] = _mesh(coord=coord, raw=b"")

    assert mesh_zstd.schedule_mesh_zstd_readbacks(renderer, max_readbacks=8) == 4

    assert len(renderer._mesh_zstd_cache) == 2
    assert list(renderer._mesh_zstd_cache) == [(2, 0, 0), (3, 0, 0)]
    stats = mesh_zstd.mesh_zstd_runtime_stats(renderer)
    assert stats["cache_entries"] == 2
    assert stats["cache_limit"] == 2


def test_mesh_zstd_finalize_compresses_offscreen_mesh_and_releases_allocation() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import mesh_zstd

    raw = bytes([7]) * int(render_consts.VERTEX_STRIDE)
    key = (4, 0, 5)
    renderer = _fake_renderer()
    renderer._visible_chunk_coord_set = {(0, 0, 0)}
    mesh = _mesh(coord=key, raw=raw, allocation_id=1)
    renderer.chunk_cache[key] = mesh
    _install_mesh_allocation(renderer, mesh, len(raw))
    renderer._cached_tile_draw_batches[(1, 1, 1)] = (0.0, [], 0, 0, 0)
    renderer._cached_visible_render_batches[(1, 1, 1)] = (
        0.0,
        [(mesh.vertex_buffer, mesh.binding_offset, mesh.vertex_count, mesh.first_vertex)],
        1,
        0,
        1,
        mesh.vertex_count,
    )

    assert mesh_zstd.schedule_mesh_zstd_readbacks(renderer) == 1
    assert mesh_zstd.finalize_mesh_zstd_readbacks(renderer) == 1

    assert key not in renderer.chunk_cache
    assert key in renderer._mesh_zstd_cache
    assert mesh_zstd.decompress_chunk_mesh_bytes(renderer._mesh_zstd_cache[key]) == raw
    assert renderer._mesh_allocations == {}
    assert list(renderer._deferred_mesh_output_frees)
    assert renderer._pending_mesh_zstd_readback_keys == set()
    assert renderer._cached_tile_draw_batches == {}
    assert renderer._cached_visible_render_batches == {}
    assert renderer._tile_dirty_keys


def test_mesh_zstd_restore_visible_mesh_uses_compressed_cache_without_remesh() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import mesh_zstd

    raw = bytes([9]) * int(render_consts.VERTEX_STRIDE)
    key = (6, 0, 7)
    renderer = _fake_renderer()
    renderer._visible_chunk_coord_set = {key}
    renderer._visible_missing_coords = {key}
    source_mesh = _mesh(coord=key, raw=raw)
    renderer._mesh_zstd_cache[key] = mesh_zstd.compress_chunk_mesh_bytes(source_mesh, raw)

    restored = mesh_zstd.restore_visible_mesh_zstd(renderer, [key])

    assert restored == 1
    assert key in renderer.chunk_cache
    assert key not in renderer._mesh_zstd_cache
    assert key not in renderer._visible_missing_coords
    assert key in renderer._visible_displayed_coords
    mesh = renderer.chunk_cache[key]
    actual = bytes(mesh.vertex_buffer.data[mesh.vertex_offset : mesh.vertex_offset + len(raw)])
    assert actual == raw
    assert renderer._chunk_request_queue_dirty is False


def test_mesh_zstd_mode_keeps_offscreen_meshes_until_readback_can_compress() -> None:
    pytest.importorskip("zstandard")
    from engine.cache.tile_mesh_cache import store_chunk_meshes

    raw = bytes([3]) * int(render_consts.VERTEX_STRIDE)
    renderer = _fake_renderer()
    renderer.max_cached_chunks = 1
    offscreen_key = (1, 0, 0)
    visible_key = (2, 0, 0)
    offscreen_mesh = _mesh(coord=offscreen_key, raw=raw)
    visible_mesh = _mesh(coord=visible_key, raw=raw)
    renderer.chunk_cache[offscreen_key] = offscreen_mesh
    renderer._visible_chunk_coord_set = {visible_key}

    store_chunk_meshes(renderer, [visible_mesh])

    assert set(renderer.chunk_cache) == {offscreen_key, visible_key}
    assert renderer.released_buffers == []


def test_mesh_zstd_clear_destroys_pending_readbacks() -> None:
    pytest.importorskip("zstandard")
    from engine.cache import mesh_zstd

    renderer = _fake_renderer()
    key = (8, 0, 9)
    buffer = FakeBuffer(size=16)
    buffer.map_state = "mapped"
    renderer._pending_mesh_zstd_readbacks.append(
        mesh_zstd.PendingMeshZstdReadback(
            coord=key,
            mesh=_mesh(coord=key, raw=b""),
            readback_buffer=buffer,
            raw_nbytes=0,
        )
    )
    renderer._pending_mesh_zstd_readback_keys.add(key)
    renderer._mesh_zstd_cache[key] = mesh_zstd.compress_chunk_mesh_bytes(_mesh(coord=key, raw=b""), b"")

    mesh_zstd.clear_mesh_zstd_cache(renderer)

    assert buffer.map_state == "unmapped"
    assert buffer.destroyed is True
    assert renderer._pending_mesh_zstd_readbacks == deque()
    assert renderer._pending_mesh_zstd_readback_keys == set()
    assert renderer._mesh_zstd_cache == OrderedDict()


def test_mesh_slab_compaction_packs_allocations_and_retires_source_slabs() -> None:
    from engine.cache import mesh_output_allocator

    renderer, source_slabs, meshes, raw_payloads = _compaction_renderer_with_two_sparse_slabs()
    old_buffers = [slab.buffer for slab in source_slabs]

    stats = mesh_output_allocator.compact_mesh_output_slabs(
        renderer,
        enabled=True,
        min_reclaim_bytes=1,
        max_copy_bytes=10_000,
        max_source_slabs=32,
    )

    assert stats["source_slabs"] == 2
    assert stats["retired_slab_bytes"] == int(render_consts.VERTEX_STRIDE) * 8
    assert stats["new_slab_bytes"] == int(render_consts.VERTEX_STRIDE) * 4
    assert stats["net_reclaimed_bytes"] == int(render_consts.VERTEX_STRIDE) * 4
    assert set(renderer._mesh_output_slabs) == {3}
    assert all(buffer.destroyed is False for buffer in old_buffers)
    assert len(renderer._gpu_mesh_deferred_buffer_cleanup) == 1
    assert renderer._gpu_mesh_deferred_buffer_cleanup[0][1] == old_buffers

    allocation_a = renderer._mesh_allocations[10]
    allocation_b = renderer._mesh_allocations[20]
    assert allocation_a.allocation_id == 10
    assert allocation_b.allocation_id == 20
    assert allocation_a.slab_id == 3
    assert allocation_b.slab_id == 3
    assert meshes[1].vertex_offset - meshes[0].vertex_offset == int(render_consts.VERTEX_STRIDE)
    assert meshes[0].vertex_buffer is allocation_a.buffer
    assert meshes[1].vertex_buffer is allocation_a.buffer
    assert meshes[2].vertex_buffer is allocation_b.buffer
    copied_a = bytes(allocation_a.buffer.data[allocation_a.offset_bytes : allocation_a.offset_bytes + allocation_a.size_bytes])
    copied_b = bytes(allocation_b.buffer.data[allocation_b.offset_bytes : allocation_b.offset_bytes + allocation_b.size_bytes])
    assert copied_a == raw_payloads[0]
    assert copied_b == raw_payloads[1]


def test_mesh_slab_compaction_skips_slab_with_pending_zstd_readback() -> None:
    from types import SimpleNamespace

    from engine.cache import mesh_output_allocator

    renderer, source_slabs, _meshes, _raw_payloads = _compaction_renderer_with_two_sparse_slabs()
    renderer._pending_mesh_zstd_readbacks.append(
        SimpleNamespace(vertex_buffer_id=id(source_slabs[0].buffer), allocation_id=10)
    )

    stats = mesh_output_allocator.compact_mesh_output_slabs(
        renderer,
        enabled=True,
        min_reclaim_bytes=1,
        max_copy_bytes=10_000,
        max_source_slabs=32,
    )

    assert stats["source_slabs"] == 0
    assert set(renderer._mesh_output_slabs) == {1, 2}
    assert renderer._gpu_mesh_deferred_buffer_cleanup == deque()


def test_mesh_slab_compaction_invalidates_cached_batches_and_marks_tiles_dirty() -> None:
    from engine.cache import mesh_output_allocator

    renderer, _source_slabs, _meshes, _raw_payloads = _compaction_renderer_with_two_sparse_slabs()
    renderer._cached_tile_draw_batches[(0, 0, 0)] = (0.0, [], 0, 0, 0)
    renderer._cached_visible_render_batches[(0, 0, 0)] = (0.0, [], 0, 0, 0, 0)

    stats = mesh_output_allocator.compact_mesh_output_slabs(
        renderer,
        enabled=True,
        min_reclaim_bytes=1,
        max_copy_bytes=10_000,
        max_source_slabs=32,
    )

    assert stats["moved_meshes"] == 3
    assert renderer._cached_tile_draw_batches == {}
    assert renderer._cached_visible_render_batches == {}
    assert renderer._tile_dirty_keys


def test_mesh_slab_compaction_skips_when_net_savings_below_threshold() -> None:
    from engine.cache import mesh_output_allocator

    renderer, _source_slabs, _meshes, _raw_payloads = _compaction_renderer_with_two_sparse_slabs()

    stats = mesh_output_allocator.compact_mesh_output_slabs(
        renderer,
        enabled=True,
        min_reclaim_bytes=10_000,
        max_copy_bytes=10_000,
        max_source_slabs=32,
    )

    assert stats["source_slabs"] == 0
    assert set(renderer._mesh_output_slabs) == {1, 2}
    assert renderer._gpu_mesh_deferred_buffer_cleanup == deque()


def test_mesh_zstd_cli_defaults_and_launcher_flags() -> None:
    from benchmark_launcher import LauncherConfig, build_entrypoint_command
    from engine import renderer_config as cfg
    from main import _build_arg_parser

    assert cfg.MESH_ZSTD_ENABLED is False
    parser = _build_arg_parser()
    assert parser.parse_args([]).mesh_zstd is False
    assert parser.parse_args(["--mesh-zstd"]).mesh_zstd is True
    assert parser.parse_args(["--no-mesh-zstd"]).mesh_zstd is False

    enabled_command = build_entrypoint_command(LauncherConfig(name="test", mode="interactive", mesh_zstd_enabled=True))
    default_command = build_entrypoint_command(LauncherConfig(name="test", mode="interactive"))
    disabled_command = build_entrypoint_command(
        LauncherConfig(name="test", mode="interactive", mesh_zstd_enabled=False)
    )
    assert "--mesh-zstd" in enabled_command
    assert "--no-mesh-zstd" in default_command
    assert "--no-mesh-zstd" in disabled_command
