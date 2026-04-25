from __future__ import annotations

import math
import struct
import time
from collections import deque

import numpy as np
import wgpu

from ..cache import mesh_allocator as mesh_cache
from .. import meshing_types as mt
from ..terrain.types import ChunkSurfaceGpuBatch, ChunkVoxelResult

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


def _renderer_module():
    from .. import renderer

    return renderer


def _chunk_half() -> float:
    return float(_renderer_module().CHUNK_WORLD_SIZE) * 0.5


def _storage_height(local_height: int) -> int:
    return int(local_height) + 2


def _storage_binding_size(size_bytes: int) -> int:
    # WGPU storage buffer binding sizes must be at least 4 bytes and
    # must be 4-byte aligned.
    size_bytes = max(4, int(size_bytes))
    return (size_bytes + 3) & ~3


def _emit_vertex_binding_size(size_bytes: int) -> int:
    # The emit shader's output binding is an array of packed ChunkVertex
    # records. Even when a high-altitude batch emits zero vertices, WGPU
    # validates the declared storage binding against the minimum element
    # footprint, so a dummy binding must cover at least one full vertex.
    vertex_stride = int(_renderer_module().VERTEX_STRIDE)
    return _storage_binding_size(max(vertex_stride, int(size_bytes)))


def _mesh_output_request_bytes(renderer, size_bytes: int) -> int:
    # Keep output allocations friendly to dynamic storage-buffer offset
    # alignment even when a whole high-altitude batch emits zero vertices.
    alignment = max(4, int(getattr(renderer, "_mesh_output_binding_alignment", 256)))
    return max(alignment, _emit_vertex_binding_size(size_bytes))


def _normalize_chunk_coord(coord) -> tuple[int, int, int]:
    if len(coord) >= 3:
        return int(coord[0]), int(coord[1]), int(coord[2])
    if len(coord) == 2:
        return int(coord[0]), 0, int(coord[1])
    raise ValueError(f"Invalid chunk coordinate: {coord!r}")


def _normalize_chunk_coords(coords) -> list[tuple[int, int, int]]:
    return [_normalize_chunk_coord(coord) for coord in coords]


@profile
def ensure_voxel_mesh_batch_scratch(renderer, sample_size: int, height_limit: int, chunk_capacity: int | None = None) -> None:
    capacity = max(1, int(chunk_capacity if chunk_capacity is not None else renderer.mesh_batch_size))
    height_limit = int(height_limit)
    if height_limit > 128:
        raise RuntimeError("WGPU voxel mesher supports a local chunk height of at most 128. Use stacked chunks for tall worlds.")
    storage_height = _storage_height(height_limit)
    if (
        renderer._voxel_mesh_scratch_capacity >= capacity
        and renderer._voxel_mesh_scratch_sample_size == sample_size
        and renderer._voxel_mesh_scratch_height_limit == height_limit
        and renderer._voxel_mesh_scratch_blocks_buffer is not None
    ):
        return

    renderer._voxel_mesh_scratch_capacity = capacity
    renderer._voxel_mesh_scratch_sample_size = int(sample_size)
    renderer._voxel_mesh_scratch_height_limit = int(height_limit)
    max_chunk_count = capacity
    max_column_plane = max(1, (sample_size - 2) * (sample_size - 2))
    blocks_bytes = max_chunk_count * storage_height * sample_size * sample_size * 4
    coords_bytes = max_chunk_count * 4 * 4
    column_totals_bytes = max_chunk_count * max_column_plane * 4
    chunk_totals_bytes = max_chunk_count * 4

    renderer._voxel_mesh_scratch_blocks_buffer = renderer.device.create_buffer(
        size=max(1, blocks_bytes),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_materials_buffer = renderer.device.create_buffer(
        size=max(1, blocks_bytes),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_coords_buffer = renderer.device.create_buffer(
        size=max(1, coords_bytes),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_column_totals_buffer = renderer.device.create_buffer(
        size=max(1, column_totals_bytes),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_chunk_totals_buffer = renderer.device.create_buffer(
        size=max(1, chunk_totals_bytes),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_chunk_offsets_buffer = renderer.device.create_buffer(
        size=max(1, chunk_totals_bytes),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_chunk_metadata_readback_buffer = renderer.device.create_buffer(
        size=max(8, max_chunk_count * 8),
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    renderer._voxel_mesh_scratch_params_buffer = renderer.device.create_buffer(
        size=32,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_batch_vertex_buffer = renderer.device.create_buffer(
        size=_renderer_module().VERTEX_STRIDE,
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    renderer._voxel_mesh_scratch_blocks_array = np.empty(
        (max_chunk_count, storage_height, sample_size, sample_size),
        dtype=np.uint32,
    )
    renderer._voxel_mesh_scratch_materials_array = np.empty(
        (max_chunk_count, storage_height, sample_size, sample_size),
        dtype=np.uint32,
    )
    renderer._voxel_mesh_scratch_coords_array = np.empty((max_chunk_count, 4), dtype=np.int32)
    renderer._voxel_mesh_scratch_chunk_totals_array = np.empty(max_chunk_count, dtype=np.uint32)
    renderer._voxel_mesh_scratch_chunk_offsets_array = np.empty(max_chunk_count, dtype=np.uint32)


@profile
def create_async_voxel_mesh_batch_resources(renderer, sample_size: int, height_limit: int, chunk_count: int):
    ensure_voxel_mesh_batch_scratch(renderer, sample_size, height_limit, 1)
    chunk_capacity = max(1, int(chunk_count))
    storage_height = _storage_height(int(height_limit))
    column_capacity = max(1, (sample_size - 2) * (sample_size - 2) * chunk_capacity)
    blocks_bytes = chunk_capacity * storage_height * sample_size * sample_size * 4
    coords_bytes = chunk_capacity * 4 * 4
    chunk_totals_bytes = chunk_capacity * 4
    emit_vertex_bytes = _renderer_module().VERTEX_STRIDE

    resources = mt.AsyncVoxelMeshBatchResources(
        sample_size=int(sample_size),
        height_limit=int(height_limit),
        chunk_capacity=chunk_capacity,
        column_capacity=column_capacity,
        blocks_buffer=renderer.device.create_buffer(
            size=max(1, blocks_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        ),
        materials_buffer=renderer.device.create_buffer(
            size=max(1, blocks_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        ),
        coords_buffer=renderer.device.create_buffer(
            size=max(1, coords_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        ),
        column_totals_buffer=renderer.device.create_buffer(
            size=max(1, column_capacity * 4),
            usage=wgpu.BufferUsage.STORAGE,
        ),
        chunk_totals_buffer=renderer.device.create_buffer(
            size=max(1, chunk_totals_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        ),
        chunk_offsets_buffer=renderer.device.create_buffer(
            size=max(1, chunk_totals_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        ),
        params_buffer=renderer.device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        ),
        expand_params_buffer=renderer.device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        ),
        readback_buffer=renderer.device.create_buffer(
            size=max(4, chunk_totals_bytes),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        ),
        emit_vertex_buffer=renderer.device.create_buffer(
            size=max(_renderer_module().VERTEX_STRIDE, int(emit_vertex_bytes)),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        ),
        coords_array=np.empty((chunk_capacity, 4), dtype=np.int32),
        zero_counts_array=np.zeros(chunk_capacity, dtype=np.uint32),
    )

    shared_entries = [
        {"binding": 0, "resource": {"buffer": resources.blocks_buffer}},
        {"binding": 1, "resource": {"buffer": resources.materials_buffer}},
        {"binding": 2, "resource": {"buffer": resources.coords_buffer}},
        {"binding": 3, "resource": {"buffer": resources.column_totals_buffer}},
        {"binding": 4, "resource": {"buffer": resources.chunk_totals_buffer}},
        {"binding": 5, "resource": {"buffer": resources.chunk_offsets_buffer}},
        {"binding": 6, "resource": {"buffer": resources.emit_vertex_buffer}},
        {"binding": 8, "resource": {"buffer": resources.params_buffer, "offset": 0, "size": 32}},
    ]
    resources.count_bind_group = renderer.device.create_bind_group(
        layout=renderer.voxel_mesh_count_bind_group_layout,
        entries=shared_entries,
    )
    resources.scan_bind_group = renderer.device.create_bind_group(
        layout=renderer.voxel_mesh_scan_bind_group_layout,
        entries=shared_entries,
    )
    resources.emit_bind_group = renderer.device.create_bind_group(
        layout=renderer.voxel_mesh_emit_bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": resources.blocks_buffer}},
            {"binding": 1, "resource": {"buffer": resources.materials_buffer}},
            {"binding": 2, "resource": {"buffer": resources.coords_buffer}},
            {"binding": 3, "resource": {"buffer": resources.column_totals_buffer}},
            {"binding": 4, "resource": {"buffer": resources.chunk_totals_buffer}},
            {"binding": 5, "resource": {"buffer": resources.chunk_offsets_buffer}},
            {"binding": 6, "resource": {"buffer": resources.emit_vertex_buffer}},
            {"binding": 8, "resource": {"buffer": resources.params_buffer, "offset": 0, "size": 32}},
        ],
    )
    return resources


def async_voxel_mesh_batch_resources_match(resources, sample_size: int, height_limit: int, chunk_count: int) -> bool:
    return (
        resources.sample_size == int(sample_size)
        and resources.height_limit == int(height_limit)
        and resources.chunk_capacity >= max(1, int(chunk_count))
    )


@profile
def destroy_async_voxel_mesh_batch_resources(resources) -> None:
    if resources.readback_buffer.map_state != "unmapped":
        try:
            resources.readback_buffer.unmap()
        except Exception:
            pass
    for buffer in (
        resources.blocks_buffer,
        resources.materials_buffer,
        resources.coords_buffer,
        resources.column_totals_buffer,
        resources.chunk_totals_buffer,
        resources.chunk_offsets_buffer,
        resources.params_buffer,
        getattr(resources, "expand_params_buffer", None),
        resources.readback_buffer,
        resources.emit_vertex_buffer,
    ):
        if buffer is None:
            continue
        try:
            buffer.destroy()
        except Exception:
            pass


@profile
def acquire_async_voxel_mesh_batch_resources(renderer, sample_size: int, height_limit: int, chunk_count: int):
    target_chunk_count = max(1, int(chunk_count))
    pool_size = len(renderer._async_voxel_mesh_batch_pool)
    for _ in range(pool_size):
        resources = renderer._async_voxel_mesh_batch_pool.popleft()
        if async_voxel_mesh_batch_resources_match(resources, sample_size, height_limit, target_chunk_count):
            return resources
        renderer._async_voxel_mesh_batch_pool.append(resources)
    return create_async_voxel_mesh_batch_resources(renderer, sample_size, height_limit, target_chunk_count)


@profile
def release_async_voxel_mesh_batch_resources(renderer, resources) -> None:
    if len(renderer._async_voxel_mesh_batch_pool) >= renderer._async_voxel_mesh_batch_pool_limit:
        destroy_async_voxel_mesh_batch_resources(resources)
        return
    renderer._async_voxel_mesh_batch_pool.append(resources)


def schedule_async_voxel_mesh_batch_resource_release(renderer, resources, frames: int = 2) -> None:
    renderer._gpu_mesh_deferred_batch_resource_releases.append((max(1, int(frames)), resources))


@profile
def get_cached_async_voxel_mesh_emit_bind_group(renderer, resources, batch_allocation) -> object | None:
    # The final output allocation size is only known after the count pass.
    # Binding directly to that final slab avoids a fixed per-chunk emit buffer
    # capacity and prevents overflow on cave-heavy chunks.
    _ = renderer, resources, batch_allocation
    return None


def schedule_gpu_buffer_cleanup(renderer, buffers: list[wgpu.GPUBuffer], frames: int = 2) -> None:
    if buffers:
        renderer._gpu_mesh_deferred_buffer_cleanup.append((max(1, int(frames)), list(buffers)))


@profile
def process_gpu_buffer_cleanup(renderer) -> None:
    if renderer._gpu_mesh_deferred_buffer_cleanup:
        next_queue: deque[tuple[int, list[wgpu.GPUBuffer]]] = deque()
        while renderer._gpu_mesh_deferred_buffer_cleanup:
            frames_left, buffers = renderer._gpu_mesh_deferred_buffer_cleanup.popleft()
            frames_left -= 1
            if frames_left <= 0:
                for buffer in buffers:
                    try:
                        buffer.destroy()
                    except Exception:
                        pass
            else:
                next_queue.append((frames_left, buffers))
        renderer._gpu_mesh_deferred_buffer_cleanup = next_queue

    if renderer._gpu_mesh_deferred_batch_resource_releases:
        next_resource_queue: deque[tuple[int, object]] = deque()
        while renderer._gpu_mesh_deferred_batch_resource_releases:
            frames_left, resources = renderer._gpu_mesh_deferred_batch_resource_releases.popleft()
            frames_left -= 1
            if frames_left <= 0:
                release_async_voxel_mesh_batch_resources(renderer, resources)
            else:
                next_resource_queue.append((frames_left, resources))
        renderer._gpu_mesh_deferred_batch_resource_releases = next_resource_queue

    if renderer._gpu_mesh_deferred_surface_batch_releases:
        next_surface_release_queue: deque[tuple[int, list[object]]] = deque()
        while renderer._gpu_mesh_deferred_surface_batch_releases:
            frames_left, callbacks = renderer._gpu_mesh_deferred_surface_batch_releases.popleft()
            frames_left -= 1
            if frames_left <= 0:
                for callback in callbacks:
                    if callable(callback):
                        try:
                            callback()
                        except Exception:
                            pass
            else:
                next_surface_release_queue.append((frames_left, callbacks))
        renderer._gpu_mesh_deferred_surface_batch_releases = next_surface_release_queue


@profile
def get_voxel_surface_expand_bind_group(renderer, surface_batch: ChunkSurfaceGpuBatch, blocks_buffer, materials_buffer, params_buffer, coords_buffer) -> object:
    device_kind = str(getattr(surface_batch, "device_kind", "") or "").strip().lower()
    source = str(getattr(surface_batch, "source", "") or "").strip().lower()
    if device_kind and device_kind != "wgpu":
        raise TypeError(f"WGPU mesher cannot bind {device_kind!r} terrain surface buffers; use ChunkVoxelResult fallback.")
    if not device_kind and "metal" in source:
        raise TypeError("WGPU mesher cannot bind Metal terrain surface buffers; use ChunkVoxelResult fallback.")
    cache_key = (
        id(surface_batch.heights_buffer),
        id(surface_batch.materials_buffer),
        id(blocks_buffer),
        id(materials_buffer),
        id(params_buffer),
        id(coords_buffer),
    )
    bind_group = renderer._voxel_surface_expand_bind_group_cache.get(cache_key)
    if bind_group is not None:
        renderer._voxel_surface_expand_bind_group_cache.move_to_end(cache_key)
        return bind_group

    bind_group = renderer.device.create_bind_group(
        layout=renderer.voxel_surface_expand_bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": surface_batch.heights_buffer}},
            {"binding": 1, "resource": {"buffer": surface_batch.materials_buffer}},
            {"binding": 2, "resource": {"buffer": blocks_buffer}},
            {"binding": 3, "resource": {"buffer": materials_buffer}},
            {"binding": 4, "resource": {"buffer": params_buffer, "offset": 0, "size": 32}},
            {"binding": 5, "resource": {"buffer": coords_buffer}},
        ],
    )
    renderer._voxel_surface_expand_bind_group_cache[cache_key] = bind_group
    while len(renderer._voxel_surface_expand_bind_group_cache) > _renderer_module().VOXEL_SURFACE_EXPAND_BIND_GROUP_CACHE_LIMIT:
        renderer._voxel_surface_expand_bind_group_cache.popitem(last=False)
    return bind_group


def schedule_surface_gpu_batch_release(renderer, callbacks: list[object], frames: int = 2) -> None:
    active_callbacks = [callback for callback in callbacks if callable(callback)]
    if active_callbacks:
        renderer._gpu_mesh_deferred_surface_batch_releases.append((max(1, int(frames)), active_callbacks))


def release_surface_gpu_batch_immediately(renderer, surface_batch: ChunkSurfaceGpuBatch) -> None:
    callback = getattr(surface_batch, "_release_callback", None)
    if not callable(callback):
        return
    try:
        callback()
    finally:
        try:
            setattr(surface_batch, "_release_callback", None)
        except Exception:
            pass


def pending_surface_gpu_batches_chunk_count(renderer) -> int:
    return int(renderer._pending_surface_gpu_batches_chunk_total)


@profile
def enqueue_surface_gpu_batches_for_meshing(renderer, surface_batches: list[ChunkSurfaceGpuBatch]) -> None:
    if not surface_batches:
        return
    if not renderer._pending_surface_gpu_batches:
        renderer._pending_surface_gpu_batches_first_enqueued_at = time.perf_counter()

    chunk_total = 0
    for batch in surface_batches:
        chunk_total += len(batch.chunks)

    renderer._pending_surface_gpu_batches.extend(surface_batches)
    renderer._pending_surface_gpu_batches_chunk_total += chunk_total


@profile
def drain_pending_surface_gpu_batches_to_meshing(renderer) -> None:
    if not renderer._pending_surface_gpu_batches:
        renderer._pending_surface_gpu_batches_chunk_total = 0
        return

    inflight_limit = max(1, int(renderer._gpu_mesh_async_inflight_limit))
    if len(renderer._pending_gpu_mesh_batches) >= inflight_limit:
        return

    target_chunks = max(1, int(renderer._pending_surface_gpu_batch_target_chunks))

    while renderer._pending_surface_gpu_batches:
        if len(renderer._pending_gpu_mesh_batches) >= inflight_limit:
            break

        submit_batches: list[ChunkSurfaceGpuBatch] = []
        submit_chunks = 0
        while renderer._pending_surface_gpu_batches and (submit_chunks < target_chunks or not submit_batches):
            batch = renderer._pending_surface_gpu_batches.popleft()
            batch_chunk_count = len(batch.chunks)
            submit_batches.append(batch)
            submit_chunks += batch_chunk_count
            renderer._pending_surface_gpu_batches_chunk_total -= batch_chunk_count

        if renderer._pending_surface_gpu_batches:
            renderer._pending_surface_gpu_batches_first_enqueued_at = time.perf_counter()
        else:
            renderer._pending_surface_gpu_batches_chunk_total = 0
            renderer._pending_surface_gpu_batches_first_enqueued_at = 0.0

        make_chunk_mesh_batches_from_surface_gpu_batches(renderer, submit_batches)


@profile
def enqueue_gpu_chunk_mesh_batch_from_gpu_buffers(
    renderer,
    chunk_coords: list[tuple[int, int, int]],
    resources,
    sample_size: int,
    height_limit: int,
    *,
    params_already_uploaded: bool = False,
    encoder=None,
    submit: bool = True,
    surface_release_callbacks: list[object] | None = None,
) -> None:
    if renderer.voxel_mesh_count_pipeline is None or renderer.voxel_mesh_emit_pipeline is None:
        raise RuntimeError("GPU meshing pipeline is unavailable.")
    if not chunk_coords:
        return

    chunk_coords_list = _normalize_chunk_coords(chunk_coords if isinstance(chunk_coords, list) else list(chunk_coords))
    chunk_count = len(chunk_coords_list)
    columns_per_side = sample_size - 2
    if not async_voxel_mesh_batch_resources_match(resources, sample_size, height_limit, chunk_count):
        raise RuntimeError("Async voxel mesh batch resources do not match requested batch size.")
    coords_view = resources.coords_array[:chunk_count]
    for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunk_coords_list):
        coords_view[index, 0] = int(chunk_x)
        coords_view[index, 1] = int(chunk_y)
        coords_view[index, 2] = int(chunk_z)
        coords_view[index, 3] = 0
    count_bind_group = resources.count_bind_group
    scan_bind_group = resources.scan_bind_group
    assert count_bind_group is not None
    assert scan_bind_group is not None

    zero_view = resources.zero_counts_array[:chunk_count]
    zero_view.fill(0)
    params_bytes = struct.pack(
        "<4I4f",
        int(sample_size),
        int(height_limit),
        int(chunk_count),
        int(_renderer_module().CHUNK_SIZE),
        float(_renderer_module().BLOCK_SIZE),
        0.0,
        0.0,
        0.0,
    )

    if not params_already_uploaded:
        renderer.device.queue.write_buffer(resources.params_buffer, 0, params_bytes)
    renderer.device.queue.write_buffer(resources.coords_buffer, 0, memoryview(coords_view))
    renderer.device.queue.write_buffer(resources.chunk_totals_buffer, 0, memoryview(zero_view))
    renderer.device.queue.write_buffer(resources.chunk_offsets_buffer, 0, memoryview(zero_view))

    if encoder is None:
        encoder = renderer.device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(renderer.voxel_mesh_count_pipeline)
    compute_pass.set_bind_group(0, count_bind_group)
    compute_pass.dispatch_workgroups(columns_per_side, columns_per_side, chunk_count)
    compute_pass.end()

    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(renderer.voxel_mesh_scan_pipeline)
    compute_pass.set_bind_group(0, scan_bind_group)
    compute_pass.dispatch_workgroups(1, 1, chunk_count)
    compute_pass.end()

    encoder.copy_buffer_to_buffer(resources.chunk_totals_buffer, 0, resources.readback_buffer, 0, chunk_count * 4)
    if submit:
        renderer.device.queue.submit([encoder.finish()])

    metadata_promise = resources.readback_buffer.map_async(wgpu.MapMode.READ, 0, chunk_count * 4)
    renderer._pending_gpu_mesh_batches.append(
        mt.PendingChunkMeshBatch(
            chunk_coords=chunk_coords_list,
            chunk_count=chunk_count,
            sample_size=int(sample_size),
            height_limit=int(height_limit),
            columns_per_side=int(columns_per_side),
            blocks_buffer=resources.blocks_buffer,
            materials_buffer=resources.materials_buffer,
            coords_buffer=resources.coords_buffer,
            column_totals_buffer=resources.column_totals_buffer,
            chunk_totals_buffer=resources.chunk_totals_buffer,
            chunk_offsets_buffer=resources.chunk_offsets_buffer,
            params_buffer=resources.params_buffer,
            readback_buffer=resources.readback_buffer,
            resources=resources,
            surface_release_callbacks=list(surface_release_callbacks or ()),
            metadata_promise=metadata_promise,
            submitted_at=time.perf_counter(),
        )
    )


@profile
def read_chunk_mesh_batch_metadata(renderer, chunk_totals_buffer, chunk_offsets_buffer, chunk_count: int, *, include_offsets: bool = False):
    if chunk_count <= 0:
        empty = np.empty(0, dtype=np.uint32)
        return empty, empty if include_offsets else None

    readback_buffer = renderer._voxel_mesh_scratch_chunk_metadata_readback_buffer
    assert readback_buffer is not None

    totals_nbytes = chunk_count * 4
    metadata_nbytes = totals_nbytes * (2 if include_offsets else 1)
    if readback_buffer.map_state != "unmapped":
        readback_buffer.unmap()

    encoder = renderer.device.create_command_encoder()
    encoder.copy_buffer_to_buffer(chunk_totals_buffer, 0, readback_buffer, 0, totals_nbytes)
    if include_offsets:
        encoder.copy_buffer_to_buffer(chunk_offsets_buffer, 0, readback_buffer, totals_nbytes, totals_nbytes)
    renderer.device.queue.submit([encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ, 0, metadata_nbytes)
    try:
        metadata_view = readback_buffer.read_mapped(0, metadata_nbytes, copy=False)
        totals = np.frombuffer(metadata_view, dtype=np.uint32, count=chunk_count).copy()
        if include_offsets:
            offsets = np.frombuffer(metadata_view, dtype=np.uint32, count=chunk_count, offset=totals_nbytes).copy()
        else:
            offsets = None
    finally:
        readback_buffer.unmap()

    return totals, offsets


@profile
def make_chunk_mesh_batch_from_terrain_results(
    renderer,
    chunk_results: list[ChunkVoxelResult],
    *,
    defer_finalize: bool = False,
):
    if (
        renderer.voxel_mesh_count_pipeline is None
        or renderer.voxel_mesh_scan_pipeline is None
        or renderer.voxel_mesh_emit_pipeline is None
    ):
        raise RuntimeError("GPU meshing pipeline is unavailable.")
    if not chunk_results:
        return []

    chunk_count = len(chunk_results)
    sample_size = int(chunk_results[0].blocks.shape[1])
    height_limit = int(chunk_results[0].blocks.shape[0])
    ensure_voxel_mesh_batch_scratch(renderer, sample_size, height_limit, chunk_count)

    blocks = renderer._voxel_mesh_scratch_blocks_array
    materials = renderer._voxel_mesh_scratch_materials_array
    chunk_coords = renderer._voxel_mesh_scratch_coords_array
    chunk_totals = renderer._voxel_mesh_scratch_chunk_totals_array
    chunk_offsets = renderer._voxel_mesh_scratch_chunk_offsets_array

    blocks_buffer = renderer._voxel_mesh_scratch_blocks_buffer
    materials_buffer = renderer._voxel_mesh_scratch_materials_buffer
    coords_buffer = renderer._voxel_mesh_scratch_coords_buffer
    column_totals_buffer = renderer._voxel_mesh_scratch_column_totals_buffer
    chunk_totals_buffer = renderer._voxel_mesh_scratch_chunk_totals_buffer
    chunk_offsets_buffer = renderer._voxel_mesh_scratch_chunk_offsets_buffer
    params_buffer = renderer._voxel_mesh_scratch_params_buffer

    assert blocks is not None
    assert materials is not None
    assert chunk_coords is not None
    assert chunk_totals is not None
    assert chunk_offsets is not None
    assert blocks_buffer is not None
    assert materials_buffer is not None
    assert coords_buffer is not None
    assert column_totals_buffer is not None
    assert chunk_totals_buffer is not None
    assert chunk_offsets_buffer is not None
    assert params_buffer is not None

    chunk_coords_list: list[tuple[int, int, int]] = []
    blocks[:chunk_count].fill(0)
    materials[:chunk_count].fill(0)
    for index, result in enumerate(chunk_results):
        chunk_x = int(result.chunk_x)
        chunk_y = int(getattr(result, "chunk_y", 0))
        chunk_z = int(result.chunk_z)
        blocks[index, 1 : height_limit + 1] = result.blocks
        materials[index, 1 : height_limit + 1] = result.materials
        bottom_plane = getattr(result, "bottom_boundary", None)
        top_plane = getattr(result, "top_boundary", None)
        if bottom_plane is not None:
            blocks[index, 0] = np.asarray(bottom_plane, dtype=np.uint32)
        if top_plane is not None:
            blocks[index, height_limit + 1] = np.asarray(top_plane, dtype=np.uint32)
        chunk_coords[index, 0] = chunk_x
        chunk_coords[index, 1] = chunk_y
        chunk_coords[index, 2] = chunk_z
        chunk_coords[index, 3] = 0
        chunk_coords_list.append((chunk_x, chunk_y, chunk_z))

    if defer_finalize:
        resources = acquire_async_voxel_mesh_batch_resources(renderer, sample_size, height_limit, chunk_count)
        renderer.device.queue.write_buffer(resources.blocks_buffer, 0, memoryview(blocks[:chunk_count]))
        renderer.device.queue.write_buffer(resources.materials_buffer, 0, memoryview(materials[:chunk_count]))
        enqueue_gpu_chunk_mesh_batch_from_gpu_buffers(
            renderer,
            chunk_coords_list,
            resources,
            sample_size,
            height_limit,
        )
        return []

    renderer.device.queue.write_buffer(blocks_buffer, 0, memoryview(blocks[:chunk_count]))
    renderer.device.queue.write_buffer(materials_buffer, 0, memoryview(materials[:chunk_count]))
    return make_chunk_mesh_batch_from_gpu_buffers(
        renderer,
        chunk_coords_list,
        blocks_buffer,
        materials_buffer,
        sample_size,
        height_limit,
    )


@profile
def make_chunk_mesh_batch_from_gpu_buffers(
    renderer,
    chunk_coords: list[tuple[int, int, int]],
    blocks_buffer,
    materials_buffer,
    sample_size: int,
    height_limit: int,
):
    if (
        renderer.voxel_mesh_count_pipeline is None
        or renderer.voxel_mesh_scan_pipeline is None
        or renderer.voxel_mesh_emit_pipeline is None
    ):
        raise RuntimeError("GPU meshing pipeline is unavailable.")
    if not chunk_coords:
        return []

    chunk_coords = _normalize_chunk_coords(chunk_coords)
    chunk_count = len(chunk_coords)
    columns_per_side = sample_size - 2
    ensure_voxel_mesh_batch_scratch(renderer, sample_size, height_limit, chunk_count)

    chunk_coords_array = renderer._voxel_mesh_scratch_coords_array
    chunk_totals = renderer._voxel_mesh_scratch_chunk_totals_array
    chunk_offsets = renderer._voxel_mesh_scratch_chunk_offsets_array
    coords_buffer = renderer._voxel_mesh_scratch_coords_buffer
    column_totals_buffer = renderer._voxel_mesh_scratch_column_totals_buffer
    chunk_totals_buffer = renderer._voxel_mesh_scratch_chunk_totals_buffer
    chunk_offsets_buffer = renderer._voxel_mesh_scratch_chunk_offsets_buffer
    batch_vertex_buffer = renderer._voxel_mesh_scratch_batch_vertex_buffer
    params_buffer = renderer._voxel_mesh_scratch_params_buffer

    assert chunk_coords_array is not None
    assert chunk_totals is not None
    assert chunk_offsets is not None
    assert coords_buffer is not None
    assert column_totals_buffer is not None
    assert chunk_totals_buffer is not None
    assert chunk_offsets_buffer is not None
    assert batch_vertex_buffer is not None
    assert params_buffer is not None

    for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunk_coords):
        chunk_coords_array[index, 0] = int(chunk_x)
        chunk_coords_array[index, 1] = int(chunk_y)
        chunk_coords_array[index, 2] = int(chunk_z)
        chunk_coords_array[index, 3] = 0

    renderer.device.queue.write_buffer(coords_buffer, 0, memoryview(chunk_coords_array[:chunk_count]))
    renderer.device.queue.write_buffer(
        params_buffer,
        0,
        struct.pack(
            "<4I4f",
            int(sample_size),
            int(height_limit),
            int(chunk_count),
            int(renderer.world.chunk_size),
            float(renderer.world.block_size),
            0.0,
            0.0,
            0.0,
        ),
    )

    chunk_totals[:chunk_count].fill(0)
    chunk_offsets[:chunk_count].fill(0)
    renderer.device.queue.write_buffer(chunk_totals_buffer, 0, memoryview(chunk_totals[:chunk_count]))
    renderer.device.queue.write_buffer(chunk_offsets_buffer, 0, memoryview(chunk_offsets[:chunk_count]))

    shared_entries = [
        {"binding": 0, "resource": {"buffer": blocks_buffer}},
        {"binding": 1, "resource": {"buffer": materials_buffer}},
        {"binding": 2, "resource": {"buffer": coords_buffer}},
        {"binding": 3, "resource": {"buffer": column_totals_buffer}},
        {"binding": 4, "resource": {"buffer": chunk_totals_buffer}},
        {"binding": 5, "resource": {"buffer": chunk_offsets_buffer}},
        {"binding": 6, "resource": {"buffer": batch_vertex_buffer}},
        {"binding": 8, "resource": {"buffer": params_buffer, "offset": 0, "size": 32}},
    ]
    count_bind_group = renderer.device.create_bind_group(
        layout=renderer.voxel_mesh_count_bind_group_layout,
        entries=shared_entries,
    )
    scan_bind_group = renderer.device.create_bind_group(
        layout=renderer.voxel_mesh_scan_bind_group_layout,
        entries=shared_entries,
    )
    encoder = renderer.device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(renderer.voxel_mesh_count_pipeline)
    compute_pass.set_bind_group(0, count_bind_group)
    compute_pass.dispatch_workgroups(columns_per_side, columns_per_side, chunk_count)
    compute_pass.end()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(renderer.voxel_mesh_scan_pipeline)
    compute_pass.set_bind_group(0, scan_bind_group)
    compute_pass.dispatch_workgroups(1, 1, chunk_count)
    compute_pass.end()
    renderer.device.queue.submit([encoder.finish()])

    validate_scan = (
        renderer.voxel_mesh_scan_validate_every > 0
        and (renderer._voxel_mesh_scan_batches_processed % renderer.voxel_mesh_scan_validate_every) == 0
    )
    renderer._voxel_mesh_scan_batches_processed += 1

    chunk_totals_readback, chunk_offsets_readback = read_chunk_mesh_batch_metadata(
        renderer,
        chunk_totals_buffer,
        chunk_offsets_buffer,
        chunk_count,
        include_offsets=validate_scan,
    )

    np.copyto(chunk_totals[:chunk_count], chunk_totals_readback)
    if chunk_count > 0:
        np.cumsum(chunk_totals_readback, dtype=np.uint32, out=chunk_offsets[:chunk_count])
        chunk_offsets[:chunk_count] -= chunk_totals_readback
        if validate_scan and chunk_offsets_readback is not None:
            if not np.array_equal(chunk_offsets_readback, chunk_offsets[:chunk_count]):
                renderer.device.queue.write_buffer(chunk_offsets_buffer, 0, memoryview(chunk_offsets[:chunk_count]))
    else:
        chunk_offsets[:0] = 0

    total_vertices = int(chunk_totals_readback.sum(dtype=np.uint64))
    total_vertex_bytes = total_vertices * _renderer_module().VERTEX_STRIDE
    batch_allocation = mesh_cache.allocate_mesh_output_range(renderer, _mesh_output_request_bytes(renderer, total_vertex_bytes))
    vertex_buffer = batch_allocation.buffer
    emit_bind_group = renderer.device.create_bind_group(
        layout=renderer.voxel_mesh_emit_bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": blocks_buffer}},
            {"binding": 1, "resource": {"buffer": materials_buffer}},
            {"binding": 2, "resource": {"buffer": coords_buffer}},
            {"binding": 3, "resource": {"buffer": column_totals_buffer}},
            {"binding": 4, "resource": {"buffer": chunk_totals_buffer}},
            {"binding": 5, "resource": {"buffer": chunk_offsets_buffer}},
            {
                "binding": 6,
                "resource": {
                    "buffer": vertex_buffer,
                    "offset": batch_allocation.offset_bytes,
                    "size": _storage_binding_size(batch_allocation.size_bytes),
                },
            },
            {"binding": 8, "resource": {"buffer": params_buffer, "offset": 0, "size": 32}},
        ],
    )
    encoder = renderer.device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(renderer.voxel_mesh_emit_pipeline)
    compute_pass.set_bind_group(0, emit_bind_group)
    compute_pass.dispatch_workgroups(columns_per_side, columns_per_side, chunk_count)
    compute_pass.end()
    renderer.device.queue.submit([encoder.finish()])

    renderer_module = _renderer_module()
    meshes: list[mt.ChunkMesh] = []
    created_at = time.perf_counter()
    for chunk_index, (chunk_x, chunk_y, chunk_z) in enumerate(chunk_coords):
        meshes.append(
            mt.ChunkMesh(
                chunk_x=int(chunk_x),
                chunk_y=int(chunk_y),
                chunk_z=int(chunk_z),
                vertex_count=int(chunk_totals[chunk_index]),
                vertex_buffer=vertex_buffer,
                vertex_offset=batch_allocation.offset_bytes + int(chunk_offsets[chunk_index]) * renderer_module.VERTEX_STRIDE,
                max_height=int(chunk_y) * int(renderer_module.CHUNK_SIZE) + int(height_limit),
                created_at=created_at,
                allocation_id=batch_allocation.allocation_id,
            )
        )
    return meshes


@profile
def make_chunk_mesh_batch_from_surface_gpu_batch(
    renderer,
    surface_batch: ChunkSurfaceGpuBatch,
    *,
    defer_finalize: bool = False,
):
    if renderer.voxel_surface_expand_pipeline is None:
        raise RuntimeError("GPU surface expansion pipeline is unavailable.")
    if not surface_batch.chunks:
        return []

    if defer_finalize:
        make_chunk_mesh_batches_from_surface_gpu_batches(renderer, [surface_batch])
        return []

    chunk_coords = _normalize_chunk_coords(surface_batch.chunks if isinstance(surface_batch.chunks, list) else list(surface_batch.chunks))
    chunk_count = len(chunk_coords)
    sample_size = renderer.world.chunk_size + 2
    height_limit = renderer.world.chunk_size
    params_bytes = struct.pack(
        "<8I",
        int(sample_size),
        int(height_limit),
        int(chunk_count),
        int(renderer.world.chunk_size),
        int(renderer.world.height),
        int(renderer.world.seed) & 0xFFFFFFFF,
        0,
        0,
    )
    ensure_voxel_mesh_batch_scratch(renderer, sample_size, height_limit, chunk_count)
    blocks_buffer = renderer._voxel_mesh_scratch_blocks_buffer
    materials_buffer = renderer._voxel_mesh_scratch_materials_buffer
    params_buffer = renderer._voxel_mesh_scratch_params_buffer
    assert blocks_buffer is not None
    assert materials_buffer is not None
    assert params_buffer is not None

    coords_array = renderer._voxel_mesh_scratch_coords_array
    coords_buffer = renderer._voxel_mesh_scratch_coords_buffer
    assert coords_array is not None
    assert coords_buffer is not None
    for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunk_coords):
        coords_array[index, 0] = int(chunk_x)
        coords_array[index, 1] = int(chunk_y)
        coords_array[index, 2] = int(chunk_z)
        coords_array[index, 3] = 0
    renderer.device.queue.write_buffer(params_buffer, 0, params_bytes)
    renderer.device.queue.write_buffer(coords_buffer, 0, memoryview(coords_array[:chunk_count]))
    expand_bind_group = get_voxel_surface_expand_bind_group(
        renderer,
        surface_batch,
        blocks_buffer,
        materials_buffer,
        params_buffer,
        coords_buffer,
    )
    encoder = renderer.device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(renderer.voxel_surface_expand_pipeline)
    compute_pass.set_bind_group(0, expand_bind_group)
    workgroups = (sample_size + 7) // 8
    compute_pass.dispatch_workgroups(workgroups, workgroups, chunk_count * (height_limit + 2))
    compute_pass.end()
    renderer.device.queue.submit([encoder.finish()])

    meshes = make_chunk_mesh_batch_from_gpu_buffers(
        renderer,
        chunk_coords,
        blocks_buffer,
        materials_buffer,
        sample_size,
        height_limit,
    )
    callback = getattr(surface_batch, "_release_callback", None)
    if callable(callback):
        try:
            callback()
        finally:
            try:
                setattr(surface_batch, "_release_callback", None)
            except Exception:
                pass
    return meshes


@profile
def make_chunk_mesh_batches_from_surface_gpu_batches(renderer, surface_batches: list[ChunkSurfaceGpuBatch]) -> None:
    if renderer.voxel_surface_expand_pipeline is None:
        raise RuntimeError("GPU surface expansion pipeline is unavailable.")
    if renderer.voxel_mesh_count_pipeline is None or renderer.voxel_mesh_scan_pipeline is None:
        raise RuntimeError("GPU meshing pipeline is unavailable.")
    if not surface_batches:
        return

    sample_size = renderer.world.chunk_size + 2
    height_limit = renderer.world.chunk_size
    workgroups = (sample_size + 7) // 8

    for surface_batch in surface_batches:
        if not surface_batch.chunks:
            release_surface_gpu_batch_immediately(renderer, surface_batch)
            continue

        chunk_coords = _normalize_chunk_coords(surface_batch.chunks if isinstance(surface_batch.chunks, list) else list(surface_batch.chunks))
        chunk_count = len(chunk_coords)
        resources = acquire_async_voxel_mesh_batch_resources(renderer, sample_size, height_limit, chunk_count)
        params_bytes = struct.pack(
            "<8I",
            int(sample_size),
            int(height_limit),
            int(chunk_count),
            int(renderer.world.chunk_size),
            int(renderer.world.height),
            int(renderer.world.seed) & 0xFFFFFFFF,
            0,
            0,
        )
        coords_view = resources.coords_array[:chunk_count]
        for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunk_coords):
            coords_view[index, 0] = int(chunk_x)
            coords_view[index, 1] = int(chunk_y)
            coords_view[index, 2] = int(chunk_z)
            coords_view[index, 3] = 0

        expand_params_buffer = getattr(resources, "expand_params_buffer", None) or resources.params_buffer
        renderer.device.queue.write_buffer(expand_params_buffer, 0, params_bytes)
        renderer.device.queue.write_buffer(resources.coords_buffer, 0, memoryview(coords_view))
        expand_bind_group = get_voxel_surface_expand_bind_group(
            renderer,
            surface_batch,
            resources.blocks_buffer,
            resources.materials_buffer,
            expand_params_buffer,
            resources.coords_buffer,
        )

        encoder = renderer.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(renderer.voxel_surface_expand_pipeline)
        compute_pass.set_bind_group(0, expand_bind_group)
        compute_pass.dispatch_workgroups(workgroups, workgroups, chunk_count * (height_limit + 2))
        compute_pass.end()

        surface_release_callbacks: list[object] = []
        callback = getattr(surface_batch, "_release_callback", None)
        if callable(callback):
            surface_release_callbacks.append(callback)
            try:
                setattr(surface_batch, "_release_callback", None)
            except Exception:
                pass

        enqueue_gpu_chunk_mesh_batch_from_gpu_buffers(
            renderer,
            chunk_coords,
            resources,
            sample_size,
            height_limit,
            params_already_uploaded=False,
            encoder=encoder,
            submit=True,
            surface_release_callbacks=surface_release_callbacks,
        )


@profile
def release_pending_chunk_mesh_readback(renderer, pending) -> None:
    owner = pending.readback_owner
    if owner is None:
        if pending.readback_buffer.map_state != "unmapped":
            try:
                pending.readback_buffer.unmap()
            except Exception:
                pass
    else:
        owner.remaining_batches -= 1
        if owner.remaining_batches > 0:
            return
        if owner.buffer.map_state != "unmapped":
            try:
                owner.buffer.unmap()
            except Exception:
                pass
        try:
            owner.buffer.destroy()
        except Exception:
            pass

    callbacks = getattr(pending, "surface_release_callbacks", None) or ()
    for callback in callbacks:
        if not callable(callback):
            continue
        try:
            callback()
        except Exception:
            pass
    try:
        pending.surface_release_callbacks.clear()
    except Exception:
        pass


@profile
def finalize_pending_gpu_mesh_batches(renderer, budget: int | None = None) -> int:
    if not renderer._pending_gpu_mesh_batches:
        return 0

    budget = max(1, int(renderer._gpu_mesh_async_finalize_budget if budget is None else budget))
    completed = 0
    remaining: deque[object] = deque()
    shared_readback_views: dict[int, memoryview] = {}
    pending_infos: list[tuple[object, np.ndarray, np.ndarray, int]] = []

    while renderer._pending_gpu_mesh_batches:
        if completed >= budget:
            remaining.extend(renderer._pending_gpu_mesh_batches)
            break

        pending = renderer._pending_gpu_mesh_batches.popleft()
        if pending.readback_buffer.map_state != "mapped":
            remaining.append(pending)
            continue

        totals_nbytes = pending.chunk_count * 4
        owner = pending.readback_owner
        if owner is not None:
            owner_key = id(owner.buffer)
            metadata_view = shared_readback_views.get(owner_key)
            if metadata_view is None:
                metadata_view = owner.buffer.read_mapped(0, owner.total_nbytes, copy=False)
                shared_readback_views[owner_key] = memoryview(metadata_view)
            totals_slice = shared_readback_views[owner_key][pending.readback_offset : pending.readback_offset + totals_nbytes]
            chunk_totals = np.frombuffer(totals_slice, dtype=np.uint32, count=pending.chunk_count).copy()
        else:
            metadata_view = pending.readback_buffer.read_mapped(0, totals_nbytes, copy=False)
            chunk_totals = np.frombuffer(metadata_view, dtype=np.uint32, count=pending.chunk_count).copy()

        chunk_offsets = np.empty(pending.chunk_count, dtype=np.uint32)
        if pending.chunk_count > 0:
            np.cumsum(chunk_totals, dtype=np.uint32, out=chunk_offsets)
            chunk_offsets -= chunk_totals
        else:
            chunk_offsets = np.empty(0, dtype=np.uint32)

        total_vertices = int(chunk_totals.sum(dtype=np.uint64))
        total_vertex_bytes = total_vertices * _renderer_module().VERTEX_STRIDE
        pending_infos.append((pending, chunk_totals, chunk_offsets, total_vertex_bytes))
        completed += 1
        release_pending_chunk_mesh_readback(renderer, pending)

    meshes_to_store: list[object] = []
    if pending_infos:
        batch_alignment = max(1, int(getattr(renderer, "_mesh_output_binding_alignment", 256)))
        batch_relative_offsets: list[int] = []
        cursor_bytes = 0
        for _pending, _chunk_totals, _chunk_offsets, total_vertex_bytes in pending_infos:
            cursor_bytes = renderer._align_up(cursor_bytes, batch_alignment)
            batch_relative_offsets.append(cursor_bytes)
            cursor_bytes += int(total_vertex_bytes)
        total_ready_vertex_bytes = _mesh_output_request_bytes(renderer, cursor_bytes)
        shared_allocation = mesh_cache.allocate_mesh_output_range(renderer, total_ready_vertex_bytes)
        ready_batches: list[tuple[object, np.ndarray, np.ndarray, int, object | None]] = []
        copy_batches: list[tuple[wgpu.GPUBuffer, int, wgpu.GPUBuffer, int]] = []
        for (pending, chunk_totals, chunk_offsets, total_vertex_bytes), batch_relative_offset in zip(pending_infos, batch_relative_offsets):
            if pending.chunk_count > 0:
                renderer.device.queue.write_buffer(pending.chunk_offsets_buffer, 0, memoryview(chunk_offsets))
            batch_base_offset = int(shared_allocation.offset_bytes) + int(batch_relative_offset)
            emit_bind_group = None
            emit_vertex_buffer = None
            resources = pending.resources

            # Air-only/high-altitude batches have no vertices to emit. Do not
            # dispatch the emit shader for them; just cache zero-vertex meshes.
            # This avoids binding a tiny dummy output range to a shader storage
            # array whose minimum validated footprint is one full vertex.
            if total_vertex_bytes > 0:
                if resources is not None:
                    emit_bind_group = get_cached_async_voxel_mesh_emit_bind_group(renderer, resources, shared_allocation)
                    emit_vertex_buffer = resources.emit_vertex_buffer
                if emit_bind_group is None:
                    emit_bind_group = renderer.device.create_bind_group(
                        layout=renderer.voxel_mesh_emit_bind_group_layout,
                        entries=[
                            {"binding": 0, "resource": {"buffer": pending.blocks_buffer}},
                            {"binding": 1, "resource": {"buffer": pending.materials_buffer}},
                            {"binding": 2, "resource": {"buffer": pending.coords_buffer}},
                            {"binding": 3, "resource": {"buffer": pending.column_totals_buffer}},
                            {"binding": 4, "resource": {"buffer": pending.chunk_totals_buffer}},
                            {"binding": 5, "resource": {"buffer": pending.chunk_offsets_buffer}},
                            {
                                "binding": 6,
                                "resource": {
                                    "buffer": shared_allocation.buffer,
                                    "offset": batch_base_offset,
                                    "size": _emit_vertex_binding_size(total_vertex_bytes),
                                },
                            },
                            {"binding": 8, "resource": {"buffer": pending.params_buffer, "offset": 0, "size": 32}},
                        ],
                    )
                elif emit_vertex_buffer is not None:
                    copy_batches.append((emit_vertex_buffer, int(total_vertex_bytes), shared_allocation.buffer, int(batch_base_offset)))
            ready_batches.append((pending, chunk_totals, chunk_offsets, batch_base_offset, emit_bind_group))

        encoder = renderer.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(renderer.voxel_mesh_emit_pipeline)
        for pending, _, _, _, emit_bind_group in ready_batches:
            if emit_bind_group is None:
                continue
            compute_pass.set_bind_group(0, emit_bind_group)
            compute_pass.dispatch_workgroups(pending.columns_per_side, pending.columns_per_side, pending.chunk_count)
        compute_pass.end()
        for src_buffer, copy_nbytes, dst_buffer, dst_offset in copy_batches:
            encoder.copy_buffer_to_buffer(src_buffer, 0, dst_buffer, dst_offset, copy_nbytes)
        renderer.device.queue.submit([encoder.finish()])

        created_at = time.perf_counter()
        shared_vertex_buffer = shared_allocation.buffer
        shared_allocation_id = shared_allocation.allocation_id
        height_cache: dict[tuple[int, int], tuple[float, float, float, int]] = {}
        for pending, chunk_totals, chunk_offsets, batch_base_offset, _ in ready_batches:
            height_limit_int = int(pending.height_limit)
            for chunk_index, (chunk_x, chunk_y, chunk_z) in enumerate(pending.chunk_coords):
                chunk_x_int = int(chunk_x)
                chunk_y_int = int(chunk_y)
                chunk_z_int = int(chunk_z)
                cached_height = height_cache.get((chunk_y_int, height_limit_int))
                if cached_height is None:
                    min_y = float(chunk_y_int * _renderer_module().CHUNK_SIZE) * float(_renderer_module().BLOCK_SIZE)
                    max_height_int = int(chunk_y_int * _renderer_module().CHUNK_SIZE + height_limit_int)
                    max_y = float(max_height_int) * float(_renderer_module().BLOCK_SIZE)
                    half_height = max(0.0, max_y - min_y) * 0.5
                    center_y = min_y + half_height
                    radius = float(math.sqrt(_chunk_half() * _chunk_half() * 2.0 + half_height * half_height))
                    cached_height = (center_y, half_height, radius, max_height_int)
                    height_cache[(chunk_y_int, height_limit_int)] = cached_height
                center_y, _half_height, radius, max_height_int = cached_height
                vertex_count = int(chunk_totals[chunk_index])
                vertex_offset = batch_base_offset + int(chunk_offsets[chunk_index]) * _renderer_module().VERTEX_STRIDE
                binding_offset = int(vertex_offset % _renderer_module().VERTEX_STRIDE)
                first_vertex = int((vertex_offset - binding_offset) // _renderer_module().VERTEX_STRIDE)
                mesh = mt.ChunkMesh.__new__(mt.ChunkMesh)
                mesh.chunk_x = chunk_x_int
                mesh.chunk_y = chunk_y_int
                mesh.chunk_z = chunk_z_int
                mesh.vertex_count = vertex_count
                mesh.vertex_buffer = shared_vertex_buffer
                mesh.max_height = int(max_height_int)
                mesh.vertex_offset = vertex_offset
                mesh.created_at = float(created_at)
                mesh.allocation_id = shared_allocation_id
                mesh.bounds = (
                    float(chunk_x_int * _renderer_module().CHUNK_WORLD_SIZE + _chunk_half()),
                    float(center_y),
                    float(chunk_z_int * _renderer_module().CHUNK_WORLD_SIZE + _chunk_half()),
                    radius,
                )
                mesh.binding_offset = binding_offset
                mesh.first_vertex = first_vertex
                meshes_to_store.append(mesh)
                try:
                    renderer._pending_chunk_coords.discard((chunk_x_int, chunk_y_int, chunk_z_int))
                except Exception:
                    pass
            if pending.resources is not None:
                schedule_async_voxel_mesh_batch_resource_release(renderer, pending.resources, frames=2)
            else:
                schedule_gpu_buffer_cleanup(
                    renderer,
                    [
                        pending.blocks_buffer,
                        pending.materials_buffer,
                        pending.coords_buffer,
                        pending.column_totals_buffer,
                        pending.chunk_totals_buffer,
                        pending.chunk_offsets_buffer,
                        pending.params_buffer,
                        pending.readback_buffer,
                    ],
                    frames=2,
                )

    if meshes_to_store:
        mesh_cache.store_chunk_meshes(renderer, meshes_to_store)

    renderer._pending_gpu_mesh_batches = remaining
    return completed
