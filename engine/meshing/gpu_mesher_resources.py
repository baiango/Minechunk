from __future__ import annotations

from collections import deque

import numpy as np
import wgpu

from .. import meshing_types as mt
from .gpu_mesher_common import profile, _renderer_module, _storage_height


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
