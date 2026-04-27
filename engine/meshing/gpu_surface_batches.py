from __future__ import annotations

import struct
import time

from ..terrain.types import ChunkSurfaceGpuBatch
from .gpu_mesher_batches import (
    enqueue_gpu_chunk_mesh_batch_from_gpu_buffers,
    make_chunk_mesh_batch_from_gpu_buffers,
)
from .gpu_mesher_common import profile, _normalize_chunk_coords, _renderer_module
from .gpu_mesher_resources import (
    acquire_async_voxel_mesh_batch_resources,
    ensure_voxel_mesh_batch_scratch,
)


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
