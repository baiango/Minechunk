from __future__ import annotations

import struct
import time

import numpy as np
import wgpu

from .. import meshing_types as mt
from ..cache import mesh_allocator as mesh_cache
from ..terrain.types import ChunkVoxelResult
from .gpu_mesher_common import (
    profile,
    _mesh_output_request_bytes,
    _normalize_chunk_coords,
    _renderer_module,
    _storage_binding_size,
)
from .gpu_mesher_resources import (
    acquire_async_voxel_mesh_batch_resources,
    async_voxel_mesh_batch_resources_match,
    ensure_voxel_mesh_batch_scratch,
)


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
