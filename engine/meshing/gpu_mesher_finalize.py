from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import wgpu

from .. import meshing_types as mt
from .. import render_contract as render_consts
from ..cache import mesh_allocator as mesh_cache
from .gpu_mesher_common import (
    profile,
    _chunk_half,
    _emit_vertex_binding_size,
    _mesh_output_request_bytes,
    _renderer_module,
)
from .gpu_mesher_resources import (
    get_cached_async_voxel_mesh_emit_bind_group,
    schedule_async_voxel_mesh_batch_resource_release,
    schedule_gpu_buffer_cleanup,
)


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
            cursor_bytes = render_consts.align_up(cursor_bytes, batch_alignment)
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
