from __future__ import annotations

import time
from collections import deque

import numpy as np

from .. import render_contract as render_consts
from ..cache import mesh_allocator as mesh_cache
from ..meshing import cpu_mesher
from ..meshing_types import ChunkMesh
from ..terrain.types import ChunkVoxelResult
from .metal_mesher_common import (
    AsyncMetalMeshBatchResources,
    _command_buffer_completed,
    _ensure_renderer_async_state,
    _renderer_module,
    profile,
)


def _cpu_fallback_mesh_for_coord(renderer, coord: tuple[int, int, int]) -> ChunkMesh:
    chunk_x, chunk_y, chunk_z = coord
    blocks, materials = renderer.world.chunk_voxel_grid(chunk_x, chunk_y, chunk_z)
    result = ChunkVoxelResult(
        chunk_x=chunk_x,
        chunk_y=chunk_y,
        chunk_z=chunk_z,
        blocks=np.ascontiguousarray(blocks),
        materials=np.ascontiguousarray(materials),
        source="metal_overflow_cpu_fallback",
    )
    meshes = cpu_mesher.cpu_make_chunk_mesh_batch_from_terrain_results(renderer, [result])
    return meshes[0]


@profile
def _build_completed_meshes_from_resources(renderer, resources: AsyncMetalMeshBatchResources) -> list[object]:
    if resources.error is not None:
        raise resources.error
    slot = resources.slot
    if slot is None:
        return list(resources.completed_meshes)
    chunk_count = len(resources.chunk_coords)
    if chunk_count <= 0:
        return []

    chunk_totals = np.frombuffer(slot.counts_buffer.contents().as_buffer(chunk_count * 4), dtype=np.uint32, count=chunk_count).copy()
    overflow_flags = np.frombuffer(slot.overflow_buffer.contents().as_buffer(chunk_count * 4), dtype=np.uint32, count=chunk_count).copy()
    total_vertices = int(chunk_totals.sum(dtype=np.uint64))
    created_at = time.perf_counter()
    renderer_module = _renderer_module()
    vertex_stride = int(getattr(slot, "vertex_stride", renderer_module.VERTEX_STRIDE))
    height_limit = int(slot.height_limit)
    max_height_by_chunk_y: dict[int, int] = {}
    meshes: list[object] = []

    if total_vertices <= 0:
        empty_buffer = cpu_mesher._shared_empty_chunk_vertex_buffer(renderer)
        for coord_index, coord in enumerate(resources.chunk_coords):
            chunk_x, chunk_y, chunk_z = coord
            if int(overflow_flags[coord_index]) != 0:
                meshes.append(_cpu_fallback_mesh_for_coord(renderer, coord))
                continue
            chunk_max_height = max_height_by_chunk_y.get(chunk_y)
            if chunk_max_height is None:
                chunk_max_height = int(chunk_y * renderer_module.CHUNK_SIZE + height_limit)
                max_height_by_chunk_y[chunk_y] = chunk_max_height
            meshes.append(
                cpu_mesher.make_chunk_mesh_fast(
                    renderer,
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    chunk_z=chunk_z,
                    vertex_count=0,
                    vertex_buffer=empty_buffer,
                    vertex_offset=0,
                    max_height=chunk_max_height,
                    created_at=created_at,
                    allocation_id=None,
                )
            )
        return meshes

    chunk_offsets_vertices = np.empty(chunk_count, dtype=np.uint64)
    running_vertices = 0
    for index, count in enumerate(chunk_totals):
        chunk_offsets_vertices[index] = running_vertices
        running_vertices += int(count)
    total_vertex_bytes = int(running_vertices) * vertex_stride
    allocation = mesh_cache.allocate_mesh_output_range(renderer, max(vertex_stride, total_vertex_bytes))
    upload = np.empty(total_vertex_bytes, dtype=np.uint8)
    max_vertices_per_chunk = int(getattr(slot, "max_vertices_per_chunk", getattr(resources.mesher, "max_vertices_per_chunk", renderer_module.MAX_VERTICES_PER_CHUNK)))
    vertex_pool = memoryview(slot.vertex_pool_buffer.contents().as_buffer(slot.chunk_capacity * max_vertices_per_chunk * vertex_stride))

    for index, coord in enumerate(resources.chunk_coords):
        chunk_x, chunk_y, chunk_z = coord
        vertex_count = int(chunk_totals[index])
        if int(overflow_flags[index]) != 0:
            meshes.append(_cpu_fallback_mesh_for_coord(renderer, coord))
            continue
        chunk_max_height = max_height_by_chunk_y.get(chunk_y)
        if chunk_max_height is None:
            chunk_max_height = int(chunk_y * renderer_module.CHUNK_SIZE + height_limit)
            max_height_by_chunk_y[chunk_y] = chunk_max_height
        if vertex_count <= 0:
            meshes.append(
                cpu_mesher.make_chunk_mesh_fast(
                    renderer,
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    chunk_z=chunk_z,
                    vertex_count=0,
                    vertex_buffer=cpu_mesher._shared_empty_chunk_vertex_buffer(renderer),
                    vertex_offset=0,
                    max_height=chunk_max_height,
                    created_at=created_at,
                    allocation_id=None,
                )
            )
            continue
        src_start = int(index) * max_vertices_per_chunk * vertex_stride
        copy_nbytes = vertex_count * vertex_stride
        dst_start = int(chunk_offsets_vertices[index]) * vertex_stride
        upload[dst_start : dst_start + copy_nbytes] = np.frombuffer(vertex_pool[src_start : src_start + copy_nbytes], dtype=np.uint8, count=copy_nbytes)
        meshes.append(
            cpu_mesher.make_chunk_mesh_fast(
                renderer,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                chunk_z=chunk_z,
                vertex_count=vertex_count,
                vertex_buffer=allocation.buffer,
                vertex_offset=int(allocation.offset_bytes) + dst_start,
                max_height=chunk_max_height,
                created_at=created_at,
                allocation_id=allocation.allocation_id,
            )
        )

    if total_vertex_bytes > 0:
        renderer.device.queue.write_buffer(allocation.buffer, int(allocation.offset_bytes), memoryview(upload))
    return meshes


@profile
def _append_completed_meshes_to_renderer(renderer, meshes: list[object]) -> None:
    if not meshes:
        return
    mesh_cache.store_chunk_meshes(renderer, meshes)


@profile
def process_gpu_buffer_cleanup(renderer) -> None:
    # Metal buffers are tied to slot lifetimes; WGPU mesh-output frees are handled by mesh_cache.
    return None


@profile
def finalize_pending_gpu_mesh_batches(renderer, budget: int | None = None) -> int:
    _ensure_renderer_async_state(renderer)
    if not renderer._pending_gpu_mesh_batches:
        return 0
    budget = max(1, int(getattr(renderer, "_gpu_mesh_async_finalize_budget", 4) if budget is None else budget))
    completed = 0
    remaining: deque[object] = deque()
    first_error: Exception | None = None

    while renderer._pending_gpu_mesh_batches:
        resources = renderer._pending_gpu_mesh_batches.popleft()
        kept_pending = False
        if completed >= budget:
            remaining.append(resources)
            continue
        try:
            if resources.command_buffer is not None and not _command_buffer_completed(resources.command_buffer):
                remaining.append(resources)
                kept_pending = True
                continue
            meshes = _build_completed_meshes_from_resources(renderer, resources)
            resources.completed_meshes = list(meshes)
            resources.completed_at = time.perf_counter()
            if resources.on_complete is not None:
                resources.on_complete(meshes)
            if resources.deliver_to_renderer:
                _append_completed_meshes_to_renderer(renderer, meshes)
            completed += 1
        except Exception as exc:
            first_error = exc if first_error is None else first_error
        finally:
            if not kept_pending:
                destroy_async_voxel_mesh_batch_resources(resources)

    renderer._pending_gpu_mesh_batches = remaining
    if first_error is not None:
        raise first_error
    return completed


@profile
def destroy_async_voxel_mesh_batch_resources(resources) -> None:
    if resources is None or getattr(resources, "cleaned_up", False):
        return
    resources.cleaned_up = True
    callbacks = list(getattr(resources, "surface_release_callbacks", []) or [])
    for callback in callbacks:
        if callable(callback):
            try:
                callback()
            except Exception:
                pass
    try:
        resources.surface_release_callbacks.clear()
    except Exception:
        pass
    mesher = getattr(resources, "mesher", None)
    slot = getattr(resources, "slot", None)
    if mesher is not None and slot is not None:
        mesher._release_slot(slot)
    resources.slot = None
    resources.mesher = None
    resources.command_buffer = None
    resources.chunk_results.clear()
    resources.chunk_coords.clear()


__all__ = [
    "destroy_async_voxel_mesh_batch_resources",
    "finalize_pending_gpu_mesh_batches",
    "process_gpu_buffer_cleanup",
]
