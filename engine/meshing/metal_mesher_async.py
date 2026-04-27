from __future__ import annotations

from collections import deque
from typing import Callable

from ..meshing import cpu_mesher
from ..terrain.types import ChunkSurfaceGpuBatch, ChunkVoxelResult
from .metal_mesher_cache import get_metal_chunk_mesher
from .metal_mesher_common import _ensure_renderer_async_state, profile, release_surface_gpu_batch_immediately
from .metal_mesher_finalize import (
    _build_completed_meshes_from_resources,
    destroy_async_voxel_mesh_batch_resources,
    finalize_pending_gpu_mesh_batches,
)


@profile
def shutdown_renderer_async_state(renderer) -> None:
    pending = list(getattr(renderer, "_pending_gpu_mesh_batches", ()) or ())
    for resources in pending:
        destroy_async_voxel_mesh_batch_resources(resources)
    if hasattr(renderer, "_pending_gpu_mesh_batches"):
        renderer._pending_gpu_mesh_batches.clear()
    if hasattr(renderer, "_pending_surface_gpu_batches"):
        while renderer._pending_surface_gpu_batches:
            release_surface_gpu_batch_immediately(renderer._pending_surface_gpu_batches.popleft())
    renderer._pending_surface_gpu_batches_chunk_total = 0
    cache = getattr(renderer, "_metal_chunk_mesher_cache", None)
    if cache:
        for mesher in list(cache.values()):
            try:
                mesher.destroy()
            except Exception:
                pass
        cache.clear()


@profile
def submit_chunk_mesh_batch_async(renderer, chunk_results: list[ChunkVoxelResult], on_complete: Callable[[list[object]], None] | None = None):
    mesher = get_metal_chunk_mesher(renderer, block=True, min_chunk_capacity=len(chunk_results))
    resources = mesher.submit_chunk_mesh_batch(renderer, chunk_results, on_complete)
    if resources is not None:
        _ensure_renderer_async_state(renderer)
        renderer._pending_gpu_mesh_batches.append(resources)
    return resources


@profile
def make_chunk_mesh_batch_from_terrain_results(renderer, terrain_results: list[ChunkVoxelResult]):
    if not terrain_results:
        return []
    mesher = get_metal_chunk_mesher(renderer, block=True, min_chunk_capacity=len(terrain_results))
    resources = mesher.submit_chunk_mesh_batch(renderer, terrain_results, None)
    if resources is None:
        return cpu_mesher.cpu_make_chunk_mesh_batch_from_terrain_results(renderer, terrain_results)
    if resources.command_buffer is not None:
        resources.command_buffer.waitUntilCompleted()
    try:
        return _build_completed_meshes_from_resources(renderer, resources)
    finally:
        destroy_async_voxel_mesh_batch_resources(resources)


@profile
def enqueue_surface_gpu_batches_for_meshing(renderer, surface_batches: list[ChunkSurfaceGpuBatch]) -> list[object]:
    _ensure_renderer_async_state(renderer)
    for surface_batch in surface_batches:
        if not getattr(surface_batch, "chunks", None):
            release_surface_gpu_batch_immediately(surface_batch)
            continue
        renderer._pending_surface_gpu_batches.append(surface_batch)
        renderer._pending_surface_gpu_batches_chunk_total += len(surface_batch.chunks)
    return []


@profile
def pending_surface_gpu_batches_chunk_count(renderer) -> int:
    return int(getattr(renderer, "_pending_surface_gpu_batches_chunk_total", 0))


@profile
def drain_pending_surface_gpu_batches_to_meshing(renderer) -> int:
    _ensure_renderer_async_state(renderer)
    if not renderer._pending_surface_gpu_batches:
        return 0
    drained = 0
    kept: deque[ChunkSurfaceGpuBatch] = deque()
    while renderer._pending_surface_gpu_batches:
        surface_batch = renderer._pending_surface_gpu_batches.popleft()
        chunk_count = len(getattr(surface_batch, "chunks", []) or [])
        if chunk_count <= 0:
            release_surface_gpu_batch_immediately(surface_batch)
            continue
        mesher = get_metal_chunk_mesher(renderer, block=True, min_chunk_capacity=chunk_count)
        resources = mesher.submit_surface_gpu_batch(renderer, surface_batch, None)
        if resources is None:
            kept.appendleft(surface_batch)
            break
        renderer._pending_gpu_mesh_batches.append(resources)
        drained += len(resources.chunk_coords)
        renderer._pending_surface_gpu_batches_chunk_total = max(
            0,
            int(renderer._pending_surface_gpu_batches_chunk_total) - len(resources.chunk_coords),
        )
    while renderer._pending_surface_gpu_batches:
        kept.append(renderer._pending_surface_gpu_batches.popleft())
    renderer._pending_surface_gpu_batches = kept
    return drained


@profile
def make_chunk_mesh_batch_from_surface_gpu_batch(renderer, surface_batch: ChunkSurfaceGpuBatch, *, defer_finalize: bool = False):
    enqueue_surface_gpu_batches_for_meshing(renderer, [surface_batch])
    drain_pending_surface_gpu_batches_to_meshing(renderer)
    if defer_finalize:
        return []
    finalize_pending_gpu_mesh_batches(renderer, budget=999999)
    return []


@profile
def make_chunk_mesh_batches_from_surface_gpu_batches(renderer, surface_batches: list[ChunkSurfaceGpuBatch]) -> None:
    enqueue_surface_gpu_batches_for_meshing(renderer, surface_batches)
    drain_pending_surface_gpu_batches_to_meshing(renderer)


__all__ = [
    "drain_pending_surface_gpu_batches_to_meshing",
    "enqueue_surface_gpu_batches_for_meshing",
    "make_chunk_mesh_batch_from_surface_gpu_batch",
    "make_chunk_mesh_batch_from_terrain_results",
    "make_chunk_mesh_batches_from_surface_gpu_batches",
    "pending_surface_gpu_batches_chunk_count",
    "shutdown_renderer_async_state",
    "submit_chunk_mesh_batch_async",
]
