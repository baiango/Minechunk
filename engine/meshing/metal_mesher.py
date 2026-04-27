"""Compatibility façade for the Metal voxel mesher.

The implementation is split by responsibility:

* metal_mesher_common: shared dataclasses, constants, and renderer async-state helpers
* metal_chunk_mesher: MetalChunkMesher slot/pipeline/dispatch implementation
* metal_mesher_cache: renderer-local mesher cache and prewarming
* metal_mesher_async: enqueue/drain/synchronous batch entry points
* metal_mesher_finalize: completed-buffer readback, mesh materialization, and cleanup

Keep this module as the stable public import path for existing renderer and
pipeline code.
"""

from __future__ import annotations

from .metal_chunk_mesher import MetalChunkMesher
from .metal_mesher_async import (
    drain_pending_surface_gpu_batches_to_meshing,
    enqueue_surface_gpu_batches_for_meshing,
    make_chunk_mesh_batch_from_surface_gpu_batch,
    make_chunk_mesh_batch_from_terrain_results,
    make_chunk_mesh_batches_from_surface_gpu_batches,
    pending_surface_gpu_batches_chunk_count,
    shutdown_renderer_async_state,
    submit_chunk_mesh_batch_async,
)
from .metal_mesher_cache import get_metal_chunk_mesher, prewarm_metal_chunk_mesher
from .metal_mesher_common import (
    AsyncMetalMeshBatchResources,
    MetalMesherSlot,
    PendingMetalChunkMesherInit,
    release_surface_gpu_batch_immediately,
)
from .metal_mesher_finalize import (
    destroy_async_voxel_mesh_batch_resources,
    finalize_pending_gpu_mesh_batches,
    process_gpu_buffer_cleanup,
)

__all__ = [
    "AsyncMetalMeshBatchResources",
    "MetalChunkMesher",
    "MetalMesherSlot",
    "PendingMetalChunkMesherInit",
    "destroy_async_voxel_mesh_batch_resources",
    "drain_pending_surface_gpu_batches_to_meshing",
    "enqueue_surface_gpu_batches_for_meshing",
    "finalize_pending_gpu_mesh_batches",
    "get_metal_chunk_mesher",
    "make_chunk_mesh_batch_from_surface_gpu_batch",
    "make_chunk_mesh_batch_from_terrain_results",
    "make_chunk_mesh_batches_from_surface_gpu_batches",
    "pending_surface_gpu_batches_chunk_count",
    "prewarm_metal_chunk_mesher",
    "process_gpu_buffer_cleanup",
    "release_surface_gpu_batch_immediately",
    "shutdown_renderer_async_state",
    "submit_chunk_mesh_batch_async",
]
