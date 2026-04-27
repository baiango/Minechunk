from __future__ import annotations

# Compatibility facade for the split WGPU voxel mesher modules.
# Keep importing from engine.meshing.gpu_mesher working while the implementation
# lives in responsibility-focused files.

from .gpu_mesher_common import (
    _chunk_half,
    _emit_vertex_binding_size,
    _mesh_output_request_bytes,
    _normalize_chunk_coord,
    _normalize_chunk_coords,
    _renderer_module,
    _storage_binding_size,
    _storage_height,
    profile,
)
from .gpu_mesher_resources import *
from .gpu_mesher_batches import *
from .gpu_surface_batches import *
from .gpu_mesher_finalize import *

__all__ = [
    "ensure_voxel_mesh_batch_scratch",
    "create_async_voxel_mesh_batch_resources",
    "async_voxel_mesh_batch_resources_match",
    "destroy_async_voxel_mesh_batch_resources",
    "acquire_async_voxel_mesh_batch_resources",
    "release_async_voxel_mesh_batch_resources",
    "schedule_async_voxel_mesh_batch_resource_release",
    "get_cached_async_voxel_mesh_emit_bind_group",
    "schedule_gpu_buffer_cleanup",
    "process_gpu_buffer_cleanup",
    "get_voxel_surface_expand_bind_group",
    "schedule_surface_gpu_batch_release",
    "release_surface_gpu_batch_immediately",
    "pending_surface_gpu_batches_chunk_count",
    "enqueue_surface_gpu_batches_for_meshing",
    "drain_pending_surface_gpu_batches_to_meshing",
    "enqueue_gpu_chunk_mesh_batch_from_gpu_buffers",
    "read_chunk_mesh_batch_metadata",
    "make_chunk_mesh_batch_from_terrain_results",
    "make_chunk_mesh_batch_from_gpu_buffers",
    "make_chunk_mesh_batch_from_surface_gpu_batch",
    "make_chunk_mesh_batches_from_surface_gpu_batches",
    "release_pending_chunk_mesh_readback",
    "finalize_pending_gpu_mesh_batches",
]
