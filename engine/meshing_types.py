from __future__ import annotations

from dataclasses import dataclass, field

import math
import numpy as np
import wgpu


@dataclass
class ChunkMesh:
    chunk_x: int
    chunk_z: int
    vertex_count: int
    vertex_buffer: wgpu.GPUBuffer
    max_height: int
    chunk_y: int = 0
    vertex_offset: int = 0
    created_at: float = 0.0
    allocation_id: int | None = None
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    binding_offset: int = 0
    first_vertex: int = 0

    def __post_init__(self) -> None:
        try:
            from . import renderer as renderer_module

            chunk_world_size = float(renderer_module.CHUNK_WORLD_SIZE)
            block_size = float(renderer_module.BLOCK_SIZE)
            chunk_size = int(renderer_module.CHUNK_SIZE)
            vertex_stride = int(renderer_module.VERTEX_STRIDE)
        except Exception:
            chunk_size = 64
            chunk_world_size = 6.4
            block_size = 0.1
            vertex_stride = 48
        half_chunk = chunk_world_size * 0.5
        min_y = float(self.chunk_y * chunk_size) * block_size
        max_y = float(self.max_height) * block_size
        center_y = min_y + max(0.0, max_y - min_y) * 0.5
        half_height = max(0.0, max_y - min_y) * 0.5
        self.bounds = (
            float(self.chunk_x * chunk_world_size + half_chunk),
            float(center_y),
            float(self.chunk_z * chunk_world_size + half_chunk),
            float(math.sqrt(half_chunk * half_chunk * 2.0 + half_height * half_height)),
        )
        self.binding_offset = int(self.vertex_offset % vertex_stride)
        self.first_vertex = int((self.vertex_offset - self.binding_offset) // vertex_stride)


@dataclass
class MeshOutputSlab:
    slab_id: int
    buffer: wgpu.GPUBuffer
    size_bytes: int
    free_ranges: list[tuple[int, int]]
    append_offset: int = 0
    size_class_bytes: int = 0


@dataclass
class MeshBufferAllocation:
    allocation_id: int
    buffer: wgpu.GPUBuffer
    offset_bytes: int
    size_bytes: int
    slab_id: int | None = None
    refcount: int = 0


@dataclass
class ChunkRenderBatch:
    signature: tuple[tuple[int, int, int], ...]
    vertex_count: int
    vertex_buffer: wgpu.GPUBuffer
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    chunk_count: int = 0
    complete_tile: bool = False
    all_mature: bool = False
    visible_mask: int = 0
    source_version: int = 0


@dataclass
class ChunkDrawBatch:
    vertex_buffer: wgpu.GPUBuffer
    binding_offset: int
    vertex_count: int
    first_vertex: int
    bounds: tuple[float, float, float, float]
    chunk_count: int = 1


@dataclass
class AsyncVoxelMeshBatchResources:
    sample_size: int
    height_limit: int
    chunk_capacity: int
    column_capacity: int
    blocks_buffer: wgpu.GPUBuffer
    materials_buffer: wgpu.GPUBuffer
    coords_buffer: wgpu.GPUBuffer
    column_totals_buffer: wgpu.GPUBuffer
    chunk_totals_buffer: wgpu.GPUBuffer
    chunk_offsets_buffer: wgpu.GPUBuffer
    params_buffer: wgpu.GPUBuffer
    readback_buffer: wgpu.GPUBuffer
    emit_vertex_buffer: wgpu.GPUBuffer
    coords_array: np.ndarray
    zero_counts_array: np.ndarray
    count_bind_group: object | None = None
    scan_bind_group: object | None = None
    emit_bind_group: object | None = None
    emit_bind_group_cache: dict[int, object] = field(default_factory=dict)


@dataclass
class PendingChunkMeshReadbackGroup:
    buffer: wgpu.GPUBuffer
    total_nbytes: int
    remaining_batches: int
    map_requested: bool = False


@dataclass
class PendingChunkMeshBatch:
    chunk_coords: list[tuple[int, int, int]]
    chunk_count: int
    sample_size: int
    height_limit: int
    columns_per_side: int
    blocks_buffer: wgpu.GPUBuffer
    materials_buffer: wgpu.GPUBuffer
    coords_buffer: wgpu.GPUBuffer
    column_totals_buffer: wgpu.GPUBuffer
    chunk_totals_buffer: wgpu.GPUBuffer
    chunk_offsets_buffer: wgpu.GPUBuffer
    params_buffer: wgpu.GPUBuffer
    readback_buffer: wgpu.GPUBuffer
    readback_offset: int = 0
    readback_owner: PendingChunkMeshReadbackGroup | None = None
    owned_surface_buffers: list[wgpu.GPUBuffer] = field(default_factory=list)
    surface_release_callbacks: list[object] = field(default_factory=list)
    resources: AsyncVoxelMeshBatchResources | None = None
    metadata_promise: object | None = None
    submitted_at: float = 0.0
