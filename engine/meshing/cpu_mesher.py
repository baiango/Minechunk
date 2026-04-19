from __future__ import annotations

import math
import time

import numpy as np
import wgpu

from ..cache import mesh_allocator as mesh_cache
from ..meshing_types import ChunkMesh
from ..terrain.types import ChunkVoxelResult
from ..terrain.kernels import (
    build_chunk_vertex_array_from_voxels_with_boundaries,
    fill_stacked_chunk_vertical_neighbor_planes,
)

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func

__all__ = [
    "build_chunk_vertex_array",
    "cpu_make_chunk_mesh_batch_from_voxels",
    "cpu_make_chunk_mesh_from_voxels",
    "make_chunk_mesh_fast",
]

def _renderer_module():
    from .. import renderer as renderer_module

    return renderer_module

def _shared_empty_chunk_vertex_buffer(renderer) -> wgpu.GPUBuffer:
    buffer = getattr(renderer, "_shared_empty_chunk_vertex_buffer", None)
    if buffer is not None:
        return buffer
    buffer = renderer.device.create_buffer(
        size=max(1, int(_renderer_module().VERTEX_STRIDE)),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    setattr(renderer, "_shared_empty_chunk_vertex_buffer", buffer)
    return buffer

def make_chunk_mesh_fast(
    renderer,
    *,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    vertex_count: int,
    vertex_buffer: wgpu.GPUBuffer,
    vertex_offset: int,
    max_height: int,
    created_at: float,
    allocation_id: int | None,
) -> ChunkMesh:
    renderer_module = _renderer_module()
    chunk_world_size = float(renderer_module.CHUNK_WORLD_SIZE)
    vertex_stride = int(renderer_module.VERTEX_STRIDE)
    half_chunk = chunk_world_size * 0.5
    half_height = float(max_height) * float(renderer_module.BLOCK_SIZE) * 0.5
    radius = float(math.sqrt(half_chunk * half_chunk * 2.0 + half_height * half_height))
    binding_offset = int(vertex_offset % vertex_stride)
    first_vertex = int((vertex_offset - binding_offset) // vertex_stride)

    mesh = ChunkMesh.__new__(ChunkMesh)
    mesh.chunk_x = int(chunk_x)
    mesh.chunk_y = int(chunk_y)
    mesh.chunk_z = int(chunk_z)
    mesh.vertex_count = int(vertex_count)
    mesh.vertex_buffer = vertex_buffer
    mesh.max_height = int(max_height)
    mesh.vertex_offset = int(vertex_offset)
    mesh.created_at = float(created_at)
    mesh.allocation_id = allocation_id
    mesh.bounds = (
        float(int(chunk_x) * chunk_world_size + half_chunk),
        float((int(chunk_y) * renderer_module.CHUNK_SIZE) * renderer_module.BLOCK_SIZE + half_height),
        float(int(chunk_z) * chunk_world_size + half_chunk),
        radius,
    )
    mesh.binding_offset = binding_offset
    mesh.first_vertex = first_vertex
    return mesh

def _stacked_vertical_neighbor_planes(renderer, chunk_x: int, chunk_y: int, chunk_z: int, voxel_grid) -> tuple[np.ndarray, np.ndarray]:
    renderer_module = _renderer_module()
    sample_size = int(renderer_module.CHUNK_SIZE) + 2
    plane_dtype = getattr(voxel_grid, "dtype", np.uint32)
    top_plane = np.zeros((sample_size, sample_size), dtype=plane_dtype)
    bottom_plane = np.zeros((sample_size, sample_size), dtype=plane_dtype)
    if not getattr(renderer_module, "VERTICAL_CHUNK_STACK_ENABLED", False):
        return top_plane, bottom_plane
    fill_stacked_chunk_vertical_neighbor_planes(
        top_plane,
        bottom_plane,
        int(chunk_x),
        int(chunk_y),
        int(chunk_z),
        int(renderer_module.CHUNK_SIZE),
        int(renderer.world.seed),
        int(renderer.world.height),
    )
    return top_plane, bottom_plane

_EMPTY_VERTEX_ARRAY = np.empty((0, int(getattr(_renderer_module(), "VERTEX_COMPONENTS", 12))), dtype=np.float32)

def build_chunk_vertex_array(
    renderer,
    voxel_grid,
    material_grid,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    top_plane: np.ndarray | None = None,
    bottom_plane: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    renderer_module = _renderer_module()
    chunk_max_height = int(chunk_y * renderer_module.CHUNK_SIZE + voxel_grid.shape[0])
    if not np.any(voxel_grid):
        return _EMPTY_VERTEX_ARRAY, 0, chunk_max_height
    if top_plane is None or bottom_plane is None:
        top_plane, bottom_plane = _stacked_vertical_neighbor_planes(renderer, chunk_x, chunk_y, chunk_z, voxel_grid)
    if np.all(voxel_grid) and np.all(top_plane) and np.all(bottom_plane):
        return _EMPTY_VERTEX_ARRAY, 0, chunk_max_height
    vertex_array, vertex_count = build_chunk_vertex_array_from_voxels_with_boundaries(
        voxel_grid,
        material_grid,
        chunk_x,
        chunk_z,
        renderer_module.CHUNK_SIZE,
        int(voxel_grid.shape[0]),
        top_plane,
        bottom_plane,
        float(renderer_module.BLOCK_SIZE),
        int(chunk_y),
    )
    used_vertex_count = int(vertex_count)
    used_vertex_array = vertex_array[:used_vertex_count]
    if getattr(voxel_grid, "ndim", 0) != 3:
        chunk_max_height = int(chunk_y * renderer_module.CHUNK_SIZE + np.max(voxel_grid))
    return used_vertex_array, used_vertex_count, chunk_max_height

def cpu_make_chunk_mesh_batch_from_voxels(renderer, chunk_results: list[ChunkVoxelResult]) -> list[ChunkMesh]:
    if not chunk_results:
        return []

    if len(chunk_results) == 1:
        result = chunk_results[0]
        chunk_x = int(result.chunk_x)
        chunk_y = int(getattr(result, "chunk_y", 0))
        chunk_z = int(result.chunk_z)
        chunk_max_height = int(chunk_y * _renderer_module().CHUNK_SIZE + result.blocks.shape[0])
        created_at = time.perf_counter()
        if bool(getattr(result, "is_empty", False)):
            return [
                make_chunk_mesh_fast(
                    renderer,
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    chunk_z=chunk_z,
                    vertex_count=0,
                    vertex_buffer=_shared_empty_chunk_vertex_buffer(renderer),
                    vertex_offset=0,
                    max_height=int(chunk_max_height),
                    created_at=created_at,
                    allocation_id=None,
                )
            ]
        vertex_array, vertex_count, chunk_max_height = build_chunk_vertex_array(
            renderer,
            result.blocks,
            result.materials,
            chunk_x,
            chunk_y,
            chunk_z,
            getattr(result, "top_boundary", None),
            getattr(result, "bottom_boundary", None),
        )
        if int(vertex_count) <= 0:
            return [
                make_chunk_mesh_fast(
                    renderer,
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    chunk_z=chunk_z,
                    vertex_count=0,
                    vertex_buffer=_shared_empty_chunk_vertex_buffer(renderer),
                    vertex_offset=0,
                    max_height=int(chunk_max_height),
                    created_at=created_at,
                    allocation_id=None,
                )
            ]
        vertex_bytes = int(vertex_count) * int(_renderer_module().VERTEX_STRIDE)
        batch_allocation = mesh_cache.allocate_mesh_output_range(renderer, vertex_bytes)
        batch_buffer = batch_allocation.buffer
        batch_offset = batch_allocation.offset_bytes
        renderer.device.queue.write_buffer(batch_buffer, batch_offset, memoryview(vertex_array.view(np.uint8).reshape(-1)))
        return [
            make_chunk_mesh_fast(
                renderer,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                chunk_z=chunk_z,
                vertex_count=int(vertex_count),
                vertex_buffer=batch_buffer,
                vertex_offset=batch_offset,
                max_height=int(chunk_max_height),
                created_at=created_at,
                allocation_id=batch_allocation.allocation_id,
            )
        ]

    built_chunks: list[tuple[int, int, int, np.ndarray, int, int, int]] = []
    total_vertex_bytes = 0
    created_at = time.perf_counter()

    for result in chunk_results:
        chunk_x = int(result.chunk_x)
        chunk_y = int(getattr(result, "chunk_y", 0))
        chunk_z = int(result.chunk_z)
        top_plane = getattr(result, "top_boundary", None)
        bottom_plane = getattr(result, "bottom_boundary", None)
        if bool(getattr(result, "is_empty", False)):
            vertex_array = _EMPTY_VERTEX_ARRAY
            vertex_count = 0
            chunk_max_height = int(chunk_y * _renderer_module().CHUNK_SIZE + result.blocks.shape[0])
        else:
            vertex_array, vertex_count, chunk_max_height = build_chunk_vertex_array(
                renderer,
                result.blocks,
                result.materials,
                chunk_x,
                chunk_y,
                chunk_z,
                top_plane,
                bottom_plane,
            )
        vertex_bytes = int(vertex_count) * int(_renderer_module().VERTEX_STRIDE)
        built_chunks.append(
            (
                chunk_x,
                chunk_y,
                chunk_z,
                vertex_array,
                int(vertex_count),
                int(vertex_bytes),
                int(chunk_max_height),
            )
        )
        total_vertex_bytes += vertex_bytes

    if total_vertex_bytes <= 0:
        empty_buffer = _shared_empty_chunk_vertex_buffer(renderer)
        meshes: list[ChunkMesh] = []
        for chunk_x, chunk_y, chunk_z, vertex_array, vertex_count, vertex_bytes, chunk_max_height in built_chunks:
            meshes.append(
                make_chunk_mesh_fast(
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

    batch_allocation = mesh_cache.allocate_mesh_output_range(renderer, total_vertex_bytes)
    batch_buffer = batch_allocation.buffer
    batch_base_offset = batch_allocation.offset_bytes

    upload_bytes = np.empty(total_vertex_bytes, dtype=np.uint8) if total_vertex_bytes > 0 else None
    meshes: list[ChunkMesh] = []
    cursor_bytes = 0
    for chunk_x, chunk_y, chunk_z, vertex_array, vertex_count, vertex_bytes, chunk_max_height in built_chunks:
        vertex_offset = batch_base_offset + cursor_bytes
        if vertex_bytes > 0 and upload_bytes is not None:
            upload_bytes[cursor_bytes : cursor_bytes + vertex_bytes] = vertex_array.view(np.uint8).reshape(-1)
        meshes.append(
            make_chunk_mesh_fast(
                renderer,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                chunk_z=chunk_z,
                vertex_count=vertex_count,
                vertex_buffer=batch_buffer,
                vertex_offset=vertex_offset,
                max_height=chunk_max_height,
                created_at=created_at,
                allocation_id=batch_allocation.allocation_id,
            )
        )
        cursor_bytes += vertex_bytes

    if upload_bytes is not None and total_vertex_bytes > 0:
        renderer.device.queue.write_buffer(batch_buffer, batch_base_offset, memoryview(upload_bytes))

    return meshes

def cpu_make_chunk_mesh_from_voxels(renderer, chunk_x: int, chunk_y: int, chunk_z: int, voxel_grid, material_grid) -> ChunkMesh:
    meshes = cpu_make_chunk_mesh_batch_from_voxels(
        renderer,
        [
            ChunkVoxelResult(
                chunk_x=int(chunk_x),
                chunk_y=int(chunk_y),
                chunk_z=int(chunk_z),
                blocks=np.ascontiguousarray(voxel_grid, dtype=np.uint32),
                materials=np.ascontiguousarray(material_grid, dtype=np.uint32),
                source="cpu",
            )
        ],
    )
    return meshes[0]
