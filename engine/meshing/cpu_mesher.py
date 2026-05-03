from __future__ import annotations

import math
import time

import numpy as np
import wgpu

from .. import render_contract as render_consts
from ..cache import mesh_allocator as mesh_cache
from ..meshing_types import ChunkMesh
from ..terrain.types import ChunkVoxelResult
from ..terrain.kernels import (
    build_chunk_surface_run_table_from_heightmap_clipped,
    build_chunk_surface_vertex_array_from_heightmap_clipped,
    build_chunk_vertex_array_from_voxels_with_boundaries,
    emit_chunk_surface_run_table_vertices,
)

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func

__all__ = [
    "build_chunk_vertex_array",
    "cpu_make_chunk_mesh_batch_from_terrain_results",
    "cpu_make_chunk_mesh_from_terrain_result",
    "make_chunk_mesh_fast",
]

def _renderer_module():
    """Return stable render constants without importing the renderer runtime."""
    return render_consts

@profile
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

@profile
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

_EMPTY_VERTEX_ARRAY = np.empty((0, int(getattr(_renderer_module(), "VERTEX_COMPONENTS", 12))), dtype=np.float32)


def _surface_mesher_payload(result: ChunkVoxelResult) -> tuple[np.ndarray, np.ndarray] | None:
    surface_heights = getattr(result, "surface_heights", None)
    surface_materials = getattr(result, "surface_materials", None)
    if bool(getattr(result, "use_surface_mesher", False)) and surface_heights is not None and surface_materials is not None:
        return surface_heights, surface_materials
    return None

@profile
def _empty_vertical_neighbor_planes(voxel_grid) -> tuple[np.ndarray, np.ndarray]:
    sample_size = int(voxel_grid.shape[1]) if getattr(voxel_grid, "ndim", 0) >= 2 else int(_renderer_module().CHUNK_SIZE) + 2
    plane_dtype = getattr(voxel_grid, "dtype", np.uint32)
    empty_plane = np.zeros((sample_size, sample_size), dtype=plane_dtype)
    return empty_plane, empty_plane.copy()


@profile
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
        top_plane, bottom_plane = _empty_vertical_neighbor_planes(voxel_grid)
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


@profile
def build_chunk_vertex_array_from_terrain_result(
    renderer,
    result: ChunkVoxelResult,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
) -> tuple[np.ndarray, int, int]:
    renderer_module = _renderer_module()
    chunk_max_height = int(chunk_y * renderer_module.CHUNK_SIZE + result.blocks.shape[0])
    surface_heights = getattr(result, "surface_heights", None)
    surface_materials = getattr(result, "surface_materials", None)
    if bool(getattr(result, "use_surface_mesher", False)) and surface_heights is not None and surface_materials is not None:
        vertex_array, vertex_count = build_chunk_surface_vertex_array_from_heightmap_clipped(
            surface_heights,
            surface_materials,
            chunk_x,
            chunk_z,
            renderer_module.CHUNK_SIZE,
            int(result.blocks.shape[0]),
            float(renderer_module.BLOCK_SIZE),
            int(chunk_y),
        )
        used_vertex_count = int(vertex_count)
        return vertex_array[:used_vertex_count], used_vertex_count, chunk_max_height
    return build_chunk_vertex_array(
        renderer,
        result.blocks,
        result.materials,
        chunk_x,
        chunk_y,
        chunk_z,
        getattr(result, "top_boundary", None),
        getattr(result, "bottom_boundary", None),
    )


@profile
def _cpu_make_surface_chunk_mesh_batch(renderer, terrain_results: list[ChunkVoxelResult]) -> list[ChunkMesh] | None:
    renderer_module = _renderer_module()
    vertex_stride = int(renderer_module.VERTEX_STRIDE)
    vertex_components = int(renderer_module.VERTEX_COMPONENTS)
    chunk_size = int(renderer_module.CHUNK_SIZE)
    block_size = float(renderer_module.BLOCK_SIZE)
    created_at = time.perf_counter()
    chunk_entries: list[tuple[int, int, int, np.ndarray | None, int, int, int, int]] = []
    total_vertex_count = 0

    for result in terrain_results:
        chunk_x = int(result.chunk_x)
        chunk_y = int(getattr(result, "chunk_y", 0))
        chunk_z = int(result.chunk_z)
        height_limit = int(result.blocks.shape[0])
        chunk_max_height = int(chunk_y * chunk_size + height_limit)
        if bool(getattr(result, "is_empty", False)) or bool(getattr(result, "is_fully_occluded", False)):
            chunk_entries.append((chunk_x, chunk_y, chunk_z, None, 0, height_limit, chunk_max_height, 0))
            continue

        surface_payload = _surface_mesher_payload(result)
        if surface_payload is None:
            return None

        surface_heights, surface_materials = surface_payload
        run_table, run_count, vertex_count = build_chunk_surface_run_table_from_heightmap_clipped(
            surface_heights,
            surface_materials,
            chunk_size,
            height_limit,
            chunk_y,
        )
        chunk_entries.append((chunk_x, chunk_y, chunk_z, run_table, int(run_count), height_limit, chunk_max_height, int(vertex_count)))
        total_vertex_count += vertex_count

    empty_buffer = _shared_empty_chunk_vertex_buffer(renderer)
    if total_vertex_count <= 0:
        return [
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
            for chunk_x, chunk_y, chunk_z, _run_table, _run_count, _height_limit, chunk_max_height, _vertex_count in chunk_entries
        ]

    total_vertex_bytes = total_vertex_count * vertex_stride
    batch_allocation = mesh_cache.allocate_mesh_output_range(renderer, total_vertex_bytes)
    batch_buffer = batch_allocation.buffer
    batch_base_offset = int(batch_allocation.offset_bytes)
    upload_vertices = np.empty((total_vertex_count, vertex_components), dtype=np.float32)

    meshes: list[ChunkMesh] = []
    cursor_vertices = 0
    for chunk_x, chunk_y, chunk_z, run_table, run_count, height_limit, chunk_max_height, vertex_count in chunk_entries:
        if vertex_count > 0:
            if run_table is None:
                raise RuntimeError("Surface chunk has vertices but no surface run table.")
            emitted_count = int(
                emit_chunk_surface_run_table_vertices(
                    upload_vertices,
                    cursor_vertices,
                    run_table,
                    run_count,
                    chunk_x,
                    chunk_z,
                    chunk_size,
                    block_size,
                )
            )
            if emitted_count != vertex_count:
                raise RuntimeError(f"Surface mesher emitted {emitted_count} vertices, expected {vertex_count}.")
            vertex_buffer = batch_buffer
            vertex_offset = batch_base_offset + cursor_vertices * vertex_stride
            allocation_id = batch_allocation.allocation_id
            cursor_vertices += vertex_count
        else:
            vertex_buffer = empty_buffer
            vertex_offset = 0
            allocation_id = None
        meshes.append(
            make_chunk_mesh_fast(
                renderer,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                chunk_z=chunk_z,
                vertex_count=vertex_count,
                vertex_buffer=vertex_buffer,
                vertex_offset=vertex_offset,
                max_height=chunk_max_height,
                created_at=created_at,
                allocation_id=allocation_id,
            )
        )

    renderer.device.queue.write_buffer(batch_buffer, batch_base_offset, memoryview(upload_vertices.view(np.uint8).reshape(-1)))
    return meshes


@profile
def _cpu_make_chunk_mesh_batch(renderer, terrain_results: list[ChunkVoxelResult]) -> list[ChunkMesh]:
    if not terrain_results:
        return []

    surface_meshes = _cpu_make_surface_chunk_mesh_batch(renderer, terrain_results)
    if surface_meshes is not None:
        return surface_meshes

    if len(terrain_results) == 1:
        result = terrain_results[0]
        chunk_x = int(result.chunk_x)
        chunk_y = int(getattr(result, "chunk_y", 0))
        chunk_z = int(result.chunk_z)
        chunk_max_height = int(chunk_y * _renderer_module().CHUNK_SIZE + result.blocks.shape[0])
        created_at = time.perf_counter()
        if bool(getattr(result, "is_empty", False)) or bool(getattr(result, "is_fully_occluded", False)):
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
        vertex_array, vertex_count, chunk_max_height = build_chunk_vertex_array_from_terrain_result(
            renderer,
            result,
            chunk_x,
            chunk_y,
            chunk_z,
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

    for result in terrain_results:
        chunk_x = int(result.chunk_x)
        chunk_y = int(getattr(result, "chunk_y", 0))
        chunk_z = int(result.chunk_z)
        if bool(getattr(result, "is_empty", False)) or bool(getattr(result, "is_fully_occluded", False)):
            vertex_array = _EMPTY_VERTEX_ARRAY
            vertex_count = 0
            chunk_max_height = int(chunk_y * _renderer_module().CHUNK_SIZE + result.blocks.shape[0])
        else:
            vertex_array, vertex_count, chunk_max_height = build_chunk_vertex_array_from_terrain_result(
                renderer,
                result,
                chunk_x,
                chunk_y,
                chunk_z,
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
    empty_buffer = _shared_empty_chunk_vertex_buffer(renderer)
    meshes: list[ChunkMesh] = []
    cursor_bytes = 0
    for chunk_x, chunk_y, chunk_z, vertex_array, vertex_count, vertex_bytes, chunk_max_height in built_chunks:
        if vertex_bytes > 0:
            vertex_buffer = batch_buffer
            vertex_offset = batch_base_offset + cursor_bytes
            allocation_id = batch_allocation.allocation_id
        else:
            vertex_buffer = empty_buffer
            vertex_offset = 0
            allocation_id = None
        if vertex_bytes > 0 and upload_bytes is not None:
            upload_bytes[cursor_bytes : cursor_bytes + vertex_bytes] = vertex_array.view(np.uint8).reshape(-1)
        meshes.append(
            make_chunk_mesh_fast(
                renderer,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                chunk_z=chunk_z,
                vertex_count=vertex_count,
                vertex_buffer=vertex_buffer,
                vertex_offset=vertex_offset,
                max_height=chunk_max_height,
                created_at=created_at,
                allocation_id=allocation_id,
            )
        )
        cursor_bytes += vertex_bytes

    if upload_bytes is not None and total_vertex_bytes > 0:
        renderer.device.queue.write_buffer(batch_buffer, batch_base_offset, memoryview(upload_bytes))

    return meshes


@profile
def cpu_make_chunk_mesh_batch_from_terrain_results(renderer, terrain_results: list[ChunkVoxelResult]) -> list[ChunkMesh]:
    return _cpu_make_chunk_mesh_batch(renderer, terrain_results)


@profile
def cpu_make_chunk_mesh_from_terrain_result(renderer, terrain_result: ChunkVoxelResult) -> ChunkMesh:
    meshes = cpu_make_chunk_mesh_batch_from_terrain_results(renderer, [terrain_result])
    return meshes[0]
