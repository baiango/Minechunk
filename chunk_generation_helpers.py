from __future__ import annotations

from collections import deque
import math
import sys
import time

import numpy as np
import wgpu

import mesh_cache_helpers as mesh_cache
from meshing_types import ChunkMesh
from terrain_backend import ChunkVoxelResult
from terrain_kernels import build_chunk_vertex_array_from_voxels
import wgpu_chunk_mesher as wgpu_mesher


def _renderer_module():
    import renderer as renderer_module

    return renderer_module


def chunk_prep_priority(renderer, chunk_x: int, chunk_z: int, camera_chunk_x: int, camera_chunk_z: int) -> tuple[float, int, int]:
    dx = chunk_x - camera_chunk_x
    dz = chunk_z - camera_chunk_z
    distance_sq = float(dx * dx + dz * dz)
    return (distance_sq, abs(dz), abs(dx))


def make_chunk_mesh_fast(
    renderer,
    *,
    chunk_x: int,
    chunk_z: int,
    vertex_count: int,
    vertex_buffer: wgpu.GPUBuffer,
    vertex_offset: int,
    max_height: int,
    created_at: float,
    allocation_id: int | None,
) -> ChunkMesh:
    renderer_module = _renderer_module()
    chunk_size = int(renderer_module.CHUNK_SIZE)
    vertex_stride = int(renderer_module.VERTEX_STRIDE)
    half_chunk = chunk_size * 0.5
    half_height = float(max_height) * 0.5
    radius = float(math.sqrt(half_chunk * half_chunk * 2.0 + half_height * half_height))
    binding_offset = int(vertex_offset % vertex_stride)
    first_vertex = int((vertex_offset - binding_offset) // vertex_stride)

    mesh = ChunkMesh.__new__(ChunkMesh)
    mesh.chunk_x = int(chunk_x)
    mesh.chunk_z = int(chunk_z)
    mesh.vertex_count = int(vertex_count)
    mesh.vertex_buffer = vertex_buffer
    mesh.max_height = int(max_height)
    mesh.vertex_offset = int(vertex_offset)
    mesh.created_at = float(created_at)
    mesh.allocation_id = allocation_id
    mesh.bounds = (
        float(int(chunk_x) * chunk_size + half_chunk),
        half_height,
        float(int(chunk_z) * chunk_size + half_chunk),
        radius,
    )
    mesh.binding_offset = binding_offset
    mesh.first_vertex = first_vertex
    return mesh


def build_chunk_vertex_array(renderer, voxel_grid, material_grid, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, int, int]:
    renderer_module = _renderer_module()
    vertex_array, vertex_count = build_chunk_vertex_array_from_voxels(
        voxel_grid,
        material_grid,
        chunk_x,
        chunk_z,
        renderer_module.CHUNK_SIZE,
        renderer_module.WORLD_HEIGHT,
    )
    used_vertex_count = int(vertex_count)
    used_vertex_array = np.ascontiguousarray(vertex_array[:used_vertex_count])
    chunk_max_height = (
        int(voxel_grid.shape[0])
        if getattr(voxel_grid, "ndim", 0) == 3
        else int(np.max(voxel_grid))
    )
    return used_vertex_array, used_vertex_count, chunk_max_height


def cpu_make_chunk_mesh_batch_from_voxels(renderer, chunk_results: list[ChunkVoxelResult]) -> list[ChunkMesh]:
    if not chunk_results:
        return []

    built_chunks: list[tuple[int, int, np.ndarray, int, int, int]] = []
    total_vertex_bytes = 0
    created_at = time.perf_counter()

    for result in chunk_results:
        chunk_x = int(result.chunk_x)
        chunk_z = int(result.chunk_z)
        vertex_array, vertex_count, chunk_max_height = build_chunk_vertex_array(
            renderer,
            result.blocks,
            result.materials,
            chunk_x,
            chunk_z,
        )
        vertex_bytes = int(vertex_count) * int(_renderer_module().VERTEX_STRIDE)
        built_chunks.append(
            (
                chunk_x,
                chunk_z,
                vertex_array,
                int(vertex_count),
                int(vertex_bytes),
                int(chunk_max_height),
            )
        )
        total_vertex_bytes += vertex_bytes

    batch_allocation = mesh_cache.allocate_mesh_output_range(renderer, total_vertex_bytes)
    batch_buffer = batch_allocation.buffer
    batch_base_offset = batch_allocation.offset_bytes

    meshes: list[ChunkMesh] = []
    cursor_bytes = 0
    for chunk_x, chunk_z, vertex_array, vertex_count, vertex_bytes, chunk_max_height in built_chunks:
        vertex_offset = batch_base_offset + cursor_bytes
        if vertex_bytes > 0:
            vertex_view = memoryview(vertex_array.view(np.uint8).reshape(-1))
            renderer.device.queue.write_buffer(batch_buffer, vertex_offset, vertex_view)
        meshes.append(
            ChunkMesh(
                chunk_x=chunk_x,
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

    return meshes


def cpu_make_chunk_mesh_from_voxels(renderer, chunk_x: int, chunk_z: int, voxel_grid, material_grid) -> ChunkMesh:
    meshes = cpu_make_chunk_mesh_batch_from_voxels(
        renderer,
        [
            ChunkVoxelResult(
                chunk_x=int(chunk_x),
                chunk_z=int(chunk_z),
                blocks=np.ascontiguousarray(voxel_grid, dtype=np.uint32),
                materials=np.ascontiguousarray(material_grid, dtype=np.uint32),
                source="cpu",
            )
        ],
    )
    return meshes[0]


def gpu_make_chunk_mesh_from_voxels(renderer, chunk_x: int, chunk_z: int, voxel_grid, material_grid) -> ChunkMesh:
    meshes = wgpu_mesher.make_chunk_mesh_batch_from_voxels(
        renderer,
        [
            ChunkVoxelResult(
                chunk_x=chunk_x,
                chunk_z=chunk_z,
                blocks=np.ascontiguousarray(voxel_grid, dtype=np.uint32),
                materials=np.ascontiguousarray(material_grid, dtype=np.uint32),
                source="gpu",
            )
        ]
    )
    if not meshes:
        empty_buffer = renderer.device.create_buffer(
            size=1,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        return ChunkMesh(
            chunk_x=chunk_x,
            chunk_z=chunk_z,
            vertex_count=0,
            vertex_buffer=empty_buffer,
            max_height=int(voxel_grid.shape[0]),
            vertex_offset=0,
            created_at=time.perf_counter(),
        )
    return meshes[0]


def make_chunk_mesh_from_voxels(renderer, chunk_x: int, chunk_z: int, voxel_grid, material_grid) -> ChunkMesh:
    if renderer.use_gpu_meshing:
        try:
            return gpu_make_chunk_mesh_from_voxels(renderer, chunk_x, chunk_z, voxel_grid, material_grid)
        except Exception as exc:
            renderer.use_gpu_meshing = False
            renderer.mesh_backend_label = "CPU"
            print(f"Warning: GPU meshing failed ({exc!s}); using CPU meshing.", file=sys.stderr)
    return cpu_make_chunk_mesh_from_voxels(renderer, chunk_x, chunk_z, voxel_grid, material_grid)


def make_chunk_mesh(renderer, chunk_x: int, chunk_z: int) -> ChunkMesh:
    voxel_grid, material_grid = renderer.world.chunk_voxel_grid(chunk_x, chunk_z)
    return make_chunk_mesh_from_voxels(renderer, chunk_x, chunk_z, voxel_grid, material_grid)


def accept_chunk_voxel_result(renderer, result) -> None:
    key = (int(result.chunk_x), int(result.chunk_z))
    renderer._pending_chunk_coords.discard(key)
    mesh = make_chunk_mesh_from_voxels(renderer, key[0], key[1], result.blocks, result.materials)
    mesh_cache.store_chunk_mesh(renderer, mesh)


def ensure_chunk_mesh(renderer, chunk_x: int, chunk_z: int) -> ChunkMesh:
    key = (chunk_x, chunk_z)
    mesh = renderer.chunk_cache.get(key)
    if mesh is not None:
        renderer.chunk_cache.move_to_end(key)
        return mesh

    mesh = make_chunk_mesh(renderer, chunk_x, chunk_z)
    mesh_cache.store_chunk_mesh(renderer, mesh)
    return mesh


def rebuild_visible_missing_tracking(renderer) -> None:
    displayed: set[tuple[int, int]] = set()
    missing: set[tuple[int, int]] = set()
    pending = renderer._pending_chunk_coords
    for coord in renderer._visible_chunk_coords:
        if coord in renderer.chunk_cache:
            displayed.add(coord)
        elif coord not in pending:
            missing.add(coord)
    renderer._visible_displayed_coords = displayed
    renderer._visible_missing_coords = missing
    renderer._chunk_request_queue.clear()
    renderer._chunk_request_queue_origin = None
    renderer._chunk_request_queue_dirty = True


def chunk_request_queue_needs_rebuild(renderer, current_origin: tuple[int, int]) -> bool:
    if renderer._chunk_request_queue_dirty:
        return True
    if renderer._chunk_request_queue_origin != current_origin:
        return True
    return bool(renderer._visible_missing_coords) and not renderer._chunk_request_queue


def rebuild_chunk_request_queue(renderer, camera_chunk_x: int, camera_chunk_z: int) -> None:
    missing_coords = renderer._visible_missing_coords
    if not missing_coords:
        renderer._chunk_request_queue.clear()
        renderer._chunk_request_queue_origin = (camera_chunk_x, camera_chunk_z)
        renderer._chunk_request_queue_dirty = False
        return

    max_ring = max(0, int(renderer.chunk_radius))
    rings: list[list[tuple[int, int]]] = [[] for _ in range(max_ring + 1)]

    for coord in renderer._visible_chunk_coords:
        if coord not in missing_coords:
            continue
        dx = coord[0] - camera_chunk_x
        dz = coord[1] - camera_chunk_z
        ring = max(abs(dx), abs(dz))
        if ring > max_ring:
            ring = max_ring
        rings[ring].append(coord)

    ordered: list[tuple[int, int]] = []
    for ring in range(max_ring + 1):
        ordered.extend(rings[ring])

    renderer._chunk_request_queue = deque(ordered)
    renderer._chunk_request_queue_origin = (camera_chunk_x, camera_chunk_z)
    renderer._chunk_request_queue_dirty = False


def refresh_visible_chunk_set(renderer) -> float:
    visibility_start = time.perf_counter()
    renderer_module = _renderer_module()
    current_origin = (int(renderer.camera.position[0] // renderer_module.CHUNK_SIZE), int(renderer.camera.position[2] // renderer_module.CHUNK_SIZE))
    if renderer._visible_chunk_origin != current_origin or not renderer._visible_chunk_coords:
        renderer._visible_chunk_origin = current_origin
        renderer._visible_chunk_coords = renderer._chunk_coords_in_view()
        renderer._visible_chunk_coord_set = set(renderer._visible_chunk_coords)
        rebuild_visible_missing_tracking(renderer)
    renderer._warn_if_visible_exceeds_cache()
    return (time.perf_counter() - visibility_start) * 1000.0


def service_background_gpu_work(renderer) -> None:
    wgpu_mesher.process_gpu_buffer_cleanup(renderer)
    mesh_cache.process_deferred_mesh_output_frees(renderer)
    if renderer.use_gpu_meshing:
        wgpu_mesher.finalize_pending_gpu_mesh_batches(renderer)


def prepare_chunks(renderer, dt: float) -> tuple[float, float]:
    renderer_module = _renderer_module()
    visibility_lookup_ms = 0.0
    current_origin = (int(renderer.camera.position[0] // renderer_module.CHUNK_SIZE), int(renderer.camera.position[2] // renderer_module.CHUNK_SIZE))
    if renderer._visible_chunk_origin != current_origin or not renderer._visible_chunk_coords:
        visibility_lookup_ms = refresh_visible_chunk_set(renderer)

    prep_start = time.perf_counter()
    terrain_backend_label = renderer.world.terrain_backend_label()
    using_wgpu_terrain = terrain_backend_label == "Wgpu"
    using_metal_terrain = terrain_backend_label == "Metal"
    if renderer.use_gpu_meshing:
        if using_wgpu_terrain:
            ready_results = renderer.world.poll_ready_chunk_voxel_batches()
            chunk_stream_drained = 0
            chunk_stream_bytes = 0.0
            drain_budget = max(1, renderer.mesh_batch_size)
            ready_results.sort(
                key=lambda result: chunk_prep_priority(
                    renderer,
                    int(result.chunk_x),
                    int(result.chunk_z),
                    current_origin[0],
                    current_origin[1],
                )
            )
            for result in ready_results:
                chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
            for result in reversed(ready_results):
                renderer._pending_voxel_mesh_results.appendleft(result)
            while renderer._pending_voxel_mesh_results and drain_budget > 0:
                batch_size = min(renderer.mesh_batch_size, len(renderer._pending_voxel_mesh_results))
                batch_size = min(batch_size, drain_budget)
                batch = [renderer._pending_voxel_mesh_results.popleft() for _ in range(batch_size)]
                chunk_stream_drained += len(batch)
                drain_budget -= len(batch)
                for result in batch:
                    chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
                meshes = (
                    wgpu_mesher.make_chunk_mesh_batch_from_voxels(renderer, batch)
                    if using_wgpu_terrain or using_metal_terrain
                    else cpu_make_chunk_mesh_batch_from_voxels(renderer, batch)
                )
                for mesh in meshes:
                    mesh_cache.store_chunk_mesh(renderer, mesh)
        elif using_metal_terrain:
            ready_results = renderer.world.poll_ready_chunk_voxel_batches()
            chunk_stream_bytes = 0.0
            chunk_stream_drained = 0
            drain_budget = max(1, renderer.mesh_batch_size)
            ready_results.sort(
                key=lambda result: chunk_prep_priority(
                    renderer,
                    int(result.chunk_x),
                    int(result.chunk_z),
                    current_origin[0],
                    current_origin[1],
                )
            )
            for result in ready_results:
                chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
            for result in reversed(ready_results):
                renderer._pending_voxel_mesh_results.appendleft(result)
            while renderer._pending_voxel_mesh_results and drain_budget > 0:
                batch_size = min(renderer.mesh_batch_size, len(renderer._pending_voxel_mesh_results))
                batch_size = min(batch_size, drain_budget)
                batch = [renderer._pending_voxel_mesh_results.popleft() for _ in range(batch_size)]
                chunk_stream_drained += len(batch)
                drain_budget -= len(batch)
                for result in batch:
                    chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
                meshes = wgpu_mesher.make_chunk_mesh_batch_from_voxels(renderer, batch)
                for mesh in meshes:
                    mesh_cache.store_chunk_mesh(renderer, mesh)
        else:
            chunk_stream_drained = 0
            chunk_stream_bytes = 0.0
    else:
        ready_results = renderer.world.poll_ready_chunk_voxel_batches()
        chunk_stream_bytes = 0.0
        chunk_stream_drained = 0
        drain_budget = max(1, renderer.mesh_batch_size)
        ready_results.sort(
            key=lambda result: chunk_prep_priority(
                renderer,
                int(result.chunk_x),
                int(result.chunk_z),
                current_origin[0],
                current_origin[1],
            )
        )
        for result in reversed(ready_results):
            renderer._pending_voxel_mesh_results.appendleft(result)
        while renderer._pending_voxel_mesh_results and drain_budget > 0:
            batch_size = min(renderer.mesh_batch_size, len(renderer._pending_voxel_mesh_results))
            batch_size = min(batch_size, drain_budget)
            batch = [renderer._pending_voxel_mesh_results.popleft() for _ in range(batch_size)]
            chunk_stream_drained += len(batch)
            drain_budget -= len(batch)
            for result in batch:
                chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
                accept_chunk_voxel_result(renderer, result)

    missing_coords = renderer._visible_missing_coords
    missing_count = len(missing_coords)
    prep_budget = min(
        missing_count,
        max(1, min(renderer_module.chunk_prep_request_budget_cap, renderer.mesh_batch_size * 2, 32)),
    )

    camera_chunk_x = current_origin[0]
    camera_chunk_z = current_origin[1]
    pending_chunk_coords = renderer._pending_chunk_coords
    displayed_chunk_coords = renderer._visible_displayed_coords

    if prep_budget > 0 and missing_count > 0 and chunk_request_queue_needs_rebuild(renderer, current_origin):
        rebuild_chunk_request_queue(renderer, camera_chunk_x, camera_chunk_z)

    request_coords: list[tuple[int, int]] = []
    request_queue = renderer._chunk_request_queue
    while request_queue and len(request_coords) < prep_budget:
        coord = request_queue.popleft()
        if coord in pending_chunk_coords or coord not in missing_coords:
            continue
        request_coords.append(coord)

    if not request_coords and prep_budget > 0 and missing_count > 0 and renderer._visible_missing_coords:
        rebuild_chunk_request_queue(renderer, camera_chunk_x, camera_chunk_z)
        request_queue = renderer._chunk_request_queue
        while request_queue and len(request_coords) < prep_budget:
            coord = request_queue.popleft()
            if coord in pending_chunk_coords or coord not in missing_coords:
                continue
            request_coords.append(coord)

    if request_coords:
        batch_size = max(1, min(renderer.mesh_batch_size * 2, 32))
        request_batches = [
            request_coords[index : index + batch_size]
            for index in range(0, len(request_coords), batch_size)
        ]
        for batch in reversed(request_batches):
            renderer.world.request_chunk_voxel_batch(batch)
        for coord in request_coords:
            pending_chunk_coords.add(coord)
            missing_coords.discard(coord)

    renderer._last_new_displayed_chunks = len(displayed_chunk_coords - renderer._last_displayed_chunk_coords)
    renderer._last_displayed_chunk_coords = set(displayed_chunk_coords)
    chunk_stream_ms = (time.perf_counter() - prep_start) * 1000.0
    renderer._last_chunk_stream_drained = chunk_stream_drained
    import hud_profile_helpers as hud_profile
    hud_profile.record_frame_breakdown_sample(renderer, "chunk_stream_bytes", chunk_stream_bytes)
    return visibility_lookup_ms, chunk_stream_ms
