from __future__ import annotations

from collections import deque
import time

import numpy as np

from ..cache import mesh_allocator as mesh_cache
from ..meshing import gpu_mesher as wgpu_mesher
from ..meshing import metal_mesher
from ..meshing_types import ChunkMesh
from ..terrain.types import ChunkVoxelResult
from ..visibility.coord_manager import refresh_visible_chunk_set, rebuild_visible_missing_tracking
from . import terrain_stage, meshing_stage, visibility_stage, render_stage, collision_stage, cache_stage, profiling

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func

def _renderer_module():
    from .. import renderer as renderer_module

    return renderer_module

def _terrain_mesher(renderer):
    terrain_label = renderer.world.terrain_backend_label()
    if terrain_label == "Metal":
        return metal_mesher
    return wgpu_mesher

@profile
def chunk_prep_priority(renderer, chunk_x: int, chunk_y: int, chunk_z: int, camera_chunk_x: int, camera_chunk_y: int, camera_chunk_z: int) -> tuple[float, int, int, int, int]:
    dx = int(chunk_x) - int(camera_chunk_x)
    dy = int(chunk_y) - int(camera_chunk_y)
    dz = int(chunk_z) - int(camera_chunk_z)
    distance_sq = float(dx * dx + dz * dz + (dy * dy * 4))
    # Prefer chunks below the camera before equally-distant sky chunks. That gets
    # surface/ground layers meshed earlier for frozen batch views instead of
    # burning the whole request budget on empty air above the spawn chunk.
    above_bias = 1 if dy > 0 else 0
    return (distance_sq, abs(dy), above_bias, abs(dz), abs(dx))

@profile
def accept_chunk_voxel_result(renderer, result) -> None:
    key = (int(result.chunk_x), int(getattr(result, "chunk_y", 0)), int(result.chunk_z))
    renderer._pending_chunk_coords.discard(key)
    mesh = meshing_stage.make_chunk_mesh_from_terrain_result(renderer, result)
    if mesh is not None:
        mesh_cache.store_chunk_mesh(renderer, mesh)

@profile
def ensure_chunk_mesh(renderer, chunk_x: int, chunk_y: int, chunk_z: int) -> ChunkMesh:
    key = (chunk_x, chunk_y, chunk_z)
    mesh = renderer.chunk_cache.get(key)
    if mesh is not None:
        renderer.chunk_cache.move_to_end(key)
        return mesh

    voxel_grid, material_grid = renderer.world.chunk_voxel_grid(chunk_x, chunk_y, chunk_z)
    terrain_result = ChunkVoxelResult(
        chunk_x=int(chunk_x),
        chunk_y=int(chunk_y),
        chunk_z=int(chunk_z),
        blocks=np.ascontiguousarray(voxel_grid, dtype=np.uint32),
        materials=np.ascontiguousarray(material_grid, dtype=np.uint32),
        source="sync",
    )
    mesh = meshing_stage.make_chunk_mesh_from_terrain_result(renderer, terrain_result)
    mesh_cache.store_chunk_mesh(renderer, mesh)
    return mesh

def _chunk_request_view_signature(renderer, current_origin: tuple[int, int, int]) -> tuple[int, int, int]:
    horizontal_stride = max(1, int(getattr(renderer, "_chunk_request_view_stride", 1)))
    return (
        int(current_origin[0]) // horizontal_stride,
        int(current_origin[1]),
        int(current_origin[2]) // horizontal_stride,
    )

def chunk_request_queue_needs_rebuild(renderer, current_origin: tuple[int, int, int]) -> bool:
    if renderer._chunk_request_queue_dirty:
        return True
    if renderer._chunk_request_queue_view_signature != _chunk_request_view_signature(renderer, current_origin):
        return True
    return bool(renderer._visible_missing_coords) and not renderer._chunk_request_queue

@profile
def rebuild_chunk_request_queue(renderer, camera_chunk_x: int, camera_chunk_y: int, camera_chunk_z: int) -> None:
    missing_coords = renderer._visible_missing_coords
    current_origin = (int(camera_chunk_x), int(camera_chunk_y), int(camera_chunk_z))
    if not missing_coords:
        renderer._chunk_request_queue.clear()
        renderer._chunk_request_queue_origin = current_origin
        renderer._chunk_request_queue_view_signature = _chunk_request_view_signature(renderer, current_origin)
        renderer._chunk_request_queue_dirty = False
        return

    renderer_module = _renderer_module()
    backlog_target = max(1, int(_chunk_prep_backlog_target(renderer)))
    queue_target = min(
        len(missing_coords),
        max(32, backlog_target * 4, int(renderer_module.chunk_prep_request_budget_cap) * 8),
    )

    ordered = sorted(
        missing_coords,
        key=lambda coord: chunk_prep_priority(
            renderer,
            int(coord[0]),
            int(coord[1]),
            int(coord[2]),
            current_origin[0],
            current_origin[1],
            current_origin[2],
        ),
    )[:queue_target]

    renderer._chunk_request_target_coords = set(ordered)
    renderer._chunk_request_queue = deque(ordered)
    renderer._chunk_request_queue_origin = current_origin
    renderer._chunk_request_queue_view_signature = _chunk_request_view_signature(renderer, current_origin)
    renderer._chunk_request_queue_dirty = False

def _chunk_prep_backlog_target(renderer) -> int:
    base_target = max(1, int(max(renderer.mesh_batch_size, renderer.terrain_batch_size)))
    backend = getattr(getattr(renderer, "world", None), "_backend", None)
    if backend is None:
        return base_target

    chunks_per_poll = max(1, int(getattr(backend, "chunks_per_poll", base_target)))
    submit_target = max(chunks_per_poll, int(getattr(backend, "_submit_target_chunks", chunks_per_poll)))
    max_in_flight = max(1, int(getattr(backend, "_max_in_flight_batches", 1)))
    return max(base_target, submit_target * max_in_flight)

def _chunk_prep_pipeline_backlog(renderer, terrain_backend_label: str) -> int:
    backlog = len(renderer._pending_chunk_coords) + len(renderer._pending_voxel_mesh_results)
    if terrain_backend_label == "Wgpu":
        backlog += int(getattr(renderer, "_pending_surface_gpu_batches_chunk_total", 0))
    return backlog

@profile
def service_background_gpu_work(renderer) -> None:
    if getattr(renderer, "_device_lost", False):
        return
    mesher = _terrain_mesher(renderer)
    if hasattr(mesher, "process_gpu_buffer_cleanup"):
        mesher.process_gpu_buffer_cleanup(renderer)
    mesh_cache.process_deferred_mesh_output_frees(renderer)
    if renderer.use_gpu_meshing:
        if renderer.world.terrain_backend_label() == "Wgpu" and hasattr(mesher, "drain_pending_surface_gpu_batches_to_meshing"):
            mesher.drain_pending_surface_gpu_batches_to_meshing(renderer)
        if hasattr(mesher, "finalize_pending_gpu_mesh_batches"):
            mesher.finalize_pending_gpu_mesh_batches(renderer)

@profile
def prepare_chunks(renderer, dt: float) -> tuple[float, float]:
    if getattr(renderer, "_device_lost", False):
        return 0.0, 0.0
    renderer_module = _renderer_module()
    visibility_lookup_ms = 0.0
    current_origin = renderer._current_chunk_origin()
    if renderer._visible_chunk_origin != current_origin or not renderer._visible_chunk_coords:
        visibility_lookup_ms = refresh_visible_chunk_set(renderer)

    prep_start = time.perf_counter()
    terrain_backend_label = renderer.world.terrain_backend_label()
    using_wgpu_terrain = terrain_backend_label == "Wgpu"
    using_metal_terrain = terrain_backend_label == "Metal"
    mesher = _terrain_mesher(renderer)
    if renderer.use_gpu_meshing:
        if using_wgpu_terrain:
            ready_surface_batches = renderer.world.poll_ready_chunk_surface_gpu_batches()
            chunk_stream_drained = 0
            chunk_stream_bytes = 0.0
            if ready_surface_batches:
                ready_surface_batches.sort(
                    key=lambda batch: min(
                        (
                            chunk_prep_priority(
                                renderer,
                                int(chunk_x),
                                0,
                                int(chunk_z),
                                current_origin[0],
                                current_origin[1],
                                current_origin[2],
                            )
                            for chunk_x, chunk_z in batch.chunks
                        ),
                        default=(0.0, 0, 0),
                    )
                )
                for surface_batch in ready_surface_batches:
                    batch_chunk_count = len(surface_batch.chunks)
                    chunk_stream_drained += batch_chunk_count
                    chunk_stream_bytes += float(batch_chunk_count * int(surface_batch.cell_count) * 8)
                staged_meshes = meshing_stage.enqueue_surface_gpu_batches_for_meshing(renderer, ready_surface_batches)
                if staged_meshes:
                    for mesh in staged_meshes:
                        mesh_cache.store_chunk_mesh(renderer, mesh)
                meshing_stage.drain_pending_surface_gpu_batches_to_meshing(renderer)
        elif using_metal_terrain:
            ready_results = renderer.world.poll_ready_chunk_voxel_batches()
            chunk_stream_bytes = 0.0
            chunk_stream_drained = 0
            drain_budget = max(1, renderer.mesh_batch_size)
            ready_results.sort(
                key=lambda result: chunk_prep_priority(
                    renderer,
                    int(result.chunk_x),
                    int(getattr(result, "chunk_y", 0)),
                    int(result.chunk_z),
                    current_origin[0],
                    current_origin[1],
                    current_origin[2],
                )
            )
            for result in ready_results:
                chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
            for result in ready_results:
                key = (int(result.chunk_x), int(getattr(result, "chunk_y", 0)), int(result.chunk_z))
                renderer._pending_chunk_coords.discard(key)
                renderer._pending_voxel_mesh_results.append(result)
            while renderer._pending_voxel_mesh_results and drain_budget > 0:
                batch_size = min(renderer.mesh_batch_size, len(renderer._pending_voxel_mesh_results))
                batch_size = min(batch_size, drain_budget)
                batch = [renderer._pending_voxel_mesh_results.popleft() for _ in range(batch_size)]
                chunk_stream_drained += len(batch)
                drain_budget -= len(batch)
                if not batch:
                    continue
                meshes = meshing_stage.make_chunk_mesh_batch_from_terrain(renderer, batch)
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
                int(getattr(result, "chunk_y", 0)),
                int(result.chunk_z),
                current_origin[0],
                current_origin[1],
                current_origin[2],
            )
        )
        for result in ready_results:
            key = (int(result.chunk_x), int(getattr(result, "chunk_y", 0)), int(result.chunk_z))
            renderer._pending_chunk_coords.discard(key)
            renderer._pending_voxel_mesh_results.append(result)
        while renderer._pending_voxel_mesh_results and drain_budget > 0:
            batch_size = min(renderer.mesh_batch_size, len(renderer._pending_voxel_mesh_results))
            batch_size = min(batch_size, drain_budget)
            batch = [renderer._pending_voxel_mesh_results.popleft() for _ in range(batch_size)]
            chunk_stream_drained += len(batch)
            drain_budget -= len(batch)
            if not batch:
                continue
            for result in batch:
                chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
            meshes = meshing_stage.make_chunk_mesh_batch_from_terrain(renderer, batch)
            mesh_cache.store_chunk_meshes(renderer, meshes)

    missing_coords = renderer._visible_missing_coords
    camera_chunk_x = current_origin[0]
    camera_chunk_y = current_origin[1]
    camera_chunk_z = current_origin[2]
    missing_count = len(missing_coords)
    base_request_budget = max(
        1,
        min(renderer_module.chunk_prep_request_budget_cap, max(renderer.mesh_batch_size, renderer.terrain_batch_size)),
    )

    pending_chunk_coords = renderer._pending_chunk_coords
    displayed_chunk_coords = renderer._visible_displayed_coords

    visible_count = max(1, len(renderer._visible_chunk_coords))
    displayed_ratio = len(displayed_chunk_coords) / float(visible_count)
    pipeline_backlog = _chunk_prep_pipeline_backlog(renderer, terrain_backend_label)
    backlog_target = _chunk_prep_backlog_target(renderer)
    bootstrap_budget = max(0, backlog_target - pipeline_backlog)
    if displayed_ratio < renderer_module.chunk_prep_bootstrap_displayed_ratio_threshold:
        prep_budget = min(missing_count, max(base_request_budget, bootstrap_budget))
    else:
        prep_budget = min(missing_count, base_request_budget)

    if prep_budget > 0 and missing_count > 0 and chunk_request_queue_needs_rebuild(renderer, current_origin):
        rebuild_chunk_request_queue(renderer, camera_chunk_x, camera_chunk_y, camera_chunk_z)

    request_coords: list[tuple[int, int, int]] = []
    request_queue = renderer._chunk_request_queue
    while request_queue and len(request_coords) < prep_budget:
        coord = request_queue.popleft()
        if coord in pending_chunk_coords or coord not in missing_coords:
            continue
        request_coords.append(coord)

    if not request_coords and prep_budget > 0 and missing_count > 0 and renderer._visible_missing_coords:
        rebuild_chunk_request_queue(renderer, camera_chunk_x, camera_chunk_y, camera_chunk_z)
        request_queue = renderer._chunk_request_queue
        while request_queue and len(request_coords) < prep_budget:
            coord = request_queue.popleft()
            if coord in pending_chunk_coords or coord not in missing_coords:
                continue
            request_coords.append(coord)

    if request_coords:
        batch_size = max(1, min(renderer_module.chunk_prep_request_budget_cap, max(renderer.mesh_batch_size, renderer.terrain_batch_size)))
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
    profiling.record_frame_breakdown_sample(renderer, "chunk_stream_bytes", chunk_stream_bytes)
    return visibility_lookup_ms, chunk_stream_ms


class ChunkPipeline:
    def __init__(self, renderer):
        self.renderer = renderer
        self.terrain = terrain_stage
        self.meshing = meshing_stage
        self.visibility = visibility_stage
        self.render = render_stage
        self.collision = collision_stage
        self.cache = cache_stage
        self.profiling = profiling

    @profile
    def prepare_chunks(self, dt: float):
        return prepare_chunks(self.renderer, dt)

    @profile
    def service_background_gpu_work(self) -> None:
        return service_background_gpu_work(self.renderer)

    def refresh_visibility(self) -> float:
        return refresh_visible_chunk_set(self.renderer)
