from __future__ import annotations

import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import Metal
import wgpu

from .. import render_contract as render_consts
from ..cache import mesh_allocator as mesh_cache
from ..meshing import cpu_mesher
from ..meshing_types import ChunkMesh
from ..terrain.types import ChunkSurfaceGpuBatch, ChunkVoxelResult

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


_MESHER_CACHE_INIT_LOCK = threading.Lock()
_ASYNC_STATE_INIT_LOCK = threading.Lock()


def _renderer_module():
    """Return stable render constants without importing the renderer runtime."""
    return render_consts


def _normalize_chunk_coord(coord) -> tuple[int, int, int]:
    if len(coord) >= 3:
        return int(coord[0]), int(coord[1]), int(coord[2])
    if len(coord) == 2:
        return int(coord[0]), 0, int(coord[1])
    raise ValueError(f"Invalid chunk coordinate: {coord!r}")


def _normalize_chunk_coords(coords) -> list[tuple[int, int, int]]:
    return [_normalize_chunk_coord(coord) for coord in coords]


def _local_mesher_height(renderer) -> int:
    renderer_module = _renderer_module()
    if bool(getattr(renderer_module, "VERTICAL_CHUNK_STACK_ENABLED", False)):
        return int(renderer.world.chunk_size)
    return int(renderer.world.height)


def _resolve_metal_device(renderer):
    world = getattr(renderer, "world", None)
    backend = getattr(world, "_backend", None)
    backend_device = getattr(backend, "device", None)
    if backend_device is not None and hasattr(backend_device, "newCommandQueue"):
        return backend_device
    renderer_device = getattr(renderer, "device", None)
    if renderer_device is not None and hasattr(renderer_device, "newCommandQueue"):
        return renderer_device
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("Metal device unavailable.")
    return device


def _command_buffer_completed(command_buffer) -> bool:
    status = int(command_buffer.status())
    completed = int(getattr(Metal, "MTLCommandBufferStatusCompleted", 4))
    errored = int(getattr(Metal, "MTLCommandBufferStatusError", 5))
    if status == completed:
        return True
    if status == errored:
        raise RuntimeError(f"Metal mesher command buffer failed: {command_buffer.error()}")
    return False


@dataclass
class MetalMesherSlot:
    slot_id: int
    chunk_capacity: int
    sample_size: int
    height_limit: int
    storage_height: int
    max_vertices_per_chunk: int
    vertex_stride: int
    blocks_buffer: object
    materials_buffer: object
    coords_buffer: object
    counts_buffer: object
    overflow_buffer: object
    column_counts_buffer: object
    column_offsets_buffer: object
    vertex_pool_buffer: object
    in_flight: bool = False


@dataclass
class AsyncMetalMeshBatchResources:
    mesher: "MetalChunkMesher | None"
    slot: MetalMesherSlot | None
    chunk_results: list[ChunkVoxelResult] = field(default_factory=list)
    chunk_coords: list[tuple[int, int, int]] = field(default_factory=list)
    on_complete: Callable[[list[object]], None] | None = None
    deliver_to_renderer: bool = False
    completed_meshes: list[object] = field(default_factory=list)
    surface_release_callbacks: list[object] = field(default_factory=list)
    command_buffer: object | None = None
    error: Exception | None = None
    created_at: float = field(default_factory=time.perf_counter)
    completed_at: float | None = None
    finalized: bool = False
    cleaned_up: bool = False


@dataclass
class PendingMetalChunkMesherInit:
    key: tuple[int, int, int, int, int]
    ready_event: threading.Event = field(default_factory=threading.Event)
    mesher: "MetalChunkMesher | None" = None
    error: Exception | None = None
    started_at: float = field(default_factory=time.perf_counter)


class MetalChunkMesher:
    """Metal compute mesher for stacked Minechunk chunks."""

    def __init__(self, device, *, chunk_capacity: int, sample_size: int, height_limit: int, inflight_slots: int = 3):
        if int(sample_size) < 3:
            raise ValueError(f"sample_size must be >= 3, got {sample_size}")
        if int(height_limit) <= 0:
            raise ValueError(f"height_limit must be > 0, got {height_limit}")
        if int(chunk_capacity) <= 0:
            raise ValueError(f"chunk_capacity must be > 0, got {chunk_capacity}")
        if int(inflight_slots) <= 0:
            raise ValueError(f"inflight_slots must be > 0, got {inflight_slots}")

        renderer_module = _renderer_module()
        self.device = device
        self.command_queue = device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal mesher command queue.")
        self.chunk_capacity = int(chunk_capacity)
        self.sample_size = int(sample_size)
        self.height_limit = int(height_limit)
        self.storage_height = self.height_limit + 2
        self.inflight_slots = int(inflight_slots)
        self.max_vertices_per_chunk = int(renderer_module.MAX_VERTICES_PER_CHUNK)
        self.vertex_stride = int(renderer_module.VERTEX_STRIDE)

        self._lock = threading.Lock()
        self._slots: list[MetalMesherSlot] = []
        self._expand_pipeline = self._build_pipeline("expand_surface_to_voxels")
        self._count_pipeline = self._build_pipeline("count_columns_fixed_slice")
        self._scan_serial_pipeline = self._build_pipeline("scan_columns_fixed_slice_serial")
        self._scan_parallel_pipeline = self._build_pipeline("scan_columns_fixed_slice_parallel")
        self._emit_pipeline = self._build_pipeline("emit_columns_fixed_slice")
        self._destroyed = False

        upload_options = int(getattr(Metal, "MTLResourceStorageModeShared")) | int(getattr(Metal, "MTLResourceCPUCacheModeWriteCombined"))
        meta_options = int(getattr(Metal, "MTLResourceStorageModeShared"))
        vertex_options = int(getattr(Metal, "MTLResourceStorageModeShared"))

        plane = self.sample_size * self.sample_size
        blocks_bytes = self.chunk_capacity * self.storage_height * plane * 4
        coords_bytes = self.chunk_capacity * 4 * 4
        meta_bytes = self.chunk_capacity * 4
        column_count_bytes = self.chunk_capacity * (self.sample_size - 2) * (self.sample_size - 2) * 4
        vertex_pool_bytes = self.chunk_capacity * self.max_vertices_per_chunk * self.vertex_stride

        for slot_id in range(self.inflight_slots):
            self._slots.append(
                MetalMesherSlot(
                    slot_id=slot_id,
                    chunk_capacity=self.chunk_capacity,
                    sample_size=self.sample_size,
                    height_limit=self.height_limit,
                    storage_height=self.storage_height,
                    max_vertices_per_chunk=self.max_vertices_per_chunk,
                    vertex_stride=self.vertex_stride,
                    blocks_buffer=device.newBufferWithLength_options_(max(4, blocks_bytes), upload_options),
                    materials_buffer=device.newBufferWithLength_options_(max(4, blocks_bytes), upload_options),
                    coords_buffer=device.newBufferWithLength_options_(max(16, coords_bytes), upload_options),
                    counts_buffer=device.newBufferWithLength_options_(max(4, meta_bytes), meta_options),
                    overflow_buffer=device.newBufferWithLength_options_(max(4, meta_bytes), meta_options),
                    column_counts_buffer=device.newBufferWithLength_options_(max(4, column_count_bytes), meta_options),
                    column_offsets_buffer=device.newBufferWithLength_options_(max(4, column_count_bytes), meta_options),
                    vertex_pool_buffer=device.newBufferWithLength_options_(max(self.vertex_stride, vertex_pool_bytes), vertex_options),
                )
            )

    def destroy(self) -> None:
        with self._lock:
            if self._destroyed:
                return
            self._destroyed = True
            for slot in self._slots:
                slot.in_flight = False
                slot.blocks_buffer = None
                slot.materials_buffer = None
                slot.coords_buffer = None
                slot.counts_buffer = None
                slot.overflow_buffer = None
                slot.column_counts_buffer = None
                slot.column_offsets_buffer = None
                slot.vertex_pool_buffer = None
            self._slots.clear()
            self.command_queue = None
            self.device = None

    def _build_pipeline(self, entry_point: str):
        src = Path(__file__).parents[1].joinpath("metal_voxel_mesher.metal").read_text(encoding="utf-8")
        library, err = self.device.newLibraryWithSource_options_error_(src, None, None)
        if err is not None or library is None:
            raise RuntimeError(f"Metal library compile failed: {err}")
        fn = library.newFunctionWithName_(entry_point)
        if fn is None:
            raise RuntimeError(f"Metal function not found: {entry_point}")
        pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err is not None or pso is None:
            raise RuntimeError(f"Metal pipeline creation failed for {entry_point}: {err}")
        return pso

    def _acquire_slot(self) -> MetalMesherSlot | None:
        with self._lock:
            if self._destroyed:
                raise RuntimeError("MetalChunkMesher is destroyed")
            for slot in self._slots:
                if not slot.in_flight:
                    slot.in_flight = True
                    return slot
        return None

    def _release_slot(self, slot: MetalMesherSlot | None) -> None:
        if slot is None:
            return
        with self._lock:
            slot.in_flight = False

    @staticmethod
    def _u32_view(buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(max(0, count) * 4), dtype=np.uint32, count=count)

    @staticmethod
    def _i32_view(buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(max(0, count) * 4), dtype=np.int32, count=count)

    def _pack_mesher_params(self, renderer, chunk_count: int) -> bytes:
        return struct.pack(
            "<5If2I",
            int(self.sample_size),
            int(self.height_limit),
            int(chunk_count),
            int(renderer.world.chunk_size),
            int(self.max_vertices_per_chunk),
            float(renderer.world.block_size),
            0,
            0,
        )

    def _pack_expand_params(self, renderer, chunk_count: int) -> bytes:
        return struct.pack(
            "<8I",
            int(self.sample_size),
            int(self.height_limit),
            int(chunk_count),
            int(renderer.world.chunk_size),
            int(renderer.world.height),
            int(renderer.world.seed) & 0xFFFFFFFF,
            0,
            0,
        )

    def _zero_metadata(self, slot: MetalMesherSlot, chunk_count: int) -> None:
        columns_per_side = self.sample_size - 2
        column_count = int(chunk_count) * columns_per_side * columns_per_side
        self._u32_view(slot.counts_buffer, int(chunk_count)).fill(0)
        self._u32_view(slot.overflow_buffer, int(chunk_count)).fill(0)
        self._u32_view(slot.column_counts_buffer, column_count).fill(0)
        self._u32_view(slot.column_offsets_buffer, column_count).fill(0)

    def _write_coords(self, slot: MetalMesherSlot, chunk_coords: list[tuple[int, int, int]]) -> None:
        coords_view = self._i32_view(slot.coords_buffer, len(chunk_coords) * 4).reshape((len(chunk_coords), 4))
        for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunk_coords):
            coords_view[index, 0] = int(chunk_x)
            coords_view[index, 1] = int(chunk_y)
            coords_view[index, 2] = int(chunk_z)
            coords_view[index, 3] = 0

    def _encode_mesh_passes(self, encoder, slot: MetalMesherSlot, chunk_count: int, params_bytes: bytes) -> None:
        columns_per_side = self.sample_size - 2

        count_tew = int(self._count_pipeline.threadExecutionWidth())
        count_max_total = int(self._count_pipeline.maxTotalThreadsPerThreadgroup())
        count_tg_width = max(1, count_tew)
        count_tg_height = max(1, count_max_total // count_tg_width)
        count_threads_per_tg = Metal.MTLSizeMake(count_tg_width, count_tg_height, 1)
        count_grid = Metal.MTLSizeMake(columns_per_side, columns_per_side, chunk_count)

        encoder.setComputePipelineState_(self._count_pipeline)
        encoder.setBuffer_offset_atIndex_(slot.blocks_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(slot.column_counts_buffer, 0, 1)
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(count_grid, count_threads_per_tg)

        columns_per_chunk = columns_per_side * columns_per_side
        scan_parallel_max_total = int(self._scan_parallel_pipeline.maxTotalThreadsPerThreadgroup())
        if columns_per_chunk <= 1024 and scan_parallel_max_total >= 1024:
            scan_threads_per_tg = Metal.MTLSizeMake(1024, 1, 1)
            scan_grid = Metal.MTLSizeMake(1024, 1, chunk_count)
            scan_pipeline = self._scan_parallel_pipeline
        else:
            scan_tew = int(self._scan_serial_pipeline.threadExecutionWidth())
            scan_threads_per_tg = Metal.MTLSizeMake(max(1, scan_tew), 1, 1)
            scan_grid = Metal.MTLSizeMake(chunk_count, 1, 1)
            scan_pipeline = self._scan_serial_pipeline

        encoder.setComputePipelineState_(scan_pipeline)
        encoder.setBuffer_offset_atIndex_(slot.column_counts_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(slot.column_offsets_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(slot.counts_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(slot.overflow_buffer, 0, 3)
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 4)
        encoder.dispatchThreads_threadsPerThreadgroup_(scan_grid, scan_threads_per_tg)

        emit_tew = int(self._emit_pipeline.threadExecutionWidth())
        emit_max_total = int(self._emit_pipeline.maxTotalThreadsPerThreadgroup())
        emit_tg_width = max(1, emit_tew)
        emit_tg_height = max(1, emit_max_total // emit_tg_width)
        emit_threads_per_tg = Metal.MTLSizeMake(emit_tg_width, emit_tg_height, 1)
        emit_grid = Metal.MTLSizeMake(columns_per_side, columns_per_side, chunk_count)

        encoder.setComputePipelineState_(self._emit_pipeline)
        encoder.setBuffer_offset_atIndex_(slot.blocks_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(slot.materials_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(slot.coords_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(slot.column_counts_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(slot.column_offsets_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(slot.overflow_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(slot.vertex_pool_buffer, 0, 6)
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 7)
        encoder.dispatchThreads_threadsPerThreadgroup_(emit_grid, emit_threads_per_tg)

    @profile
    def submit_chunk_mesh_batch(
        self,
        renderer,
        chunk_results: list[ChunkVoxelResult],
        on_complete: Callable[[list[object]], None] | None = None,
    ) -> AsyncMetalMeshBatchResources | None:
        if not chunk_results:
            return AsyncMetalMeshBatchResources(mesher=self, slot=None, on_complete=on_complete, completed_meshes=[])

        chunk_count = len(chunk_results)
        if chunk_count > self.chunk_capacity:
            raise ValueError(f"chunk batch {chunk_count} exceeds slot capacity {self.chunk_capacity}")

        sample_size = int(chunk_results[0].blocks.shape[1])
        height_limit = int(chunk_results[0].blocks.shape[0])
        if sample_size != self.sample_size or height_limit != self.height_limit:
            raise ValueError(
                f"chunk shape mismatch: mesher expects sample={self.sample_size}, height={self.height_limit}, "
                f"got sample={sample_size}, height={height_limit}"
            )

        slot = self._acquire_slot()
        if slot is None:
            return None

        chunk_coords = [(int(r.chunk_x), int(getattr(r, "chunk_y", 0)), int(r.chunk_z)) for r in chunk_results]
        resources = AsyncMetalMeshBatchResources(
            mesher=self,
            slot=slot,
            chunk_results=list(chunk_results),
            chunk_coords=chunk_coords,
            on_complete=on_complete,
            deliver_to_renderer=on_complete is None,
        )

        try:
            plane = self.sample_size * self.sample_size
            total_voxels = chunk_count * self.storage_height * plane
            blocks_view = np.frombuffer(
                slot.blocks_buffer.contents().as_buffer(total_voxels * 4), dtype=np.uint32, count=total_voxels
            ).reshape((chunk_count, self.storage_height, self.sample_size, self.sample_size))
            materials_view = np.frombuffer(
                slot.materials_buffer.contents().as_buffer(total_voxels * 4), dtype=np.uint32, count=total_voxels
            ).reshape((chunk_count, self.storage_height, self.sample_size, self.sample_size))
            blocks_view.fill(0)
            materials_view.fill(0)
            for i, result in enumerate(chunk_results):
                blocks_view[i, 1 : 1 + self.height_limit] = np.ascontiguousarray(result.blocks, dtype=np.uint32)
                materials_view[i, 1 : 1 + self.height_limit] = np.ascontiguousarray(result.materials, dtype=np.uint32)
                bottom = getattr(result, "bottom_boundary", None)
                top = getattr(result, "top_boundary", None)
                if bottom is not None:
                    blocks_view[i, 0] = np.ascontiguousarray(bottom, dtype=np.uint32)
                if top is not None:
                    blocks_view[i, self.height_limit + 1] = np.ascontiguousarray(top, dtype=np.uint32)

            self._write_coords(slot, chunk_coords)
            self._zero_metadata(slot, chunk_count)
            params_bytes = self._pack_mesher_params(renderer, chunk_count)
            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            self._encode_mesh_passes(encoder, slot, chunk_count, params_bytes)
            encoder.endEncoding()
            command_buffer.commit()
            resources.command_buffer = command_buffer
        except Exception:
            self._release_slot(slot)
            raise
        return resources

    @profile
    def submit_surface_gpu_batch(
        self,
        renderer,
        surface_batch: ChunkSurfaceGpuBatch,
        on_complete: Callable[[list[object]], None] | None = None,
    ) -> AsyncMetalMeshBatchResources | None:
        device_kind = str(getattr(surface_batch, "device_kind", "") or "").strip().lower()
        source = str(getattr(surface_batch, "source", "") or "").strip().lower()
        if device_kind and device_kind != "metal":
            raise TypeError(f"Metal mesher cannot bind {device_kind!r} terrain surface buffers; use ChunkVoxelResult fallback.")
        if not device_kind and "wgpu" in source:
            raise TypeError("Metal mesher cannot bind WGPU terrain surface buffers; use ChunkVoxelResult fallback.")
        chunk_coords = _normalize_chunk_coords(surface_batch.chunks if isinstance(surface_batch.chunks, list) else list(surface_batch.chunks))
        if not chunk_coords:
            release_surface_gpu_batch_immediately(surface_batch)
            return None
        chunk_count = len(chunk_coords)
        if chunk_count > self.chunk_capacity:
            raise ValueError(f"surface batch {chunk_count} exceeds slot capacity {self.chunk_capacity}")

        slot = self._acquire_slot()
        if slot is None:
            return None

        callbacks: list[object] = []
        callback = getattr(surface_batch, "_release_callback", None)
        if callable(callback):
            callbacks.append(callback)
            try:
                setattr(surface_batch, "_release_callback", None)
            except Exception:
                pass

        resources = AsyncMetalMeshBatchResources(
            mesher=self,
            slot=slot,
            chunk_results=[],
            chunk_coords=chunk_coords,
            on_complete=on_complete,
            deliver_to_renderer=on_complete is None,
            surface_release_callbacks=callbacks,
        )
        try:
            self._write_coords(slot, chunk_coords)
            self._zero_metadata(slot, chunk_count)
            expand_params = self._pack_expand_params(renderer, chunk_count)
            mesher_params = self._pack_mesher_params(renderer, chunk_count)
            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()

            encoder.setComputePipelineState_(self._expand_pipeline)
            encoder.setBuffer_offset_atIndex_(surface_batch.heights_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(surface_batch.materials_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(slot.blocks_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(slot.materials_buffer, 0, 3)
            encoder.setBytes_length_atIndex_(expand_params, len(expand_params), 4)
            encoder.setBuffer_offset_atIndex_(slot.coords_buffer, 0, 5)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(self.sample_size, self.sample_size, chunk_count * self.storage_height),
                Metal.MTLSizeMake(8, 8, 1),
            )

            self._encode_mesh_passes(encoder, slot, chunk_count, mesher_params)
            encoder.endEncoding()
            command_buffer.commit()
            resources.command_buffer = command_buffer
        except Exception:
            self._release_slot(slot)
            resources.surface_release_callbacks = callbacks
            destroy_async_voxel_mesh_batch_resources(resources)
            raise
        return resources


@profile
def _get_renderer_async_state_lock(renderer) -> threading.Lock:
    lock = getattr(renderer, "_metal_mesh_async_state_lock", None)
    if lock is None:
        with _ASYNC_STATE_INIT_LOCK:
            lock = getattr(renderer, "_metal_mesh_async_state_lock", None)
            if lock is None:
                lock = threading.Lock()
                setattr(renderer, "_metal_mesh_async_state_lock", lock)
    return lock


@profile
def _ensure_renderer_async_state(renderer) -> None:
    if not hasattr(renderer, "_pending_gpu_mesh_batches") or renderer._pending_gpu_mesh_batches is None:
        renderer._pending_gpu_mesh_batches = deque()
    if not hasattr(renderer, "_pending_surface_gpu_batches") or renderer._pending_surface_gpu_batches is None:
        renderer._pending_surface_gpu_batches = deque()
    if not hasattr(renderer, "_pending_surface_gpu_batches_chunk_total"):
        renderer._pending_surface_gpu_batches_chunk_total = 0


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


def _mesher_cache_lock(renderer) -> threading.Lock:
    lock = getattr(renderer, "_metal_chunk_mesher_cache_lock", None)
    if lock is None:
        with _MESHER_CACHE_INIT_LOCK:
            lock = getattr(renderer, "_metal_chunk_mesher_cache_lock", None)
            if lock is None:
                lock = threading.Lock()
                setattr(renderer, "_metal_chunk_mesher_cache_lock", lock)
    return lock


def _mesher_cache(renderer):
    cache = getattr(renderer, "_metal_chunk_mesher_cache", None)
    if cache is None:
        cache = {}
        setattr(renderer, "_metal_chunk_mesher_cache", cache)
    return cache


def _requested_metal_mesher_capacity(renderer, min_chunk_capacity: int = 1) -> int:
    return max(
        1,
        int(getattr(renderer, "mesh_batch_size", 1)),
        int(getattr(renderer, "terrain_batch_size", 1)),
        int(min_chunk_capacity),
    )


def _mesher_cache_key(renderer, metal_device, min_chunk_capacity: int = 1) -> tuple[int, int, int, int, int]:
    return (
        id(metal_device),
        int(_requested_metal_mesher_capacity(renderer, min_chunk_capacity)),
        int(renderer.world.chunk_size + 2),
        int(_local_mesher_height(renderer)),
        int(getattr(renderer, "metal_mesh_inflight_slots", _renderer_module().METAL_MESH_INFLIGHT_SLOTS)),
    )


@profile
def prewarm_metal_chunk_mesher(renderer) -> None:
    get_metal_chunk_mesher(renderer, block=True)


@profile
def get_metal_chunk_mesher(
    renderer,
    *,
    block: bool = False,
    timeout: float | None = None,
    min_chunk_capacity: int = 1,
) -> MetalChunkMesher | None:
    metal_device = _resolve_metal_device(renderer)
    key = _mesher_cache_key(renderer, metal_device, min_chunk_capacity)
    lock = _mesher_cache_lock(renderer)
    cache = _mesher_cache(renderer)
    with lock:
        mesher = cache.get(key)
        if mesher is not None:
            return mesher
        mesher = MetalChunkMesher(
            metal_device,
            chunk_capacity=int(_requested_metal_mesher_capacity(renderer, min_chunk_capacity)),
            sample_size=int(renderer.world.chunk_size + 2),
            height_limit=int(_local_mesher_height(renderer)),
            inflight_slots=int(getattr(renderer, "metal_mesh_inflight_slots", _renderer_module().METAL_MESH_INFLIGHT_SLOTS)),
        )
        cache[key] = mesher
        return mesher


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
def release_surface_gpu_batch_immediately(surface_batch: ChunkSurfaceGpuBatch) -> None:
    callback = getattr(surface_batch, "_release_callback", None)
    if callable(callback):
        try:
            callback()
        finally:
            try:
                setattr(surface_batch, "_release_callback", None)
            except Exception:
                pass


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
