from __future__ import annotations

import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import Metal
import wgpu

from . import mesh_cache_helpers as mesh_cache
from .terrain_backend import ChunkVoxelResult


try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


_MESHER_CACHE_INIT_LOCK = threading.Lock()
_ASYNC_STATE_INIT_LOCK = threading.Lock()


def _renderer_module():
    from . import renderer
    return renderer


def _resolve_metal_device(renderer):
    world = getattr(renderer, "world", None)
    backend = getattr(world, "_backend", None)
    backend_device = getattr(backend, "device", None)
    if backend_device is not None and hasattr(backend_device, "newCommandQueue"):
        return backend_device
    renderer_device = getattr(renderer, "device", None)
    if renderer_device is not None and hasattr(renderer_device, "newCommandQueue"):
        return renderer_device
    return None


def _command_buffer_status_name(status: int) -> str:
    mapping = {
        int(getattr(Metal, "MTLCommandBufferStatusNotEnqueued", 0)): "not_enqueued",
        int(getattr(Metal, "MTLCommandBufferStatusEnqueued", 1)): "enqueued",
        int(getattr(Metal, "MTLCommandBufferStatusCommitted", 2)): "committed",
        int(getattr(Metal, "MTLCommandBufferStatusScheduled", 3)): "scheduled",
        int(getattr(Metal, "MTLCommandBufferStatusCompleted", 4)): "completed",
        int(getattr(Metal, "MTLCommandBufferStatusError", 5)): "error",
    }
    return mapping.get(int(status), f"unknown({status})")


@dataclass
class MetalMesherSlot:
    slot_id: int
    chunk_capacity: int
    sample_size: int
    height_limit: int
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
class AsyncVoxelMeshBatchResources:
    mesher: "MetalChunkMesher | None"
    slot: MetalMesherSlot | None
    chunk_results: list[ChunkVoxelResult]
    on_complete: Callable[[list[object]], None] | None = None
    deliver_to_renderer: bool = False
    completed_meshes: list[object] = field(default_factory=list)
    done_event: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    created_at: float = field(default_factory=time.perf_counter)
    completed_at: float | None = None
    callback_invoked: bool = False
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
    """
    Apple-silicon/shared-buffer path.
    One slot = one in-flight batch. No buffer reuse race across batches.
    """

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
        self.chunk_capacity = int(chunk_capacity)
        self.sample_size = int(sample_size)
        self.height_limit = int(height_limit)
        self.inflight_slots = int(inflight_slots)
        self.max_vertices_per_chunk = int(renderer_module.MAX_VERTICES_PER_CHUNK)
        self.vertex_stride = int(renderer_module.VERTEX_STRIDE)

        self._lock = threading.Lock()
        self._slots: list[MetalMesherSlot] = []
        self._count_pipeline = self._build_pipeline("count_columns_fixed_slice")
        self._scan_serial_pipeline = self._build_pipeline("scan_columns_fixed_slice_serial")
        self._scan_parallel_pipeline = self._build_pipeline("scan_columns_fixed_slice_parallel")
        self._emit_pipeline = self._build_pipeline("emit_columns_fixed_slice")
        self._destroyed = False

        upload_options = (
            int(getattr(Metal, "MTLResourceStorageModeShared"))
            | int(getattr(Metal, "MTLResourceCPUCacheModeWriteCombined"))
        )
        meta_options = int(getattr(Metal, "MTLResourceStorageModeShared"))
        vertex_options = int(getattr(Metal, "MTLResourceStorageModeShared"))

        blocks_bytes = self.chunk_capacity * self.height_limit * self.sample_size * self.sample_size * 4
        coords_bytes = self.chunk_capacity * 2 * 4
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
                    blocks_buffer=device.newBufferWithLength_options_(blocks_bytes, upload_options),
                    materials_buffer=device.newBufferWithLength_options_(blocks_bytes, upload_options),
                    coords_buffer=device.newBufferWithLength_options_(coords_bytes, upload_options),
                    counts_buffer=device.newBufferWithLength_options_(meta_bytes, meta_options),
                    overflow_buffer=device.newBufferWithLength_options_(meta_bytes, meta_options),
                    column_counts_buffer=device.newBufferWithLength_options_(column_count_bytes, meta_options),
                    column_offsets_buffer=device.newBufferWithLength_options_(column_count_bytes, meta_options),
                    vertex_pool_buffer=device.newBufferWithLength_options_(vertex_pool_bytes, vertex_options),
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
            self._count_pipeline = None
            self._scan_serial_pipeline = None
            self._scan_parallel_pipeline = None
            self._emit_pipeline = None
            self.command_queue = None
            self.device = None

    def _build_pipeline(self, entry_point: str):
        src = Path(__file__).with_name("metal_voxel_mesher.metal").read_text(encoding="utf-8")
        library, err = self.device.newLibraryWithSource_options_error_(src, None, None)
        if err is not None or library is None:
            raise RuntimeError(f"Metal library compile failed: {err}")
        fn = library.newFunctionWithName_(entry_point)
        if fn is None:
            raise RuntimeError(f"Metal function not found: {entry_point}")
        pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err is not None or pso is None:
            raise RuntimeError(f"Metal pipeline creation failed: {err}")
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
        return np.frombuffer(buffer.contents().as_buffer(count * 4), dtype=np.uint32, count=count)

    @staticmethod
    def _i32_view(buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(count * 4), dtype=np.int32, count=count)

@profile
def submit_chunk_mesh_batch(
        self,
        renderer,
        chunk_results: list[ChunkVoxelResult],
        on_complete: Callable[[list[object]], None] | None,
    ) -> AsyncVoxelMeshBatchResources | None:
        if not chunk_results:
            resources = AsyncVoxelMeshBatchResources(
                mesher=self,
                slot=None,
                chunk_results=[],
                on_complete=on_complete,
                deliver_to_renderer=on_complete is None,
                completed_meshes=[],
            )
            resources.callback_invoked = True
            resources.completed_at = time.perf_counter()
            resources.done_event.set()
            if on_complete is not None:
                on_complete([])
            return resources

        chunk_count = len(chunk_results)
        if chunk_count > self.chunk_capacity:
            raise ValueError(
                f"chunk batch {chunk_count} exceeds slot capacity {self.chunk_capacity}"
            )

        sample_size = int(chunk_results[0].blocks.shape[1])
        height_limit = int(chunk_results[0].blocks.shape[0])
        if sample_size != self.sample_size or height_limit != self.height_limit:
            raise ValueError(
                f"chunk shape mismatch: mesher expects sample={self.sample_size}, height={self.height_limit}, "
                f"got sample={sample_size}, height={height_limit}"
            )
        if sample_size < 3:
            raise ValueError(f"sample_size must be >= 3, got {sample_size}")

        slot = self._acquire_slot()
        if slot is None:
            return None

        resources = AsyncVoxelMeshBatchResources(
            mesher=self,
            slot=slot,
            chunk_results=list(chunk_results),
            on_complete=on_complete,
            deliver_to_renderer=on_complete is None,
        )
        _register_pending_gpu_mesh_batch(renderer, resources)

        try:
            blocks_view = np.frombuffer(
                slot.blocks_buffer.contents().as_buffer(
                    chunk_count * height_limit * sample_size * sample_size * 4
                ),
                dtype=np.uint32,
                count=chunk_count * height_limit * sample_size * sample_size,
            ).reshape((chunk_count, height_limit, sample_size, sample_size))

            materials_view = np.frombuffer(
                slot.materials_buffer.contents().as_buffer(
                    chunk_count * height_limit * sample_size * sample_size * 4
                ),
                dtype=np.uint32,
                count=chunk_count * height_limit * sample_size * sample_size,
            ).reshape((chunk_count, height_limit, sample_size, sample_size))

            coords_view = self._i32_view(slot.coords_buffer, chunk_count * 2).reshape((chunk_count, 2))
            counts_view = self._u32_view(slot.counts_buffer, chunk_count)
            overflow_view = self._u32_view(slot.overflow_buffer, chunk_count)
            columns_per_side = sample_size - 2
            column_count = chunk_count * columns_per_side * columns_per_side
            column_counts_view = self._u32_view(slot.column_counts_buffer, column_count)
            column_offsets_view = self._u32_view(slot.column_offsets_buffer, column_count)

            for i, result in enumerate(chunk_results):
                blocks_view[i] = np.ascontiguousarray(result.blocks, dtype=np.uint32)
                materials_view[i] = np.ascontiguousarray(result.materials, dtype=np.uint32)
                coords_view[i, 0] = int(result.chunk_x)
                coords_view[i, 1] = int(result.chunk_z)

            counts_view.fill(0)
            overflow_view.fill(0)
            column_counts_view.fill(0)
            column_offsets_view.fill(0)

            params_bytes = struct.pack(
                "<5If2I",
                sample_size,
                height_limit,
                chunk_count,
                int(renderer.world.chunk_size),
                self.max_vertices_per_chunk,
                float(renderer.world.block_size),
                0,
                0,
            )

            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()

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
                scan_threads = 1024
                scan_threads_per_tg = Metal.MTLSizeMake(scan_threads, 1, 1)
                scan_grid = Metal.MTLSizeMake(scan_threads, 1, chunk_count)
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
            encoder.endEncoding()
        except Exception as exc:
            self._release_slot(slot)
            _complete_gpu_mesh_batch(renderer, resources, error=exc, meshes=None, callback_invoked=False)
            raise

        slot_ref = slot
        chunk_results_snapshot = resources.chunk_results

        def _done(cb):
            error: Exception | None = None
            try:
                status = int(cb.status())
                completed_status = int(getattr(Metal, "MTLCommandBufferStatusCompleted", 4))
                if status != completed_status:
                    err = cb.error()
                    status_name = _command_buffer_status_name(status)
                    raise RuntimeError(f"Metal mesher command buffer failed: status={status_name}, error={err}")
            except Exception as exc:
                error = exc
            finally:
                _complete_gpu_mesh_batch(
                    renderer,
                    resources,
                    meshes=None,
                    error=error,
                    callback_invoked=False,
                )

        try:
            command_buffer.addCompletedHandler_(_done)
            command_buffer.commit()
        except Exception as exc:
            self._release_slot(slot)
            _complete_gpu_mesh_batch(
                renderer,
                resources,
                meshes=None,
                error=exc,
                callback_invoked=False,
            )
            raise

        return resources


@profile
def _get_renderer_async_state_lock(renderer) -> threading.Lock:
    lock = getattr(renderer, "_metal_chunk_mesher_async_lock", None)
    if lock is not None:
        return lock
    with _ASYNC_STATE_INIT_LOCK:
        lock = getattr(renderer, "_metal_chunk_mesher_async_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(renderer, "_metal_chunk_mesher_async_lock", lock)
        return lock


@profile
def _ensure_renderer_async_state(renderer) -> None:
    lock = _get_renderer_async_state_lock(renderer)
    with lock:
        if getattr(renderer, "_metal_pending_gpu_mesh_batches", None) is None:
            setattr(renderer, "_metal_pending_gpu_mesh_batches", [])
        if getattr(renderer, "_metal_completed_gpu_mesh_batches", None) is None:
            setattr(renderer, "_metal_completed_gpu_mesh_batches", deque())
        if getattr(renderer, "_metal_gpu_buffer_cleanup_queue", None) is None:
            setattr(renderer, "_metal_gpu_buffer_cleanup_queue", deque())
        if getattr(renderer, "_metal_gpu_mesh_async_errors", None) is None:
            setattr(renderer, "_metal_gpu_mesh_async_errors", deque())


@profile
def shutdown_renderer_async_state(renderer) -> None:
    completed_queue = getattr(renderer, "_metal_completed_gpu_mesh_batches", None)
    if completed_queue is not None:
        while True:
            try:
                resources = completed_queue.popleft()
            except Exception:
                break
            destroy_async_voxel_mesh_batch_resources(resources)

    cleanup_queue = getattr(renderer, "_metal_gpu_buffer_cleanup_queue", None)
    if cleanup_queue is not None:
        while True:
            try:
                resources = cleanup_queue.popleft()
            except Exception:
                break
            destroy_async_voxel_mesh_batch_resources(resources)

    pending = getattr(renderer, "_metal_pending_gpu_mesh_batches", None)
    if pending is not None:
        pending.clear()

    lock = _mesher_cache_lock(renderer)
    with lock:
        cache = dict(_mesher_cache(renderer))
        _mesher_cache(renderer).clear()
    for entry in cache.values():
        if isinstance(entry, MetalChunkMesher):
            try:
                entry.destroy()
            except Exception:
                pass


@profile
def _register_pending_gpu_mesh_batch(renderer, resources: AsyncVoxelMeshBatchResources) -> None:
    _ensure_renderer_async_state(renderer)
    lock = _get_renderer_async_state_lock(renderer)
    with lock:
        pending = getattr(renderer, "_metal_pending_gpu_mesh_batches")
        pending.append(resources)


@profile
def _complete_gpu_mesh_batch(
    renderer,
    resources: AsyncVoxelMeshBatchResources,
    *,
    meshes: list[object] | None,
    error: Exception | None,
    callback_invoked: bool,
) -> None:
    _ensure_renderer_async_state(renderer)
    resources.completed_meshes = [] if meshes is None else list(meshes)
    resources.error = error
    resources.callback_invoked = callback_invoked
    resources.completed_at = time.perf_counter()
    resources.done_event.set()

    lock = _get_renderer_async_state_lock(renderer)
    with lock:
        pending = getattr(renderer, "_metal_pending_gpu_mesh_batches")
        if resources in pending:
            pending.remove(resources)
        completed = getattr(renderer, "_metal_completed_gpu_mesh_batches")
        completed.append(resources)
        if error is not None:
            errors = getattr(renderer, "_metal_gpu_mesh_async_errors")
            errors.append(error)


def _mesher_cache_lock(renderer) -> threading.Lock:
    lock = getattr(renderer, "_metal_chunk_mesher_cache_lock", None)
    if lock is not None:
        return lock
    with _MESHER_CACHE_INIT_LOCK:
        lock = getattr(renderer, "_metal_chunk_mesher_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(renderer, "_metal_chunk_mesher_cache_lock", lock)
        return lock


def _mesher_cache(renderer):
    cache = getattr(renderer, "_metal_chunk_mesher_cache", None)
    if cache is not None:
        return cache
    cache = {}
    setattr(renderer, "_metal_chunk_mesher_cache", cache)
    return cache


def _mesher_cache_key(renderer, metal_device) -> tuple[int, int, int, int, int]:
    inflight_slots = max(1, int(getattr(renderer, "_gpu_mesh_async_inflight_limit", 3)))
    return (
        id(metal_device),
        int(renderer.world.chunk_size),
        int(renderer.world.height),
        max(1, int(renderer.mesh_batch_size)),
        inflight_slots,
    )


def _start_background_mesher_init(renderer, metal_device, key: tuple[int, int, int, int, int]) -> PendingMetalChunkMesherInit:
    pending = PendingMetalChunkMesherInit(key=key)

    def _worker():
        try:
            pending.mesher = MetalChunkMesher(
                metal_device,
                chunk_capacity=max(1, int(renderer.mesh_batch_size)),
                sample_size=int(renderer.world.chunk_size) + 2,
                height_limit=int(renderer.world.height),
                inflight_slots=max(1, int(getattr(renderer, "_gpu_mesh_async_inflight_limit", 3))),
            )
        except Exception as exc:
            pending.error = exc
        finally:
            pending.ready_event.set()

    thread = threading.Thread(target=_worker, name="metal-chunk-mesher-init", daemon=True)
    thread.start()
    return pending


@profile
def prewarm_metal_chunk_mesher(renderer) -> None:
    try:
        get_metal_chunk_mesher(renderer, block=False)
    except Exception:
        return


@profile
def get_metal_chunk_mesher(renderer, *, block: bool = False, timeout: float | None = None) -> MetalChunkMesher | None:
    metal_device = _resolve_metal_device(renderer)
    if metal_device is None:
        if block:
            raise RuntimeError("Metal mesher unavailable: no Metal MTLDevice is active.")
        return None

    key = _mesher_cache_key(renderer, metal_device)
    lock = _mesher_cache_lock(renderer)

    while True:
        with lock:
            cache = _mesher_cache(renderer)
            entry = cache.get(key)
            if isinstance(entry, MetalChunkMesher):
                return entry
            if isinstance(entry, PendingMetalChunkMesherInit):
                pending = entry
            else:
                pending = _start_background_mesher_init(renderer, metal_device, key)
                cache[key] = pending

        if not block:
            if pending.ready_event.is_set():
                with lock:
                    cache = _mesher_cache(renderer)
                    current = cache.get(key)
                    if current is pending:
                        if pending.error is not None:
                            cache.pop(key, None)
                            setattr(renderer, "_metal_mesher_last_error", pending.error)
                            return None
                        if pending.mesher is not None:
                            cache[key] = pending.mesher
                            return pending.mesher
            return None

        if not pending.ready_event.wait(timeout):
            return None

        with lock:
            cache = _mesher_cache(renderer)
            current = cache.get(key)
            if current is pending:
                if pending.error is not None:
                    cache.pop(key, None)
                    if block:
                        raise pending.error
                    setattr(renderer, "_metal_mesher_last_error", pending.error)
                    return None
                if pending.mesher is not None:
                    cache[key] = pending.mesher
                    return pending.mesher
            elif isinstance(current, MetalChunkMesher):
                return current


@profile
def submit_chunk_mesh_batch_async(
    renderer,
    chunk_results: list[ChunkVoxelResult],
    on_complete: Callable[[list[object]], None] | None = None,
) -> AsyncVoxelMeshBatchResources | None:
    mesher = get_metal_chunk_mesher(renderer, block=False)
    if mesher is None:
        return None
    return mesher.submit_chunk_mesh_batch(renderer, chunk_results, on_complete)


@profile
def make_chunk_mesh_batch_from_voxels(renderer, chunk_results: list[ChunkVoxelResult]):
    if not chunk_results:
        return []
    prewarm_metal_chunk_mesher(renderer)
    mesher = get_metal_chunk_mesher(renderer, block=False)
    if mesher is None:
        mesher = get_metal_chunk_mesher(renderer, block=True, timeout=2.0)
    if mesher is None:
        from .chunk_generation_helpers import cpu_make_chunk_mesh_batch_from_voxels

        return cpu_make_chunk_mesh_batch_from_voxels(renderer, chunk_results)
    resources = mesher.submit_chunk_mesh_batch(renderer, chunk_results, None)
    if resources is None:
        from .chunk_generation_helpers import cpu_make_chunk_mesh_batch_from_voxels

        return cpu_make_chunk_mesh_batch_from_voxels(renderer, chunk_results)
    return []



@profile
def _build_completed_meshes_from_resources(renderer, resources: AsyncVoxelMeshBatchResources) -> list[object]:
    slot = resources.slot
    mesher = resources.mesher
    if slot is None or mesher is None:
        return []

    chunk_count = len(resources.chunk_results)
    if chunk_count <= 0:
        return []

    counts = mesher._u32_view(slot.counts_buffer, chunk_count).copy()
    overflow = mesher._u32_view(slot.overflow_buffer, chunk_count).copy()
    pool_nbytes = chunk_count * mesher.max_vertices_per_chunk * mesher.vertex_stride
    pool_view = memoryview(slot.vertex_pool_buffer.contents().as_buffer(pool_nbytes))

    meshes: list[object] = []
    overflow_results: list[ChunkVoxelResult] = []
    overflow_coords: list[tuple[int, int]] = []
    for i, result in enumerate(resources.chunk_results):
        if int(overflow[i]) != 0:
            overflow_results.append(result)
            overflow_coords.append((int(result.chunk_x), int(result.chunk_z)))
            continue

        vertex_count = int(counts[i])
        if vertex_count <= 0:
            continue
        if vertex_count > mesher.max_vertices_per_chunk:
            raise RuntimeError(
                f"Metal mesher produced vertex_count={vertex_count} beyond per-chunk max={mesher.max_vertices_per_chunk}"
            )

        vertex_offset = i * mesher.max_vertices_per_chunk * mesher.vertex_stride
        vertex_nbytes = vertex_count * mesher.vertex_stride
        vertex_bytes = bytes(pool_view[vertex_offset : vertex_offset + vertex_nbytes])
        meshes.append(
            {
                "chunk_x": int(result.chunk_x),
                "chunk_z": int(result.chunk_z),
                "vertex_count": vertex_count,
                "max_height": int(resources.chunk_results[i].blocks.shape[0]),
                "vertex_bytes": vertex_bytes,
            }
        )

    if overflow_results:
        from .chunk_generation_helpers import cpu_make_chunk_mesh_batch_from_voxels

        preview = ", ".join(f"({x},{z})" for x, z in overflow_coords[:8])
        more = "" if len(overflow_coords) <= 8 else f" ... +{len(overflow_coords) - 8} more"
        print(
            f"Warning: Metal mesher overflow on {len(overflow_results)} chunk(s); falling back to CPU meshing for {preview}{more}.",
            file=sys.stderr,
        )
        meshes.extend(cpu_make_chunk_mesh_batch_from_voxels(renderer, overflow_results))

    return meshes


@profile
def _append_completed_meshes_to_renderer(renderer, meshes: list[object]) -> None:
    if not meshes or getattr(renderer, "_device_lost", False):
        return
    from .chunk_generation_helpers import make_chunk_mesh_fast

    created_at = time.perf_counter()
    packed_meshes: list[tuple[int, int, int, int, bytes]] = []
    passthrough_meshes: list[object] = []

    for mesh in meshes:
        if isinstance(mesh, dict) and "vertex_bytes" in mesh:
            vertex_bytes = bytes(mesh["vertex_bytes"])
            packed_meshes.append(
                (
                    int(mesh["chunk_x"]),
                    int(mesh["chunk_z"]),
                    int(mesh["vertex_count"]),
                    int(mesh["max_height"]),
                    vertex_bytes,
                )
            )
        else:
            passthrough_meshes.append(mesh)

    chunk_meshes: list[object] = []
    upload_batch_bytes = max(1, int(getattr(renderer, "_mesh_output_upload_batch_bytes", 32 * 1024 * 1024)))
    pending_batch: list[tuple[int, int, int, int, bytes]] = []
    pending_bytes = 0

    def _flush_pending() -> None:
        nonlocal pending_batch, pending_bytes
        if not pending_batch:
            return
        batch_allocation = mesh_cache.allocate_mesh_output_range(renderer, pending_bytes)
        batch_buffer = batch_allocation.buffer
        cursor_bytes = 0
        for chunk_x, chunk_z, vertex_count, max_height, vertex_bytes in pending_batch:
            vertex_offset = batch_allocation.offset_bytes + cursor_bytes
            if vertex_bytes:
                renderer.device.queue.write_buffer(batch_buffer, vertex_offset, vertex_bytes)
            chunk_meshes.append(
                make_chunk_mesh_fast(
                    renderer,
                    chunk_x=chunk_x,
                    chunk_z=chunk_z,
                    vertex_count=vertex_count,
                    vertex_buffer=batch_buffer,
                    vertex_offset=vertex_offset,
                    max_height=max_height,
                    created_at=created_at,
                    allocation_id=batch_allocation.allocation_id,
                )
            )
            cursor_bytes += len(vertex_bytes)
        pending_batch = []
        pending_bytes = 0

    for packed in packed_meshes:
        vertex_bytes = len(packed[4])
        if pending_batch and pending_bytes + vertex_bytes > upload_batch_bytes:
            _flush_pending()
        if vertex_bytes > upload_batch_bytes:
            batch_allocation = mesh_cache.allocate_mesh_output_range(renderer, vertex_bytes)
            batch_buffer = batch_allocation.buffer
            vertex_offset = batch_allocation.offset_bytes
            if packed[4]:
                renderer.device.queue.write_buffer(batch_buffer, vertex_offset, packed[4])
            chunk_meshes.append(
                make_chunk_mesh_fast(
                    renderer,
                    chunk_x=packed[0],
                    chunk_z=packed[1],
                    vertex_count=packed[2],
                    vertex_buffer=batch_buffer,
                    vertex_offset=vertex_offset,
                    max_height=packed[3],
                    created_at=created_at,
                    allocation_id=batch_allocation.allocation_id,
                )
            )
            continue
        pending_batch.append(packed)
        pending_bytes += vertex_bytes
    _flush_pending()

    for mesh in passthrough_meshes:
        chunk_meshes.append(mesh)

    if chunk_meshes:
        mesh_cache.store_chunk_meshes(renderer, chunk_meshes)


@profile
def process_gpu_buffer_cleanup(renderer) -> None:
    _ensure_renderer_async_state(renderer)
    cleanup_queue = getattr(renderer, "_metal_gpu_buffer_cleanup_queue")
    while True:
        try:
            resources = cleanup_queue.popleft()
        except IndexError:
            break
        destroy_async_voxel_mesh_batch_resources(resources)


@profile
def finalize_pending_gpu_mesh_batches(renderer, budget: int | None = None) -> int:
    _ensure_renderer_async_state(renderer)
    process_gpu_buffer_cleanup(renderer)

    if budget is None:
        limit = max(1, int(getattr(renderer, "_metal_gpu_mesh_async_finalize_budget", 1)))
    else:
        limit = max(0, int(budget))
        if limit == 0:
            return 0

    processed = 0
    first_error: Exception | None = None
    completed_queue = getattr(renderer, "_metal_completed_gpu_mesh_batches")
    cleanup_queue = getattr(renderer, "_metal_gpu_buffer_cleanup_queue")

    while limit is None or processed < limit:
        try:
            resources = completed_queue.popleft()
        except IndexError:
            break

        if resources.error is not None and first_error is None:
            first_error = resources.error

        if resources.error is None and not resources.completed_meshes:
            resources.completed_meshes = _build_completed_meshes_from_resources(renderer, resources)

        if resources.on_complete is not None and not resources.callback_invoked:
            resources.on_complete(resources.completed_meshes)
            resources.callback_invoked = True
        elif resources.deliver_to_renderer and resources.completed_meshes:
            _append_completed_meshes_to_renderer(renderer, resources.completed_meshes)

        resources.finalized = True
        cleanup_queue.append(resources)
        processed += 1

    process_gpu_buffer_cleanup(renderer)

    if first_error is not None:
        raise first_error
    return processed


def destroy_async_voxel_mesh_batch_resources(resources) -> None:
    if resources is None:
        return
    if getattr(resources, "cleaned_up", False):
        return

    done_event = getattr(resources, "done_event", None)
    if done_event is not None and not done_event.is_set():
        return

    resources.cleaned_up = True
    resources.finalized = True
    if resources.mesher is not None:
        resources.mesher._release_slot(resources.slot)
    resources.slot = None
    resources.mesher = None
    resources.on_complete = None
    resources.error = None
    resources.chunk_results.clear()
    resources.completed_meshes.clear()
    resources.done_event = None
