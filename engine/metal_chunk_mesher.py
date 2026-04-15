from __future__ import annotations

import struct
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
    vertex_pool_buffer: object
    in_flight: bool = False


@dataclass
class AsyncVoxelMeshBatchResources:
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
        self._pipeline = self._build_pipeline("mesh_columns_fixed_slice")
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
                slot.vertex_pool_buffer = None
            self._slots.clear()
            self._pipeline = None
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

    def submit_chunk_mesh_batch(
        self,
        renderer,
        chunk_results: list[ChunkVoxelResult],
        on_complete: Callable[[list[object]], None] | None,
    ) -> AsyncVoxelMeshBatchResources | None:
        if not chunk_results:
            resources = AsyncVoxelMeshBatchResources(
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

            for i, result in enumerate(chunk_results):
                blocks_view[i] = np.ascontiguousarray(result.blocks, dtype=np.uint32)
                materials_view[i] = np.ascontiguousarray(result.materials, dtype=np.uint32)
                coords_view[i, 0] = int(result.chunk_x)
                coords_view[i, 1] = int(result.chunk_z)

            counts_view.fill(0)
            overflow_view.fill(0)

            params_bytes = struct.pack(
                "<8I",
                sample_size,
                height_limit,
                chunk_count,
                int(renderer.world.chunk_size),
                self.max_vertices_per_chunk,
                0,
                0,
                0,
            )

            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipeline)
            encoder.setBuffer_offset_atIndex_(slot.blocks_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(slot.materials_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(slot.coords_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(slot.counts_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(slot.overflow_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(slot.vertex_pool_buffer, 0, 5)
            encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 6)

            tew = int(self._pipeline.threadExecutionWidth())
            max_total = int(self._pipeline.maxTotalThreadsPerThreadgroup())
            tg_width = max(1, tew)
            tg_height = max(1, max_total // tg_width)
            threads_per_tg = Metal.MTLSizeMake(tg_width, tg_height, 1)
            grid = Metal.MTLSizeMake(sample_size - 2, sample_size - 2, chunk_count)

            encoder.dispatchThreads_threadsPerThreadgroup_(grid, threads_per_tg)
            encoder.endEncoding()
        except Exception as exc:
            self._release_slot(slot)
            _complete_gpu_mesh_batch(renderer, resources, error=exc, meshes=None, callback_invoked=False)
            raise

        def _done(cb):
            meshes: list[object] = []
            error: Exception | None = None
            callback_invoked = False
            try:
                status = int(cb.status())
                completed_status = int(getattr(Metal, "MTLCommandBufferStatusCompleted", 4))
                if status != completed_status:
                    err = cb.error()
                    status_name = _command_buffer_status_name(status)
                    raise RuntimeError(f"Metal mesher command buffer failed: status={status_name}, error={err}")

                counts = self._u32_view(slot.counts_buffer, chunk_count).copy()
                overflow = self._u32_view(slot.overflow_buffer, chunk_count).copy()

                pool_nbytes = chunk_count * self.max_vertices_per_chunk * self.vertex_stride
                pool_view = memoryview(slot.vertex_pool_buffer.contents().as_buffer(pool_nbytes))
                for i, result in enumerate(chunk_results):
                    if int(overflow[i]) != 0:
                        continue

                    vertex_count = int(counts[i])
                    if vertex_count <= 0:
                        continue
                    if vertex_count > self.max_vertices_per_chunk:
                        raise RuntimeError(
                            f"Metal mesher produced vertex_count={vertex_count} beyond per-chunk max={self.max_vertices_per_chunk}"
                        )

                    vertex_offset = i * self.max_vertices_per_chunk * self.vertex_stride
                    vertex_nbytes = vertex_count * self.vertex_stride
                    vertex_bytes = bytes(pool_view[vertex_offset : vertex_offset + vertex_nbytes])
                    meshes.append(
                        {
                            "chunk_x": int(result.chunk_x),
                            "chunk_z": int(result.chunk_z),
                            "vertex_count": vertex_count,
                            "max_height": int(height_limit),
                            "vertex_bytes": vertex_bytes,
                        }
                    )

                if on_complete is not None:
                    callback_invoked = True
                    on_complete(meshes)
            except Exception as exc:
                error = exc
            finally:
                self._release_slot(slot)
                _complete_gpu_mesh_batch(
                    renderer,
                    resources,
                    meshes=meshes,
                    error=error,
                    callback_invoked=callback_invoked,
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


def _register_pending_gpu_mesh_batch(renderer, resources: AsyncVoxelMeshBatchResources) -> None:
    _ensure_renderer_async_state(renderer)
    lock = _get_renderer_async_state_lock(renderer)
    with lock:
        pending = getattr(renderer, "_metal_pending_gpu_mesh_batches")
        pending.append(resources)


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


def prewarm_metal_chunk_mesher(renderer) -> None:
    try:
        get_metal_chunk_mesher(renderer, block=False)
    except Exception:
        return


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


def submit_chunk_mesh_batch_async(
    renderer,
    chunk_results: list[ChunkVoxelResult],
    on_complete: Callable[[list[object]], None] | None = None,
) -> AsyncVoxelMeshBatchResources | None:
    mesher = get_metal_chunk_mesher(renderer, block=False)
    if mesher is None:
        return None
    return mesher.submit_chunk_mesh_batch(renderer, chunk_results, on_complete)


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


def _append_completed_meshes_to_renderer(renderer, meshes: list[object]) -> None:
    if not meshes:
        return
    from .chunk_generation_helpers import make_chunk_mesh_fast

    created_at = time.perf_counter()
    for mesh in meshes:
        if isinstance(mesh, dict) and "vertex_bytes" in mesh:
            vertex_bytes = mesh["vertex_bytes"]
            size_bytes = max(1, len(vertex_bytes))
            vertex_buffer = renderer.device.create_buffer(
                size=size_bytes,
                usage=wgpu.BufferUsage.VERTEX
                | wgpu.BufferUsage.COPY_DST
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.STORAGE,
            )
            if vertex_bytes:
                renderer.device.queue.write_buffer(vertex_buffer, 0, vertex_bytes)
            chunk_mesh = make_chunk_mesh_fast(
                renderer,
                chunk_x=int(mesh["chunk_x"]),
                chunk_z=int(mesh["chunk_z"]),
                vertex_count=int(mesh["vertex_count"]),
                vertex_buffer=vertex_buffer,
                vertex_offset=0,
                max_height=int(mesh["max_height"]),
                created_at=created_at,
                allocation_id=None,
            )
            mesh_cache.store_chunk_mesh(renderer, chunk_mesh)
        else:
            mesh_cache.store_chunk_mesh(renderer, mesh)


def process_gpu_buffer_cleanup(renderer) -> None:
    _ensure_renderer_async_state(renderer)
    cleanup_queue = getattr(renderer, "_metal_gpu_buffer_cleanup_queue")
    while True:
        try:
            resources = cleanup_queue.popleft()
        except IndexError:
            break
        destroy_async_voxel_mesh_batch_resources(resources)


def finalize_pending_gpu_mesh_batches(renderer, budget: int | None = None) -> int:
    _ensure_renderer_async_state(renderer)
    process_gpu_buffer_cleanup(renderer)

    if budget is None:
        limit = None
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

        if resources.deliver_to_renderer and resources.completed_meshes:
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
    resources.slot = None
    resources.on_complete = None
    resources.error = None
    resources.chunk_results.clear()
    resources.completed_meshes.clear()
    resources.done_event = None
