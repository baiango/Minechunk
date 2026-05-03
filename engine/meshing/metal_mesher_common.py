from __future__ import annotations

import builtins
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import Metal

from .. import render_contract as render_consts
from ..terrain.types import ChunkVoxelResult

_kernprof_profile = getattr(builtins, "profile", None)
if callable(_kernprof_profile):
    profile = _kernprof_profile
else:  # pragma: no cover - only used outside kernprof
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


@profile
def release_surface_gpu_batch_immediately(surface_batch) -> None:
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


__all__ = [
    "AsyncMetalMeshBatchResources",
    "MetalMesherSlot",
    "PendingMetalChunkMesherInit",
    "release_surface_gpu_batch_immediately",
]
