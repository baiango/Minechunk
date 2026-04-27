from __future__ import annotations

import threading

from .metal_chunk_mesher import MetalChunkMesher
from .metal_mesher_common import (
    _MESHER_CACHE_INIT_LOCK,
    _local_mesher_height,
    _renderer_module,
    _resolve_metal_device,
    profile,
)


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


__all__ = [
    "get_metal_chunk_mesher",
    "prewarm_metal_chunk_mesher",
]
