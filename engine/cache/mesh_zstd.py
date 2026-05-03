from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import time
from typing import Iterable

import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkMesh
from . import mesh_output_allocator

ZSTD_LEVEL = 1
DEFAULT_MAX_READBACKS_PER_FRAME = 8
DEFAULT_MAX_READBACK_BYTES_PER_FRAME = 32 * 1024 * 1024
DEFAULT_FINALIZE_BUDGET = 16

_zstd = None
_compressor = None
_decompressor = None


def _zstd_module():
    global _zstd
    if _zstd is not None:
        return _zstd
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Mesh zstd compression requires the 'zstandard' package. "
            "Install dependencies with `python3 -m pip install -r requirements.txt`."
        ) from exc
    _zstd = zstd
    return zstd


def _zstd_compressor():
    global _compressor
    if _compressor is None:
        _compressor = _zstd_module().ZstdCompressor(level=ZSTD_LEVEL)
    return _compressor


def _zstd_decompressor():
    global _decompressor
    if _decompressor is None:
        _decompressor = _zstd_module().ZstdDecompressor()
    return _decompressor


@dataclass(frozen=True)
class CompressedChunkMesh:
    chunk_x: int
    chunk_y: int
    chunk_z: int
    vertex_count: int
    max_height: int
    bounds: tuple[float, float, float, float]
    raw_nbytes: int
    payload: bytes

    @property
    def compressed_nbytes(self) -> int:
        return len(self.payload)

    @property
    def coord(self) -> tuple[int, int, int]:
        return int(self.chunk_x), int(self.chunk_y), int(self.chunk_z)


@dataclass
class PendingMeshZstdReadback:
    coord: tuple[int, int, int]
    mesh: ChunkMesh
    readback_buffer: object
    raw_nbytes: int
    map_promise: object | None = None
    vertex_buffer_id: int = 0
    vertex_offset: int = 0
    vertex_count: int = 0
    allocation_id: int | None = None
    submitted_at: float = 0.0


def mesh_zstd_enabled(renderer) -> bool:
    return bool(getattr(renderer, "mesh_zstd_enabled", False))


def _mesh_key(mesh: ChunkMesh) -> tuple[int, int, int]:
    return int(mesh.chunk_x), int(getattr(mesh, "chunk_y", 0)), int(mesh.chunk_z)


def _ensure_mesh_zstd_state(renderer) -> None:
    if not hasattr(renderer, "_mesh_zstd_cache") or renderer._mesh_zstd_cache is None:
        renderer._mesh_zstd_cache = OrderedDict()
    if not hasattr(renderer, "_pending_mesh_zstd_readbacks") or renderer._pending_mesh_zstd_readbacks is None:
        renderer._pending_mesh_zstd_readbacks = deque()
    if not hasattr(renderer, "_pending_mesh_zstd_readback_keys") or renderer._pending_mesh_zstd_readback_keys is None:
        renderer._pending_mesh_zstd_readback_keys = set()


def compress_chunk_mesh_bytes(mesh: ChunkMesh, raw_bytes: bytes | bytearray | memoryview) -> CompressedChunkMesh:
    raw_view = memoryview(raw_bytes)
    raw_nbytes = int(raw_view.nbytes)
    if raw_nbytes <= 0:
        compressed = b""
    else:
        source = raw_view if raw_view.contiguous else raw_view.tobytes()
        compressed = _zstd_compressor().compress(source)
    return CompressedChunkMesh(
        chunk_x=int(mesh.chunk_x),
        chunk_y=int(getattr(mesh, "chunk_y", 0)),
        chunk_z=int(mesh.chunk_z),
        vertex_count=int(mesh.vertex_count),
        max_height=int(mesh.max_height),
        bounds=tuple(float(value) for value in getattr(mesh, "bounds", (0.0, 0.0, 0.0, 0.0))),
        raw_nbytes=raw_nbytes,
        payload=compressed,
    )


def decompress_chunk_mesh_bytes(mesh: CompressedChunkMesh) -> bytes:
    if int(mesh.raw_nbytes) <= 0:
        return b""
    return _zstd_decompressor().decompress(mesh.payload, max_output_size=int(mesh.raw_nbytes))


def compressed_chunk_meshes_stats(values: Iterable[object]) -> dict[str, int]:
    entries = 0
    raw_bytes = 0
    compressed_bytes = 0
    for value in values:
        if not isinstance(value, CompressedChunkMesh):
            continue
        entries += 1
        raw_bytes += int(value.raw_nbytes)
        compressed_bytes += int(value.compressed_nbytes)
    return {"entries": entries, "raw_bytes": raw_bytes, "compressed_bytes": compressed_bytes}


def _store_compressed_mesh(renderer, compressed: CompressedChunkMesh) -> None:
    _ensure_mesh_zstd_state(renderer)
    cache = renderer._mesh_zstd_cache
    cache[compressed.coord] = compressed
    cache.move_to_end(compressed.coord)
    _enforce_mesh_zstd_cache_limit(renderer)


def _mesh_zstd_cache_limit(renderer) -> int:
    limit = getattr(renderer, "mesh_zstd_cache_limit", None)
    if limit is None:
        limit = getattr(renderer, "max_cached_chunks", 0)
    try:
        return max(0, int(limit))
    except Exception:
        return 0


def _enforce_mesh_zstd_cache_limit(renderer) -> None:
    limit = _mesh_zstd_cache_limit(renderer)
    if limit <= 0:
        return
    cache = renderer._mesh_zstd_cache
    while len(cache) > limit:
        cache.popitem(last=False)


def _destroy_readback_buffer(buffer) -> None:
    if buffer is None:
        return
    try:
        if getattr(buffer, "map_state", "unmapped") != "unmapped":
            buffer.unmap()
    except Exception:
        pass
    try:
        buffer.destroy()
    except Exception:
        pass


def clear_mesh_zstd_cache(renderer) -> None:
    _ensure_mesh_zstd_state(renderer)
    while renderer._pending_mesh_zstd_readbacks:
        pending = renderer._pending_mesh_zstd_readbacks.popleft()
        _destroy_readback_buffer(getattr(pending, "readback_buffer", None))
    renderer._pending_mesh_zstd_readback_keys.clear()
    renderer._mesh_zstd_cache.clear()


def mesh_zstd_runtime_stats(renderer) -> dict[str, int | bool]:
    _ensure_mesh_zstd_state(renderer)
    cache_stats = compressed_chunk_meshes_stats(renderer._mesh_zstd_cache.values())
    pending_entries = 0
    pending_raw_bytes = 0
    for pending in renderer._pending_mesh_zstd_readbacks:
        pending_entries += 1
        pending_raw_bytes += int(getattr(pending, "raw_nbytes", 0))
    return {
        "enabled": bool(mesh_zstd_enabled(renderer)),
        "cache_entries": int(cache_stats["entries"]),
        "cache_raw_bytes": int(cache_stats["raw_bytes"]),
        "cache_compressed_bytes": int(cache_stats["compressed_bytes"]),
        "cache_limit": int(_mesh_zstd_cache_limit(renderer)),
        "pending_entries": int(pending_entries),
        "pending_raw_bytes": int(pending_raw_bytes),
    }


def should_keep_chunk_mesh_for_zstd(renderer, key: tuple[int, int, int]) -> bool:
    if not mesh_zstd_enabled(renderer):
        return False
    _ensure_mesh_zstd_state(renderer)
    key = (int(key[0]), int(key[1]), int(key[2]))
    if key in getattr(renderer, "_visible_chunk_coord_set", set()):
        return True
    return key not in renderer._mesh_zstd_cache


def _mesh_still_matches_pending(renderer, pending: PendingMeshZstdReadback) -> bool:
    mesh = renderer.chunk_cache.get(pending.coord)
    if mesh is not pending.mesh:
        return False
    if id(mesh.vertex_buffer) != int(pending.vertex_buffer_id):
        return False
    if int(mesh.vertex_offset) != int(pending.vertex_offset):
        return False
    if int(mesh.vertex_count) != int(pending.vertex_count):
        return False
    return getattr(mesh, "allocation_id", None) == pending.allocation_id


def _remove_chunk_mesh_after_compression(renderer, key: tuple[int, int, int], mesh: ChunkMesh) -> None:
    from .tile_mesh_cache import remove_chunk_mesh_from_cache

    remove_chunk_mesh_from_cache(renderer, key, mesh, mark_visible_missing=False)


def _compress_zero_vertex_mesh(renderer, key: tuple[int, int, int], mesh: ChunkMesh) -> bool:
    compressed = compress_chunk_mesh_bytes(mesh, b"")
    _store_compressed_mesh(renderer, compressed)
    _remove_chunk_mesh_after_compression(renderer, key, mesh)
    return True


def schedule_mesh_zstd_readbacks(
    renderer,
    *,
    max_readbacks: int = DEFAULT_MAX_READBACKS_PER_FRAME,
    max_raw_bytes: int = DEFAULT_MAX_READBACK_BYTES_PER_FRAME,
) -> int:
    if not mesh_zstd_enabled(renderer):
        return 0
    _ensure_mesh_zstd_state(renderer)
    visible = set(getattr(renderer, "_visible_chunk_coord_set", set()) or ())
    if not visible:
        return 0

    scheduled = 0
    scheduled_raw_bytes = 0
    pending_keys = renderer._pending_mesh_zstd_readback_keys
    cache = renderer._mesh_zstd_cache
    for key, mesh in list(renderer.chunk_cache.items()):
        key = (int(key[0]), int(key[1]), int(key[2]))
        if key in visible or key in pending_keys or key in cache:
            continue
        if int(getattr(mesh, "vertex_count", 0)) <= 0:
            _compress_zero_vertex_mesh(renderer, key, mesh)
            scheduled += 1
            if scheduled >= int(max_readbacks):
                break
            continue

        raw_nbytes = int(mesh.vertex_count) * int(render_consts.VERTEX_STRIDE)
        if raw_nbytes <= 0:
            continue
        if scheduled >= int(max_readbacks):
            break
        if scheduled_raw_bytes + raw_nbytes > int(max_raw_bytes):
            break

        readback_buffer = renderer.device.create_buffer(
            size=max(1, raw_nbytes),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        try:
            encoder = renderer.device.create_command_encoder()
            encoder.copy_buffer_to_buffer(mesh.vertex_buffer, int(mesh.vertex_offset), readback_buffer, 0, raw_nbytes)
            renderer.device.queue.submit([encoder.finish()])
            map_promise = readback_buffer.map_async(wgpu.MapMode.READ, 0, raw_nbytes)
        except Exception:
            _destroy_readback_buffer(readback_buffer)
            continue

        renderer._pending_mesh_zstd_readbacks.append(
            PendingMeshZstdReadback(
                coord=key,
                mesh=mesh,
                readback_buffer=readback_buffer,
                raw_nbytes=raw_nbytes,
                map_promise=map_promise,
                vertex_buffer_id=id(mesh.vertex_buffer),
                vertex_offset=int(mesh.vertex_offset),
                vertex_count=int(mesh.vertex_count),
                allocation_id=getattr(mesh, "allocation_id", None),
                submitted_at=time.perf_counter(),
            )
        )
        pending_keys.add(key)
        scheduled += 1
        scheduled_raw_bytes += raw_nbytes
    return scheduled


def finalize_mesh_zstd_readbacks(renderer, *, budget: int = DEFAULT_FINALIZE_BUDGET) -> int:
    if not mesh_zstd_enabled(renderer):
        return 0
    _ensure_mesh_zstd_state(renderer)
    completed = 0
    remaining = deque()
    visible = set(getattr(renderer, "_visible_chunk_coord_set", set()) or ())

    while renderer._pending_mesh_zstd_readbacks:
        pending = renderer._pending_mesh_zstd_readbacks.popleft()
        if completed >= int(budget):
            remaining.append(pending)
            continue
        buffer = pending.readback_buffer
        if getattr(buffer, "map_state", "unmapped") != "mapped":
            remaining.append(pending)
            continue

        compressed = None
        try:
            if pending.coord not in visible and _mesh_still_matches_pending(renderer, pending):
                mapped = buffer.read_mapped(0, int(pending.raw_nbytes), copy=False)
                compressed = compress_chunk_mesh_bytes(pending.mesh, mapped)
        finally:
            _destroy_readback_buffer(buffer)
            renderer._pending_mesh_zstd_readback_keys.discard(pending.coord)

        if compressed is not None:
            _store_compressed_mesh(renderer, compressed)
            _remove_chunk_mesh_after_compression(renderer, pending.coord, pending.mesh)
        completed += 1

    renderer._pending_mesh_zstd_readbacks = remaining
    renderer._pending_mesh_zstd_readback_keys = {pending.coord for pending in remaining}
    return completed


def restore_visible_mesh_zstd(renderer, coords: Iterable[tuple[int, int, int]] | None = None) -> int:
    if not mesh_zstd_enabled(renderer):
        return 0
    _ensure_mesh_zstd_state(renderer)
    visible = set(getattr(renderer, "_visible_chunk_coord_set", set()) or ())
    if not visible:
        return 0
    if coords is None:
        candidate_coords = visible
    else:
        candidate_coords = {(int(x), int(y), int(z)) for x, y, z in coords}

    restored_meshes: list[ChunkMesh] = []
    restored_keys: list[tuple[int, int, int]] = []
    for key in list(candidate_coords):
        if key not in visible or key in renderer.chunk_cache:
            continue
        compressed = renderer._mesh_zstd_cache.get(key)
        if compressed is None:
            continue
        try:
            raw = decompress_chunk_mesh_bytes(compressed)
            created_at = time.perf_counter()
            if int(compressed.vertex_count) <= 0 or int(compressed.raw_nbytes) <= 0:
                from ..meshing import cpu_mesher

                mesh = cpu_mesher.make_chunk_mesh_fast(
                    renderer,
                    chunk_x=compressed.chunk_x,
                    chunk_y=compressed.chunk_y,
                    chunk_z=compressed.chunk_z,
                    vertex_count=0,
                    vertex_buffer=cpu_mesher._shared_empty_chunk_vertex_buffer(renderer),
                    vertex_offset=0,
                    max_height=int(compressed.max_height),
                    created_at=created_at,
                    allocation_id=None,
                )
            else:
                allocation = mesh_output_allocator.allocate_mesh_output_range(renderer, int(compressed.raw_nbytes))
                renderer.device.queue.write_buffer(allocation.buffer, int(allocation.offset_bytes), memoryview(raw))
                mesh = ChunkMesh(
                    chunk_x=int(compressed.chunk_x),
                    chunk_y=int(compressed.chunk_y),
                    chunk_z=int(compressed.chunk_z),
                    vertex_count=int(compressed.vertex_count),
                    vertex_buffer=allocation.buffer,
                    vertex_offset=int(allocation.offset_bytes),
                    max_height=int(compressed.max_height),
                    created_at=created_at,
                    allocation_id=allocation.allocation_id,
                )
                mesh.bounds = tuple(float(value) for value in compressed.bounds)
            restored_meshes.append(mesh)
            restored_keys.append(key)
        except Exception:
            renderer._mesh_zstd_cache.pop(key, None)

    if restored_meshes:
        from .tile_mesh_cache import store_chunk_meshes

        for key in restored_keys:
            renderer._mesh_zstd_cache.pop(key, None)
        store_chunk_meshes(renderer, restored_meshes)
    return len(restored_meshes)


def service_mesh_zstd(renderer) -> None:
    if not mesh_zstd_enabled(renderer):
        return
    finalize_mesh_zstd_readbacks(renderer)
    mesh_output_allocator.compact_mesh_output_slabs(renderer)
    schedule_mesh_zstd_readbacks(renderer)
