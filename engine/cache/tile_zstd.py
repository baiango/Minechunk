from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import time
from typing import Iterable

import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkDrawBatch, ChunkRenderBatch
from .direct_render_batches import _group_render_batches

ZSTD_LEVEL = 1
DEFAULT_FINALIZE_BUDGET = 8

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
            "Tile zstd compression requires the 'zstandard' package. "
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
class CompressedTileRenderBatch:
    tile_key: tuple[int, int, int]
    signature: tuple[tuple[int, int, int], ...]
    vertex_count: int
    bounds: tuple[float, float, float, float]
    chunk_count: int
    complete_tile: bool
    visible_mask: int
    source_version: int
    next_refresh_at: float
    visible_chunk_count: int
    merged_chunk_count: int
    visible_vertex_count: int
    raw_nbytes: int
    payload: bytes

    @property
    def compressed_nbytes(self) -> int:
        return len(self.payload)


@dataclass
class PendingTileZstdReadback:
    tile_key: tuple[int, int, int]
    batch: ChunkRenderBatch
    source_buffer: object
    source_capacity_bytes: int
    readback_buffer: object
    raw_nbytes: int
    map_promise: object | None = None
    source_buffer_id: int = 0
    submitted_at: float = 0.0


def tile_zstd_enabled(renderer) -> bool:
    return bool(getattr(renderer, "tile_zstd_enabled", False) and getattr(renderer, "mesh_zstd_enabled", False))


def _ensure_tile_zstd_state(renderer) -> None:
    if not hasattr(renderer, "_tile_zstd_cache") or renderer._tile_zstd_cache is None:
        renderer._tile_zstd_cache = OrderedDict()
    if not hasattr(renderer, "_pending_tile_zstd_readbacks") or renderer._pending_tile_zstd_readbacks is None:
        renderer._pending_tile_zstd_readbacks = deque()
    if not hasattr(renderer, "_pending_tile_zstd_readback_keys") or renderer._pending_tile_zstd_readback_keys is None:
        renderer._pending_tile_zstd_readback_keys = set()


def _tile_zstd_cache_limit(renderer) -> int:
    limit = getattr(renderer, "tile_zstd_cache_limit", None)
    if limit is None:
        limit = getattr(renderer, "max_cached_chunks", 0)
    try:
        return max(0, int(limit))
    except Exception:
        return 0


def _destroy_buffer(buffer) -> None:
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


def _enforce_tile_zstd_cache_limit(renderer) -> None:
    limit = _tile_zstd_cache_limit(renderer)
    if limit <= 0:
        return
    cache = renderer._tile_zstd_cache
    while len(cache) > limit:
        cache.popitem(last=False)


def compressed_tile_render_batches_stats(values: Iterable[object]) -> dict[str, int]:
    entries = 0
    raw_bytes = 0
    compressed_bytes = 0
    for value in values:
        if not isinstance(value, CompressedTileRenderBatch):
            continue
        entries += 1
        raw_bytes += int(value.raw_nbytes)
        compressed_bytes += int(value.compressed_nbytes)
    return {"entries": entries, "raw_bytes": raw_bytes, "compressed_bytes": compressed_bytes}


def compress_tile_render_batch_bytes(
    tile_key: tuple[int, int, int],
    batch: ChunkRenderBatch,
    raw_bytes: bytes | bytearray | memoryview,
) -> CompressedTileRenderBatch:
    raw_view = memoryview(raw_bytes)
    raw_nbytes = int(raw_view.nbytes)
    if raw_nbytes <= 0:
        compressed = b""
    else:
        source = raw_view if raw_view.contiguous else raw_view.tobytes()
        compressed = _zstd_compressor().compress(source)
    return CompressedTileRenderBatch(
        tile_key=(int(tile_key[0]), int(tile_key[1]), int(tile_key[2])),
        signature=tuple(tuple(int(value) for value in coord) for coord in getattr(batch, "signature", ())),
        vertex_count=int(getattr(batch, "vertex_count", 0)),
        bounds=tuple(float(value) for value in getattr(batch, "bounds", (0.0, 0.0, 0.0, 0.0))),
        chunk_count=int(getattr(batch, "chunk_count", 0)),
        complete_tile=bool(getattr(batch, "complete_tile", False)),
        visible_mask=int(getattr(batch, "visible_mask", 0)),
        source_version=int(getattr(batch, "source_version", 0)),
        next_refresh_at=float(getattr(batch, "next_refresh_at", 0.0)),
        visible_chunk_count=int(getattr(batch, "visible_chunk_count", 0)),
        merged_chunk_count=int(getattr(batch, "merged_chunk_count", 0)),
        visible_vertex_count=int(getattr(batch, "visible_vertex_count", 0)),
        raw_nbytes=raw_nbytes,
        payload=compressed,
    )


def decompress_tile_render_batch_bytes(batch: CompressedTileRenderBatch) -> bytes:
    if int(batch.raw_nbytes) <= 0:
        return b""
    return _zstd_decompressor().decompress(batch.payload, max_output_size=int(batch.raw_nbytes))


def _store_compressed_tile_batch(renderer, compressed: CompressedTileRenderBatch) -> None:
    _ensure_tile_zstd_state(renderer)
    renderer._tile_zstd_cache[compressed.tile_key] = compressed
    renderer._tile_zstd_cache.move_to_end(compressed.tile_key)
    _enforce_tile_zstd_cache_limit(renderer)


def _batch_is_tile_zstd_candidate(batch: ChunkRenderBatch) -> bool:
    if not bool(getattr(batch, "owns_vertex_buffer", False)):
        return False
    if getattr(batch, "vertex_buffer", None) is None:
        return False
    if not bool(getattr(batch, "all_mature", False)):
        return False
    if int(getattr(batch, "vertex_count", 0)) <= 0:
        return False
    render_batches = tuple(getattr(batch, "cached_render_batches", ()) or ())
    if len(render_batches) != 1:
        return False
    render_buffer, binding_offset, vertex_count, first_vertex = render_batches[0]
    return (
        render_buffer is batch.vertex_buffer
        and int(binding_offset) == 0
        and int(first_vertex) == 0
        and int(vertex_count) == int(getattr(batch, "vertex_count", 0))
    )


def schedule_tile_zstd_readback(renderer, tile_key: tuple[int, int, int], batch: ChunkRenderBatch) -> bool:
    if not tile_zstd_enabled(renderer):
        return False
    _ensure_tile_zstd_state(renderer)
    tile_key = (int(tile_key[0]), int(tile_key[1]), int(tile_key[2]))
    if tile_key in renderer._pending_tile_zstd_readback_keys or tile_key in renderer._tile_zstd_cache:
        return False
    if not _batch_is_tile_zstd_candidate(batch):
        return False
    raw_nbytes = int(batch.vertex_count) * int(render_consts.VERTEX_STRIDE)
    if raw_nbytes <= 0:
        return False
    source_buffer = batch.vertex_buffer
    readback_buffer = renderer.device.create_buffer(
        size=max(1, raw_nbytes),
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    try:
        encoder = renderer.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(source_buffer, 0, readback_buffer, 0, raw_nbytes)
        renderer.device.queue.submit([encoder.finish()])
        map_promise = readback_buffer.map_async(wgpu.MapMode.READ, 0, raw_nbytes)
    except Exception:
        _destroy_buffer(readback_buffer)
        return False
    renderer._pending_tile_zstd_readbacks.append(
        PendingTileZstdReadback(
            tile_key=tile_key,
            batch=batch,
            source_buffer=source_buffer,
            source_capacity_bytes=int(getattr(batch, "owned_vertex_buffer_capacity_bytes", raw_nbytes) or raw_nbytes),
            readback_buffer=readback_buffer,
            raw_nbytes=int(raw_nbytes),
            map_promise=map_promise,
            source_buffer_id=id(source_buffer),
            submitted_at=time.perf_counter(),
        )
    )
    renderer._pending_tile_zstd_readback_keys.add(tile_key)
    return True


def finalize_tile_zstd_readbacks(renderer, *, budget: int = DEFAULT_FINALIZE_BUDGET) -> int:
    if not tile_zstd_enabled(renderer):
        return 0
    _ensure_tile_zstd_state(renderer)
    completed = 0
    remaining = deque()
    while renderer._pending_tile_zstd_readbacks:
        pending = renderer._pending_tile_zstd_readbacks.popleft()
        if completed >= int(budget):
            remaining.append(pending)
            continue
        readback_buffer = pending.readback_buffer
        if getattr(readback_buffer, "map_state", "unmapped") != "mapped":
            remaining.append(pending)
            continue

        try:
            mapped = readback_buffer.read_mapped(0, int(pending.raw_nbytes), copy=False)
            compressed = compress_tile_render_batch_bytes(pending.tile_key, pending.batch, mapped)
            _store_compressed_tile_batch(renderer, compressed)
        finally:
            _destroy_buffer(readback_buffer)
            _destroy_buffer(pending.source_buffer)
            renderer._pending_tile_zstd_readback_keys.discard(pending.tile_key)
        completed += 1

    renderer._pending_tile_zstd_readbacks = remaining
    renderer._pending_tile_zstd_readback_keys = {pending.tile_key for pending in remaining}
    return completed


def restore_tile_zstd_batch(
    renderer,
    tile_key: tuple[int, int, int],
    *,
    visible_mask: int,
    source_version: int,
) -> ChunkRenderBatch | None:
    if not tile_zstd_enabled(renderer):
        return None
    _ensure_tile_zstd_state(renderer)
    tile_key = (int(tile_key[0]), int(tile_key[1]), int(tile_key[2]))
    compressed = renderer._tile_zstd_cache.get(tile_key)
    if compressed is None:
        return None
    if int(compressed.source_version) != int(source_version) or int(compressed.visible_mask) != int(visible_mask):
        renderer._tile_zstd_cache.pop(tile_key, None)
        return None
    try:
        raw = decompress_tile_render_batch_bytes(compressed)
        buffer = renderer.device.create_buffer(
            size=max(1, int(compressed.raw_nbytes)),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        if raw:
            renderer.device.queue.write_buffer(buffer, 0, memoryview(raw))
    except Exception:
        renderer._tile_zstd_cache.pop(tile_key, None)
        return None

    draw_batch = ChunkDrawBatch(
        vertex_buffer=buffer,
        binding_offset=0,
        vertex_count=int(compressed.vertex_count),
        first_vertex=0,
        bounds=compressed.bounds,
        chunk_count=int(compressed.merged_chunk_count),
    )
    cached_render_batches = ((buffer, 0, int(compressed.vertex_count), 0),)
    batch = ChunkRenderBatch(
        signature=compressed.signature,
        vertex_count=int(compressed.vertex_count),
        vertex_buffer=buffer,
        bounds=compressed.bounds,
        chunk_count=int(compressed.chunk_count),
        complete_tile=bool(compressed.complete_tile),
        all_mature=True,
        visible_mask=int(compressed.visible_mask),
        source_version=int(compressed.source_version),
        cached_draw_batches=(draw_batch,),
        cached_render_batches=cached_render_batches,
        cached_grouped_render_batches=_group_render_batches(cached_render_batches),
        next_refresh_at=float(compressed.next_refresh_at),
        visible_chunk_count=int(compressed.visible_chunk_count),
        merged_chunk_count=int(compressed.merged_chunk_count),
        visible_vertex_count=int(compressed.visible_vertex_count),
        owns_vertex_buffer=True,
        owned_vertex_buffer_capacity_bytes=int(compressed.raw_nbytes),
    )
    renderer._tile_zstd_cache.pop(tile_key, None)
    renderer._tile_render_batches[tile_key] = batch
    renderer._tile_dirty_keys.discard(tile_key)
    renderer._visible_tile_dirty_keys.discard(tile_key)
    return batch


def clear_tile_zstd_cache(renderer) -> None:
    _ensure_tile_zstd_state(renderer)
    while renderer._pending_tile_zstd_readbacks:
        pending = renderer._pending_tile_zstd_readbacks.popleft()
        _destroy_buffer(getattr(pending, "readback_buffer", None))
        _destroy_buffer(getattr(pending, "source_buffer", None))
    renderer._pending_tile_zstd_readback_keys.clear()
    renderer._tile_zstd_cache.clear()


def tile_zstd_runtime_stats(renderer) -> dict[str, int | bool]:
    _ensure_tile_zstd_state(renderer)
    cache_stats = compressed_tile_render_batches_stats(renderer._tile_zstd_cache.values())
    pending_entries = 0
    pending_raw_bytes = 0
    pending_source_bytes = 0
    for pending in renderer._pending_tile_zstd_readbacks:
        pending_entries += 1
        pending_raw_bytes += int(getattr(pending, "raw_nbytes", 0))
        pending_source_bytes += int(getattr(pending, "source_capacity_bytes", 0))
    return {
        "enabled": bool(tile_zstd_enabled(renderer)),
        "cache_entries": int(cache_stats["entries"]),
        "cache_raw_bytes": int(cache_stats["raw_bytes"]),
        "cache_compressed_bytes": int(cache_stats["compressed_bytes"]),
        "cache_limit": int(_tile_zstd_cache_limit(renderer)),
        "pending_entries": int(pending_entries),
        "pending_raw_bytes": int(pending_raw_bytes),
        "pending_source_bytes": int(pending_source_bytes),
    }


def service_tile_zstd(renderer) -> None:
    if not tile_zstd_enabled(renderer):
        return
    finalize_tile_zstd_readbacks(renderer)
