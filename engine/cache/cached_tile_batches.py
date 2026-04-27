from __future__ import annotations

import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkDrawBatch, ChunkMesh, ChunkRenderBatch


def _renderer_module():
    return render_consts


def _cached_tile_batch_stats(batch: ChunkRenderBatch) -> tuple[int, int, int, float]:
    return (
        int(getattr(batch, "merged_chunk_count", 0)),
        int(getattr(batch, "visible_chunk_count", 0)),
        int(getattr(batch, "visible_vertex_count", 0)),
        float(getattr(batch, "next_refresh_at", 0.0)),
    )


def _build_visible_tile_iterable(
    visible_active_tile_keys: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    active_key_set: set[tuple[int, int, int]],
    tile_render_batches: dict[tuple[int, int, int], ChunkRenderBatch],
    visible_tile_dirty_keys: set[tuple[int, int, int]],
) -> list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...]:
    if not visible_tile_dirty_keys:
        return visible_active_tile_keys
    tile_iterable = list(visible_active_tile_keys)
    seen = set(tile_iterable)
    seen_add = seen.add
    tile_iterable_append = tile_iterable.append
    for tile_key_value in tile_render_batches.keys():
        if tile_key_value not in seen:
            seen_add(tile_key_value)
            tile_iterable_append(tile_key_value)
    for tile_key_value in visible_tile_dirty_keys:
        if tile_key_value not in seen:
            seen_add(tile_key_value)
            tile_iterable_append(tile_key_value)
    return tile_iterable


def _store_cached_tile_render_batch(
    renderer,
    tile_key_value: tuple[int, int, int],
    existing: ChunkRenderBatch | None,
    *,
    signature: tuple[tuple[int, int, int], ...],
    vertex_count: int,
    vertex_buffer,
    bounds: tuple[float, float, float, float],
    chunk_count: int,
    complete_tile: bool,
    all_mature: bool,
    visible_mask: int,
    source_version: int,
    cached_draw_batches: tuple[ChunkDrawBatch, ...],
    cached_render_batches: tuple[tuple[wgpu.GPUBuffer, int, int, int], ...],
    cached_grouped_render_batches: tuple[tuple[tuple[int, int], wgpu.GPUBuffer, int, tuple[tuple[wgpu.GPUBuffer, int, int, int], ...]], ...],
    next_refresh_at: float,
    visible_chunk_count: int,
    merged_chunk_count: int,
    visible_vertex_count: int,
    owns_vertex_buffer: bool,
    owned_vertex_buffer_capacity_bytes: int = 0,
) -> ChunkRenderBatch:
    from .tile_mesh_cache import _queue_merged_tile_buffer_for_reuse

    old_buffer = existing.vertex_buffer if (existing is not None and getattr(existing, "owns_vertex_buffer", False)) else None
    old_buffer_capacity_bytes = 0
    if old_buffer is not None and existing is not None:
        old_buffer_capacity_bytes = int(
            getattr(existing, "owned_vertex_buffer_capacity_bytes", 0)
            or (int(getattr(existing, "vertex_count", 0)) * int(_renderer_module().VERTEX_STRIDE))
        )
    batch = ChunkRenderBatch(
        signature=signature,
        vertex_count=vertex_count,
        vertex_buffer=vertex_buffer,
        bounds=bounds,
        chunk_count=chunk_count,
        complete_tile=complete_tile,
        all_mature=all_mature,
        visible_mask=visible_mask,
        source_version=source_version,
        cached_draw_batches=cached_draw_batches,
        cached_render_batches=cached_render_batches,
        cached_grouped_render_batches=cached_grouped_render_batches,
        next_refresh_at=float(next_refresh_at),
        visible_chunk_count=int(visible_chunk_count),
        merged_chunk_count=int(merged_chunk_count),
        visible_vertex_count=int(visible_vertex_count),
        owns_vertex_buffer=bool(owns_vertex_buffer),
        owned_vertex_buffer_capacity_bytes=int(owned_vertex_buffer_capacity_bytes),
    )
    renderer._tile_render_batches[tile_key_value] = batch
    renderer._tile_dirty_keys.discard(tile_key_value)
    renderer._visible_tile_dirty_keys.discard(tile_key_value)
    if old_buffer is not None and old_buffer is not vertex_buffer:
        _queue_merged_tile_buffer_for_reuse(renderer, old_buffer, old_buffer_capacity_bytes)
    return batch
