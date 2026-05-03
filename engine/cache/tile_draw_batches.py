from __future__ import annotations

from collections import OrderedDict
import time

import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkDrawBatch, ChunkMesh
from ..visibility import coord_manager
from .cached_tile_batches import (
    _build_visible_tile_iterable,
    _cached_tile_batch_stats,
    _store_cached_tile_render_batch,
)
from . import tile_zstd
from .direct_render_batches import (
    _draw_batches_to_render_batches,
    _extend_direct_render_batches_grouped,
    _extend_grouped_render_batch_groups,
    _finalize_direct_render_batch_groups,
    _group_render_batches,
    _normalize_direct_render_batches,
)
from .tile_cache_constants import MERGED_TILE_AGE_REFRESH_INTERVAL_SECONDS


def _renderer_module():
    return render_consts


try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


@profile
def build_tile_draw_batches(
    renderer,
    meshes: list[ChunkMesh] | None,
    encoder,
    *,
    age_gate: bool,
    direct_render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]] | None = None,
    direct_render_batch_groups: OrderedDict[tuple[int, int], list[object]] | None = None,
) -> tuple[list[ChunkDrawBatch], int, int, int, float]:
    from .tile_mesh_cache import (
        _flush_merged_tile_buffer_reuse_queue,
        _queue_merged_tile_buffer_for_reuse,
        merge_chunk_bounds,
        merge_tile_meshes,
        tile_key,
        tile_visible_mask,
    )

    renderer_module = _renderer_module()
    merged_tile_min_age_seconds = float(renderer_module.MERGED_TILE_MIN_AGE_SECONDS)

    cache_key: tuple[int, int, int] | None = None
    now = 0.0
    current_tile_keys: set[tuple[int, int, int]]

    if meshes is None:
        visible_tile_dirty_keys = renderer._visible_tile_dirty_keys
        cache_key = (int(renderer._visible_layout_version), int(renderer._visible_tile_mutation_version), 1 if age_gate else 0)
        cached = renderer._cached_tile_draw_batches.get(cache_key)
        if cached is not None and not visible_tile_dirty_keys:
            cached_until, cached_batches, cached_merged, cached_visible, cached_vertices = cached
            if cached_until <= 0.0:
                return cached_batches, cached_merged, cached_visible, cached_vertices, 0.0
            now = time.perf_counter()
            if now < cached_until:
                return cached_batches, cached_merged, cached_visible, cached_vertices, cached_until
        if not renderer._visible_chunk_coords:
            coord_manager.refresh_visible_chunk_coords(renderer)
            visible_tile_dirty_keys = renderer._visible_tile_dirty_keys
        visible_tile_active_meshes = getattr(renderer, "_visible_tile_active_meshes", None) or {}
        current_tile_keys = getattr(renderer, "_visible_tile_key_set", None) or set(getattr(renderer, "_visible_tile_keys", None) or ())
    else:
        tile_groups: dict[tuple[int, int, int], list[ChunkMesh]] = {}
        for mesh in meshes:
            if mesh.vertex_count <= 0:
                continue
            tile_key_value = tile_key(renderer, mesh.chunk_x, mesh.chunk_z, getattr(mesh, "chunk_y", 0))
            group = tile_groups.get(tile_key_value)
            if group is None:
                group = []
                tile_groups[tile_key_value] = group
            group.append(mesh)
        current_tile_keys = set(tile_groups.keys())
        visible_tile_dirty_keys = set()

    tile_render_batches = renderer._tile_render_batches
    if meshes is None:
        visible_layout_version = int(getattr(renderer, "_visible_layout_version", 0))
        last_cleanup_layout_version = int(getattr(renderer, "_tile_render_batch_cleanup_layout_version", -1))
        if last_cleanup_layout_version != visible_layout_version:
            stale_keys = [tile_key_value for tile_key_value in tile_render_batches if tile_key_value not in current_tile_keys]
            for tile_key_value in stale_keys:
                batch = tile_render_batches.pop(tile_key_value)
                if getattr(batch, "owns_vertex_buffer", False) and batch.vertex_buffer is not None:
                    if not tile_zstd.schedule_tile_zstd_readback(renderer, tile_key_value, batch):
                        _queue_merged_tile_buffer_for_reuse(
                            renderer,
                            batch.vertex_buffer,
                            int(
                                getattr(batch, "owned_vertex_buffer_capacity_bytes", 0)
                                or (int(getattr(batch, "vertex_count", 0)) * int(renderer_module.VERTEX_STRIDE))
                            ),
                        )
            renderer._tile_render_batch_cleanup_layout_version = visible_layout_version
    else:
        stale_keys = [tile_key_value for tile_key_value in tile_render_batches if tile_key_value not in current_tile_keys]
        for tile_key_value in stale_keys:
            batch = tile_render_batches.pop(tile_key_value)
            if getattr(batch, "owns_vertex_buffer", False) and batch.vertex_buffer is not None:
                if not tile_zstd.schedule_tile_zstd_readback(renderer, tile_key_value, batch):
                    renderer._transient_render_buffers.append([batch.vertex_buffer])

    draw_batches: list[ChunkDrawBatch] = []
    merged_chunk_count = 0
    visible_chunk_count = 0
    next_refresh_at = 0.0
    merged_tile_max_chunks = int(renderer_module.MERGED_TILE_MAX_CHUNKS)
    visible_masks = getattr(renderer, "_visible_tile_masks", {})
    if age_gate and now <= 0.0:
        now = time.perf_counter()

    visible_vertex_count = 0

    if meshes is None:
        visible_active_tile_keys = getattr(renderer, "_visible_active_tile_keys", None) or ()
        active_key_set = getattr(renderer, "_visible_active_tile_key_set", None) or set(visible_active_tile_keys)
        tile_iterable = _build_visible_tile_iterable(
            visible_active_tile_keys,
            active_key_set,
            tile_render_batches,
            visible_tile_dirty_keys,
        )
        tile_render_batches_get = tile_render_batches.get
        tile_versions_get = renderer._tile_versions.get
        visible_masks_get = visible_masks.get
        visible_tile_active_meshes_get = visible_tile_active_meshes.get
        visible_tile_dirty_contains = visible_tile_dirty_keys.__contains__
    else:
        tile_iterable = sorted(tile_groups)
        tile_render_batches_get = tile_render_batches.get
        tile_versions_get = renderer._tile_versions.get
        visible_masks_get = visible_masks.get
        visible_tile_active_meshes_get = None
        visible_tile_dirty_contains = visible_tile_dirty_keys.__contains__

    draw_batches_extend = draw_batches.extend
    draw_batches_append = draw_batches.append
    grouped_extend = _extend_direct_render_batches_grouped
    direct_render_batches_extend = direct_render_batches.extend if direct_render_batches is not None else None

    for tile_key_value in tile_iterable:
        existing = tile_render_batches_get(tile_key_value)
        tile_version = int(tile_versions_get(tile_key_value, 0))
        current_visible_mask = int(visible_masks_get(tile_key_value, 0))
        if (
            meshes is None
            and existing is None
            and current_visible_mask != 0
            and not visible_tile_dirty_contains(tile_key_value)
        ):
            existing = tile_zstd.restore_tile_zstd_batch(
                renderer,
                tile_key_value,
                visible_mask=current_visible_mask,
                source_version=tile_version,
            )
        tile_is_dirty = visible_tile_dirty_contains(tile_key_value) or (existing is not None and existing.source_version != tile_version)

        if meshes is None and existing is not None and not tile_is_dirty and existing.visible_mask == current_visible_mask:
            tile_merged, tile_visible, tile_vertices, tile_refresh_at = _cached_tile_batch_stats(existing)
            if tile_visible > 0 and (tile_refresh_at <= 0.0 or now < tile_refresh_at):
                cached_draw_batches = getattr(existing, "cached_draw_batches", ())
                if cached_draw_batches:
                    cached_render_batches = getattr(existing, "cached_render_batches", ())
                    if direct_render_batch_groups is not None:
                        cached_grouped_render_batches = getattr(existing, "cached_grouped_render_batches", ())
                        if cached_grouped_render_batches:
                            _extend_grouped_render_batch_groups(direct_render_batch_groups, cached_grouped_render_batches)
                        else:
                            grouped_extend(direct_render_batch_groups, cached_render_batches)
                    elif direct_render_batches is not None:
                        direct_render_batches_extend(cached_render_batches)
                    else:
                        draw_batches_extend(cached_draw_batches)
                merged_chunk_count += tile_merged
                visible_chunk_count += tile_visible
                visible_vertex_count += tile_vertices
                if tile_refresh_at > 0.0 and (next_refresh_at <= 0.0 or tile_refresh_at < next_refresh_at):
                    next_refresh_at = tile_refresh_at
                continue

        if meshes is None:
            tile_meshes = visible_tile_active_meshes_get(tile_key_value)
            if not tile_meshes:
                if tile_is_dirty:
                    renderer._tile_dirty_keys.discard(tile_key_value)
                    visible_tile_dirty_keys.discard(tile_key_value)
                    stale_batch = tile_render_batches.pop(tile_key_value, None)
                    if stale_batch is not None and getattr(stale_batch, "owns_vertex_buffer", False) and stale_batch.vertex_buffer is not None:
                        _queue_merged_tile_buffer_for_reuse(
                            renderer,
                            stale_batch.vertex_buffer,
                            int(
                                getattr(stale_batch, "owned_vertex_buffer_capacity_bytes", 0)
                                or (int(getattr(stale_batch, "vertex_count", 0)) * int(renderer_module.VERTEX_STRIDE))
                            ),
                        )
                    tile_zstd_cache = getattr(renderer, "_tile_zstd_cache", None)
                    if tile_zstd_cache is not None:
                        tile_zstd_cache.pop(tile_key_value, None)
                continue
        else:
            tile_meshes = tile_groups[tile_key_value]
            if current_visible_mask == 0:
                current_visible_mask = int(tile_visible_mask(renderer, tile_key_value, tile_meshes))

        tile_mesh_count = len(tile_meshes)
        if (
            existing is not None
            and not tile_is_dirty
            and existing.chunk_count == tile_mesh_count
            and existing.visible_mask == current_visible_mask
        ):
            tile_merged, tile_visible, tile_vertices, tile_refresh_at = _cached_tile_batch_stats(existing)
            if tile_visible > 0 and (tile_refresh_at <= 0.0 or now < tile_refresh_at):
                cached_draw_batches = getattr(existing, "cached_draw_batches", ())
                if cached_draw_batches:
                    if direct_render_batch_groups is not None:
                        cached_grouped_render_batches = getattr(existing, "cached_grouped_render_batches", ())
                        if cached_grouped_render_batches:
                            _extend_grouped_render_batch_groups(direct_render_batch_groups, cached_grouped_render_batches)
                        else:
                            _extend_direct_render_batches_grouped(direct_render_batch_groups, getattr(existing, "cached_render_batches", ()))
                    elif direct_render_batches is not None:
                        direct_render_batches_extend(getattr(existing, "cached_render_batches", ()))
                    else:
                        draw_batches.extend(cached_draw_batches)
                merged_chunk_count += tile_merged
                visible_chunk_count += tile_visible
                visible_vertex_count += tile_vertices
                if tile_refresh_at > 0.0 and (next_refresh_at <= 0.0 or tile_refresh_at < next_refresh_at):
                    next_refresh_at = tile_refresh_at
                continue

        mature_meshes: list[ChunkMesh] = []
        immature_meshes: list[ChunkMesh] = []
        tile_next_refresh = 0.0
        for mesh in tile_meshes:
            if age_gate and now - float(mesh.created_at) < merged_tile_min_age_seconds:
                immature_meshes.append(mesh)
                mesh_refresh = float(mesh.created_at) + merged_tile_min_age_seconds
                if tile_next_refresh <= 0.0 or mesh_refresh < tile_next_refresh:
                    tile_next_refresh = mesh_refresh
            else:
                mature_meshes.append(mesh)
        if meshes is not None:
            mature_meshes.sort(key=lambda mesh: (mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z))
            immature_meshes.sort(key=lambda mesh: (mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z))
        if tile_next_refresh > 0.0 and (next_refresh_at <= 0.0 or tile_next_refresh < next_refresh_at):
            next_refresh_at = tile_next_refresh

        if not bool(getattr(renderer, "tile_merging_enabled", getattr(renderer_module, "TILE_MERGING_ENABLED", False))):
            tile_draw_batches: list[ChunkDrawBatch] = []
            tile_visible_vertex_count = 0
            for mesh in tile_meshes:
                tile_draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                tile_visible_vertex_count += int(mesh.vertex_count)
            tile_cached_draw_batches = tuple(tile_draw_batches)
            tile_cached_render_batches = _draw_batches_to_render_batches(tile_draw_batches)
            batch = _store_cached_tile_render_batch(
                renderer,
                tile_key_value,
                existing,
                signature=tuple((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z) for mesh in tile_meshes),
                vertex_count=tile_visible_vertex_count,
                vertex_buffer=tile_draw_batches[0].vertex_buffer if tile_draw_batches else None,
                bounds=merge_chunk_bounds(renderer, tile_meshes),
                chunk_count=tile_mesh_count,
                complete_tile=False,
                all_mature=(len(immature_meshes) == 0),
                visible_mask=current_visible_mask,
                source_version=tile_version,
                cached_draw_batches=tile_cached_draw_batches,
                cached_render_batches=tile_cached_render_batches,
                cached_grouped_render_batches=_group_render_batches(tile_cached_render_batches),
                next_refresh_at=tile_next_refresh,
                visible_chunk_count=tile_mesh_count,
                merged_chunk_count=0,
                visible_vertex_count=tile_visible_vertex_count,
                owns_vertex_buffer=False,
            )
            if direct_render_batch_groups is not None:
                _extend_grouped_render_batch_groups(direct_render_batch_groups, batch.cached_grouped_render_batches)
            elif direct_render_batches is not None:
                direct_render_batches_extend(batch.cached_render_batches)
            else:
                draw_batches_extend(tile_draw_batches)
            visible_chunk_count += tile_mesh_count
            visible_vertex_count += tile_visible_vertex_count
            if tile_next_refresh > 0.0 and (next_refresh_at <= 0.0 or tile_next_refresh < next_refresh_at):
                next_refresh_at = tile_next_refresh
            continue

        if tile_mesh_count == 1:
            mesh = tile_meshes[0]
            single_draw_batch = ChunkDrawBatch(
                vertex_buffer=mesh.vertex_buffer,
                binding_offset=mesh.binding_offset,
                vertex_count=mesh.vertex_count,
                first_vertex=mesh.first_vertex,
                bounds=mesh.bounds,
            )
            single_cached_render_batches = ((mesh.vertex_buffer, int(mesh.binding_offset), int(mesh.vertex_count), int(mesh.first_vertex)),)
            batch = _store_cached_tile_render_batch(
                renderer,
                tile_key_value,
                existing,
                signature=((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z),),
                vertex_count=int(mesh.vertex_count),
                vertex_buffer=mesh.vertex_buffer,
                bounds=mesh.bounds,
                chunk_count=1,
                complete_tile=False,
                all_mature=not age_gate or now - float(mesh.created_at) >= merged_tile_min_age_seconds,
                visible_mask=current_visible_mask,
                source_version=tile_version,
                cached_draw_batches=(single_draw_batch,),
                cached_render_batches=single_cached_render_batches,
                cached_grouped_render_batches=_group_render_batches(single_cached_render_batches),
                next_refresh_at=0.0 if (not age_gate or now - float(mesh.created_at) >= merged_tile_min_age_seconds) else float(mesh.created_at) + merged_tile_min_age_seconds,
                visible_chunk_count=1,
                merged_chunk_count=0,
                visible_vertex_count=int(mesh.vertex_count),
                owns_vertex_buffer=False,
            )
            if direct_render_batch_groups is not None:
                _extend_grouped_render_batch_groups(direct_render_batch_groups, batch.cached_grouped_render_batches)
            elif direct_render_batches is not None:
                direct_render_batches_extend(batch.cached_render_batches)
            else:
                draw_batches_append(single_draw_batch)
            visible_chunk_count += 1
            visible_vertex_count += int(mesh.vertex_count)
            mesh_refresh_at = 0.0 if (not age_gate or now - float(mesh.created_at) >= merged_tile_min_age_seconds) else float(mesh.created_at) + merged_tile_min_age_seconds
            if mesh_refresh_at > 0.0 and (next_refresh_at <= 0.0 or mesh_refresh_at < next_refresh_at):
                next_refresh_at = mesh_refresh_at
            continue

        if len(mature_meshes) < 2:
            tile_draw_batches: list[ChunkDrawBatch] = []
            tile_visible_chunk_count = 0
            tile_visible_vertex_count = 0
            for mesh in immature_meshes:
                tile_draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                tile_visible_chunk_count += 1
                tile_visible_vertex_count += int(mesh.vertex_count)
            for mesh in mature_meshes:
                tile_draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                tile_visible_chunk_count += 1
                tile_visible_vertex_count += int(mesh.vertex_count)
            tile_cached_draw_batches = tuple(tile_draw_batches)
            tile_cached_render_batches = _draw_batches_to_render_batches(tile_draw_batches)
            batch = _store_cached_tile_render_batch(
                renderer,
                tile_key_value,
                existing,
                signature=tuple((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z) for mesh in tile_meshes),
                vertex_count=tile_visible_vertex_count,
                vertex_buffer=tile_draw_batches[0].vertex_buffer if tile_draw_batches else None,
                bounds=merge_chunk_bounds(renderer, tile_meshes),
                chunk_count=tile_visible_chunk_count,
                complete_tile=False,
                all_mature=(len(immature_meshes) == 0),
                visible_mask=current_visible_mask,
                source_version=tile_version,
                cached_draw_batches=tile_cached_draw_batches,
                cached_render_batches=tile_cached_render_batches,
                cached_grouped_render_batches=_group_render_batches(tile_cached_render_batches),
                next_refresh_at=tile_next_refresh,
                visible_chunk_count=tile_visible_chunk_count,
                merged_chunk_count=0,
                visible_vertex_count=tile_visible_vertex_count,
                owns_vertex_buffer=False,
            )
            if direct_render_batch_groups is not None:
                _extend_grouped_render_batch_groups(direct_render_batch_groups, batch.cached_grouped_render_batches)
            elif direct_render_batches is not None:
                direct_render_batches_extend(batch.cached_render_batches)
            else:
                draw_batches_extend(tile_draw_batches)
            visible_chunk_count += tile_visible_chunk_count
            visible_vertex_count += tile_visible_vertex_count
            if tile_next_refresh > 0.0 and (next_refresh_at <= 0.0 or tile_next_refresh < next_refresh_at):
                next_refresh_at = tile_next_refresh
            continue

        signature = tuple((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z) for mesh in mature_meshes)
        batch_vertex_count = sum(mesh.vertex_count for mesh in mature_meshes)
        batch_vertex_bytes = int(batch_vertex_count) * int(renderer_module.VERTEX_STRIDE)
        batch_bounds = merge_chunk_bounds(renderer, mature_meshes)
        rebuild_merged = (
            existing is None
            or tile_is_dirty
            or existing.signature != signature
            or existing.vertex_count != batch_vertex_count
            or existing.chunk_count != len(mature_meshes)
            or existing.vertex_buffer is None
            or not getattr(existing, "owns_vertex_buffer", False)
        )
        merged_buffer = existing.vertex_buffer if not rebuild_merged else None
        merged_buffer_capacity_bytes = int(batch_vertex_bytes)
        if not rebuild_merged and existing is not None:
            merged_buffer_capacity_bytes = int(
                getattr(existing, "owned_vertex_buffer_capacity_bytes", batch_vertex_bytes)
                or batch_vertex_bytes
            )
        if rebuild_merged:
            reuse_merged_buffer = None
            reuse_capacity_bytes = 0
            if (
                existing is not None
                and getattr(existing, "owns_vertex_buffer", False)
                and existing.vertex_buffer is not None
            ):
                existing_capacity_bytes = int(
                    getattr(existing, "owned_vertex_buffer_capacity_bytes", 0)
                    or (int(existing.vertex_count) * int(renderer_module.VERTEX_STRIDE))
                )
                if existing_capacity_bytes >= batch_vertex_bytes:
                    reuse_merged_buffer = existing.vertex_buffer
                    reuse_capacity_bytes = existing_capacity_bytes
            merged_buffer, merged_buffer_capacity_bytes = merge_tile_meshes(
                renderer,
                mature_meshes,
                encoder,
                reuse_merged_buffer,
                reuse_capacity_bytes,
            )

        tile_draw_batches: list[ChunkDrawBatch] = [
            ChunkDrawBatch(
                vertex_buffer=merged_buffer,
                binding_offset=0,
                vertex_count=batch_vertex_count,
                first_vertex=0,
                bounds=batch_bounds,
                chunk_count=len(mature_meshes),
            )
        ]
        tile_visible_vertex_count = int(batch_vertex_count)
        for mesh in immature_meshes:
            tile_draw_batches.append(
                ChunkDrawBatch(
                    vertex_buffer=mesh.vertex_buffer,
                    binding_offset=mesh.binding_offset,
                    vertex_count=mesh.vertex_count,
                    first_vertex=mesh.first_vertex,
                    bounds=mesh.bounds,
                )
            )
            tile_visible_vertex_count += int(mesh.vertex_count)

        tile_cached_draw_batches = tuple(tile_draw_batches)
        tile_cached_render_batches = _draw_batches_to_render_batches(tile_draw_batches)
        batch = _store_cached_tile_render_batch(
            renderer,
            tile_key_value,
            existing,
            signature=signature,
            vertex_count=batch_vertex_count,
            vertex_buffer=merged_buffer,
            bounds=batch_bounds,
            chunk_count=len(mature_meshes),
            complete_tile=(tile_mesh_count == merged_tile_max_chunks and len(mature_meshes) == tile_mesh_count),
            all_mature=(len(mature_meshes) == tile_mesh_count),
            visible_mask=current_visible_mask,
            source_version=tile_version,
            cached_draw_batches=tile_cached_draw_batches,
            cached_render_batches=tile_cached_render_batches,
            cached_grouped_render_batches=_group_render_batches(tile_cached_render_batches),
            next_refresh_at=tile_next_refresh,
            visible_chunk_count=tile_mesh_count,
            merged_chunk_count=len(mature_meshes),
            visible_vertex_count=tile_visible_vertex_count,
            owns_vertex_buffer=True,
            owned_vertex_buffer_capacity_bytes=merged_buffer_capacity_bytes,
        )
        if direct_render_batch_groups is not None:
            _extend_grouped_render_batch_groups(direct_render_batch_groups, batch.cached_grouped_render_batches)
        elif direct_render_batches is not None:
            direct_render_batches_extend(batch.cached_render_batches)
        else:
            draw_batches_extend(batch.cached_draw_batches)
        merged_chunk_count += batch.merged_chunk_count
        visible_chunk_count += batch.visible_chunk_count
        visible_vertex_count += batch.visible_vertex_count
        if tile_next_refresh > 0.0 and (next_refresh_at <= 0.0 or tile_next_refresh < next_refresh_at):
            next_refresh_at = tile_next_refresh

    _flush_merged_tile_buffer_reuse_queue(renderer)
    while len(renderer._transient_render_buffers) > 3:
        old_buffers = renderer._transient_render_buffers.pop(0)
        for buffer in old_buffers:
            buffer.destroy()

    if cache_key is not None and not renderer._visible_tile_dirty_keys:
        cache_until = float(next_refresh_at)
        if cache_until > 0.0:
            if now <= 0.0:
                now = time.perf_counter()
            cache_until = max(cache_until, now + MERGED_TILE_AGE_REFRESH_INTERVAL_SECONDS)
        renderer._cached_tile_draw_batches[cache_key] = (
            cache_until,
            draw_batches,
            merged_chunk_count,
            visible_chunk_count,
            visible_vertex_count,
        )
    return draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count, next_refresh_at
