from __future__ import annotations

import sys
import time
from typing import Any

from ..renderer_config import MERGED_TILE_SIZE_CHUNKS
from . import tile_layout

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


@profile
def rebuild_visible_missing_tracking(renderer: Any) -> None:
    visible = getattr(renderer, "_visible_chunk_coord_set", None)
    if visible is None:
        visible = set(renderer._visible_chunk_coords)
        renderer._visible_chunk_coord_set = visible

    chunk_cache = renderer.chunk_cache
    displayed = getattr(renderer, "_visible_prefetched_displayed_coords", None)
    if displayed is None:
        displayed = visible.intersection(chunk_cache.keys())
    else:
        renderer._visible_prefetched_displayed_coords = None

    if displayed and len(visible) <= int(getattr(renderer, "max_cached_chunks", 0)):
        move_to_end = chunk_cache.move_to_end
        for coord in displayed:
            move_to_end(coord)

    missing = renderer._visible_missing_coords
    missing.clear()
    missing.update(visible)
    missing.difference_update(displayed)
    pending_chunk_coords = renderer._pending_chunk_coords
    if pending_chunk_coords:
        missing.difference_update(pending_chunk_coords)

    renderer._visible_displayed_coords = displayed
    renderer._visible_display_state_dirty = False
    renderer._chunk_request_target_coords = set()
    renderer._chunk_request_queue.clear()
    renderer._chunk_request_queue_origin = None
    renderer._chunk_request_queue_view_signature = None
    renderer._chunk_request_queue_dirty = True


@profile
def refresh_visible_chunk_set(renderer: Any) -> float:
    visibility_start = time.perf_counter()
    current_origin = renderer._current_chunk_origin()
    if renderer._visible_chunk_origin != current_origin or not renderer._visible_chunk_coords:
        refresh_visible_chunk_coords(renderer)
        rebuild_visible_missing_tracking(renderer)
    elif bool(getattr(renderer, "_visible_display_state_dirty", False)):
        rebuild_visible_missing_tracking(renderer)
    warn_if_visible_exceeds_cache(renderer)
    return (time.perf_counter() - visibility_start) * 1000.0


@profile
def rebuild_visible_tile_layout_from_coords(renderer: Any) -> None:
    origin = renderer._visible_chunk_origin if renderer._visible_chunk_origin is not None else (0, 0, 0)
    (
        _,
        renderer._visible_tile_keys,
        renderer._visible_tile_masks,
        renderer._visible_rel_coord_to_tile_slot,
        renderer._visible_rel_tile_slot_sizes,
        renderer._visible_tile_base,
    ) = tile_layout.tile_layout_in_view_for_origin(renderer, origin)
    renderer._visible_tile_coords = {}


@profile
def apply_visible_chunk_coord_delta(renderer: Any, new_origin: tuple[int, int, int]) -> bool:
    old_origin = renderer._visible_chunk_origin
    if old_origin is None or not renderer._visible_chunk_coords:
        return False
    dx = int(new_origin[0]) - int(old_origin[0])
    dy = int(new_origin[1]) - int(old_origin[1])
    dz = int(new_origin[2]) - int(old_origin[2])
    if dx == 0 and dy == 0 and dz == 0:
        return True
    shifted_coords = [
        (chunk_x + dx, chunk_y + dy, chunk_z + dz)
        for chunk_x, chunk_y, chunk_z in renderer._visible_chunk_coords
    ]
    renderer._visible_chunk_coords = shifted_coords
    renderer._visible_chunk_coord_set = set(shifted_coords)
    return True


@profile
def refresh_visible_chunk_coords(renderer: Any) -> None:
    new_origin = renderer._current_chunk_origin()
    previous_origin = renderer._visible_chunk_origin
    (
        rel_coords,
        renderer._visible_tile_keys,
        renderer._visible_tile_masks,
        renderer._visible_rel_coord_to_tile_slot,
        renderer._visible_rel_tile_slot_sizes,
        renderer._visible_tile_base,
    ) = tile_layout.tile_layout_in_view_for_origin(renderer, new_origin)

    new_rel_y_bounds = tile_layout.visible_rel_y_bounds_for_origin_y(renderer, int(new_origin[1]))
    shifted_visible_coords = False
    if previous_origin is not None and renderer._visible_chunk_coords:
        visible_coord_set = renderer._visible_chunk_coord_set
        visible_coords = renderer._visible_chunk_coords
        if (
            visible_coord_set
            and len(visible_coord_set) == len(visible_coords)
            and renderer._visible_rel_y_bounds == new_rel_y_bounds
        ):
            shifted_visible_coords = apply_visible_chunk_coord_delta(renderer, new_origin)
    renderer._visible_chunk_origin = new_origin
    renderer._visible_rel_y_bounds = new_rel_y_bounds

    if shifted_visible_coords:
        visible_coords = renderer._visible_chunk_coords
        visible_coord_set = renderer._visible_chunk_coord_set
    else:
        origin_x = int(new_origin[0])
        origin_y = int(new_origin[1])
        origin_z = int(new_origin[2])
        visible_coords = [(origin_x + dx, origin_y + dy, origin_z + dz) for dx, dy, dz in rel_coords]
        renderer._visible_chunk_coords = visible_coords
        visible_coord_set = set(visible_coords)
        renderer._visible_chunk_coord_set = visible_coord_set

    renderer._visible_tile_coords = {}
    renderer._visible_tile_key_set = set(renderer._visible_tile_keys)
    # Keep visible slot arrays lazy. The direct render path only needs per-tile
    # active meshes, so eagerly allocating and filling slot arrays here just
    # duplicates bookkeeping on every origin hop.
    renderer._visible_tile_mesh_slots = {}
    renderer._visible_tile_active_meshes = {}
    renderer._visible_active_tile_key_set = set()
    renderer._visible_tile_slot_index_cache = {}

    chunk_cache = renderer.chunk_cache
    displayed_coords = visible_coord_set.intersection(chunk_cache.keys())
    chunk_cache_getitem = chunk_cache.__getitem__

    tile_size = int(MERGED_TILE_SIZE_CHUNKS)
    visible_tile_active_meshes: dict[tuple[int, int, int], list[Any]] = {}

    for coord in displayed_coords:
        mesh = chunk_cache_getitem(coord)
        if mesh.vertex_count <= 0:
            continue
        tile_key_value = (int(coord[0]) // tile_size, int(coord[1]), int(coord[2]) // tile_size)
        active_list = visible_tile_active_meshes.get(tile_key_value)
        if active_list is None:
            active_list = []
            visible_tile_active_meshes[tile_key_value] = active_list
        active_list.append(mesh)

    renderer._visible_tile_active_meshes = visible_tile_active_meshes
    renderer._visible_active_tile_key_set = set(visible_tile_active_meshes)
    renderer._visible_active_tile_keys = [tile_key_value for tile_key_value in renderer._visible_tile_keys if tile_key_value in visible_tile_active_meshes]

    renderer._visible_prefetched_displayed_coords = displayed_coords
    renderer._visible_displayed_coords = set(displayed_coords)
    renderer._visible_display_state_dirty = False
    renderer._visible_tile_dirty_keys = {
        key for key in renderer._tile_dirty_keys if key in renderer._visible_tile_key_set
    }
    renderer._visible_layout_version += 1
    renderer._visible_tile_mutation_version += 1
    renderer._cached_tile_draw_batches.clear()
    renderer._cached_visible_render_batches.clear()


def warn_if_visible_exceeds_cache(renderer: Any) -> None:
    visible_count = len(renderer._visible_chunk_coords)
    if renderer._cache_capacity_warned or visible_count <= renderer.max_cached_chunks:
        return
    renderer._cache_capacity_warned = True
    print(
        f"Warning: visible chunk count ({visible_count}) exceeds cache capacity "
        f"({renderer.max_cached_chunks}). Expect missing chunks, evictions, or flashing.",
        file=sys.stderr,
    )
