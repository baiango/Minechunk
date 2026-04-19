from __future__ import annotations

import time

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func

@profile
def rebuild_visible_missing_tracking(renderer) -> None:
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
def refresh_visible_chunk_set(renderer) -> float:
    visibility_start = time.perf_counter()
    from ..pipelines.chunk_pipeline import _renderer_module
    renderer_module = _renderer_module()
    current_origin = renderer._current_chunk_origin()
    if renderer._visible_chunk_origin != current_origin or not renderer._visible_chunk_coords:
        renderer._refresh_visible_chunk_coords()
        rebuild_visible_missing_tracking(renderer)
    elif bool(getattr(renderer, "_visible_display_state_dirty", False)):
        rebuild_visible_missing_tracking(renderer)
    renderer._warn_if_visible_exceeds_cache()
    return (time.perf_counter() - visibility_start) * 1000.0


@profile
def refresh_visible_chunk_coords(renderer) -> None:
    return renderer._refresh_visible_chunk_coords()


@profile
def apply_visible_chunk_coord_delta(renderer, new_origin):
    return renderer._apply_visible_chunk_coord_delta(new_origin)
