from __future__ import annotations

import sys
from typing import Any

from rendercanvas.auto import loop


def is_device_lost_error(exc: Exception) -> bool:
    text = str(exc)
    if "Parent device is lost" in text or "device is lost" in text.lower():
        return True
    return exc.__class__.__name__ == "GPUValidationError" and "lost" in text.lower()


def pending_chunk_work_count(renderer: Any) -> int:
    return (
        len(renderer._pending_chunk_coords)
        + len(renderer._pending_voxel_mesh_results)
        + len(renderer._pending_surface_gpu_batches)
        + int(renderer._pending_surface_gpu_batches_chunk_total)
        + len(renderer._pending_gpu_mesh_batches)
    )


def view_ready_for_auto_exit(renderer: Any) -> bool:
    if renderer._device_lost:
        return False
    target_coords = renderer._visible_chunk_coord_set if renderer._visible_chunk_coord_set else set(renderer._visible_chunk_coords)
    if not target_coords:
        return False
    if renderer._visible_missing_coords:
        return False
    if pending_chunk_work_count(renderer) > 0:
        return False
    chunk_cache = renderer.chunk_cache
    visible_nonempty_chunks = 0
    for coord in target_coords:
        mesh = chunk_cache.get(coord)
        if mesh is None:
            return False
        if int(getattr(mesh, "vertex_count", 0)) > 0:
            visible_nonempty_chunks += 1
    if visible_nonempty_chunks <= 0:
        return False
    if int(renderer._last_frame_draw_calls) <= 0:
        return False
    if int(renderer._last_frame_visible_vertices) <= 0:
        return False
    return True


def request_auto_exit(renderer: Any) -> None:
    if renderer._auto_exit_requested:
        return
    renderer._auto_exit_requested = True
    target_count = len(renderer._visible_chunk_coord_set) if renderer._visible_chunk_coord_set else len(renderer._visible_chunk_coords)
    print(
        f"Info: fixed-size render complete; loaded {target_count} target chunks. Closing window.",
        file=sys.stderr,
    )
    try:
        renderer.canvas.close()
    except Exception:
        pass
    stop_loop = getattr(loop, "stop", None)
    if callable(stop_loop):
        try:
            stop_loop()
        except Exception:
            pass
    raise SystemExit(0)


def service_auto_exit(renderer: Any) -> None:
    if not renderer.exit_when_view_ready or renderer._auto_exit_requested:
        return
    if view_ready_for_auto_exit(renderer):
        renderer._auto_exit_frame_count += 1
    else:
        renderer._auto_exit_frame_count = 0
    if renderer._auto_exit_frame_count >= 2:
        request_auto_exit(renderer)
