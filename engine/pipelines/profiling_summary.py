from __future__ import annotations

import time

from .. import render_contract as render_consts
from ..cache import mesh_allocator as mesh_cache
from ..rendering import worldspace_rc
from .hud_overlay import _ensure_hud_vertex_buffer, build_frame_breakdown_hud_vertices, build_profile_hud_vertices
from .profiling_stats import frame_breakdown_average, profile_average_fps, profile_frame_time_percentiles


def _renderer_module():
    """Return stable render constants without importing the renderer runtime."""
    return render_consts


def refresh_profile_summary(renderer, now: float) -> None:
    renderer_module = _renderer_module()
    avg_cpu_ms = renderer.profile_window_cpu_ms / max(1, renderer.profile_window_frames)
    avg_fps = profile_average_fps(renderer)
    frame_p50_ms, frame_p95_ms, frame_p99_ms = profile_frame_time_percentiles(renderer)

    slab_count, slab_total_bytes, slab_used_bytes, slab_free_bytes, slab_largest_free_bytes, slab_alloc_count = mesh_cache.mesh_output_allocator_stats(renderer)
    lines = [
        f"AVG FPS {avg_fps:5.1f}  CPU {avg_cpu_ms:5.1f}MS",
        f"FRAME P50 {frame_p50_ms:5.1f}MS  P95 {frame_p95_ms:5.1f}MS  P99 {frame_p99_ms:5.1f}MS",
        f"RENDER API  {renderer.render_api_label}",
        f"RENDER BACKEND {renderer.render_backend_label}",
        f"ENGINE MODE {renderer.engine_mode_label}",
        f"PRESENT     FPS {renderer_module.SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if renderer_module.SWAPCHAIN_USE_VSYNC else 'OFF'}",
        f"MESH SLABS {slab_count:2d}  USED {slab_used_bytes / (1024.0 * 1024.0):4.1f} MIB  FREE {slab_free_bytes / (1024.0 * 1024.0):4.1f} MIB",
        f"MESH BIGGEST GAP {slab_largest_free_bytes / (1024.0 * 1024.0):4.1f} MIB  ALLOCS {slab_alloc_count:3d}",
    ]
    if lines != renderer.profile_hud_lines or renderer.profile_hud_vertex_count <= 0:
        renderer.profile_hud_lines = lines
        renderer.profile_hud_vertex_bytes, renderer.profile_hud_vertex_count = build_profile_hud_vertices(renderer, lines)
        _ensure_hud_vertex_buffer(renderer, "profile_hud", renderer.profile_hud_vertex_bytes)
    renderer.profile_window_start = now
    renderer.profile_next_report = now + renderer_module.PROFILE_REPORT_INTERVAL
    renderer.profile_window_cpu_ms = 0.0
    renderer.profile_window_frames = 0
    renderer.profile_window_frame_times = []


def refresh_frame_breakdown_summary(renderer, now: float | None = None) -> None:
    if not renderer.profiling_enabled:
        return
    if now is None:
        now = time.perf_counter()
    next_refresh = float(getattr(renderer, "_frame_breakdown_next_refresh", 0.0))
    if renderer.frame_breakdown_vertex_count > 0 and now < next_refresh:
        return
    renderer._frame_breakdown_next_refresh = float(now) + 0.2
    renderer_module = _renderer_module()
    avg_world_update = frame_breakdown_average(renderer, "world_update")
    avg_visibility_lookup = frame_breakdown_average(renderer, "visibility_lookup")
    avg_chunk_stream = frame_breakdown_average(renderer, "chunk_stream")
    avg_chunk_stream_bytes = frame_breakdown_average(renderer, "chunk_stream_bytes")
    avg_new_displayed_chunks = frame_breakdown_average(renderer, "chunk_displayed_added")
    avg_camera_upload = frame_breakdown_average(renderer, "camera_upload")
    avg_swapchain_acquire = frame_breakdown_average(renderer, "swapchain_acquire")
    avg_render_encode = frame_breakdown_average(renderer, "render_encode")
    avg_command_finish = frame_breakdown_average(renderer, "command_finish")
    avg_queue_submit = frame_breakdown_average(renderer, "queue_submit")
    avg_wall_frame = frame_breakdown_average(renderer, "wall_frame")
    pending_chunk_requests = int(round(frame_breakdown_average(renderer, "pending_chunk_requests")))
    visible_vertices = int(round(frame_breakdown_average(renderer, "visible_vertices")))
    avg_issue_encode = (
        avg_world_update
        + avg_visibility_lookup
        + avg_chunk_stream
        + avg_camera_upload
        + avg_swapchain_acquire
        + avg_render_encode
        + avg_command_finish
        + avg_queue_submit
    )
    draw_calls = int(round(frame_breakdown_average(renderer, "draw_calls")))
    merged_chunks = int(round(frame_breakdown_average(renderer, "merged_chunks")))
    visible_chunk_targets = int(round(frame_breakdown_average(renderer, "visible_chunk_targets")))
    visible_chunks = int(round(frame_breakdown_average(renderer, "visible_chunks")))
    visible_but_not_ready = max(0, visible_chunk_targets - visible_chunks)
    chunk_memory_bytes = mesh_cache.chunk_cache_memory_bytes(renderer)
    chunk_memory_mib = chunk_memory_bytes / (1024.0 * 1024.0)
    slab_count, slab_total_bytes, slab_used_bytes, slab_free_bytes, slab_largest_free_bytes, slab_alloc_count = mesh_cache.mesh_output_allocator_stats(renderer)
    chunk_stream_bandwidth_mib_s = 0.0
    if avg_chunk_stream > 0.0:
        chunk_stream_bandwidth_mib_s = (avg_chunk_stream_bytes / (1024.0 * 1024.0)) / max(avg_chunk_stream / 1000.0, 1e-9)
    chunk_generation_per_s = 0.0
    if avg_wall_frame > 0.0:
        chunk_generation_per_s = avg_new_displayed_chunks / max(avg_wall_frame / 1000.0, 1e-9)

    camera_x = float(renderer.camera.position[0])
    camera_y = float(renderer.camera.position[1])
    camera_z = float(renderer.camera.position[2])
    camera_block_x = camera_x / max(renderer.world.block_size, 1e-9)
    camera_block_y = camera_y / max(renderer.world.block_size, 1e-9)
    camera_block_z = camera_z / max(renderer.world.block_size, 1e-9)

    rc_debug_mode = int(getattr(renderer, "rc_debug_mode", 0))
    rc_debug_names = tuple(getattr(renderer, "rc_debug_mode_names", ("off",)))
    rc_debug_name = rc_debug_names[rc_debug_mode] if 0 <= rc_debug_mode < len(rc_debug_names) else "unknown"
    rc_frame = int(getattr(renderer, "_worldspace_rc_frame_index", 0))
    rc_burst = int(getattr(renderer, "_worldspace_rc_convergence_frames_remaining", 0))
    rc_kind = str(getattr(renderer, "_worldspace_rc_last_update_kind", "unknown"))

    def _fmt_cascade_list(values) -> str:
        vals = [int(v) for v in list(values or [])]
        return "-" if not vals else ",".join(f"C{v}" for v in vals)

    rc_scheduled = _fmt_cascade_list(getattr(renderer, "_worldspace_rc_last_scheduled_updates", []))
    rc_dirty = _fmt_cascade_list(getattr(renderer, "_worldspace_rc_last_dirty_indices", []))
    rc_reject = _fmt_cascade_list(getattr(renderer, "_worldspace_rc_last_history_reject_updates", []))
    rc_last_updates = list(getattr(renderer, "_worldspace_rc_last_update_frame", []))
    rc_ages: list[str] = []
    for idx in range(4):
        try:
            last_update = int(rc_last_updates[idx])
        except Exception:
            last_update = -1000000
        rc_ages.append("--" if last_update < -999000 else str(max(0, rc_frame - last_update)))
    rc_age_text = "/".join(rc_ages)
    rc_resolution = int(getattr(render_consts, "WORLDSPACE_RC_GRID_RESOLUTION", 0))
    rc_directions = int(getattr(render_consts, "WORLDSPACE_RC_DIRECTION_COUNT", 0))
    rc_filter_passes = int(getattr(render_consts, "WORLDSPACE_RC_SPATIAL_FILTER_PASSES", 0))
    rc_temporal_alpha = float(getattr(render_consts, "WORLDSPACE_RC_TEMPORAL_BLEND_ALPHA", 0.0))
    rc_interval_bands = list(getattr(renderer, "_worldspace_rc_last_interval_bands", []))
    if len(rc_interval_bands) < 4:
        rc_interval_bands = [worldspace_rc.interval_band(i) for i in range(4)]
    rc_interval_text = " ".join(
        f"C{idx}:{float(band[0]):.1f}-{float(band[1]):.1f}"
        for idx, band in enumerate(rc_interval_bands[:4])
    )
    rc_snapshot_path = str(getattr(renderer, "_worldspace_rc_last_snapshot_path", "") or "-")
    if len(rc_snapshot_path) > 54:
        rc_snapshot_path = "..." + rc_snapshot_path[-51:]

    lines = [
        f"FRAME BREAKDOWN @ DIMENSION {renderer.render_dimension_chunks}x{renderer.render_dimension_chunks} CHUNKS",
        f"MOVE SPEED: {renderer._current_move_speed / max(renderer.world.block_size, 1e-9):5.1f} B/S",
        f"CAM POS M: {camera_x:7.2f} {camera_y:7.2f} {camera_z:7.2f}",
        f"CAM POS B: {camera_block_x:7.1f} {camera_block_y:7.1f} {camera_block_z:7.1f}",
        f"RENDER BACKEND: {renderer.render_backend_label}",
        f"TERRAIN BACKEND: {renderer.world.terrain_backend_label()}",
        f"MESH BACKEND: {renderer.mesh_backend_label}",
        f"CHUNK DIMS: {renderer_module.CHUNK_SIZE}x{renderer_module.CHUNK_SIZE}x{renderer_module.CHUNK_SIZE}",
        f"BACKEND POLL SIZE: {renderer.terrain_batch_size}",
        f"MESH DRAIN SIZE: {renderer.mesh_batch_size}",
        f"RC: {'ON' if renderer.radiance_cascades_enabled else 'OFF'} DEBUG {rc_debug_mode}:{rc_debug_name}",
        f"RC FIELD: RES {rc_resolution} DIRS {rc_directions} FILTER {rc_filter_passes} TEMP {rc_temporal_alpha:.2f}",
        f"RC UPDATE: {rc_kind.upper()} SCHED {rc_scheduled} DIRTY {rc_dirty} REJECT {rc_reject} BURST {rc_burst}",
        f"RC AGE C0/C1/C2/C3: {rc_age_text} FRAMES",
        f"RC INTERVALS: {rc_interval_text}",
        f"RC SNAPSHOT F7: {rc_snapshot_path}",
        f"MESH SLABS: {slab_count}  USED {slab_used_bytes / (1024.0 * 1024.0):4.1f} MIB  FREE {slab_free_bytes / (1024.0 * 1024.0):4.1f} MIB",
        f"MESH BIGGEST GAP: {slab_largest_free_bytes / (1024.0 * 1024.0):4.1f} MIB  ALLOCS {slab_alloc_count}",
        f"CPU FRAME ISSUE: {avg_issue_encode:5.1f} MS",
        f"  WORLD UPDATE: {avg_world_update:5.1f} MS",
        f"  VISIBILITY LOOKUP: {avg_visibility_lookup:5.1f} MS",
        f"  CHUNK STREAM: {avg_chunk_stream:5.1f} MS",
        f"  CHUNK STREAM BANDWIDTH: {chunk_stream_bandwidth_mib_s:5.1f} MIB/S",
        f"  NEW GENERATED CHUNKS / S: {chunk_generation_per_s:5.1f}",
        f"  CAMERA UPLOAD: {avg_camera_upload:5.1f} MS",
        f"  SWAPCHAIN ACQUIRE: {avg_swapchain_acquire:5.1f} MS",
        f"  RENDER ENCODE: {avg_render_encode:5.1f} MS",
        f"  COMMAND FINISH: {avg_command_finish:5.1f} MS",
        f"  QUEUE SUBMIT: {avg_queue_submit:5.1f} MS",
        f"WALL FRAME: {avg_wall_frame:5.1f} MS",
        f"CHUNK VRAM: {chunk_memory_bytes:,} BYTES ({chunk_memory_mib:5.2f} MIB)",
        f"TOTAL DRAW VERTICES: {visible_vertices:,}",
        f"VISIBLE BUT NOT READY: {visible_but_not_ready}",
        f"PENDING CHUNK REQUESTS: {pending_chunk_requests}",
        f"DRAW CALLS: {draw_calls}",
        f"VISIBLE MERGED CHUNKS (VISIBLE ONLY): {merged_chunks}",
    ]
    if lines != renderer.frame_breakdown_lines or renderer.frame_breakdown_vertex_count <= 0:
        renderer.frame_breakdown_lines = lines
        renderer.frame_breakdown_vertex_bytes, renderer.frame_breakdown_vertex_count = build_frame_breakdown_hud_vertices(renderer, lines)
        _ensure_hud_vertex_buffer(renderer, "frame_breakdown", renderer.frame_breakdown_vertex_bytes)

