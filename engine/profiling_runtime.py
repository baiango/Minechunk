from __future__ import annotations

import time

from .pipelines import profiling as hud_profile
from .renderer_config import PROFILE_REPORT_INTERVAL, SWAPCHAIN_MAX_FPS, SWAPCHAIN_USE_VSYNC
from .world_constants import CHUNK_SIZE


def toggle(renderer) -> None:
    if renderer.profiling_enabled:
        disable(renderer)
    else:
        enable(renderer)


def enable(renderer) -> None:
    renderer.profiling_enabled = True
    now = time.perf_counter()
    renderer.profile_window_start = now
    renderer.profile_next_report = now + PROFILE_REPORT_INTERVAL
    renderer.profile_window_cpu_ms = 0.0
    renderer.profile_window_frames = 0
    renderer.profile_window_frame_times = []
    for name, samples in renderer.frame_breakdown_samples.items():
        samples.clear()
        renderer.frame_breakdown_sample_sums[name] = 0.0
    renderer.profile_hud_lines = []
    renderer.profile_hud_vertex_bytes = b""
    renderer.profile_hud_vertex_count = 0
    renderer._frame_breakdown_next_refresh = now
    renderer.frame_breakdown_lines = [
        f"FRAME BREAKDOWN @ DIMENSION {renderer.render_dimension_chunks}x{renderer.render_dimension_chunks} CHUNKS",
        "CPU FRAME ISSUE: --.- MS",
        "  WORLD UPDATE: --.- MS",
        "  VISIBILITY LOOKUP: --.- MS",
        "  CHUNK STREAM: --.- MS",
        "  CAMERA UPLOAD: --.- MS",
        "  SWAPCHAIN ACQUIRE: --.- MS",
        "  RENDER ENCODE: --.- MS",
        "  COMMAND FINISH: --.- MS",
        "  QUEUE SUBMIT: --.- MS",
        f"CHUNK DIMS: {CHUNK_SIZE}x{CHUNK_SIZE}x{CHUNK_SIZE}",
        f"BACKEND POLL SIZE: {renderer.terrain_batch_size}",
        f"MESH DRAIN SIZE: {renderer.mesh_batch_size}",
        f"PRESENT PACING: FPS {SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if SWAPCHAIN_USE_VSYNC else 'OFF'}",
        "MESH SLABS: --  USED --.- MIB  FREE --.- MIB",
        "MESH BIGGEST GAP: --.- MIB  ALLOCS --",
        "TOTAL DRAW VERTICES: --",
        "WALL FRAME: --.- MS",
        "PROCESS MEM: FOOT --  RSS --  PEAK --",
        "MEM MAC: INT --  IO --  GFX --  REUSE --  COMP --  RELIEF --",
        "MEM CPU: TRACK --  TERR --  COLL --/--  SCR --",
        "MEM CPU: OTHER FOOT --  OTHER RSS --  RAWQ --/--",
        "MEM GPU: EST --  SLABS --  TILE --  TRANS --",
        "MESH ZSTD: --  CACHE --/-- RAW -- COMP -- RATIO --.-X  PENDING --/--",
        "TILE ZSTD: --  CACHE --/-- RAW -- COMP -- RATIO --.-X  PENDING --/--",
        "MESH COMPACT: SLABS --  ---->--  RECLAIM --  TOTAL ---->--  COPY --  PEND --",
        "DRAW CALLS: --",
        "VISIBLE MERGED CHUNKS (VISIBLE ONLY): --",
    ]
    renderer.frame_breakdown_vertex_bytes, renderer.frame_breakdown_vertex_count = hud_profile.build_frame_breakdown_hud_vertices(renderer, renderer.frame_breakdown_lines)


def disable(renderer) -> None:
    renderer.profiling_enabled = False
    renderer.profile_window_start = 0.0
    renderer.profile_next_report = 0.0
    renderer._frame_breakdown_next_refresh = 0.0
    renderer.profile_window_cpu_ms = 0.0
    renderer.profile_window_frames = 0
    renderer.profile_window_frame_times = []
    renderer.profile_hud_lines = []
    renderer.profile_hud_vertex_bytes = b""
    renderer.profile_hud_vertex_count = 0
    renderer.frame_breakdown_lines = []
    renderer.frame_breakdown_vertex_bytes = b""
    renderer.frame_breakdown_vertex_count = 0
    renderer._hud_geometry_cache.clear()
    for name, samples in renderer.frame_breakdown_samples.items():
        samples.clear()
        renderer.frame_breakdown_sample_sums[name] = 0.0
