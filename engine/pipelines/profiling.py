from __future__ import annotations

# Compatibility façade for the profiling/HUD subsystem.
# Keep imports through engine.pipelines.profiling working while the
# implementation lives in smaller, responsibility-specific modules.

from .hud_font import (
    HUD_FONT,
    HUD_FONT_FALLBACK,
    _build_hud_font_from_freetype,
    _find_freetype_library_path,
    _find_hud_font_path,
    build_hud_font,
    get_hud_font,
    hud_glyph_rows,
)
from .hud_overlay import (
    _ensure_hud_vertex_buffer,
    build_frame_breakdown_hud_vertices,
    build_hud_vertices,
    build_profile_hud_vertices,
    draw_frame_breakdown_hud,
    draw_hud_overlay,
    draw_profile_hud,
)
from .profiling_stats import (
    frame_breakdown_average,
    profile_average_fps,
    profile_begin_frame,
    profile_end_frame,
    profile_frame_time_percentiles,
    record_frame_breakdown_sample,
)


def refresh_profile_summary(renderer, now: float) -> None:
    from .profiling_summary import refresh_profile_summary as _impl

    return _impl(renderer, now)


def refresh_frame_breakdown_summary(renderer, now: float | None = None) -> None:
    from .profiling_summary import refresh_frame_breakdown_summary as _impl

    return _impl(renderer, now)

__all__ = [
    "HUD_FONT",
    "HUD_FONT_FALLBACK",
    "_build_hud_font_from_freetype",
    "_ensure_hud_vertex_buffer",
    "_find_freetype_library_path",
    "_find_hud_font_path",
    "build_frame_breakdown_hud_vertices",
    "build_hud_font",
    "build_hud_vertices",
    "get_hud_font",
    "build_profile_hud_vertices",
    "draw_frame_breakdown_hud",
    "draw_hud_overlay",
    "draw_profile_hud",
    "frame_breakdown_average",
    "hud_glyph_rows",
    "profile_average_fps",
    "profile_begin_frame",
    "profile_end_frame",
    "profile_frame_time_percentiles",
    "record_frame_breakdown_sample",
    "refresh_frame_breakdown_summary",
    "refresh_profile_summary",
]
