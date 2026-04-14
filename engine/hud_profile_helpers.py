from __future__ import annotations

import ctypes
import os
import math
import subprocess
import sys
import time
from ctypes.util import find_library
from pathlib import Path

from . import mesh_cache_helpers as mesh_cache
from . import render_constants as render_consts
import wgpu
from .render_utils import pack_vertex, screen_to_ndc


def _renderer_module():
    from . import renderer as renderer_module

    return renderer_module


HUD_FONT_FALLBACK: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000"),
    "?": ("111", "001", "010", "000", "010"),
    ".": ("000", "000", "000", "000", "010"),
    ",": ("000", "000", "000", "010", "100"),
    ":": ("000", "010", "000", "010", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "/": ("001", "001", "010", "100", "100"),
    "(": ("010", "100", "100", "100", "010"),
    ")": ("010", "001", "001", "001", "010"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "101", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


def _find_hud_font_path() -> str | None:
    bundled_font = Path(__file__).resolve().with_name("res") / "Roboto-VariableFont_wdth,wght.ttf"
    if os.path.exists(bundled_font):
        return str(bundled_font)
    try:
        font_path = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n", "Roboto:style=Regular"],
            text=True,
        ).strip()
        if font_path and os.path.exists(font_path):
            return font_path
    except Exception:
        pass
    for candidate in (
        "/System/Library/Fonts/Supplemental/Roboto Regular.ttf",
        "/Library/Fonts/Roboto Regular.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _find_freetype_library_path() -> str | None:
    candidates: list[str] = []

    found = find_library("freetype")
    if found:
        candidates.append(found)

    if sys.platform == "darwin":
        candidates.extend(
            [
                "/opt/homebrew/opt/freetype/lib/libfreetype.dylib",
                "/usr/local/opt/freetype/lib/libfreetype.dylib",
                "/opt/homebrew/lib/libfreetype.dylib",
                "/usr/local/lib/libfreetype.dylib",
                "/usr/lib/libfreetype.dylib",
            ]
        )
    elif sys.platform.startswith("linux"):
        candidates.extend(
            [
                "/usr/lib/x86_64-linux-gnu/libfreetype.so.6",
                "/usr/lib64/libfreetype.so.6",
                "/usr/lib/libfreetype.so.6",
                "/lib/x86_64-linux-gnu/libfreetype.so.6",
                "/lib64/libfreetype.so.6",
                "/lib/libfreetype.so.6",
            ]
        )
    elif sys.platform.startswith("win"):
        candidates.extend(
            [
                "libfreetype-6.dll",
                "freetype.dll",
            ]
        )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    for candidate in candidates:
        if candidate:
            return candidate

    return None


def _build_hud_font_from_freetype() -> dict[str, tuple[str, ...]]:
    font_path = _find_hud_font_path()
    if not font_path:
        raise RuntimeError("No usable HUD font file found")

    library_path = _find_freetype_library_path()
    if not library_path or not os.path.exists(library_path):
        raise RuntimeError("FreeType library not available")

    freetype = ctypes.CDLL(library_path)
    c_void_p = ctypes.c_void_p

    class FT_Generic(ctypes.Structure):
        _fields_ = [("data", c_void_p), ("finalizer", c_void_p)]

    class FT_Vector(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class FT_BBox(ctypes.Structure):
        _fields_ = [
            ("xMin", ctypes.c_long),
            ("yMin", ctypes.c_long),
            ("xMax", ctypes.c_long),
            ("yMax", ctypes.c_long),
        ]

    class FT_Bitmap(ctypes.Structure):
        _fields_ = [
            ("rows", ctypes.c_uint),
            ("width", ctypes.c_uint),
            ("pitch", ctypes.c_int),
            ("buffer", ctypes.POINTER(ctypes.c_ubyte)),
            ("num_grays", ctypes.c_ushort),
            ("pixel_mode", ctypes.c_ubyte),
            ("palette_mode", ctypes.c_ubyte),
            ("palette", c_void_p),
        ]

    class FT_Glyph_Metrics(ctypes.Structure):
        _fields_ = [
            ("width", ctypes.c_long),
            ("height", ctypes.c_long),
            ("horiBearingX", ctypes.c_long),
            ("horiBearingY", ctypes.c_long),
            ("horiAdvance", ctypes.c_long),
            ("vertBearingX", ctypes.c_long),
            ("vertBearingY", ctypes.c_long),
            ("vertAdvance", ctypes.c_long),
        ]

    class FT_Outline(ctypes.Structure):
        _fields_ = [
            ("n_contours", ctypes.c_ushort),
            ("n_points", ctypes.c_ushort),
            ("points", c_void_p),
            ("tags", c_void_p),
            ("contours", c_void_p),
            ("flags", ctypes.c_int),
        ]

    class FT_SizeRec(ctypes.Structure):
        pass

    class FT_FaceRec(ctypes.Structure):
        pass

    class FT_GlyphSlotRec(ctypes.Structure):
        pass

    FT_Library = c_void_p
    FT_Size = ctypes.POINTER(FT_SizeRec)
    FT_CharMap = c_void_p
    FT_Driver = c_void_p
    FT_Memory = c_void_p
    FT_Stream = c_void_p
    FT_ListRec = c_void_p
    FT_Face_Internal = c_void_p
    FT_Slot_Internal = c_void_p
    FT_GlyphSlot = ctypes.POINTER(FT_GlyphSlotRec)
    FT_Face = ctypes.POINTER(FT_FaceRec)

    class FT_Size_Metrics(ctypes.Structure):
        _fields_ = [
            ("x_ppem", ctypes.c_ushort),
            ("y_ppem", ctypes.c_ushort),
            ("x_scale", ctypes.c_long),
            ("y_scale", ctypes.c_long),
            ("ascender", ctypes.c_long),
            ("descender", ctypes.c_long),
            ("height", ctypes.c_long),
            ("max_advance", ctypes.c_long),
        ]

    FT_SizeRec._fields_ = [
        ("face", FT_Face),
        ("generic", FT_Generic),
        ("metrics", FT_Size_Metrics),
        ("internal", c_void_p),
    ]

    FT_GlyphSlotRec._fields_ = [
        ("library", FT_Library),
        ("face", FT_Face),
        ("next", FT_GlyphSlot),
        ("glyph_index", ctypes.c_uint),
        ("generic", FT_Generic),
        ("metrics", FT_Glyph_Metrics),
        ("linearHoriAdvance", ctypes.c_long),
        ("linearVertAdvance", ctypes.c_long),
        ("advance", FT_Vector),
        ("format", ctypes.c_uint),
        ("bitmap", FT_Bitmap),
        ("bitmap_left", ctypes.c_int),
        ("bitmap_top", ctypes.c_int),
        ("outline", FT_Outline),
        ("num_subglyphs", ctypes.c_uint),
        ("subglyphs", c_void_p),
        ("control_data", c_void_p),
        ("control_len", ctypes.c_long),
        ("lsb_delta", ctypes.c_long),
        ("rsb_delta", ctypes.c_long),
        ("other", c_void_p),
        ("internal", FT_Slot_Internal),
    ]

    FT_FaceRec._fields_ = [
        ("num_faces", ctypes.c_long),
        ("face_index", ctypes.c_long),
        ("face_flags", ctypes.c_long),
        ("style_flags", ctypes.c_long),
        ("num_glyphs", ctypes.c_long),
        ("family_name", ctypes.c_char_p),
        ("style_name", ctypes.c_char_p),
        ("num_fixed_sizes", ctypes.c_int),
        ("available_sizes", c_void_p),
        ("num_charmaps", ctypes.c_int),
        ("charmaps", c_void_p),
        ("generic", FT_Generic),
        ("bbox", FT_BBox),
        ("units_per_EM", ctypes.c_ushort),
        ("ascender", ctypes.c_short),
        ("descender", ctypes.c_short),
        ("height", ctypes.c_short),
        ("max_advance_width", ctypes.c_short),
        ("max_advance_height", ctypes.c_short),
        ("underline_position", ctypes.c_short),
        ("underline_thickness", ctypes.c_short),
        ("glyph", FT_GlyphSlot),
        ("size", FT_Size),
        ("charmap", FT_CharMap),
        ("driver", FT_Driver),
        ("memory", FT_Memory),
        ("stream", FT_Stream),
        ("sizes_list", FT_ListRec),
        ("autohint", c_void_p),
        ("extensions", c_void_p),
        ("internal", FT_Face_Internal),
    ]

    face = FT_Face()
    lib_obj = FT_Library()

    FT_LOAD_RENDER = 0x4

    freetype.FT_Init_FreeType.argtypes = [ctypes.POINTER(FT_Library)]
    freetype.FT_Init_FreeType.restype = ctypes.c_int
    freetype.FT_New_Face.argtypes = [FT_Library, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(FT_Face)]
    freetype.FT_New_Face.restype = ctypes.c_int
    freetype.FT_Set_Pixel_Sizes.argtypes = [FT_Face, ctypes.c_uint, ctypes.c_uint]
    freetype.FT_Set_Pixel_Sizes.restype = ctypes.c_int
    freetype.FT_Load_Char.argtypes = [FT_Face, ctypes.c_ulong, ctypes.c_int]
    freetype.FT_Load_Char.restype = ctypes.c_int
    freetype.FT_Done_Face.argtypes = [FT_Face]
    freetype.FT_Done_FreeType.argtypes = [FT_Library]

    error = freetype.FT_Init_FreeType(ctypes.byref(lib_obj))
    if error != 0:
        raise RuntimeError(f"FT_Init_FreeType failed: {error}")
    try:
        error = freetype.FT_New_Face(lib_obj, font_path.encode("utf-8"), 0, ctypes.byref(face))
        if error != 0:
            raise RuntimeError(f"FT_New_Face failed: {error}")
        try:
            error = freetype.FT_Set_Pixel_Sizes(face, 0, 7)
            if error != 0:
                raise RuntimeError(f"FT_Set_Pixel_Sizes failed: {error}")
            size_metrics = face.contents.size.contents.metrics
            cell_width = max(1, int(round(size_metrics.max_advance / 64.0)) + 1)
            cell_height = max(1, int(round((size_metrics.ascender - size_metrics.descender) / 64.0)))
            baseline = int(round(size_metrics.ascender / 64.0))
            font: dict[str, tuple[str, ...]] = {}
            for code in range(32, 127):
                char = chr(code)
                error = freetype.FT_Load_Char(face, code, FT_LOAD_RENDER)
                if error != 0:
                    font[char] = HUD_FONT_FALLBACK.get(char, HUD_FONT_FALLBACK["?"])
                    continue
                slot = face.contents.glyph.contents
                bitmap = slot.bitmap
                rows = [["0"] * cell_width for _ in range(cell_height)]
                if bitmap.width > 0 and bitmap.rows > 0 and bool(bitmap.buffer):
                    buffer = ctypes.cast(bitmap.buffer, ctypes.POINTER(ctypes.c_ubyte))
                    row_offset = baseline - slot.bitmap_top
                    for y in range(bitmap.rows):
                        dest_y = row_offset + y
                        if not (0 <= dest_y < cell_height):
                            continue
                        for x in range(bitmap.width):
                            dest_x = slot.bitmap_left + x
                            if not (0 <= dest_x < cell_width):
                                continue
                            value = buffer[y * bitmap.pitch + x]
                            if value > 64:
                                rows[dest_y][dest_x] = "1"
                font[char] = tuple("".join(row) for row in rows)
            if "?" not in font:
                font["?"] = HUD_FONT_FALLBACK["?"]
            return font
        finally:
            freetype.FT_Done_Face(face)
    finally:
        freetype.FT_Done_FreeType(lib_obj)


def build_hud_font() -> dict[str, tuple[str, ...]]:
    try:
        return _build_hud_font_from_freetype()
    except Exception:
        return dict(HUD_FONT_FALLBACK)


HUD_FONT = build_hud_font()


def hud_glyph_rows(char: str) -> tuple[str, ...]:
    return HUD_FONT.get(char, HUD_FONT["?"])


def profile_begin_frame(renderer) -> float | None:
    if not renderer.profiling_enabled:
        return None
    return time.perf_counter()


def profile_end_frame(renderer, started_at: float | None, frame_dt: float) -> None:
    if started_at is None:
        return
    ended_at = time.perf_counter()
    renderer.profile_window_cpu_ms += (ended_at - started_at) * 1000.0
    renderer.profile_window_frames += 1
    renderer.profile_window_frame_times.append(frame_dt)
    if ended_at >= renderer.profile_next_report:
        refresh_profile_summary(renderer, ended_at)


def profile_frame_time_percentiles(renderer) -> tuple[float, float, float]:
    if not renderer.profile_window_frame_times:
        return 0.0, 0.0, 0.0
    ordered = sorted(renderer.profile_window_frame_times)
    count = len(ordered)

    def pick(percentile: float) -> float:
        index = max(0, min(count - 1, math.ceil(percentile * count) - 1))
        return ordered[index] * 1000.0

    return pick(0.50), pick(0.95), pick(0.99)


def profile_average_fps(renderer) -> float:
    if not renderer.profile_window_frame_times:
        return 0.0
    avg_frame_time = sum(renderer.profile_window_frame_times) / len(renderer.profile_window_frame_times)
    return 1.0 / max(1e-6, avg_frame_time)


def record_frame_breakdown_sample(renderer, name: str, value: float) -> None:
    samples = renderer.frame_breakdown_samples.get(name)
    if samples is not None:
        samples.append(value)


def frame_breakdown_average(renderer, name: str) -> float:
    samples = renderer.frame_breakdown_samples.get(name)
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def build_hud_vertices(renderer, lines: list[str], *, align_right: bool = False) -> tuple[bytes, int]:
    renderer_module = _renderer_module()
    if not lines:
        return b"", 0

    screen_w, screen_h = renderer.canvas.get_physical_size()
    if screen_w <= 0 or screen_h <= 0:
        return b"", 0

    cache_key = (bool(align_right), int(screen_w), int(screen_h), tuple(lines))
    cached = renderer._hud_geometry_cache.get(cache_key)
    if cached is not None:
        renderer._hud_geometry_cache.move_to_end(cache_key)
        return cached

    scale = render_consts.HUD_FONT_SCALE
    glyph_w = render_consts.HUD_FONT_CHAR_WIDTH * scale
    glyph_h = render_consts.HUD_FONT_CHAR_HEIGHT * scale
    advance_x = glyph_w + render_consts.HUD_GLYPH_SPACING * scale
    line_step = glyph_h + render_consts.HUD_LINE_SPACING
    max_chars = max(len(line) for line in lines)
    panel_w = render_consts.HUD_PANEL_PADDING * 2 + max_chars * advance_x - (render_consts.HUD_GLYPH_SPACING * scale if max_chars else 0)
    panel_h = render_consts.HUD_PANEL_PADDING * 2 + len(lines) * line_step - render_consts.HUD_LINE_SPACING

    vertices: list[bytes] = []

    def add_quad(px0: float, py0: float, px1: float, py1: float, color: tuple[float, float, float], alpha: float) -> None:
        x0, y0 = screen_to_ndc(px0, py0, screen_w, screen_h)
        x1, y1 = screen_to_ndc(px1, py1, screen_w, screen_h)
        vertices.append(pack_vertex((x0, y0, 0.0), (0.0, 0.0, 1.0), color, alpha))
        vertices.append(pack_vertex((x1, y0, 0.0), (0.0, 0.0, 1.0), color, alpha))
        vertices.append(pack_vertex((x1, y1, 0.0), (0.0, 0.0, 1.0), color, alpha))
        vertices.append(pack_vertex((x0, y0, 0.0), (0.0, 0.0, 1.0), color, alpha))
        vertices.append(pack_vertex((x1, y1, 0.0), (0.0, 0.0, 1.0), color, alpha))
        vertices.append(pack_vertex((x0, y1, 0.0), (0.0, 0.0, 1.0), color, alpha))

    panel_x = 12.0
    panel_y = 12.0
    if align_right:
        panel_x = max(12.0, float(screen_w) - panel_w - 12.0)
    add_quad(panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, (0.05, 0.07, 0.09), 0.78)
    add_quad(panel_x, panel_y, panel_x + panel_w, panel_y + 4.0, (0.12, 0.72, 0.85), 0.85)

    for line_index, raw_line in enumerate(lines):
        line = raw_line.upper()
        cursor_x = panel_x + render_consts.HUD_PANEL_PADDING
        cursor_y = panel_y + render_consts.HUD_PANEL_PADDING + line_index * line_step
        text_color = (0.92, 0.97, 1.0) if line_index == 0 else (0.84, 0.90, 0.84)
        for char in line:
            glyph = hud_glyph_rows(char)
            for row_index, row in enumerate(glyph):
                for col_index, bit in enumerate(row):
                    if bit != "1":
                        continue
                    px0 = cursor_x + col_index * scale
                    py0 = cursor_y + row_index * scale
                    add_quad(px0, py0, px0 + scale, py0 + scale, text_color, 1.0)
            cursor_x += advance_x

    built = b"".join(vertices), len(vertices)
    renderer._hud_geometry_cache[cache_key] = built
    while len(renderer._hud_geometry_cache) > 8:
        renderer._hud_geometry_cache.popitem(last=False)
    return built


def build_profile_hud_vertices(renderer, lines: list[str]) -> tuple[bytes, int]:
    return build_hud_vertices(renderer, lines)


def build_frame_breakdown_hud_vertices(renderer, lines: list[str]) -> tuple[bytes, int]:
    return build_hud_vertices(renderer, lines, align_right=True)


def draw_hud_overlay(renderer, encoder, color_view, vertex_bytes: bytes, vertex_count: int) -> None:
    if not vertex_bytes:
        return
    hud_buffer = renderer.device.create_buffer_with_data(
        data=vertex_bytes,
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    hud_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": color_view,
                "resolve_target": None,
                "load_op": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            }
        ]
    )
    hud_pass.set_pipeline(renderer.profile_hud_pipeline)
    hud_pass.set_vertex_buffer(0, hud_buffer)
    hud_pass.draw(vertex_count, 1, 0, 0)
    hud_pass.end()


def draw_profile_hud(renderer, encoder, color_view) -> None:
    if not renderer.profiling_enabled:
        return
    draw_hud_overlay(renderer, encoder, color_view, renderer.profile_hud_vertex_bytes, renderer.profile_hud_vertex_count)


def draw_frame_breakdown_hud(renderer, encoder, color_view) -> None:
    if not renderer.profiling_enabled:
        return
    draw_hud_overlay(renderer, encoder, color_view, renderer.frame_breakdown_vertex_bytes, renderer.frame_breakdown_vertex_count)


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
        f"ENGINE MODE {renderer.engine_mode_label}",
        f"PRESENT     FPS {renderer_module.SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if renderer_module.SWAPCHAIN_USE_VSYNC else 'OFF'}",
        f"MESH SLABS {slab_count:2d}  USED {slab_used_bytes / (1024.0 * 1024.0):4.1f} MIB  FREE {slab_free_bytes / (1024.0 * 1024.0):4.1f} MIB",
        f"MESH BIGGEST GAP {slab_largest_free_bytes / (1024.0 * 1024.0):4.1f} MIB  ALLOCS {slab_alloc_count:3d}",
    ]
    renderer.profile_hud_lines = lines
    renderer.profile_hud_vertex_bytes, renderer.profile_hud_vertex_count = build_profile_hud_vertices(renderer, lines)
    renderer.profile_window_start = now
    renderer.profile_next_report = now + renderer_module.PROFILE_REPORT_INTERVAL
    renderer.profile_window_cpu_ms = 0.0
    renderer.profile_window_frames = 0
    renderer.profile_window_frame_times = []


def refresh_frame_breakdown_summary(renderer) -> None:
    if not renderer.profiling_enabled:
        return
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

    lines = [
        f"FRAME BREAKDOWN @ DIMENSION {renderer.render_dimension_chunks}x{renderer.render_dimension_chunks} CHUNKS",
        f"MOVE SPEED: {renderer._current_move_speed:5.1f} B/S",
        f"TERRAIN BACKEND: {renderer.world.terrain_backend_label()}",
        f"MESH BACKEND: {renderer.mesh_backend_label}",
        f"CHUNK DIMS: {renderer_module.CHUNK_SIZE}x{renderer_module.WORLD_HEIGHT}x{renderer_module.CHUNK_SIZE}",
        f"BACKEND POLL SIZE: {renderer.terrain_batch_size}",
        f"MESH DRAIN SIZE: {renderer.mesh_batch_size}",
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
        f"CHUNK PAYLOAD: {chunk_memory_bytes:,} BYTES ({chunk_memory_mib:5.2f} MIB)",
        f"TOTAL DRAW VERTICES: {visible_vertices:,}",
        f"VISIBLE BUT NOT READY: {visible_but_not_ready}",
        f"PENDING CHUNK REQUESTS: {pending_chunk_requests}",
        f"DRAW CALLS: {draw_calls}",
        f"VISIBLE MERGED CHUNKS (VISIBLE ONLY): {merged_chunks}",
    ]
    renderer.frame_breakdown_lines = lines
    renderer.frame_breakdown_vertex_bytes, renderer.frame_breakdown_vertex_count = build_frame_breakdown_hud_vertices(renderer, lines)
