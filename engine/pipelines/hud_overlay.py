from __future__ import annotations

import wgpu

from .. import render_contract as render_consts
from ..render_utils import pack_vertex, screen_to_ndc
from .hud_font import hud_glyph_rows


def build_hud_vertices(renderer, lines: list[str], *, align_right: bool = False) -> tuple[bytes, int]:
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


def _ensure_hud_vertex_buffer(renderer, attr_prefix: str, vertex_bytes: bytes):
    buffer_attr = f"{attr_prefix}_vertex_buffer"
    capacity_attr = f"{attr_prefix}_vertex_buffer_capacity"
    if not vertex_bytes:
        setattr(renderer, buffer_attr, None)
        setattr(renderer, capacity_attr, 0)
        return None

    existing = getattr(renderer, buffer_attr, None)
    capacity = int(getattr(renderer, capacity_attr, 0) or 0)
    size = len(vertex_bytes)
    if existing is None or capacity < size:
        if existing is not None:
            existing.destroy()
        capacity = max(4096, 1 << (size - 1).bit_length())
        existing = renderer.device.create_buffer(
            size=capacity,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        setattr(renderer, buffer_attr, existing)
        setattr(renderer, capacity_attr, capacity)
    renderer.device.queue.write_buffer(existing, 0, vertex_bytes)
    return existing


def draw_hud_overlay(renderer, encoder, color_view, hud_buffer, vertex_count: int) -> None:
    if hud_buffer is None or vertex_count <= 0:
        return
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
    draw_hud_overlay(renderer, encoder, color_view, renderer.profile_hud_vertex_buffer, renderer.profile_hud_vertex_count)


def draw_frame_breakdown_hud(renderer, encoder, color_view) -> None:
    if not renderer.profiling_enabled:
        return
    draw_hud_overlay(renderer, encoder, color_view, renderer.frame_breakdown_vertex_buffer, renderer.frame_breakdown_vertex_count)

