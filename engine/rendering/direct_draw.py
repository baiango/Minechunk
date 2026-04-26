from __future__ import annotations

"""Direct/indirect visible-batch draw helpers."""

from typing import Any

from ..renderer_config import INDIRECT_DRAW_COMMAND_STRIDE

try:
    from wgpu.backends.wgpu_native import multi_draw_indirect as wgpu_native_multi_draw_indirect
except Exception:  # pragma: no cover - backend optional
    wgpu_native_multi_draw_indirect = None


def draw_visible_batches_to_pass(
    renderer: Any,
    render_pass: Any,
    visible_batches: Any,
    use_gpu_visibility: bool,
    use_indirect: bool,
) -> None:
    current_vertex_buffer = None
    current_binding_offset = None
    set_vertex_buffer = render_pass.set_vertex_buffer
    if use_gpu_visibility or use_indirect:
        indirect_buffer = renderer._mesh_draw_indirect_buffer
        assert indirect_buffer is not None
        draw_indirect = render_pass.draw_indirect
        for vertex_buffer, binding_offset, batch_start, batch_count in visible_batches:
            if vertex_buffer is not current_vertex_buffer or binding_offset != current_binding_offset:
                set_vertex_buffer(0, vertex_buffer, binding_offset)
                current_vertex_buffer = vertex_buffer
                current_binding_offset = binding_offset
            if wgpu_native_multi_draw_indirect is not None and batch_count > 1:
                wgpu_native_multi_draw_indirect(
                    render_pass,
                    indirect_buffer,
                    offset=batch_start * INDIRECT_DRAW_COMMAND_STRIDE,
                    count=batch_count,
                )
                continue
            for batch_index in range(batch_count):
                indirect_offset = (batch_start + batch_index) * INDIRECT_DRAW_COMMAND_STRIDE
                draw_indirect(indirect_buffer, indirect_offset)
    else:
        draw = render_pass.draw
        for vertex_buffer, binding_offset, vertex_count, first_vertex in visible_batches:
            if vertex_buffer is not current_vertex_buffer or binding_offset != current_binding_offset:
                set_vertex_buffer(0, vertex_buffer, binding_offset)
                current_vertex_buffer = vertex_buffer
                current_binding_offset = binding_offset
            draw(vertex_count, 1, first_vertex, 0)
