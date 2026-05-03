from __future__ import annotations

import math
import time

import wgpu

from ..cache import mesh_allocator as mesh_cache
from ..renderer_config import *
from ..render_utils import pack_camera_uniform
from ..world_constants import BLOCK_SIZE
from . import direct_draw, postprocess_targets, rc_debug_capture, worldspace_rc

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


@profile
def submit_render(renderer, meshes=None):
    """Encode the scene pass, RC composition, final blit, and render stats.

    The renderer still owns the long-lived GPU resources; this module owns the
    per-frame command encoding path so TerrainRenderer can stay closer to an
    orchestration object.
    """
    self = renderer
    use_postprocess = bool(self.final_present_enabled)
    physical_width, physical_height = self.canvas.get_physical_size()
    if not use_postprocess and (physical_width, physical_height) != self.depth_size:
        self._ensure_depth_buffer((physical_width, physical_height))
    right, up, forward = self._camera_basis()
    camera_position = tuple(float(value) for value in self.camera.position)
    focal = 1.0 / math.tan(math.radians(90.0) * 0.5)
    aspect = max(1.0, float(physical_width) / max(1.0, float(physical_height)))
    near = max(0.02, 0.1 * BLOCK_SIZE)
    far = max(128.0 * BLOCK_SIZE, DEFAULT_RENDER_DISTANCE_WORLD * 1.25)
    camera_uniform_signature = (camera_position, right, up, forward, focal, aspect, near, far)
    camera_upload_start = time.perf_counter()
    if getattr(self, "_camera_uniform_signature", None) != camera_uniform_signature:
        self.device.queue.write_buffer(
            self.camera_buffer,
            0,
            pack_camera_uniform(
                camera_position,
                right,
                up,
                forward,
                focal,
                aspect,
                near,
                far,
                LIGHT_DIRECTION,
            ),
        )
        self._camera_uniform_signature = camera_uniform_signature
    camera_upload_ms = (time.perf_counter() - camera_upload_start) * 1000.0

    encoder = self.device.create_command_encoder()
    use_gpu_visibility = bool(
        self.use_gpu_indirect_render
        and self.use_gpu_visibility_culling
        and self.mesh_visibility_pipeline is not None
        and self._mesh_output_slabs
        and hasattr(wgpu.GPURenderCommandsMixin, "draw_indirect")
    )
    use_indirect = bool(
        self.use_gpu_indirect_render
        and self._mesh_allocations
        and hasattr(wgpu.GPURenderCommandsMixin, "draw_indirect")
    )
    if meshes is not None:
        if use_gpu_visibility:
            gpu_visibility_start = time.perf_counter()
            visible_batches, draw_calls, merged_batches, visible_chunks, visible_vertices = self.pipeline.render.build_gpu_visibility_records_for_meshes(self, meshes)
            if draw_calls > 0:
                metadata_buffer = self._mesh_visibility_record_buffer
                indirect_buffer = self._mesh_draw_indirect_buffer
                params_buffer = self._mesh_visibility_params_buffer
                assert metadata_buffer is not None
                assert indirect_buffer is not None
                assert params_buffer is not None
                visibility_bind_group = self.device.create_bind_group(
                    layout=self.mesh_visibility_bind_group_layout,
                    entries=[
                        {"binding": 0, "resource": {"buffer": metadata_buffer}},
                        {"binding": 1, "resource": {"buffer": indirect_buffer}},
                        {"binding": 2, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": CAMERA_UNIFORM_BYTES}},
                        {"binding": 3, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
                    ],
                )
                compute_pass = encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.mesh_visibility_pipeline)
                compute_pass.set_bind_group(0, visibility_bind_group)
                compute_pass.dispatch_workgroups((draw_calls + GPU_VISIBILITY_WORKGROUP_SIZE - 1) // GPU_VISIBILITY_WORKGROUP_SIZE, 1, 1)
                compute_pass.end()
            render_encode_ms = (time.perf_counter() - gpu_visibility_start) * 1000.0
        elif use_indirect:
            visible_batches, render_encode_ms, draw_calls, merged_batches, visible_chunks, visible_vertices = self.pipeline.render.visible_render_batches_indirect_for_meshes(self, meshes)
        else:
            visible_batches, render_encode_ms, draw_calls, merged_batches, visible_chunks, visible_vertices = self.pipeline.render.visible_render_batches_for_meshes(meshes)
    else:
        if use_gpu_visibility:
            gpu_visibility_start = time.perf_counter()
            visible_batches, draw_calls, merged_batches, visible_chunks, visible_vertices = mesh_cache.build_gpu_visibility_records(self, encoder)
            if draw_calls > 0:
                metadata_buffer = self._mesh_visibility_record_buffer
                indirect_buffer = self._mesh_draw_indirect_buffer
                params_buffer = self._mesh_visibility_params_buffer
                assert metadata_buffer is not None
                assert indirect_buffer is not None
                assert params_buffer is not None
                visibility_bind_group = self.device.create_bind_group(
                    layout=self.mesh_visibility_bind_group_layout,
                    entries=[
                        {"binding": 0, "resource": {"buffer": metadata_buffer}},
                        {"binding": 1, "resource": {"buffer": indirect_buffer}},
                        {"binding": 2, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": CAMERA_UNIFORM_BYTES}},
                        {"binding": 3, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
                    ],
                )
                compute_pass = encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.mesh_visibility_pipeline)
                compute_pass.set_bind_group(0, visibility_bind_group)
                compute_pass.dispatch_workgroups((draw_calls + GPU_VISIBILITY_WORKGROUP_SIZE - 1) // GPU_VISIBILITY_WORKGROUP_SIZE, 1, 1)
                compute_pass.end()
            render_encode_ms = (time.perf_counter() - gpu_visibility_start) * 1000.0
        elif use_indirect:
            visible_batches, render_encode_ms, draw_calls, merged_batches, visible_chunks, visible_vertices = mesh_cache.visible_render_batches_indirect(self, encoder)
        else:
            visible_batches, render_encode_ms, draw_calls, merged_batches, visible_chunks, visible_vertices = mesh_cache.visible_render_batches(self, encoder)

    # Drawable acquisition can block when GPU/display work backs up, so keep it
    # measured separately in the HUD instead of folding it into generic encode time.
    acquire_start = time.perf_counter()
    current_texture = self.context.get_current_texture()
    swapchain_acquire_ms = (time.perf_counter() - acquire_start) * 1000.0
    color_view = current_texture.create_view()

    if use_postprocess:
        postprocess_targets.ensure_postprocess_targets(self)
        assert self.scene_color_view is not None
        assert self.scene_render_pipeline is not None

        scene_color_attachment_view = self.scene_color_msaa_view if self.postprocess_msaa_sample_count > 1 else self.scene_color_view
        scene_resolve_target = self.scene_color_view if self.postprocess_msaa_sample_count > 1 else None
        scene_gbuffer_attachment_view = self.scene_gbuffer_msaa_view if self.postprocess_msaa_sample_count > 1 else self.scene_gbuffer_view
        scene_gbuffer_resolve_target = self.scene_gbuffer_view if self.postprocess_msaa_sample_count > 1 else None
        scene_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": scene_color_attachment_view,
                    "resolve_target": scene_resolve_target,
                    "clear_value": (0.60, 0.80, 0.98, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                },
                {
                    "view": scene_gbuffer_attachment_view,
                    "resolve_target": scene_gbuffer_resolve_target,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.postprocess_depth_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        scene_pass.set_pipeline(self.scene_render_pipeline)
        scene_pass.set_bind_group(0, self.camera_bind_group)
        direct_draw.draw_visible_batches_to_pass(self, scene_pass, visible_batches, use_gpu_visibility, use_indirect)
        scene_pass.end()

        if self.radiance_cascades_enabled:
            assert self.gi_color_view is not None
            assert self.gi_compose_pipeline is not None
            assert self.gi_compose_bind_group is not None
            worldspace_rc.update(self, encoder)
            gi_compose_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.gi_color_view,
                        "resolve_target": None,
                        "clear_value": (0.0, 0.0, 0.0, 1.0),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    }
                ],
            )
            gi_compose_pass.set_pipeline(self.gi_compose_pipeline)
            gi_compose_pass.set_bind_group(0, self.gi_compose_bind_group)
            gi_compose_pass.draw(3, 1, 0, 0)
            gi_compose_pass.end()

        rc_debug_capture.encode_pending_captures(self, encoder)

        if self.final_present_enabled:
            if self.radiance_cascades_enabled:
                source_bind_group = self.final_gi_bind_group
            else:
                source_bind_group = self.final_scene_bind_group
            assert source_bind_group is not None
            assert self.final_present_pipeline is not None
            final_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": color_view,
                        "resolve_target": None,
                        "clear_value": (0.0, 0.0, 0.0, 1.0),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    }
                ],
            )
            final_pass.set_pipeline(self.final_present_pipeline)
            final_pass.set_bind_group(0, source_bind_group)
            final_pass.draw(3, 1, 0, 0)
            final_pass.end()
    else:
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": color_view,
                    "resolve_target": None,
                    "clear_value": (0.60, 0.80, 0.98, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_bind_group(0, self.camera_bind_group)
        direct_draw.draw_visible_batches_to_pass(self, render_pass, visible_batches, use_gpu_visibility, use_indirect)
        render_pass.end()
    stats = {
        "camera_upload_ms": camera_upload_ms,
        "visibility_lookup_ms": 0.0,
        "swapchain_acquire_ms": swapchain_acquire_ms,
        "render_encode_ms": render_encode_ms,
        "draw_calls": draw_calls,
        "merged_chunks": merged_batches,
        "visible_chunks": visible_chunks,
        "visible_vertices": visible_vertices,
    }
    return encoder, color_view, stats
