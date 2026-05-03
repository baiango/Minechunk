from __future__ import annotations

import math

import numpy as np
import wgpu

from ..renderer_config import *
from ..render_shaders import WORLDSPACE_RC_UPDATE_PARAMS_BYTES


def write_final_present_params(renderer, width: int, height: int) -> None:
    self = renderer
    params = np.array([
        1.0 / max(1.0, float(width)),
        1.0 / max(1.0, float(height)),
        0.0,
        0.0,
    ], dtype=np.float32)
    self.device.queue.write_buffer(self.final_present_params_buffer, 0, params.tobytes())

def make_gi_params(renderer, rc_debug_mode: int | None = None) -> np.ndarray:
    self = renderer
    # merge_control.x is used by the world-space RC compose shader as a
    # lightweight debug visualization mode. The legacy screen-space cascade
    # pass is not submitted in the current WGPU RC path.
    debug_mode = int(self.rc_debug_mode if rc_debug_mode is None else rc_debug_mode)
    return np.array([
        float(RADIANCE_CASCADES_STRENGTH),
        float(RADIANCE_CASCADES_BIAS),
        float(RADIANCE_CASCADES_SKY_STRENGTH),
        float(RADIANCE_CASCADES_HIT_THICKNESS),
        float(debug_mode),
        float(RADIANCE_CASCADES_MERGE_STRENGTH),
        float(RADIANCE_CASCADES_CASCADE_COUNT),
        float(WORLDSPACE_RC_GRID_RESOLUTION),
    ], dtype=np.float32)

def write_gi_params(renderer) -> None:
    self = renderer
    if self.gi_params_buffer is None:
        return
    self.device.queue.write_buffer(self.gi_params_buffer, 0, make_gi_params(self).tobytes())

def ensure_postprocess_targets(renderer) -> None:
    self = renderer
    width, height = self.canvas.get_physical_size()
    target_size = (max(1, int(width)), max(1, int(height)))
    cascade_count = max(1, int(RADIANCE_CASCADES_CASCADE_COUNT))
    if (
        target_size == self._postprocess_size
        and self.scene_color_view is not None
        and self.scene_gbuffer_view is not None
        and self.gi_color_view is not None
        and len(self.gi_cascade_views) == cascade_count
        and len(self.gi_cascade_bind_groups) == cascade_count
        and len(self.worldspace_rc_views) == 4
        and len(self.worldspace_rc_visibility_views) == 4
        and len(self.worldspace_rc_scratch_views) == 4
        and len(self.worldspace_rc_visibility_scratch_views) == 4
        and len(self.worldspace_rc_update_param_buffers) == 4
        and len(self.worldspace_rc_trace_bind_groups) == 4
        and len(self.worldspace_rc_filter_bind_groups) == 4
        and len(self.worldspace_rc_filter_pingpong_bind_groups) == 4
        and self.gi_compose_bind_group is not None
        and self.final_scene_bind_group is not None
        and self.final_gi_bind_group is not None
        and self.postprocess_depth_view is not None
        and (self.postprocess_msaa_sample_count <= 1 or self.scene_color_msaa_view is not None)
        and (self.postprocess_msaa_sample_count <= 1 or self.scene_gbuffer_msaa_view is not None)
    ):
        return
    self._postprocess_size = target_size
    gi_compose_scale = min(1.0, max(0.25, float(RADIANCE_CASCADES_COMPOSE_SCALE)))
    gi_target_size = (
        max(1, int(math.ceil(float(target_size[0]) * gi_compose_scale))),
        max(1, int(math.ceil(float(target_size[1]) * gi_compose_scale))),
    )
    texture_usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
    self.scene_color_texture = self.device.create_texture(
        size=(target_size[0], target_size[1], 1),
        format=POSTPROCESS_SCENE_FORMAT,
        usage=texture_usage,
    )
    self.scene_color_view = self.scene_color_texture.create_view()
    if self.postprocess_msaa_sample_count > 1:
        self.scene_color_msaa_texture = self.device.create_texture(
            size=(target_size[0], target_size[1], 1),
            format=POSTPROCESS_SCENE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.postprocess_msaa_sample_count,
        )
        self.scene_color_msaa_view = self.scene_color_msaa_texture.create_view()
    else:
        self.scene_color_msaa_texture = None
        self.scene_color_msaa_view = self.scene_color_view
    self.scene_gbuffer_texture = self.device.create_texture(
        size=(target_size[0], target_size[1], 1),
        format=POSTPROCESS_GBUFFER_FORMAT,
        usage=texture_usage,
    )
    self.scene_gbuffer_view = self.scene_gbuffer_texture.create_view()
    if self.postprocess_msaa_sample_count > 1:
        self.scene_gbuffer_msaa_texture = self.device.create_texture(
            size=(target_size[0], target_size[1], 1),
            format=POSTPROCESS_GBUFFER_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.postprocess_msaa_sample_count,
        )
        self.scene_gbuffer_msaa_view = self.scene_gbuffer_msaa_texture.create_view()
    else:
        self.scene_gbuffer_msaa_texture = None
        self.scene_gbuffer_msaa_view = self.scene_gbuffer_view
    self.gi_color_texture = self.device.create_texture(
        size=(gi_target_size[0], gi_target_size[1], 1),
        format=POSTPROCESS_GI_FORMAT,
        usage=texture_usage,
    )
    self.gi_color_view = self.gi_color_texture.create_view()
    self._gi_color_size = (gi_target_size[0], gi_target_size[1])
    worldspace_resolution = max(4, int(WORLDSPACE_RC_GRID_RESOLUTION))
    worldspace_direction_count = max(1, int(WORLDSPACE_RC_DIRECTION_COUNT))
    worldspace_radiance_width = worldspace_resolution * worldspace_direction_count
    self.worldspace_rc_textures = []
    self.worldspace_rc_views = []
    self.worldspace_rc_visibility_textures = []
    self.worldspace_rc_visibility_views = []
    self.worldspace_rc_scratch_textures = []
    self.worldspace_rc_scratch_views = []
    self.worldspace_rc_visibility_scratch_textures = []
    self.worldspace_rc_visibility_scratch_views = []
    self.worldspace_rc_trace_bind_groups = []
    self.worldspace_rc_filter_bind_groups = []
    self.worldspace_rc_filter_pingpong_bind_groups = []
    rc_texture_usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.STORAGE_BINDING
    for cascade_index in range(4):
        rc_texture = self.device.create_texture(
            size=(worldspace_radiance_width, worldspace_resolution, worldspace_resolution),
            dimension="3d",
            format=POSTPROCESS_GI_FORMAT,
            usage=rc_texture_usage,
        )
        vis_texture = self.device.create_texture(
            size=(worldspace_resolution, worldspace_resolution, worldspace_resolution),
            dimension="3d",
            format=POSTPROCESS_GI_FORMAT,
            usage=rc_texture_usage,
        )
        scratch_texture = self.device.create_texture(
            size=(worldspace_radiance_width, worldspace_resolution, worldspace_resolution),
            dimension="3d",
            format=POSTPROCESS_GI_FORMAT,
            usage=rc_texture_usage,
        )
        scratch_vis_texture = self.device.create_texture(
            size=(worldspace_resolution, worldspace_resolution, worldspace_resolution),
            dimension="3d",
            format=POSTPROCESS_GI_FORMAT,
            usage=rc_texture_usage,
        )
        self.worldspace_rc_textures.append(rc_texture)
        self.worldspace_rc_views.append(rc_texture.create_view(dimension="3d"))
        self.worldspace_rc_visibility_textures.append(vis_texture)
        self.worldspace_rc_visibility_views.append(vis_texture.create_view(dimension="3d"))
        self.worldspace_rc_scratch_textures.append(scratch_texture)
        self.worldspace_rc_scratch_views.append(scratch_texture.create_view(dimension="3d"))
        self.worldspace_rc_visibility_scratch_textures.append(scratch_vis_texture)
        self.worldspace_rc_visibility_scratch_views.append(scratch_vis_texture.create_view(dimension="3d"))
    self._worldspace_rc_signature = None
    self._worldspace_rc_active_signatures = [None, None, None, None]
    self._worldspace_rc_active_mins = [(0.0, 0.0, 0.0) for _ in range(4)]
    self._worldspace_rc_active_inv_extents = [(0.0, 0.0, 0.0) for _ in range(4)]
    self._worldspace_rc_last_update_frame = [-1000000, -1000000, -1000000, -1000000]
    self._worldspace_rc_update_cursor = 0
    self.gi_cascade_textures = []
    self.gi_cascade_views = []
    self.gi_cascade_param_buffers = []
    self.gi_cascade_bind_groups = []
    for cascade_index in range(cascade_count):
        cascade_size = (
            max(1, target_size[0] >> cascade_index),
            max(1, target_size[1] >> cascade_index),
            1,
        )
        cascade_texture = self.device.create_texture(
            size=cascade_size,
            format=POSTPROCESS_GI_FORMAT,
            usage=texture_usage,
        )
        self.gi_cascade_textures.append(cascade_texture)
        self.gi_cascade_views.append(cascade_texture.create_view())
    self.postprocess_depth_texture = self.device.create_texture(
        size=(target_size[0], target_size[1], 1),
        format=DEPTH_FORMAT,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        sample_count=self.postprocess_msaa_sample_count,
    )
    self.postprocess_depth_view = self.postprocess_depth_texture.create_view()

    write_final_present_params(self, target_size[0], target_size[1])

    write_gi_params(self)

    for cascade_index in range(cascade_count):
        cascade_params = np.array([
            float(cascade_index),
            1.0 if cascade_index + 1 < cascade_count else 0.0,
            float(cascade_count),
            0.0,
            float(RADIANCE_CASCADES_BASE_INTERVAL),
            float(RADIANCE_CASCADES_INTERVAL_SCALE),
            float(RADIANCE_CASCADES_STEPS_PER_CASCADE),
            0.0,
        ], dtype=np.float32)
        cascade_param_buffer = self.device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(cascade_param_buffer, 0, cascade_params.tobytes())
        self.gi_cascade_param_buffers.append(cascade_param_buffer)
        prev_view = self.gi_cascade_views[cascade_index + 1] if cascade_index + 1 < cascade_count else self.scene_color_view
        self.gi_cascade_bind_groups.append(
            self.device.create_bind_group(
                layout=self.gi_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": self.scene_color_view},
                    {"binding": 1, "resource": self.scene_gbuffer_view},
                    {"binding": 2, "resource": prev_view},
                    {"binding": 3, "resource": self.postprocess_sampler},
                    {"binding": 4, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": CAMERA_UNIFORM_BYTES}},
                    {"binding": 5, "resource": {"buffer": self.gi_params_buffer, "offset": 0, "size": 32}},
                    {"binding": 6, "resource": {"buffer": cascade_param_buffer, "offset": 0, "size": 32}},
                ],
            )
        )
    self.gi_bind_group = self.gi_cascade_bind_groups[0] if self.gi_cascade_bind_groups else None
    worldspace_rc_params = np.zeros((8, 4), dtype=np.float32)
    self.device.queue.write_buffer(self.worldspace_rc_volume_params_buffer, 0, worldspace_rc_params.tobytes())
    for cascade_index in range(4):
        update_param_buffer = self.worldspace_rc_update_param_buffers[cascade_index]
        self.worldspace_rc_trace_bind_groups.append(
            self.device.create_bind_group(
                layout=self.worldspace_rc_trace_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": self.worldspace_rc_scratch_views[cascade_index]},
                    {"binding": 1, "resource": self.worldspace_rc_visibility_scratch_views[cascade_index]},
                    {"binding": 2, "resource": {"buffer": update_param_buffer, "offset": 0, "size": WORLDSPACE_RC_UPDATE_PARAMS_BYTES}},
                    {"binding": 3, "resource": self.worldspace_rc_views[0]},
                    {"binding": 4, "resource": self.worldspace_rc_views[1]},
                    {"binding": 5, "resource": self.worldspace_rc_views[2]},
                    {"binding": 6, "resource": self.worldspace_rc_views[3]},
                    {"binding": 7, "resource": self.worldspace_rc_visibility_views[0]},
                    {"binding": 8, "resource": self.worldspace_rc_visibility_views[1]},
                    {"binding": 9, "resource": self.worldspace_rc_visibility_views[2]},
                    {"binding": 10, "resource": self.worldspace_rc_visibility_views[3]},
                    {"binding": 11, "resource": {"buffer": self.worldspace_rc_volume_params_buffer, "offset": 0, "size": 256}},
                ],
            )
        )
        self.worldspace_rc_filter_bind_groups.append(
            self.device.create_bind_group(
                layout=self.worldspace_rc_filter_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": self.worldspace_rc_scratch_views[cascade_index]},
                    {"binding": 1, "resource": self.worldspace_rc_visibility_scratch_views[cascade_index]},
                    {"binding": 2, "resource": self.worldspace_rc_views[cascade_index]},
                    {"binding": 3, "resource": self.worldspace_rc_visibility_views[cascade_index]},
                    {"binding": 4, "resource": {"buffer": update_param_buffer, "offset": 0, "size": WORLDSPACE_RC_UPDATE_PARAMS_BYTES}},
                ],
            )
        )
        self.worldspace_rc_filter_pingpong_bind_groups.append(
            self.device.create_bind_group(
                layout=self.worldspace_rc_filter_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": self.worldspace_rc_views[cascade_index]},
                    {"binding": 1, "resource": self.worldspace_rc_visibility_views[cascade_index]},
                    {"binding": 2, "resource": self.worldspace_rc_scratch_views[cascade_index]},
                    {"binding": 3, "resource": self.worldspace_rc_visibility_scratch_views[cascade_index]},
                    {"binding": 4, "resource": {"buffer": update_param_buffer, "offset": 0, "size": WORLDSPACE_RC_UPDATE_PARAMS_BYTES}},
                ],
            )
        )
    self.gi_compose_bind_group = self.device.create_bind_group(
        layout=self.gi_compose_bind_group_layout,
        entries=[
            {"binding": 0, "resource": self.scene_color_view},
            {"binding": 1, "resource": self.scene_gbuffer_view},
            {"binding": 2, "resource": self.worldspace_rc_views[0]},
            {"binding": 3, "resource": self.worldspace_rc_views[1]},
            {"binding": 4, "resource": self.worldspace_rc_views[2]},
            {"binding": 5, "resource": self.worldspace_rc_views[3]},
            {"binding": 6, "resource": self.worldspace_rc_visibility_views[0]},
            {"binding": 7, "resource": self.worldspace_rc_visibility_views[1]},
            {"binding": 8, "resource": self.worldspace_rc_visibility_views[2]},
            {"binding": 9, "resource": self.worldspace_rc_visibility_views[3]},
            {"binding": 10, "resource": self.postprocess_sampler},
            {"binding": 11, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": CAMERA_UNIFORM_BYTES}},
            {"binding": 12, "resource": {"buffer": self.gi_params_buffer, "offset": 0, "size": 32}},
            {"binding": 13, "resource": {"buffer": self.worldspace_rc_volume_params_buffer, "offset": 0, "size": 256}},
        ],
    )
    self.final_scene_bind_group = self.device.create_bind_group(
        layout=self.final_present_bind_group_layout,
        entries=[
            {"binding": 0, "resource": self.scene_color_view},
            {"binding": 1, "resource": self.postprocess_sampler},
            {"binding": 2, "resource": {"buffer": self.final_present_params_buffer, "offset": 0, "size": 32}},
            {"binding": 3, "resource": self.scene_gbuffer_view},
        ],
    )
    self.final_gi_bind_group = self.device.create_bind_group(
        layout=self.final_present_bind_group_layout,
        entries=[
            {"binding": 0, "resource": self.gi_color_view},
            {"binding": 1, "resource": self.postprocess_sampler},
            {"binding": 2, "resource": {"buffer": self.final_present_params_buffer, "offset": 0, "size": 32}},
            {"binding": 3, "resource": self.scene_gbuffer_view},
        ],
    )
