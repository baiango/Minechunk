from __future__ import annotations

import wgpu

from ..renderer_config import *
from ..render_shaders import WORLDSPACE_RC_UPDATE_PARAMS_BYTES


def create_gpu_buffers(renderer) -> None:
    """Create long-lived renderer buffers and samplers."""
    self = renderer
    self.camera_buffer = self.device.create_buffer(
        size=CAMERA_UNIFORM_BYTES,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    self._mesh_visibility_params_buffer = self.device.create_buffer(
        size=16,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    self._tile_merge_dummy_buffer = self.device.create_buffer(
        size=VERTEX_STRIDE,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    self.gi_params_buffer = self.device.create_buffer(
        size=32,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    self.worldspace_rc_volume_params_buffer = self.device.create_buffer(
        size=256,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    self.worldspace_rc_update_param_buffers = [
        self.device.create_buffer(
            size=WORLDSPACE_RC_UPDATE_PARAMS_BYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        for _ in range(4)
    ]
    self.final_present_params_buffer = self.device.create_buffer(
        size=32,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    self.postprocess_sampler = self.device.create_sampler(
        address_mode_u="clamp-to-edge",
        address_mode_v="clamp-to-edge",
        address_mode_w="clamp-to-edge",
        mag_filter="linear",
        min_filter="linear",
        mipmap_filter="linear",
    )
