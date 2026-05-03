from __future__ import annotations

import wgpu

from .. import render_contract
from ..renderer_config import *


def create_gpu_bind_group_layouts(renderer) -> None:
    """Create bind group layouts used by render, GI, RC, and meshing pipelines."""
    self = renderer
    self.voxel_mesh_count_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 6,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 8,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
        ]
    )
    self.voxel_mesh_scan_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 6,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 8,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
        ]
    )
    self.voxel_mesh_emit_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 6,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 8,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
        ]
    )
    self.voxel_surface_expand_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
        ]
    )
    self.mesh_visibility_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
        ]
    )
    self.tile_merge_bind_group_layout = None
    self.tile_merge_pipeline = None
    if render_contract.device_limit(self.device, "max_storage_buffers_per_shader_stage", 0) >= MERGED_TILE_MAX_CHUNKS + 2:
        self.tile_merge_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": index,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                }
                for index in range(MERGED_TILE_MAX_CHUNKS)
            ]
            + [
                {
                    "binding": MERGED_TILE_MAX_CHUNKS,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": MERGED_TILE_MAX_CHUNKS + 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": MERGED_TILE_MAX_CHUNKS + 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
            ]
        )
    self.render_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            }
        ]
    )
    self.camera_bind_group = self.device.create_bind_group(
        layout=self.render_bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": self.camera_buffer, "offset": 0, "size": CAMERA_UNIFORM_BYTES},
            }
        ],
    )
    self.gi_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 6,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
        ],
    )
    self.gi_compose_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 6,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 7,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 8,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 9,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 10,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            },
            {
                "binding": 11,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 12,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 13,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
        ],
    )
    self.worldspace_rc_trace_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {"access": "write-only", "format": POSTPROCESS_GI_FORMAT, "view_dimension": "3d"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {"access": "write-only", "format": POSTPROCESS_GI_FORMAT, "view_dimension": "3d"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {"binding": 8, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {"binding": 9, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {"binding": 10, "visibility": wgpu.ShaderStage.COMPUTE, "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False}},
            {
                "binding": 11,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
        ],
    )
    self.worldspace_rc_filter_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {"sample_type": "float", "view_dimension": "3d", "multisampled": False},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {"access": "write-only", "format": POSTPROCESS_GI_FORMAT, "view_dimension": "3d"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {"access": "write-only", "format": POSTPROCESS_GI_FORMAT, "view_dimension": "3d"},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform"},
            },
        ],
    )
    self.final_present_bind_group_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
            },
        ],
    )
