from __future__ import annotations

import sys

import wgpu

from ..renderer_config import *
from ..render_shaders import (
    FINAL_BLIT_SHADER,
    GI_CASCADE_SHADER,
    GI_GBUFFER_SHADER,
    GI_POSTPROCESS_SHADER,
    GPU_VISIBILITY_SHADER,
    HUD_SHADER,
    RENDER_SHADER,
    TILE_MERGE_SHADER,
    VOXEL_MESH_BATCH_SHADER,
    VOXEL_SURFACE_EXPAND_SHADER,
    WORLDSPACE_RC_FILTER_SHADER,
    WORLDSPACE_RC_TRACE_SHADER,
)
from ..pipelines.hud_overlay import HUD_VERTEX_STRIDE


def create_gpu_pipelines(renderer) -> None:
    """Create compute and render pipelines after layouts have been created."""
    self = renderer
    if self.tile_merge_bind_group_layout is not None:
        try:
            self.tile_merge_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.tile_merge_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=TILE_MERGE_SHADER), "entry_point": "combine_main"},
            )
        except Exception as exc:
            self.tile_merge_pipeline = None
            print(f"Warning: GPU tile merge pipeline could not be created ({exc!s}); using copy-based tile merges.", file=sys.stderr)
    
    try:
        self.mesh_visibility_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.mesh_visibility_bind_group_layout]),
            compute={"module": self.device.create_shader_module(code=GPU_VISIBILITY_SHADER), "entry_point": "main"},
        )
    except Exception as exc:
        self.mesh_visibility_pipeline = None
        self.use_gpu_visibility_culling = False
        print(f"Warning: GPU visibility pipeline could not be created ({exc!s}); using CPU visibility.", file=sys.stderr)
    
    
    worldspace_rc_trace_module = self.device.create_shader_module(code=WORLDSPACE_RC_TRACE_SHADER)
    worldspace_rc_filter_module = self.device.create_shader_module(code=WORLDSPACE_RC_FILTER_SHADER)
    self.worldspace_rc_trace_pipeline = self.device.create_compute_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[self.worldspace_rc_trace_bind_group_layout]),
        compute={"module": worldspace_rc_trace_module, "entry_point": "trace_main"},
    )
    self.worldspace_rc_filter_pipeline = self.device.create_compute_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[self.worldspace_rc_filter_bind_group_layout]),
        compute={"module": worldspace_rc_filter_module, "entry_point": "filter_main"},
    )
    
    if self._using_metal_meshing:
        self.voxel_mesh_count_pipeline = None
        self.voxel_mesh_scan_pipeline = None
        self.voxel_mesh_emit_pipeline = None
        self.voxel_surface_expand_pipeline = None
    else:
        try:
            self.voxel_mesh_count_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_mesh_count_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_MESH_BATCH_SHADER), "entry_point": "count_main"},
            )
            self.voxel_mesh_scan_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_mesh_scan_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_MESH_BATCH_SHADER), "entry_point": "scan_main"},
            )
            self.voxel_mesh_emit_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_mesh_emit_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_MESH_BATCH_SHADER), "entry_point": "emit_main"},
            )
            self.voxel_surface_expand_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_surface_expand_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_SURFACE_EXPAND_SHADER), "entry_point": "expand_main"},
            )
        except Exception as exc:
            self.voxel_mesh_count_pipeline = None
            self.voxel_mesh_scan_pipeline = None
            self.voxel_mesh_emit_pipeline = None
            self.voxel_surface_expand_pipeline = None
            self.use_gpu_meshing = False
            self.mesh_backend_label = "CPU"
            self._using_metal_meshing = False
            print(f"Warning: GPU meshing could not be created ({exc!s}); using CPU meshing.", file=sys.stderr)
    self.render_pipeline = self.device.create_render_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[self.render_bind_group_layout]),
        vertex={
            "module": self.device.create_shader_module(code=RENDER_SHADER),
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": VERTEX_STRIDE,
                    "step_mode": "vertex",
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": "float32x3"},
                        {"shader_location": 1, "offset": 12, "format": "float32x3"},
                        {"shader_location": 2, "offset": 24, "format": "float32x3"},
                    ],
                }
            ],
        },
        fragment={
            "module": self.device.create_shader_module(code=RENDER_SHADER),
            "entry_point": "fs_main",
            "targets": [{"format": self.color_format}],
        },
        primitive={
            "topology": "triangle-list",
            "cull_mode": "none",
        },
        depth_stencil={
            "format": DEPTH_FORMAT,
            "depth_write_enabled": True,
            "depth_compare": "less",
        },
    )
    self.scene_render_pipeline = self.device.create_render_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[self.render_bind_group_layout]),
        vertex={
            "module": self.device.create_shader_module(code=GI_GBUFFER_SHADER),
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": VERTEX_STRIDE,
                    "step_mode": "vertex",
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": "float32x3"},
                        {"shader_location": 1, "offset": 12, "format": "float32x3"},
                        {"shader_location": 2, "offset": 24, "format": "float32x3"},
                    ],
                }
            ],
        },
        fragment={
            "module": self.device.create_shader_module(code=GI_GBUFFER_SHADER),
            "entry_point": "fs_main",
            "targets": [
                {"format": POSTPROCESS_SCENE_FORMAT},
                {"format": POSTPROCESS_GBUFFER_FORMAT},
            ],
        },
        primitive={
            "topology": "triangle-list",
            "cull_mode": "none",
        },
        depth_stencil={
            "format": DEPTH_FORMAT,
            "depth_write_enabled": True,
            "depth_compare": "less",
        },
        multisample={"count": self.postprocess_msaa_sample_count},
    )
    self.gi_pipeline = self.device.create_render_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[self.gi_bind_group_layout]),
        vertex={
            "module": self.device.create_shader_module(code=GI_CASCADE_SHADER),
            "entry_point": "vs_main",
            "buffers": [],
        },
        fragment={
            "module": self.device.create_shader_module(code=GI_CASCADE_SHADER),
            "entry_point": "fs_main",
            "targets": [{"format": POSTPROCESS_GI_FORMAT}],
        },
        primitive={
            "topology": "triangle-list",
            "cull_mode": "none",
        },
    )
    self.gi_compose_pipeline = self.device.create_render_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[self.gi_compose_bind_group_layout]),
        vertex={
            "module": self.device.create_shader_module(code=GI_POSTPROCESS_SHADER),
            "entry_point": "vs_main",
            "buffers": [],
        },
        fragment={
            "module": self.device.create_shader_module(code=GI_POSTPROCESS_SHADER),
            "entry_point": "fs_main",
            "targets": [{"format": POSTPROCESS_GI_FORMAT}],
        },
        primitive={
            "topology": "triangle-list",
            "cull_mode": "none",
        },
    )
    self.final_present_pipeline = self.device.create_render_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[self.final_present_bind_group_layout]),
        vertex={
            "module": self.device.create_shader_module(code=FINAL_BLIT_SHADER),
            "entry_point": "vs_main",
            "buffers": [],
        },
        fragment={
            "module": self.device.create_shader_module(code=FINAL_BLIT_SHADER),
            "entry_point": "fs_main",
            "targets": [{"format": self.color_format}],
        },
        primitive={
            "topology": "triangle-list",
            "cull_mode": "none",
        },
    )
    
    self.profile_hud_pipeline = self.device.create_render_pipeline(
        layout=self.device.create_pipeline_layout(bind_group_layouts=[]),
        vertex={
            "module": self.device.create_shader_module(code=HUD_SHADER),
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": HUD_VERTEX_STRIDE,
                    "step_mode": "vertex",
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": "float32x3"},
                        {"shader_location": 1, "offset": 12, "format": "float32x4"},
                    ],
                }
            ],
        },
        fragment={
            "module": self.device.create_shader_module(code=HUD_SHADER),
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": self.color_format,
                    "blend": {
                        "color": {
                            "operation": "add",
                            "src_factor": "src-alpha",
                            "dst_factor": "one-minus-src-alpha",
                        },
                        "alpha": {
                            "operation": "add",
                            "src_factor": "one",
                            "dst_factor": "one-minus-src-alpha",
                        },
                    },
                }
            ],
        },
        primitive={
            "topology": "triangle-list",
            "cull_mode": "none",
        },
    )
