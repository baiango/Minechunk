from __future__ import annotations

import math
import time
import sys
from collections import OrderedDict, deque

import numpy as np
import wgpu
from . import chunk_generation_helpers as chunk_gen
from . import hud_profile_helpers as hud_profile
from . import mesh_cache_helpers as mesh_cache
from .renderer_config import *
from .render_shaders import (
    GPU_VISIBILITY_SHADER,
    HUD_SHADER,
    RENDER_SHADER,
    TILE_MERGE_SHADER,
    VOXEL_MESH_BATCH_SHADER,
    VOXEL_SURFACE_EXPAND_SHADER,
)
from .render_utils import (
    clamp,
    cross3,
    flat_forward_vector,
    forward_vector,
    normalize3,
    pack_camera_uniform,
    right_vector,
)
from . import wgpu_chunk_mesher as wgpu_mesher
from . import metal_chunk_mesher as metal_mesher
try:
    from wgpu.backends.wgpu_native import multi_draw_indirect as wgpu_native_multi_draw_indirect
except Exception:
    wgpu_native_multi_draw_indirect = None
from rendercanvas.auto import RenderCanvas, loop

from .meshing_types import (
    AsyncVoxelMeshBatchResources,
    ChunkDrawBatch,
    ChunkMesh,
    ChunkRenderBatch,
    MeshBufferAllocation,
    MeshOutputSlab,
    PendingChunkMeshBatch,
)
from .voxel_world import CHUNK_SIZE, WORLD_HEIGHT, VoxelWorld
from .terrain_backend import ChunkSurfaceGpuBatch

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


def _engine_mode_uses_gpu_path() -> bool:
    return engine_mode != ENGINE_MODE_CPU


class TerrainRenderer:
    def __init__(
        self,
        seed: int = 1337,
        use_gpu_terrain: bool | None = None,
        use_gpu_meshing: bool | None = None,
        terrain_batch_size: int = 256,
        mesh_batch_size: int | None = None,
    ) -> None:
        default_use_gpu = _engine_mode_uses_gpu_path()
        self.use_gpu_terrain = default_use_gpu if use_gpu_terrain is None else bool(use_gpu_terrain)
        self.use_gpu_meshing = default_use_gpu if use_gpu_meshing is None else bool(use_gpu_meshing)
        self.terrain_batch_size = max(1, int(terrain_batch_size))
        if mesh_batch_size is None:
            self.mesh_batch_size = max(1, int(self.terrain_batch_size))
        else:
            self.mesh_batch_size = max(1, int(mesh_batch_size))
        self._pending_surface_gpu_batch_target_multiplier = 10
        self.base_title = "Minechunk"
        self.engine_mode_label = engine_mode
        self.canvas = RenderCanvas(
            title=self.base_title,
            size=(1280, 800),
            update_mode="continuous",
            max_fps=SWAPCHAIN_MAX_FPS,
            vsync=SWAPCHAIN_USE_VSYNC,
        )
        request_adapter = getattr(wgpu.gpu, "request_adapter_sync", wgpu.gpu.request_adapter)
        self.adapter = request_adapter(canvas=self.canvas, power_preference="high-performance")
        if self.adapter is None:
            raise RuntimeError("No compatible GPU adapter was found.")
        request_device = getattr(self.adapter, "request_device_sync", self.adapter.request_device)
        supported_features = set(getattr(self.adapter, "features", []) or [])
        requested_features = []
        if {"timestamp-query", "timestamp-query-inside-passes"}.issubset(supported_features):
            requested_features = ["timestamp-query", "timestamp-query-inside-passes"]
        self.timestamp_query_supported = bool(requested_features)
        if requested_features:
            self.device = request_device(required_features=requested_features)
        else:
            self.device = request_device()
        self.context = self.canvas.get_wgpu_context()
        self.color_format = self.context.get_preferred_format(self.adapter)
        self.render_api_label = self._describe_render_api()
        self.render_backend_label = "Wgpu"
        self.context.configure(
            device=self.device,
            format=self.color_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            alpha_mode="opaque",
        )

        self.world = VoxelWorld(
            seed,
            gpu_device=self.device,
            prefer_gpu_terrain=self.use_gpu_terrain,
            prefer_metal_backend=engine_mode == ENGINE_MODE_METAL,
            terrain_batch_size=self.terrain_batch_size,
        )
        self.mesh_backend_label = "CPU"
        self._using_metal_meshing = bool(self.use_gpu_meshing and self.world.terrain_backend_label() == "Metal")
        if self.use_gpu_meshing:
            self.mesh_backend_label = "Metal" if self._using_metal_meshing else "Wgpu"
        if self._using_metal_meshing:
            metal_mesher.prewarm_metal_chunk_mesher(self)
        if self.use_gpu_terrain and self.world.terrain_backend_label() == "Metal":
            print(
                "Info: Metal terrain backend active; renderer will use the Metal mesher for chunk meshing.",
                file=sys.stderr,
            )
        self._log_backend_diagnostics()

        self.camera = Camera(position=[0.0, 200.0, 0.0], yaw=math.pi, pitch=-1.20)
        self.keys_down: set[str] = set()
        self.dragging = False
        self.last_pointer: tuple[float, float] | None = None
        self.last_frame_time = time.perf_counter()
        self.depth_texture = None
        self.depth_view = None
        self.depth_size = (0, 0)
        # 512 blocks rounds up to a whole number of chunks.
        self.chunk_radius = max(1, math.ceil(DEFAULT_RENDER_DISTANCE_BLOCKS / CHUNK_SIZE))
        self.render_dimension_chunks = self.chunk_radius * 2 + 1
        self.chunk_cache: OrderedDict[tuple[int, int], ChunkMesh] = OrderedDict()
        self._visible_chunk_coords: list[tuple[int, int]] = []
        self._visible_chunk_coord_set: set[tuple[int, int]] = set()
        self._visible_chunk_origin: tuple[int, int] | None = None
        self._visible_tile_keys: list[tuple[int, int]] = []
        self._visible_tile_coords: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        self._visible_tile_masks: dict[tuple[int, int], int] = {}
        self._visible_displayed_coords: set[tuple[int, int]] = set()
        self._visible_missing_coords: set[tuple[int, int]] = set()
        self._chunk_request_queue: deque[tuple[int, int]] = deque()
        self._chunk_request_queue_origin: tuple[int, int] | None = None
        self._chunk_request_queue_dirty = True
        self._pending_chunk_coords: set[tuple[int, int]] = set()
        self._transient_render_buffers: list[list[wgpu.GPUBuffer]] = []
        self._tile_render_batches: dict[tuple[int, int], ChunkRenderBatch] = {}
        self._tile_dirty_keys: set[tuple[int, int]] = set()
        self._tile_versions: dict[tuple[int, int], int] = {}
        self._visible_layout_version = 0
        self._tile_mutation_version = 0
        self._cached_tile_draw_batches: dict[tuple[int, int, int], tuple[list[ChunkDrawBatch], int, int, int]] = {}
        self._mesh_buffer_refs: dict[int, int] = {}
        self._mesh_output_slabs: OrderedDict[int, MeshOutputSlab] = OrderedDict()
        self._mesh_allocations: dict[int, MeshBufferAllocation] = {}
        self._deferred_mesh_output_frees: deque[tuple[int, int, int, int]] = deque()
        self._next_mesh_output_slab_id = 1
        self._mesh_output_append_slab_id: int | None = None
        self._next_mesh_allocation_id = 1
        self._mesh_output_binding_alignment = math.lcm(
            VERTEX_STRIDE,
            max(1, int(self._device_limit("min_storage_buffer_offset_alignment", 256))),
        )
        self._mesh_output_min_slab_bytes = max(
            MESH_OUTPUT_SLAB_MIN_BYTES,
            self._mesh_output_binding_alignment,
        )
        self.use_gpu_indirect_render = True
        self._mesh_draw_indirect_capacity = 0
        self._mesh_draw_indirect_buffer = None
        self._mesh_draw_indirect_array = np.empty((0, 4), dtype=np.uint32)
        self.use_gpu_visibility_culling = True
        self._mesh_visibility_record_capacity = 0
        self._mesh_visibility_record_buffer = None
        self._mesh_visibility_record_array = np.empty(0, dtype=MESH_VISIBILITY_RECORD_DTYPE)
        self._mesh_visibility_params_buffer = None
        self.max_cached_chunks = MAX_CACHED_CHUNKS
        self._cache_capacity_warned = False
        self._current_move_speed = self.camera.move_speed
        self.mesh_backend_label = (
            "Metal" if self._using_metal_meshing else ("Wgpu" if self.use_gpu_meshing else "CPU")
        )
        self.voxel_mesh_scan_validate_every = 0
        self._voxel_mesh_scan_batches_processed = 0
        self._pending_voxel_mesh_results: deque = deque()
        self._pending_surface_gpu_batches: deque[ChunkSurfaceGpuBatch] = deque()
        self._pending_surface_gpu_batches_chunk_total = 0
        self._pending_surface_gpu_batch_age_seconds = 1.0 / 60.0
        self._pending_surface_gpu_batch_target_chunks = (
            max(self.mesh_batch_size, self.terrain_batch_size) * self._pending_surface_gpu_batch_target_multiplier
        )
        self._async_voxel_mesh_batch_pool_limit = 48
        self._gpu_mesh_async_inflight_limit = max(32, self.mesh_batch_size * 6)
        self._gpu_mesh_async_finalize_budget = max(1, self.mesh_batch_size * 2)
        self._metal_gpu_mesh_async_finalize_budget = max(1, min(2, self.mesh_batch_size))
        self._pending_surface_gpu_batches_first_enqueued_at = 0.0
        self._pending_gpu_mesh_batches: deque[PendingChunkMeshBatch] = deque()
        self._gpu_mesh_deferred_buffer_cleanup: deque[tuple[int, list[wgpu.GPUBuffer]]] = deque()
        self._gpu_mesh_deferred_batch_resource_releases: deque[tuple[int, AsyncVoxelMeshBatchResources]] = deque()
        self._gpu_mesh_deferred_surface_batch_releases: deque[tuple[int, list[object]]] = deque()
        self._async_voxel_mesh_batch_pool: deque[AsyncVoxelMeshBatchResources] = deque()
        self._voxel_surface_expand_bind_group_cache: OrderedDict[tuple[int, int, int, int, int], object] = OrderedDict()
        self.profiling_enabled = False
        self.profile_window_start = 0.0
        self.profile_next_report = 0.0
        self.profile_window_cpu_ms = 0.0
        self.profile_window_frames = 0
        self.profile_window_frame_times: list[float] = []
        self.profile_hud_lines: list[str] = []
        self.profile_hud_vertex_bytes = b""
        self.profile_hud_vertex_count = 0
        self.profile_hud_vertex_buffer = None
        self.profile_hud_vertex_buffer_capacity = 0
        self._shutdown_complete = False
        self._hud_geometry_cache: OrderedDict[tuple[bool, int, int, tuple[str, ...]], tuple[bytes, int]] = OrderedDict()
        self.frame_breakdown_samples: dict[str, deque[float]] = {
            "world_update": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visibility_lookup": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_stream": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_stream_bytes": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_displayed_added": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "camera_upload": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "swapchain_acquire": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "render_encode": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "command_finish": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "queue_submit": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "wall_frame": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "draw_calls": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "merged_chunks": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visible_vertices": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visible_chunk_targets": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visible_chunks": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "pending_chunk_requests": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "voxel_mesh_backlog": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
        }
        self.frame_breakdown_lines: list[str] = []
        self.frame_breakdown_vertex_bytes = b""
        self.frame_breakdown_vertex_count = 0
        self.frame_breakdown_vertex_buffer = None
        self.frame_breakdown_vertex_buffer_capacity = 0
        self._last_frame_draw_calls = 0
        self._last_frame_merged_batches = 0
        self._last_new_displayed_chunks = 0
        self._last_chunk_stream_drained = 0
        self._last_frame_visible_batches = 0
        self._last_displayed_chunk_coords: set[tuple[int, int]] = set()
        self._voxel_mesh_scratch_capacity = 0
        self._voxel_mesh_scratch_sample_size = 0
        self._voxel_mesh_scratch_height_limit = 0
        self._voxel_mesh_scratch_blocks_buffer = None
        self._voxel_mesh_scratch_materials_buffer = None
        self._voxel_mesh_scratch_coords_buffer = None
        self._voxel_mesh_scratch_column_totals_buffer = None
        self._voxel_mesh_scratch_chunk_totals_buffer = None
        self._voxel_mesh_scratch_chunk_offsets_buffer = None
        self._voxel_mesh_scratch_params_buffer = None
        self._voxel_mesh_scratch_batch_vertex_buffer = None
        self._voxel_mesh_scratch_blocks_array = None
        self._voxel_mesh_scratch_materials_array = None
        self._voxel_mesh_scratch_coords_array = None
        self._voxel_mesh_scratch_chunk_totals_array = None
        self._voxel_mesh_scratch_chunk_offsets_array = None
        self._voxel_mesh_scratch_chunk_metadata_readback_buffer = None

        self.camera_buffer = self.device.create_buffer(
            size=80,
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
        if self._device_limit("max_storage_buffers_per_shader_stage", 0) >= MERGED_TILE_MAX_CHUNKS + 2:
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
                    "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 80},
                }
            ],
        )

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
                            {"shader_location": 0, "offset": 0, "format": "float32x4"},
                            {"shader_location": 1, "offset": 16, "format": "float32x4"},
                            {"shader_location": 2, "offset": 32, "format": "float32x4"},
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
        self.profile_hud_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[]),
            vertex={
                "module": self.device.create_shader_module(code=HUD_SHADER),
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": VERTEX_STRIDE,
                        "step_mode": "vertex",
                        "attributes": [
                            {"shader_location": 0, "offset": 0, "format": "float32x4"},
                            {"shader_location": 1, "offset": 16, "format": "float32x4"},
                            {"shader_location": 2, "offset": 32, "format": "float32x4"},
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

        self.canvas.add_event_handler(self._handle_key_down, "key_down")
        self.canvas.add_event_handler(self._handle_key_up, "key_up")
        self.canvas.add_event_handler(self._handle_pointer_down, "pointer_down")
        self.canvas.add_event_handler(self._handle_pointer_move, "pointer_move")
        self.canvas.add_event_handler(self._handle_pointer_up, "pointer_up")
        self.canvas.add_event_handler(self._handle_resize, "resize")

        self._disable_profiling()
        self.canvas.request_draw(self.draw_frame)

    def run(self) -> None:
        try:
            loop.run()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        try:
            metal_mesher.shutdown_renderer_async_state(self)
        except Exception:
            pass
        try:
            self.world.destroy()
        except Exception:
            pass

    def _handle_resize(self, event) -> None:
        self.depth_size = (0, 0)
        self.depth_texture = None
        self.depth_view = None
        if self.profiling_enabled and self.profile_hud_lines:
            self.profile_hud_vertex_bytes, self.profile_hud_vertex_count = hud_profile.build_profile_hud_vertices(self, self.profile_hud_lines)
        if self.profiling_enabled and self.frame_breakdown_lines:
            self.frame_breakdown_vertex_bytes, self.frame_breakdown_vertex_count = hud_profile.build_frame_breakdown_hud_vertices(self, self.frame_breakdown_lines)

    def _normalize_key(self, event) -> str:
        key = str(event.get("key", "")).strip().lower()
        if key in {" ", "spacebar"}:
            return "space"
        if key in {"controlleft", "controlright", "ctrl"}:
            return "control"
        if key == "shiftleft":
            return "shiftleft"
        if key == "shiftright":
            return "shiftright"
        return key

    def _handle_key_down(self, event) -> None:
        key = self._normalize_key(event)
        is_new_press = key not in self.keys_down
        self.keys_down.add(key)
        if is_new_press and key == "f3":
            self._toggle_profiling()
        if is_new_press and key in {"r"}:
            self.regenerate_world()

    def _handle_key_up(self, event) -> None:
        self.keys_down.discard(self._normalize_key(event))

    def _handle_pointer_down(self, event) -> None:
        if int(event.get("button", 0)) == 1:
            self.dragging = True
            self.last_pointer = (float(event.get("x", 0.0)), float(event.get("y", 0.0)))

    def _handle_pointer_move(self, event) -> None:
        if not self.dragging:
            return
        x = float(event.get("x", 0.0))
        y = float(event.get("y", 0.0))
        if self.last_pointer is None:
            self.last_pointer = (x, y)
            return
        last_x, last_y = self.last_pointer
        dx = x - last_x
        dy = y - last_y
        self.last_pointer = (x, y)
        self.camera.yaw -= dx * self.camera.look_speed
        self.camera.pitch -= dy * self.camera.look_speed
        self.camera.clamp_pitch()

    def _handle_pointer_up(self, event) -> None:
        if int(event.get("button", 0)) == 1:
            self.dragging = False
            self.last_pointer = None

    def _key_active(self, *names: str) -> bool:
        for name in names:
            if name in self.keys_down:
                return True
        return False

    def _toggle_profiling(self) -> None:
        if self.profiling_enabled:
            self._disable_profiling()
        else:
            self._enable_profiling()

    def _enable_profiling(self) -> None:
        self.profiling_enabled = True
        now = time.perf_counter()
        self.profile_window_start = now
        self.profile_next_report = now + PROFILE_REPORT_INTERVAL
        self.profile_window_cpu_ms = 0.0
        self.profile_window_frames = 0
        self.profile_window_frame_times = []
        for samples in self.frame_breakdown_samples.values():
            samples.clear()
        self.profile_hud_lines = []
        self.profile_hud_vertex_bytes = b""
        self.profile_hud_vertex_count = 0
        self.frame_breakdown_lines = [
            f"FRAME BREAKDOWN @ DIMENSION {self.render_dimension_chunks}x{self.render_dimension_chunks} CHUNKS",
            "CPU FRAME ISSUE: --.- MS",
            "  WORLD UPDATE: --.- MS",
            "  VISIBILITY LOOKUP: --.- MS",
            "  CHUNK STREAM: --.- MS",
            "  CHUNK STREAM BW: --.- MIB/S",
            "  CAMERA UPLOAD: --.- MS",
            "  SWAPCHAIN ACQUIRE: --.- MS",
            "  RENDER ENCODE: --.- MS",
            "  COMMAND FINISH: --.- MS",
            "  QUEUE SUBMIT: --.- MS",
            f"CHUNK DIMS: {CHUNK_SIZE}x{WORLD_HEIGHT}x{CHUNK_SIZE}",
            f"BACKEND POLL SIZE: {self.terrain_batch_size}",
            f"MESH DRAIN SIZE: {self.mesh_batch_size}",
            f"PRESENT PACING: FPS {SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if SWAPCHAIN_USE_VSYNC else 'OFF'}",
            "MESH SLABS: --  USED --.- MIB  FREE --.- MIB",
            "MESH BIGGEST GAP: --.- MIB  ALLOCS --",
            "TOTAL DRAW VERTICES: --",
            "WALL FRAME: --.- MS",
            "CHUNK VRAM: -- BYTES (--.- MIB)",
            "DRAW CALLS: --",
            "VISIBLE MERGED CHUNKS (VISIBLE ONLY): --",
        ]
        self.frame_breakdown_vertex_bytes, self.frame_breakdown_vertex_count = hud_profile.build_frame_breakdown_hud_vertices(self, self.frame_breakdown_lines)

    def _disable_profiling(self) -> None:
        self.profiling_enabled = False
        self.profile_window_start = 0.0
        self.profile_next_report = 0.0
        self.profile_window_cpu_ms = 0.0
        self.profile_window_frames = 0
        self.profile_window_frame_times = []
        self.profile_hud_lines = []
        self.profile_hud_vertex_bytes = b""
        self.profile_hud_vertex_count = 0
        self.frame_breakdown_lines = []
        self.frame_breakdown_vertex_bytes = b""
        self.frame_breakdown_vertex_count = 0
        self._hud_geometry_cache.clear()
        for samples in self.frame_breakdown_samples.values():
            samples.clear()

    def _describe_render_api(self) -> str:
        info = getattr(self.adapter, "info", None)
        summary = getattr(self.adapter, "summary", "")

        backend = ""
        adapter_type = ""
        description = ""

        if info is not None:
            getter = info.get if hasattr(info, "get") else None
            if getter is not None:
                backend = getter("backend_type", "") or getter("backend", "")
                adapter_type = getter("adapter_type", "") or getter("device_type", "")
                description = getter("description", "") or getter("device", "")
            else:
                backend = getattr(info, "backend_type", "") or getattr(info, "backend", "")
                adapter_type = getattr(info, "adapter_type", "") or getattr(info, "device_type", "")
                description = getattr(info, "description", "") or getattr(info, "device", "")

        if isinstance(summary, str) and summary.strip():
            if not backend and not adapter_type and not description:
                return summary.strip()

        parts = [part for part in (backend, adapter_type, description) if part]
        if parts:
            return " / ".join(parts)
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        return "unknown"

    def _log_backend_diagnostics(self) -> None:
        terrain_backend = getattr(self.world, "_backend", None)
        terrain_device = getattr(terrain_backend, "device", None)
        mesh_backend_label = getattr(self, "mesh_backend_label", "CPU")
        print(
            "Info: Backend diagnostics: "
            f"render_backend={self.render_backend_label}, "
            f"render_device={type(self.device).__name__}, "
            f"terrain_backend={type(terrain_backend).__name__}, "
            f"terrain_device={type(terrain_device).__name__}, "
            f"mesh_backend={mesh_backend_label}",
            file=sys.stderr,
        )

    def _device_limit(self, name: str, default: int) -> int:
        limits = getattr(self.device, "limits", None)
        if limits is None:
            return int(default)
        getter = limits.get if hasattr(limits, "get") else None
        if getter is not None:
            value = getter(name, default)
        else:
            value = getattr(limits, name, default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def _align_up(self, value: int, alignment: int) -> int:
        if alignment <= 1:
            return int(value)
        return ((int(value) + alignment - 1) // alignment) * alignment

    def regenerate_world(self) -> None:
        for mesh in self.chunk_cache.values():
            mesh_cache.release_chunk_mesh_storage(self, mesh)
        self.chunk_cache.clear()
        self._mesh_buffer_refs.clear()
        self._pending_chunk_coords.clear()
        self._chunk_request_queue.clear()
        self._chunk_request_queue_origin = None
        self._chunk_request_queue_dirty = True
        self._pending_voxel_mesh_results.clear()
        self._pending_surface_gpu_batches.clear()
        self._pending_surface_gpu_batches_chunk_total = 0
        self._pending_surface_gpu_batches_first_enqueued_at = 0.0
        self._voxel_surface_expand_bind_group_cache.clear()
        while self._pending_gpu_mesh_batches:
            pending = self._pending_gpu_mesh_batches.popleft()
            if pending.readback_buffer.map_state != "unmapped":
                try:
                    pending.readback_buffer.unmap()
                except Exception:
                    pass
            if pending.resources is not None:
                wgpu_mesher.destroy_async_voxel_mesh_batch_resources(pending.resources)
                continue
            for buffer in (
                pending.blocks_buffer,
                pending.materials_buffer,
                pending.coords_buffer,
                pending.column_totals_buffer,
                pending.chunk_totals_buffer,
                pending.chunk_offsets_buffer,
                pending.params_buffer,
                pending.readback_buffer,
            ):
                try:
                    buffer.destroy()
                except Exception:
                    pass
        while self._gpu_mesh_deferred_buffer_cleanup:
            _, buffers = self._gpu_mesh_deferred_buffer_cleanup.popleft()
            for buffer in buffers:
                try:
                    buffer.destroy()
                except Exception:
                    pass
        while self._gpu_mesh_deferred_batch_resource_releases:
            _, resources = self._gpu_mesh_deferred_batch_resource_releases.popleft()
            wgpu_mesher.destroy_async_voxel_mesh_batch_resources(resources)
        while self._async_voxel_mesh_batch_pool:
            wgpu_mesher.destroy_async_voxel_mesh_batch_resources(self._async_voxel_mesh_batch_pool.popleft())
        mesh_cache.clear_tile_render_batches(self)
        self._clear_transient_render_buffers()
        self._visible_chunk_origin = None
        try:
            metal_mesher.shutdown_renderer_async_state(self)
        except Exception:
            pass
        try:
            self.world.destroy()
        except Exception:
            pass
        self.world = VoxelWorld(
            int(time.time()) & 0x7FFFFFFF,
            gpu_device=self.device,
            prefer_gpu_terrain=self.use_gpu_terrain,
            prefer_metal_backend=engine_mode == ENGINE_MODE_METAL,
            terrain_batch_size=self.terrain_batch_size,
        )
        self._using_metal_meshing = bool(self.use_gpu_meshing and self.world.terrain_backend_label() == "Metal")
        if self.use_gpu_meshing:
            self.mesh_backend_label = "Metal" if self._using_metal_meshing else "Wgpu"
        if self._using_metal_meshing:
            metal_mesher.prewarm_metal_chunk_mesher(self)
        if self.use_gpu_terrain and self.world.terrain_backend_label() == "Metal":
            print(
                "Info: Metal terrain backend active; renderer will use the Metal mesher for chunk meshing.",
                file=sys.stderr,
            )
        self._log_backend_diagnostics()
        self.camera.position[:] = [0.0, 200.0, 0.0]
        self.camera.yaw = math.pi
        self.camera.pitch = -1.20
        self.camera.clamp_pitch()

    def _clear_transient_render_buffers(self) -> None:
        for buffer_group in self._transient_render_buffers:
            for buffer in buffer_group:
                buffer.destroy()
        self._transient_render_buffers.clear()

    def _retain_mesh_buffer(self, buffer: wgpu.GPUBuffer) -> None:
        key = id(buffer)
        self._mesh_buffer_refs[key] = self._mesh_buffer_refs.get(key, 0) + 1

    def _release_mesh_buffer(self, buffer: wgpu.GPUBuffer) -> None:
        key = id(buffer)
        refs = self._mesh_buffer_refs.get(key, 0)
        if refs <= 1:
            self._mesh_buffer_refs.pop(key, None)
            buffer.destroy()
        else:
            self._mesh_buffer_refs[key] = refs - 1

    def _update_camera(self, dt: float) -> None:
        sprinting = self._key_active("shift", "shiftleft", "shiftright")
        speed = SPRINT_FLY_SPEED if sprinting else self.camera.move_speed
        self._current_move_speed = float(speed)
        move = [0.0, 0.0, 0.0]

        forward = flat_forward_vector(self.camera.yaw)
        right = right_vector(self.camera.yaw)

        if self._key_active("w", "arrowup"):
            move[0] += forward[0]
            move[2] += forward[2]
        if self._key_active("s", "arrowdown"):
            move[0] -= forward[0]
            move[2] -= forward[2]
        if self._key_active("d", "arrowright"):
            move[0] += right[0]
            move[2] += right[2]
        if self._key_active("a", "arrowleft"):
            move[0] -= right[0]
            move[2] -= right[2]
        if self._key_active("x"):
            move[1] += 1.0
        if self._key_active("z"):
            move[1] -= 1.0

        length = math.sqrt(move[0] * move[0] + move[1] * move[1] + move[2] * move[2])
        if length > 0.0:
            scale = speed * dt / length
            self.camera.position[0] += move[0] * scale
            self.camera.position[1] += move[1] * scale
            self.camera.position[2] += move[2] * scale

        self.camera.position[1] = clamp(self.camera.position[1], 4.0, self.world.height + 48.0)

    def _ensure_depth_buffer(self) -> None:
        width, height = self.canvas.get_physical_size()
        if (width, height) == self.depth_size:
            return
        self.depth_size = (width, height)
        self.depth_texture = self.device.create_texture(
            size=(max(1, width), max(1, height), 1),
            format=DEPTH_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.depth_view = self.depth_texture.create_view()

    def _camera_basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        forward = normalize3(forward_vector(self.camera.yaw, self.camera.pitch))
        world_up = (0.0, 1.0, 0.0)
        right = cross3(forward, world_up)
        if right == (0.0, 0.0, 0.0):
            right = right_vector(self.camera.yaw)
        right = normalize3(right)
        up = normalize3(cross3(right, forward))
        return right, up, forward

    def _chunk_coords_in_view(self) -> list[tuple[int, int]]:
        chunk_x = int(self.camera.position[0] // CHUNK_SIZE)
        chunk_z = int(self.camera.position[2] // CHUNK_SIZE)
        coords: list[tuple[int, int]] = []
        for dz in range(-self.chunk_radius, self.chunk_radius + 1):
            for dx in range(-self.chunk_radius, self.chunk_radius + 1):
                coords.append((chunk_x + dx, chunk_z + dz))
        return coords

    def _visible_chunk_bounds(self, origin: tuple[int, int]) -> tuple[int, int, int, int]:
        radius = int(self.chunk_radius)
        return (
            int(origin[0] - radius),
            int(origin[0] + radius),
            int(origin[1] - radius),
            int(origin[1] + radius),
        )

    def _tile_key_for_chunk(self, chunk_x: int, chunk_z: int) -> tuple[int, int]:
        tile_size = int(MERGED_TILE_SIZE_CHUNKS)
        return (int(chunk_x) // tile_size, int(chunk_z) // tile_size)

    def _tile_bit_for_chunk(self, chunk_x: int, chunk_z: int) -> tuple[tuple[int, int], int]:
        tile_size = int(MERGED_TILE_SIZE_CHUNKS)
        tile_key_value = self._tile_key_for_chunk(int(chunk_x), int(chunk_z))
        local_x = int(chunk_x) - tile_key_value[0] * tile_size
        local_z = int(chunk_z) - tile_key_value[1] * tile_size
        if 0 <= local_x < tile_size and 0 <= local_z < tile_size:
            return tile_key_value, 1 << (local_z * tile_size + local_x)
        return tile_key_value, 0

    def _rebuild_visible_tile_layout_from_coords(self) -> None:
        tile_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        tile_masks: dict[tuple[int, int], int] = {}
        for chunk_x, chunk_z in self._visible_chunk_coords:
            tile_key_value, tile_bit = self._tile_bit_for_chunk(int(chunk_x), int(chunk_z))
            tile_groups.setdefault(tile_key_value, []).append((int(chunk_x), int(chunk_z)))
            if tile_bit != 0:
                tile_masks[tile_key_value] = int(tile_masks.get(tile_key_value, 0)) | int(tile_bit)
        self._visible_tile_keys = sorted(tile_groups)
        self._visible_tile_coords = {
            key: tuple(sorted(coords))
            for key, coords in tile_groups.items()
        }
        self._visible_tile_masks = tile_masks

    def _apply_visible_chunk_coord_delta(self, new_origin: tuple[int, int]) -> bool:
        old_origin = self._visible_chunk_origin
        old_coords = self._visible_chunk_coord_set
        if old_origin is None or not old_coords:
            return False

        dx = int(new_origin[0] - old_origin[0])
        dz = int(new_origin[1] - old_origin[1])
        if dx == 0 and dz == 0:
            return True
        if max(abs(dx), abs(dz)) > 4:
            return False

        old_min_x, old_max_x, old_min_z, old_max_z = self._visible_chunk_bounds(old_origin)
        new_min_x, new_max_x, new_min_z, new_max_z = self._visible_chunk_bounds(new_origin)
        leaving: set[tuple[int, int]] = set()
        entering: set[tuple[int, int]] = set()

        if dx > 0:
            for chunk_x in range(old_min_x, min(old_max_x + 1, old_min_x + dx)):
                for chunk_z in range(old_min_z, old_max_z + 1):
                    leaving.add((chunk_x, chunk_z))
            for chunk_x in range(max(old_max_x + 1, new_min_x), new_max_x + 1):
                for chunk_z in range(new_min_z, new_max_z + 1):
                    entering.add((chunk_x, chunk_z))
        elif dx < 0:
            step = -dx
            for chunk_x in range(max(old_min_x, old_max_x - step + 1), old_max_x + 1):
                for chunk_z in range(old_min_z, old_max_z + 1):
                    leaving.add((chunk_x, chunk_z))
            for chunk_x in range(new_min_x, min(new_max_x + 1, old_min_x)):
                for chunk_z in range(new_min_z, new_max_z + 1):
                    entering.add((chunk_x, chunk_z))

        if dz > 0:
            for chunk_z in range(old_min_z, min(old_max_z + 1, old_min_z + dz)):
                for chunk_x in range(old_min_x, old_max_x + 1):
                    leaving.add((chunk_x, chunk_z))
            for chunk_z in range(max(old_max_z + 1, new_min_z), new_max_z + 1):
                for chunk_x in range(new_min_x, new_max_x + 1):
                    entering.add((chunk_x, chunk_z))
        elif dz < 0:
            step = -dz
            for chunk_z in range(max(old_min_z, old_max_z - step + 1), old_max_z + 1):
                for chunk_x in range(old_min_x, old_max_x + 1):
                    leaving.add((chunk_x, chunk_z))
            for chunk_z in range(new_min_z, min(new_max_z + 1, old_min_z)):
                for chunk_x in range(new_min_x, new_max_x + 1):
                    entering.add((chunk_x, chunk_z))

        if not leaving and not entering:
            return False

        updated_coords = set(old_coords)
        updated_coords.difference_update(leaving)
        updated_coords.update(entering)

        tile_groups = {
            key: set(coords)
            for key, coords in self._visible_tile_coords.items()
        }
        tile_masks = dict(self._visible_tile_masks)

        for chunk_x, chunk_z in leaving:
            tile_key_value, tile_bit = self._tile_bit_for_chunk(chunk_x, chunk_z)
            coords = tile_groups.get(tile_key_value)
            if coords is not None:
                coords.discard((chunk_x, chunk_z))
                if not coords:
                    tile_groups.pop(tile_key_value, None)
            if tile_bit != 0:
                next_mask = int(tile_masks.get(tile_key_value, 0)) & ~int(tile_bit)
                if next_mask != 0:
                    tile_masks[tile_key_value] = next_mask
                else:
                    tile_masks.pop(tile_key_value, None)

        for chunk_x, chunk_z in entering:
            tile_key_value, tile_bit = self._tile_bit_for_chunk(chunk_x, chunk_z)
            tile_groups.setdefault(tile_key_value, set()).add((chunk_x, chunk_z))
            if tile_bit != 0:
                tile_masks[tile_key_value] = int(tile_masks.get(tile_key_value, 0)) | int(tile_bit)

        self._visible_chunk_origin = new_origin
        self._visible_chunk_coord_set = updated_coords
        self._visible_chunk_coords = list(updated_coords)
        self._visible_tile_keys = sorted(tile_groups)
        self._visible_tile_coords = {
            key: tuple(sorted(coords))
            for key, coords in tile_groups.items()
        }
        self._visible_tile_masks = tile_masks
        self._visible_layout_version += 1
        self._cached_tile_draw_batches.clear()
        return True

    def _refresh_visible_chunk_coords(self) -> None:
        new_origin = (
            int(self.camera.position[0] // CHUNK_SIZE),
            int(self.camera.position[2] // CHUNK_SIZE),
        )
        if self._apply_visible_chunk_coord_delta(new_origin):
            return
        self._visible_chunk_origin = new_origin
        self._visible_chunk_coords = self._chunk_coords_in_view()
        self._visible_chunk_coord_set = set(self._visible_chunk_coords)
        self._rebuild_visible_tile_layout_from_coords()
        self._visible_layout_version += 1
        self._cached_tile_draw_batches.clear()

    def _warn_if_visible_exceeds_cache(self) -> None:
        visible_count = len(self._visible_chunk_coords)
        if self._cache_capacity_warned or visible_count <= self.max_cached_chunks:
            return
        self._cache_capacity_warned = True
        print(
            f"Warning: visible chunk count ({visible_count}) exceeds cache capacity "
            f"({self.max_cached_chunks}). Expect missing chunks, evictions, or flashing.",
            file=sys.stderr,
        )

    @profile
    def _submit_render(self):
        self._ensure_depth_buffer()
        right, up, forward = self._camera_basis()
        camera_upload_start = time.perf_counter()
        self.device.queue.write_buffer(
            self.camera_buffer,
            0,
            pack_camera_uniform(
                tuple(self.camera.position),
                right,
                up,
                forward,
                1.0 / math.tan(math.radians(90.0) * 0.5),
                max(1.0, self.canvas.get_physical_size()[0] / max(1.0, float(self.canvas.get_physical_size()[1]))),
                0.1,
                1024.0,
                LIGHT_DIRECTION,
            ),
        )
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
                        {"binding": 2, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 80}},
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

        if use_gpu_visibility or use_indirect:
            indirect_buffer = self._mesh_draw_indirect_buffer
            assert indirect_buffer is not None
            for vertex_buffer, binding_offset, batch_start, batch_count in visible_batches:
                render_pass.set_vertex_buffer(0, vertex_buffer, binding_offset)
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
                    render_pass.draw_indirect(indirect_buffer, indirect_offset)
        else:
            for vertex_buffer, vertex_count, vertex_offset in visible_batches:
                render_pass.set_vertex_buffer(0, vertex_buffer, vertex_offset)
                render_pass.draw(vertex_count, 1, 0, 0)

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

    def draw_frame(self) -> None:
        frame_start = time.perf_counter()
        now = frame_start
        dt = min(0.05, now - self.last_frame_time)
        self.last_frame_time = now
        profile_started_at = hud_profile.profile_begin_frame(self)
        try:
            update_start = time.perf_counter()
            self._update_camera(dt)
            world_update_ms = (time.perf_counter() - update_start) * 1000.0

            visibility_lookup_ms = chunk_gen.refresh_visible_chunk_set(self)
            encoder, color_view, render_stats = self._submit_render()
            visibility_lookup_ms += render_stats["visibility_lookup_ms"]
            camera_upload_ms = render_stats["camera_upload_ms"]
            swapchain_acquire_ms = render_stats["swapchain_acquire_ms"]
            render_encode_ms = render_stats["render_encode_ms"]
            draw_calls = int(render_stats["draw_calls"])
            merged_chunks = int(render_stats["merged_chunks"])
            visible_chunks = int(render_stats["visible_chunks"])
            visible_vertices = int(render_stats["visible_vertices"])

            hud_profile.draw_profile_hud(self, encoder, color_view)
            hud_profile.draw_frame_breakdown_hud(self, encoder, color_view)

            command_finish_start = time.perf_counter()
            command_buffer = encoder.finish()
            command_finish_ms = (time.perf_counter() - command_finish_start) * 1000.0

            queue_submit_start = time.perf_counter()
            self.device.queue.submit([command_buffer])
            queue_submit_ms = (time.perf_counter() - queue_submit_start) * 1000.0

            chunk_gen.service_background_gpu_work(self)
            _, chunk_stream_ms = chunk_gen.prepare_chunks(self, dt)

            wall_frame_ms = (time.perf_counter() - frame_start) * 1000.0

            hud_profile.record_frame_breakdown_sample(self, "world_update", world_update_ms)
            hud_profile.record_frame_breakdown_sample(self, "visibility_lookup", visibility_lookup_ms)
            hud_profile.record_frame_breakdown_sample(self, "chunk_stream", chunk_stream_ms)
            hud_profile.record_frame_breakdown_sample(self, "chunk_displayed_added", float(self._last_new_displayed_chunks))
            hud_profile.record_frame_breakdown_sample(self, "camera_upload", camera_upload_ms)
            hud_profile.record_frame_breakdown_sample(self, "swapchain_acquire", swapchain_acquire_ms)
            hud_profile.record_frame_breakdown_sample(self, "render_encode", render_encode_ms)
            hud_profile.record_frame_breakdown_sample(self, "command_finish", command_finish_ms)
            hud_profile.record_frame_breakdown_sample(self, "queue_submit", queue_submit_ms)
            hud_profile.record_frame_breakdown_sample(self, "wall_frame", wall_frame_ms)
            hud_profile.record_frame_breakdown_sample(self, "draw_calls", float(draw_calls))
            hud_profile.record_frame_breakdown_sample(self, "merged_chunks", float(merged_chunks))
            hud_profile.record_frame_breakdown_sample(self, "visible_vertices", float(visible_vertices))
            hud_profile.record_frame_breakdown_sample(self, "visible_chunk_targets", float(len(self._visible_chunk_coords)))
            hud_profile.record_frame_breakdown_sample(self, "visible_chunks", float(visible_chunks))
            hud_profile.record_frame_breakdown_sample(self, "pending_chunk_requests", float(len(self._pending_chunk_coords)))

            hud_profile.refresh_frame_breakdown_summary(self)
        except Exception:
            try:
                chunk_gen.service_background_gpu_work(self)
            except Exception:
                pass
            try:
                mesh_cache.process_deferred_mesh_output_frees(self)
            except Exception:
                pass
            if profile_started_at is not None:
                hud_profile.profile_end_frame(self, profile_started_at, 0.0)
            raise
        hud_profile.profile_end_frame(self, profile_started_at, wall_frame_ms / 1000.0)
        self.canvas.request_draw(self.draw_frame)
