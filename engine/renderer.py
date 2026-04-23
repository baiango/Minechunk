from __future__ import annotations

import math
import time
import sys
from collections import OrderedDict, deque

import numpy as np
import wgpu
from .pipelines import chunk_pipeline as chunk_gen
from .pipelines import profiling as hud_profile
from .cache import mesh_allocator as mesh_cache
from .renderer_config import *
from .render_shaders import (
    COMPOSITE_SHADER,
    FINAL_BLIT_SHADER,
    GPU_VISIBILITY_SHADER,
    HUD_SHADER,
    OCCLUSION_MASK_SHADER,
    RADIAL_BLUR_SHADER,
    RENDER_SHADER,
    TILE_MERGE_SHADER,
    VOXEL_MESH_BATCH_SHADER,
    VOXEL_SURFACE_EXPAND_SHADER,
)
from .render_utils import (
    clamp,
    cross3,
    dot3,
    flat_forward_vector,
    forward_vector,
    normalize3,
    pack_camera_uniform,
    right_vector,
)
from .meshing import gpu_mesher as wgpu_mesher
from .meshing import metal_mesher as metal_mesher
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
from .world_constants import (
    BLOCK_SIZE,
    CHUNK_SIZE,
    CHUNK_WORLD_SIZE,
    VERTICAL_CHUNK_COUNT,
    VERTICAL_CHUNK_RENDER_RADIUS,
    VERTICAL_CHUNK_STACK_ENABLED,
)
from .terrain.world import VoxelWorld
from .terrain.types import ChunkSurfaceGpuBatch
from .terrain.kernels import AIR

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


@profile
def _engine_mode_uses_gpu_path() -> bool:
    return engine_mode != ENGINE_MODE_CPU


class TerrainRenderer:
    def __init__(
        self,
        seed: int = 1337,
        use_gpu_terrain: bool | None = None,
        use_gpu_meshing: bool | None = None,
        terrain_batch_size: int = DEFAULT_MESH_BATCH_SIZE,
        mesh_batch_size: int | None = None,
        chunk_radius: int | None = None,
        vertical_chunk_radius: int | None = None,
        fixed_view_dimensions: tuple[int, int, int] | None = None,
        freeze_view_origin: bool = False,
        freeze_camera: bool = False,
        exit_when_view_ready: bool = False,
    ) -> None:
        default_use_gpu = _engine_mode_uses_gpu_path()
        self.use_gpu_terrain = default_use_gpu if use_gpu_terrain is None else bool(use_gpu_terrain)
        self.use_gpu_meshing = default_use_gpu if use_gpu_meshing is None else bool(use_gpu_meshing)
        if VERTICAL_CHUNK_STACK_ENABLED:
            self.use_gpu_terrain = False
            self.use_gpu_meshing = False
        self.terrain_batch_size = max(1, int(terrain_batch_size))
        if mesh_batch_size is None:
            self.mesh_batch_size = max(1, int(self.terrain_batch_size))
        else:
            self.mesh_batch_size = max(1, int(mesh_batch_size))
        self._pending_surface_gpu_batch_target_multiplier = 2 if CHUNK_SIZE >= 64 else 10
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

        self.pipeline = chunk_gen.ChunkPipeline(self)

        spawn_y = self._default_camera_spawn_y()
        self.camera = Camera(position=[0.0, spawn_y, 0.0], yaw=math.pi, pitch=-1.20)
        self.keys_down: set[str] = set()
        self.dragging = False
        self.last_pointer: tuple[float, float] | None = None
        self.last_frame_time = time.perf_counter()
        self.depth_texture = None
        self.depth_view = None
        self.depth_size = (0, 0)
        self.volumetric_lighting_enabled = bool(VOLUMETRIC_LIGHTING_ENABLED)
        self.final_present_enabled = True
        self._postprocess_size = (0, 0)
        self.postprocess_msaa_sample_count = 4 if int(POSTPROCESS_MSAA_SAMPLE_COUNT) > 1 else 1
        self.scene_color_texture = None
        self.scene_color_view = None
        self.scene_color_msaa_texture = None
        self.scene_color_msaa_view = None
        self.composite_color_texture = None
        self.composite_color_view = None
        self.occlusion_mask_texture = None
        self.occlusion_mask_view = None
        self.occlusion_mask_msaa_texture = None
        self.occlusion_mask_msaa_view = None
        self.volumetric_shaft_texture = None
        self.volumetric_shaft_view = None
        self.postprocess_depth_texture = None
        self.postprocess_depth_view = None
        self.postprocess_sampler = None
        self.radial_blur_params_buffer = None
        self.final_present_params_buffer = None
        self.radial_blur_bind_group_layout = None
        self.radial_blur_pipeline = None
        self.radial_blur_bind_group = None
        self.composite_bind_group_layout = None
        self.composite_pipeline = None
        self.composite_bind_group = None
        self.final_present_bind_group_layout = None
        self.final_present_pipeline = None
        self.final_scene_bind_group = None
        self.final_composite_bind_group = None
        self.scene_render_pipeline = None
        self.occlusion_mask_pipeline = None
        self.freeze_view_origin = bool(freeze_view_origin)
        self.freeze_camera = bool(freeze_camera)
        self._frozen_view_origin: tuple[int, int, int] | None = None
        self.fixed_view_dimensions = None if fixed_view_dimensions is None else tuple(max(1, int(value)) for value in fixed_view_dimensions)
        if self.fixed_view_dimensions is not None and len(self.fixed_view_dimensions) != 3:
            raise ValueError("fixed_view_dimensions must contain exactly 3 integers")
        self.fixed_view_box_mode = self.fixed_view_dimensions is not None
        if self.fixed_view_box_mode:
            dim_x, dim_y, dim_z = self.fixed_view_dimensions
            self._view_extent_neg_x = int(dim_x // 2)
            self._view_extent_pos_x = int(dim_x - self._view_extent_neg_x - 1)
            self._view_extent_neg_y = int(dim_y // 2)
            self._view_extent_pos_y = int(dim_y - self._view_extent_neg_y - 1)
            self._view_extent_neg_z = int(dim_z // 2)
            self._view_extent_pos_z = int(dim_z - self._view_extent_neg_z - 1)
            self.chunk_radius = max(self._view_extent_neg_x, self._view_extent_pos_x)
            self.vertical_chunk_radius = max(self._view_extent_neg_y, self._view_extent_pos_y) if VERTICAL_CHUNK_STACK_ENABLED else 0
            self.render_dimension_chunks = int(dim_x)
            self.render_dimension_vertical_chunks = int(dim_y) if VERTICAL_CHUNK_STACK_ENABLED else 1
            self.render_dimension_depth_chunks = int(dim_z)
        else:
            effective_chunk_radius = DEFAULT_RENDER_DISTANCE_CHUNKS if chunk_radius is None else chunk_radius
            self.chunk_radius = max(1, int(effective_chunk_radius))
            effective_vertical_radius = VERTICAL_CHUNK_RENDER_RADIUS if vertical_chunk_radius is None else vertical_chunk_radius
            self.vertical_chunk_radius = max(0, int(effective_vertical_radius)) if VERTICAL_CHUNK_STACK_ENABLED else 0
            self.render_dimension_chunks = self.chunk_radius * 2 + 1
            self.render_dimension_vertical_chunks = self.vertical_chunk_radius * 2 + 1 if VERTICAL_CHUNK_STACK_ENABLED else 1
            self.render_dimension_depth_chunks = self.render_dimension_chunks
            self._view_extent_neg_x = int(self.chunk_radius)
            self._view_extent_pos_x = int(self.chunk_radius)
            self._view_extent_neg_y = int(self.vertical_chunk_radius) if VERTICAL_CHUNK_STACK_ENABLED else 0
            self._view_extent_pos_y = int(self.vertical_chunk_radius) if VERTICAL_CHUNK_STACK_ENABLED else 0
            self._view_extent_neg_z = int(self.chunk_radius)
            self._view_extent_pos_z = int(self.chunk_radius)
        self.chunk_cache: OrderedDict[tuple[int, int, int], ChunkMesh] = OrderedDict()
        self._visible_chunk_coords: list[tuple[int, int, int]] = []
        self._visible_chunk_coord_set: set[tuple[int, int, int]] = set()
        self._visible_chunk_origin: tuple[int, int, int] | None = None
        self._visible_display_state_dirty = True
        self._visible_tile_keys: list[tuple[int, int, int]] = []
        self._visible_tile_key_set: set[tuple[int, int, int]] = set()
        self._visible_tile_coords: dict[tuple[int, int, int], tuple[tuple[int, int, int], ...]] = {}
        self._visible_tile_masks: dict[tuple[int, int, int], int] = {}
        self._visible_tile_dirty_keys: set[tuple[int, int, int]] = set()
        self._visible_layout_template_cache: dict[tuple[int, int, int, int, int, int], tuple] = {}
        self._visible_rel_xz_order_cache: dict[tuple[int], tuple[tuple[int, int], ...]] = {}
        self._visible_tile_mesh_slots: dict[tuple[int, int, int], list[ChunkMesh | None]] = {}
        self._visible_tile_active_meshes: dict[tuple[int, int, int], list[ChunkMesh]] = {}
        self._visible_active_tile_key_set: set[tuple[int, int, int]] = set()
        self._visible_tile_slot_index_cache: dict[tuple[int, int, int], dict[tuple[int, int, int], int]] = {}
        self._visible_rel_coord_to_tile_slot: dict[tuple[int, int, int], tuple[tuple[int, int, int], int]] = {}
        self._visible_rel_tile_slot_sizes: dict[tuple[int, int, int], int] = {}
        self._visible_tile_base: tuple[int, int, int] = (0, 0, 0)
        self._shared_empty_chunk_vertex_buffer = None
        self._visible_displayed_coords: set[tuple[int, int, int]] = set()
        self._visible_missing_coords: set[tuple[int, int, int]] = set()
        self._chunk_request_target_coords: set[tuple[int, int, int]] = set()
        self._chunk_request_queue: deque[tuple[int, int, int]] = deque()
        self._chunk_request_queue_origin: tuple[int, int, int] | None = None
        self._chunk_request_queue_view_signature: tuple | None = None
        self._chunk_request_queue_dirty = True
        self._chunk_request_view_stride = max(1, int(MERGED_TILE_SIZE_CHUNKS) // 2)
        self._pending_chunk_coords: set[tuple[int, int, int]] = set()
        self._transient_render_buffers: list[list[wgpu.GPUBuffer]] = []
        self._tile_render_batches: dict[tuple[int, int], ChunkRenderBatch] = {}
        self._tile_dirty_keys: set[tuple[int, int]] = set()
        self._tile_versions: dict[tuple[int, int], int] = {}
        self._visible_layout_version = 0
        self._visible_tile_mutation_version = 0
        self._tile_mutation_version = 0
        self._cached_tile_draw_batches: dict[tuple[int, int, int], tuple[float, list[ChunkDrawBatch], int, int, int]] = {}
        self._cached_visible_render_batches: dict[tuple[int, int, int], tuple[float, list[tuple[wgpu.GPUBuffer, int, int, int]], int, int, int, int]] = {}
        self._mesh_buffer_refs: dict[int, int] = {}
        self._mesh_output_slabs: OrderedDict[int, MeshOutputSlab] = OrderedDict()
        self._mesh_output_slabs_by_size_class: dict[int, OrderedDict[int, MeshOutputSlab]] = {}
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
        self.use_gpu_indirect_render = False if VERTICAL_CHUNK_STACK_ENABLED else True
        self._mesh_draw_indirect_capacity = 0
        self._mesh_draw_indirect_buffer = None
        self._mesh_draw_indirect_array = np.empty((0, 4), dtype=np.uint32)
        self.use_gpu_visibility_culling = False if VERTICAL_CHUNK_STACK_ENABLED else True
        self._mesh_visibility_record_capacity = 0
        self._mesh_visibility_record_buffer = None
        self._mesh_visibility_record_array = np.empty(0, dtype=MESH_VISIBILITY_RECORD_DTYPE)
        self._mesh_visibility_params_buffer = None
        self.max_cached_chunks = MAX_CACHED_CHUNKS
        if self.fixed_view_box_mode and self.fixed_view_dimensions is not None:
            fixed_target_chunk_count = int(self.fixed_view_dimensions[0]) * int(self.fixed_view_dimensions[1]) * int(self.fixed_view_dimensions[2])
            self.max_cached_chunks = max(int(self.max_cached_chunks), fixed_target_chunk_count)
        self._cache_capacity_warned = False
        self._current_move_speed = self.camera.move_speed
        self.walk_mode = True
        self._walk_velocity = [0.0, 0.0, 0.0]
        self._camera_on_ground = False
        self._jump_queued = False
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
        self._async_voxel_mesh_batch_pool_limit = max(4, min(16, self.mesh_batch_size))
        if self._using_metal_meshing:
            self._gpu_mesh_async_inflight_limit = METAL_MESH_INFLIGHT_SLOTS
            self._gpu_mesh_async_finalize_budget = max(1, min(4, self.mesh_batch_size))
            self._metal_gpu_mesh_async_finalize_budget = 1
        else:
            self._gpu_mesh_async_inflight_limit = max(8, min(32, self.mesh_batch_size * 2))
            self._gpu_mesh_async_finalize_budget = max(1, min(16, self.mesh_batch_size * 2))
            self._metal_gpu_mesh_async_finalize_budget = max(1, min(2, self.mesh_batch_size))
        self._mesh_output_upload_batch_bytes = max(self._mesh_output_min_slab_bytes, int(MESH_OUTPUT_UPLOAD_BATCH_BYTES))
        self._device_lost = False
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
        self._frame_breakdown_next_refresh = 0.0
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
        self.frame_breakdown_sample_sums: dict[str, float] = {name: 0.0 for name in self.frame_breakdown_samples}
        self.frame_breakdown_lines: list[str] = []
        self.frame_breakdown_vertex_bytes = b""
        self.frame_breakdown_vertex_count = 0
        self.frame_breakdown_vertex_buffer = None
        self.frame_breakdown_vertex_buffer_capacity = 0
        self._frame_breakdown_next_refresh = 0.0
        self._solid_block_cache: dict[tuple[int, int, int], bool] = {}
        self.exit_when_view_ready = bool(exit_when_view_ready)
        self._auto_exit_requested = False
        self._auto_exit_frame_count = 0
        self._last_frame_draw_calls = 0
        self._last_frame_merged_batches = 0
        self._last_frame_visible_chunks = 0
        self._last_frame_visible_vertices = 0
        self._last_new_displayed_chunks = 0
        self._last_chunk_stream_drained = 0
        self._last_frame_visible_batches = 0
        self._last_displayed_chunk_coords: set[tuple[int, int, int]] = set()
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
        self.radial_blur_params_buffer = self.device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
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
        self.radial_blur_bind_group_layout = self.device.create_bind_group_layout(
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
            ],
        )
        self.composite_bind_group_layout = self.device.create_bind_group_layout(
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
                    "sampler": {"type": "filtering"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d", "multisampled": False},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
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
        self.scene_render_pipeline = self.device.create_render_pipeline(
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
                "targets": [{"format": POSTPROCESS_SCENE_FORMAT}],
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
        self.occlusion_mask_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.render_bind_group_layout]),
            vertex={
                "module": self.device.create_shader_module(code=OCCLUSION_MASK_SHADER),
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
                "module": self.device.create_shader_module(code=OCCLUSION_MASK_SHADER),
                "entry_point": "fs_main",
                "targets": [{"format": POSTPROCESS_OCCLUSION_FORMAT}],
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
        self.radial_blur_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.radial_blur_bind_group_layout]),
            vertex={
                "module": self.device.create_shader_module(code=RADIAL_BLUR_SHADER),
                "entry_point": "vs_main",
                "buffers": [],
            },
            fragment={
                "module": self.device.create_shader_module(code=RADIAL_BLUR_SHADER),
                "entry_point": "fs_main",
                "targets": [{"format": POSTPROCESS_SHAFT_FORMAT}],
            },
            primitive={
                "topology": "triangle-list",
                "cull_mode": "none",
            },
        )
        self.composite_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.composite_bind_group_layout]),
            vertex={
                "module": self.device.create_shader_module(code=COMPOSITE_SHADER),
                "entry_point": "vs_main",
                "buffers": [],
            },
            fragment={
                "module": self.device.create_shader_module(code=COMPOSITE_SHADER),
                "entry_point": "fs_main",
                "targets": [{"format": POSTPROCESS_COMPOSITE_FORMAT if self.final_present_enabled else self.color_format}],
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

    def _default_camera_spawn_y(self) -> float:
        try:
            surface_height_blocks = int(self.world.surface_height_at(0, 0))
        except Exception:
            surface_height_blocks = int(self.world.height // 2)
        surface_y = float(surface_height_blocks) * BLOCK_SIZE
        return max(
            CAMERA_EYE_HEIGHT_METERS + 0.2,
            min(float(self.world.height) * BLOCK_SIZE + CAMERA_HEADROOM_METERS, surface_y + CAMERA_EYE_HEIGHT_METERS),
        )

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
        self._postprocess_size = (0, 0)
        self.scene_color_texture = None
        self.scene_color_view = None
        self.scene_color_msaa_texture = None
        self.scene_color_msaa_view = None
        self.composite_color_texture = None
        self.composite_color_view = None
        self.occlusion_mask_texture = None
        self.occlusion_mask_view = None
        self.occlusion_mask_msaa_texture = None
        self.occlusion_mask_msaa_view = None
        self.volumetric_shaft_texture = None
        self.volumetric_shaft_view = None
        self.postprocess_depth_texture = None
        self.postprocess_depth_view = None
        self.radial_blur_bind_group = None
        self.composite_bind_group = None
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
        if is_new_press and key == "v":
            self.walk_mode = not self.walk_mode
            self._walk_velocity[:] = [0.0, 0.0, 0.0]
            self._jump_queued = False
        if is_new_press and key == "space":
            self._jump_queued = True

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

    def _write_final_present_params(self, width: int, height: int) -> None:
        params = np.array([
            1.0 / max(1.0, float(width)),
            1.0 / max(1.0, float(height)),
            0.0,
            0.0,
        ], dtype=np.float32)
        self.device.queue.write_buffer(self.final_present_params_buffer, 0, params.tobytes())

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
        for name, samples in self.frame_breakdown_samples.items():
            samples.clear()
            self.frame_breakdown_sample_sums[name] = 0.0
        self.profile_hud_lines = []
        self.profile_hud_vertex_bytes = b""
        self.profile_hud_vertex_count = 0
        self._frame_breakdown_next_refresh = now
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
            f"CHUNK DIMS: {CHUNK_SIZE}x{CHUNK_SIZE}x{CHUNK_SIZE}",
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
        self._frame_breakdown_next_refresh = 0.0
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
        for name, samples in self.frame_breakdown_samples.items():
            samples.clear()
            self.frame_breakdown_sample_sums[name] = 0.0

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

    @profile
    def regenerate_world(self) -> None:
        for mesh in self.chunk_cache.values():
            mesh_cache.release_chunk_mesh_storage(self, mesh)
        self.chunk_cache.clear()
        self._mesh_buffer_refs.clear()
        self._pending_chunk_coords.clear()
        self._chunk_request_target_coords.clear()
        self._chunk_request_queue.clear()
        self._chunk_request_queue_origin = None
        self._chunk_request_queue_view_signature = None
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
        self._cached_visible_render_batches.clear()
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

        self.pipeline = chunk_gen.ChunkPipeline(self)
        self.camera.position[:] = [0.0, self._default_camera_spawn_y(), 0.0]
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

    def _player_extents(self) -> tuple[float, float, float]:
        return (
            float(PLAYER_COLLIDER_HALF_WIDTH_METERS),
            float(PLAYER_EYE_OFFSET_METERS),
            float(max(0.0, PLAYER_COLLIDER_HEIGHT_METERS - PLAYER_EYE_OFFSET_METERS)),
        )

    def _player_aabb(self, position: list[float]) -> tuple[float, float, float, float, float, float]:
        half_width, eye_down, eye_up = self._player_extents()
        return (
            float(position[0]) - half_width,
            float(position[1]) - eye_down,
            float(position[2]) - half_width,
            float(position[0]) + half_width,
            float(position[1]) + eye_up,
            float(position[2]) + half_width,
        )

    def _is_block_solid(self, bx: int, by: int, bz: int) -> bool:
        if by < 0:
            return True
        key = (int(bx), int(by), int(bz))
        cached = self._solid_block_cache.get(key)
        if cached is not None:
            return bool(cached)
        is_solid = int(self.world.block_at(key[0], key[1], key[2])) != int(AIR)
        self._solid_block_cache[key] = bool(is_solid)
        return bool(is_solid)

    def _resolve_small_downward_snap(self, position: list[float], delta: float) -> bool:
        if delta >= 0.0 or delta < -BLOCK_SIZE * 1.01:
            return False
        eps = 1e-6
        half_width, eye_down, _ = self._player_extents()
        min_x = float(position[0]) - half_width
        max_x = float(position[0]) + half_width
        min_z = float(position[2]) - half_width
        max_z = float(position[2]) + half_width
        target_min_y = float(position[1]) + float(delta) - eye_down
        probe_by = int(math.floor(target_min_y / BLOCK_SIZE))
        min_bx = int(math.floor(min_x / BLOCK_SIZE))
        max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
        min_bz = int(math.floor(min_z / BLOCK_SIZE))
        max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
        for bz in range(min_bz, max_bz + 1):
            for bx in range(min_bx, max_bx + 1):
                if self._is_block_solid(bx, probe_by, bz):
                    position[1] = (float(probe_by) + 1.0) * BLOCK_SIZE + eye_down + eps
                    return True
        return False

    def _resolve_collision_axis(self, position: list[float], axis: int, delta: float) -> bool:
        if abs(delta) <= 1e-9:
            return False

        eps = 1e-6
        old_position = [float(position[0]), float(position[1]), float(position[2])]
        old_min_x, old_min_y, old_min_z, old_max_x, old_max_y, old_max_z = self._player_aabb(old_position)
        position[axis] += float(delta)
        min_x, min_y, min_z, max_x, max_y, max_z = self._player_aabb(position)
        half_width, eye_down, eye_up = self._player_extents()

        if axis == 0:
            min_by = int(math.floor(min_y / BLOCK_SIZE))
            max_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
            min_bz = int(math.floor(min_z / BLOCK_SIZE))
            max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
            if delta > 0.0:
                start_bx = int(math.floor((old_max_x - eps) / BLOCK_SIZE)) + 1
                end_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
                for bx in range(start_bx, end_bx + 1):
                    for by in range(min_by, max_by + 1):
                        for bz in range(min_bz, max_bz + 1):
                            if self._is_block_solid(bx, by, bz):
                                position[0] = float(bx) * BLOCK_SIZE - half_width - eps
                                return True
            else:
                start_bx = int(math.floor(old_min_x / BLOCK_SIZE)) - 1
                end_bx = int(math.floor(min_x / BLOCK_SIZE))
                for bx in range(start_bx, end_bx - 1, -1):
                    for by in range(min_by, max_by + 1):
                        for bz in range(min_bz, max_bz + 1):
                            if self._is_block_solid(bx, by, bz):
                                position[0] = (float(bx) + 1.0) * BLOCK_SIZE + half_width + eps
                                return True
            return False

        if axis == 1:
            if delta < 0.0:
                snap_probe = [float(old_position[0]), float(old_position[1]), float(old_position[2])]
                if self._resolve_small_downward_snap(snap_probe, delta):
                    position[1] = snap_probe[1]
                    return True
            min_bx = int(math.floor(min_x / BLOCK_SIZE))
            max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
            min_bz = int(math.floor(min_z / BLOCK_SIZE))
            max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
            if delta > 0.0:
                start_by = int(math.floor((old_max_y - eps) / BLOCK_SIZE)) + 1
                end_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
                for by in range(start_by, end_by + 1):
                    for bz in range(min_bz, max_bz + 1):
                        for bx in range(min_bx, max_bx + 1):
                            if self._is_block_solid(bx, by, bz):
                                position[1] = float(by) * BLOCK_SIZE - eye_up - eps
                                return True
            else:
                start_by = int(math.floor(old_min_y / BLOCK_SIZE)) - 1
                end_by = int(math.floor(min_y / BLOCK_SIZE))
                for by in range(start_by, end_by - 1, -1):
                    for bz in range(min_bz, max_bz + 1):
                        for bx in range(min_bx, max_bx + 1):
                            if self._is_block_solid(bx, by, bz):
                                position[1] = (float(by) + 1.0) * BLOCK_SIZE + eye_down + eps
                                return True
            return False

        min_bx = int(math.floor(min_x / BLOCK_SIZE))
        max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
        min_by = int(math.floor(min_y / BLOCK_SIZE))
        max_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
        if delta > 0.0:
            start_bz = int(math.floor((old_max_z - eps) / BLOCK_SIZE)) + 1
            end_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
            for bz in range(start_bz, end_bz + 1):
                for by in range(min_by, max_by + 1):
                    for bx in range(min_bx, max_bx + 1):
                        if self._is_block_solid(bx, by, bz):
                            position[2] = float(bz) * BLOCK_SIZE - half_width - eps
                            return True
        else:
            start_bz = int(math.floor(old_min_z / BLOCK_SIZE)) - 1
            end_bz = int(math.floor(min_z / BLOCK_SIZE))
            for bz in range(start_bz, end_bz - 1, -1):
                for by in range(min_by, max_by + 1):
                    for bx in range(min_bx, max_bx + 1):
                        if self._is_block_solid(bx, by, bz):
                            position[2] = (float(bz) + 1.0) * BLOCK_SIZE + half_width + eps
                            return True
        return False

    @profile
    def _position_is_clear(self, position: list[float]) -> bool:
        min_x, min_y, min_z, max_x, max_y, max_z = self._player_aabb(position)
        eps = 1e-6
        min_bx = int(math.floor(min_x / BLOCK_SIZE))
        max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
        min_by = int(math.floor(min_y / BLOCK_SIZE))
        max_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
        min_bz = int(math.floor(min_z / BLOCK_SIZE))
        max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
        for by in range(min_by, max_by + 1):
            for bz in range(min_bz, max_bz + 1):
                for bx in range(min_bx, max_bx + 1):
                    if self._is_block_solid(bx, by, bz):
                        return False
        return True

    @profile
    def _move_horizontal_with_step(self, position: list[float], axis: int, delta: float) -> bool:
        if abs(delta) <= 1e-9:
            return False
        trial = [float(position[0]), float(position[1]), float(position[2])]
        collided = self._resolve_collision_axis(trial, axis, delta)
        if not collided:
            position[:] = trial
            return False
        if not self.walk_mode or not self._camera_on_ground:
            position[:] = trial
            return True
        stepped = [float(position[0]), float(position[1]), float(position[2])]
        self._resolve_collision_axis(stepped, 1, float(PLAYER_STEP_HEIGHT_METERS))
        if stepped[1] <= position[1] + 1e-5 or not self._position_is_clear(stepped):
            position[:] = trial
            return True
        step_trial = [float(stepped[0]), float(stepped[1]), float(stepped[2])]
        if self._resolve_collision_axis(step_trial, axis, delta):
            position[:] = trial
            return True
        self._resolve_collision_axis(step_trial, 1, -float(PLAYER_GROUND_SNAP_METERS))
        position[:] = step_trial
        return False

    def _snap_to_ground(self, position: list[float]) -> bool:
        probe = [float(position[0]), float(position[1]), float(position[2])]
        collided = self._resolve_collision_axis(probe, 1, -float(PLAYER_GROUND_SNAP_METERS))
        if collided:
            position[:] = probe
            return True
        return False

    @profile
    def _update_camera_fly(self, dt: float) -> None:
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

    @profile
    def _update_camera_walk(self, dt: float) -> None:
        self._solid_block_cache.clear()
        sprinting = self._key_active("shift", "shiftleft", "shiftright")
        speed = WALK_SPRINT_SPEED if sprinting else self.camera.move_speed
        self._current_move_speed = float(speed)

        move_x = 0.0
        move_z = 0.0
        forward = flat_forward_vector(self.camera.yaw)
        right = right_vector(self.camera.yaw)
        if self._key_active("w", "arrowup"):
            move_x += forward[0]
            move_z += forward[2]
        if self._key_active("s", "arrowdown"):
            move_x -= forward[0]
            move_z -= forward[2]
        if self._key_active("d", "arrowright"):
            move_x += right[0]
            move_z += right[2]
        if self._key_active("a", "arrowleft"):
            move_x -= right[0]
            move_z -= right[2]

        move_len = math.sqrt(move_x * move_x + move_z * move_z)
        if move_len > 0.0:
            move_x /= move_len
            move_z /= move_len
        desired_dx = move_x * speed * dt
        desired_dz = move_z * speed * dt

        if self._camera_on_ground and self._jump_queued:
            self._walk_velocity[1] = float(PLAYER_JUMP_SPEED_METERS)
            self._camera_on_ground = False
        self._jump_queued = False

        if self._camera_on_ground and desired_dx == 0.0 and desired_dz == 0.0 and self._walk_velocity[1] <= 0.0:
            position = [float(self.camera.position[0]), float(self.camera.position[1]), float(self.camera.position[2])]
            if self._resolve_small_downward_snap(position, -float(PLAYER_GROUND_SNAP_METERS)):
                self._walk_velocity[1] = 0.0
                self.camera.position[:] = position
                return

        self._walk_velocity[1] -= float(PLAYER_GRAVITY_METERS) * dt

        position = [float(self.camera.position[0]), float(self.camera.position[1]), float(self.camera.position[2])]
        if desired_dx != 0.0:
            self._move_horizontal_with_step(position, 0, desired_dx)
        if desired_dz != 0.0:
            self._move_horizontal_with_step(position, 2, desired_dz)

        vertical_delta = self._walk_velocity[1] * dt
        if vertical_delta <= 0.0:
            snap_delta = min(vertical_delta, -float(PLAYER_GROUND_SNAP_METERS))
            collided_y = self._resolve_small_downward_snap(position, snap_delta)
            if not collided_y:
                collided_y = self._resolve_collision_axis(position, 1, snap_delta)
            if collided_y:
                self._camera_on_ground = True
                self._walk_velocity[1] = 0.0
            else:
                self._camera_on_ground = False
        else:
            collided_y = self._resolve_collision_axis(position, 1, vertical_delta)
            if collided_y:
                self._walk_velocity[1] = 0.0
            else:
                self._camera_on_ground = False

        self.camera.position[:] = position

    @profile
    def _update_camera(self, dt: float) -> None:
        if self.walk_mode:
            self._update_camera_walk(dt)
        else:
            self._update_camera_fly(dt)
            self._walk_velocity[:] = [0.0, 0.0, 0.0]
            self._camera_on_ground = False
            self._jump_queued = False
        self.camera.position[1] = clamp(self.camera.position[1], CAMERA_MIN_HEIGHT_METERS, self.world.height * BLOCK_SIZE + CAMERA_HEADROOM_METERS)

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

    def _camera_chunk_origin(self) -> tuple[int, int, int]:
        return (
            int(self.camera.position[0] // CHUNK_WORLD_SIZE),
            max(0, min(VERTICAL_CHUNK_COUNT - 1, int(self.camera.position[1] // CHUNK_WORLD_SIZE))) if VERTICAL_CHUNK_STACK_ENABLED else 0,
            int(self.camera.position[2] // CHUNK_WORLD_SIZE),
        )

    def _current_chunk_origin(self) -> tuple[int, int, int]:
        current = self._camera_chunk_origin()
        if self.fixed_view_box_mode and VERTICAL_CHUNK_STACK_ENABLED:
            min_origin_y = int(self._view_extent_neg_y)
            max_origin_y = max(min_origin_y, int(VERTICAL_CHUNK_COUNT - 1 - self._view_extent_pos_y))
            current = (
                int(current[0]),
                min(max(int(current[1]), min_origin_y), max_origin_y),
                int(current[2]),
            )
        if self.freeze_view_origin:
            if self._frozen_view_origin is None:
                self._frozen_view_origin = current
            return self._frozen_view_origin
        return current

    @profile
    def _build_visible_layout_template(
        self,
        origin_mod_x: int,
        origin_mod_z: int,
    ) -> tuple[
        tuple[tuple[int, int, int], ...],
        tuple[tuple[int, int, int], ...],
        tuple[int, ...],
        dict[tuple[int, int, int], tuple[tuple[int, int, int], int]],
        dict[tuple[int, int, int], int],
    ]:
        tile_size = int(MERGED_TILE_SIZE_CHUNKS)
        neg_x = int(self._view_extent_neg_x)
        pos_x = int(self._view_extent_pos_x)
        neg_z = int(self._view_extent_neg_z)
        pos_z = int(self._view_extent_pos_z)
        neg_y = int(self._view_extent_neg_y) if VERTICAL_CHUNK_STACK_ENABLED else 0
        pos_y = int(self._view_extent_pos_y) if VERTICAL_CHUNK_STACK_ENABLED else 0
        template_key = (
            neg_x,
            pos_x,
            neg_y,
            pos_y,
            neg_z,
            pos_z,
            tile_size,
            int(origin_mod_x),
            int(origin_mod_z),
            1 if VERTICAL_CHUNK_STACK_ENABLED else 0,
            1 if self.fixed_view_box_mode else 0,
        )
        cached = self._visible_layout_template_cache.get(template_key)
        if cached is not None:
            return cached

        min_rel_y = -neg_y if VERTICAL_CHUNK_STACK_ENABLED else 0
        max_rel_y = pos_y if VERTICAL_CHUNK_STACK_ENABLED else 0
        if VERTICAL_CHUNK_STACK_ENABLED:
            cy_order: list[int] = [0]
            max_offset = max(-min_rel_y, max_rel_y)
            for offset in range(1, max_offset + 1):
                up = offset
                down = -offset
                if up <= max_rel_y:
                    cy_order.append(up)
                if down >= min_rel_y:
                    cy_order.append(down)
        else:
            cy_order = [0]

        rel_coords: list[tuple[int, int, int]] = []
        rel_tile_keys: list[tuple[int, int, int]] = []
        rel_tile_masks: dict[tuple[int, int, int], int] = {}
        rel_coord_to_tile_slot: dict[tuple[int, int, int], tuple[tuple[int, int, int], int]] = {}
        rel_tile_slot_sizes: dict[tuple[int, int, int], int] = {}
        base_tile_x = int(origin_mod_x) // tile_size
        base_tile_z = int(origin_mod_z) // tile_size

        rel_xz_cache_key = (neg_x, pos_x, neg_z, pos_z, 1 if self.fixed_view_box_mode else 0)
        rel_xz_order = self._visible_rel_xz_order_cache.get(rel_xz_cache_key)
        if rel_xz_order is None:
            rel_xz = [
                (dx, dz)
                for dz in range(-neg_z, pos_z + 1)
                for dx in range(-neg_x, pos_x + 1)
            ]
            rel_xz.sort(key=lambda delta: (delta[0] * delta[0] + delta[1] * delta[1], abs(delta[1]), abs(delta[0]), delta[1], delta[0]))
            rel_xz_order = tuple(rel_xz)
            self._visible_rel_xz_order_cache[rel_xz_cache_key] = rel_xz_order

        for rel_y in cy_order:
            for dx, dz in rel_xz_order:
                rel_coord = (dx, rel_y, dz)
                rel_coords.append(rel_coord)
                abs_x = int(origin_mod_x) + dx
                abs_z = int(origin_mod_z) + dz
                tile_x = abs_x // tile_size
                tile_z = abs_z // tile_size
                local_x = abs_x - tile_x * tile_size
                local_z = abs_z - tile_z * tile_size
                rel_tile_key = (tile_x - base_tile_x, rel_y, tile_z - base_tile_z)
                slot_index = int(rel_tile_slot_sizes.get(rel_tile_key, 0))
                if slot_index == 0:
                    rel_tile_keys.append(rel_tile_key)
                rel_coord_to_tile_slot[rel_coord] = (rel_tile_key, slot_index)
                rel_tile_slot_sizes[rel_tile_key] = slot_index + 1
                if 0 <= local_x < tile_size and 0 <= local_z < tile_size:
                    rel_tile_masks[rel_tile_key] = int(rel_tile_masks.get(rel_tile_key, 0)) | int(1 << (local_z * tile_size + local_x))

        rel_tile_mask_values = tuple(int(rel_tile_masks.get(rel_tile_key, 0)) for rel_tile_key in rel_tile_keys)
        template = (
            tuple(rel_coords),
            tuple(rel_tile_keys),
            rel_tile_mask_values,
            rel_coord_to_tile_slot,
            rel_tile_slot_sizes,
        )
        self._visible_layout_template_cache[template_key] = template
        return template

    @profile
    def _tile_layout_in_view_for_origin(
        self,
        origin: tuple[int, int, int],
    ) -> tuple[
        tuple[tuple[int, int, int], ...],
        list[tuple[int, int, int]],
        dict[tuple[int, int, int], int],
        dict[tuple[int, int, int], tuple[tuple[int, int, int], int]],
        dict[tuple[int, int, int], int],
        tuple[int, int, int],
    ]:
        chunk_x = int(origin[0])
        chunk_y = int(origin[1])
        chunk_z = int(origin[2])
        tile_size = int(MERGED_TILE_SIZE_CHUNKS)
        rel_coords, rel_tile_keys, rel_tile_mask_values, rel_coord_to_tile_slot, rel_tile_slot_sizes = self._build_visible_layout_template(
            chunk_x % tile_size,
            chunk_z % tile_size,
        )
        base_tile_x = chunk_x // tile_size
        base_tile_z = chunk_z // tile_size
        tile_base = (base_tile_x, chunk_y, base_tile_z)
        visible_tile_keys: list[tuple[int, int, int]] = []
        visible_tile_masks: dict[tuple[int, int, int], int] = {}
        append_visible_tile_key = visible_tile_keys.append
        for index, (tx, ty, tz) in enumerate(rel_tile_keys):
            tile_key_value = (base_tile_x + tx, chunk_y + ty, base_tile_z + tz)
            append_visible_tile_key(tile_key_value)
            mask_value = int(rel_tile_mask_values[index])
            if mask_value != 0:
                visible_tile_masks[tile_key_value] = mask_value
        return rel_coords, visible_tile_keys, visible_tile_masks, rel_coord_to_tile_slot, rel_tile_slot_sizes, tile_base

    @profile
    def _chunk_coords_and_tile_layout_in_view_for_origin(
        self,
        origin: tuple[int, int, int],
    ) -> tuple[
        list[tuple[int, int, int]],
        list[tuple[int, int, int]],
        dict[tuple[int, int, int], int],
        dict[tuple[int, int, int], tuple[tuple[int, int, int], int]],
        dict[tuple[int, int, int], int],
        tuple[int, int, int],
    ]:
        rel_coords, visible_tile_keys, visible_tile_masks, rel_coord_to_tile_slot, rel_tile_slot_sizes, tile_base = self._tile_layout_in_view_for_origin(origin)
        chunk_x = int(origin[0])
        chunk_y = int(origin[1])
        chunk_z = int(origin[2])
        coords = [(chunk_x + dx, chunk_y + dy, chunk_z + dz) for dx, dy, dz in rel_coords]
        return coords, visible_tile_keys, visible_tile_masks, rel_coord_to_tile_slot, rel_tile_slot_sizes, tile_base

    def _chunk_coords_in_view_for_origin(self, origin: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        coords, _, _, _, _, _ = self._chunk_coords_and_tile_layout_in_view_for_origin(origin)
        return coords

    def _chunk_coords_in_view(self) -> list[tuple[int, int, int]]:
        origin = self._current_chunk_origin()
        return self._chunk_coords_in_view_for_origin(origin)

    def _tile_key_for_chunk(self, chunk_x: int, chunk_z: int, chunk_y: int = 0) -> tuple[int, int, int]:
        tile_size = int(MERGED_TILE_SIZE_CHUNKS)
        return (int(chunk_x) // tile_size, int(chunk_y), int(chunk_z) // tile_size)

    def _tile_bit_for_chunk(self, chunk_x: int, chunk_z: int, chunk_y: int = 0) -> tuple[tuple[int, int, int], int]:
        tile_size = int(MERGED_TILE_SIZE_CHUNKS)
        tile_key_value = self._tile_key_for_chunk(int(chunk_x), int(chunk_z), int(chunk_y))
        local_x = int(chunk_x) - tile_key_value[0] * tile_size
        local_z = int(chunk_z) - tile_key_value[2] * tile_size
        if 0 <= local_x < tile_size and 0 <= local_z < tile_size:
            return tile_key_value, 1 << (local_z * tile_size + local_x)
        return tile_key_value, 0

    def _visible_tile_slot_info_for_coord(
        self,
        coord: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], int, list[ChunkMesh | None]] | None:
        origin = self._visible_chunk_origin
        if origin is None:
            return None
        rel_coord = (
            int(coord[0]) - int(origin[0]),
            int(coord[1]) - int(origin[1]),
            int(coord[2]) - int(origin[2]),
        )
        rel_slot = self._visible_rel_coord_to_tile_slot.get(rel_coord)
        if rel_slot is None:
            return None
        rel_tile_key, slot_index = rel_slot
        tile_base = self._visible_tile_base
        tile_key_value = (
            int(tile_base[0]) + int(rel_tile_key[0]),
            int(tile_base[1]) + int(rel_tile_key[1]),
            int(tile_base[2]) + int(rel_tile_key[2]),
        )
        slots = self._visible_tile_mesh_slots.get(tile_key_value)
        if slots is None:
            slot_count = int(self._visible_rel_tile_slot_sizes.get(rel_tile_key, 0))
            if slot_count <= 0:
                return None
            slots = [None] * slot_count
            self._visible_tile_mesh_slots[tile_key_value] = slots
        return tile_key_value, int(slot_index), slots

    @profile
    def _rebuild_visible_tile_layout_from_coords(self) -> None:
        origin = self._visible_chunk_origin if self._visible_chunk_origin is not None else (0, 0, 0)
        (
            _,
            self._visible_tile_keys,
            self._visible_tile_masks,
            self._visible_rel_coord_to_tile_slot,
            self._visible_rel_tile_slot_sizes,
            self._visible_tile_base,
        ) = self._tile_layout_in_view_for_origin(origin)
        self._visible_tile_coords = {}

    @profile
    def _apply_visible_chunk_coord_delta(self, new_origin: tuple[int, int, int]) -> bool:
        old_origin = self._visible_chunk_origin
        if old_origin is None or not self._visible_chunk_coords:
            return False
        dx = int(new_origin[0]) - int(old_origin[0])
        dy = int(new_origin[1]) - int(old_origin[1])
        dz = int(new_origin[2]) - int(old_origin[2])
        if dx == 0 and dy == 0 and dz == 0:
            return True
        shifted_coords = [
            (chunk_x + dx, chunk_y + dy, chunk_z + dz)
            for chunk_x, chunk_y, chunk_z in self._visible_chunk_coords
        ]
        self._visible_chunk_coords = shifted_coords
        self._visible_chunk_coord_set = set(shifted_coords)
        return True

    @profile
    def _refresh_visible_chunk_coords(self) -> None:
        new_origin = self._current_chunk_origin()
        previous_origin = self._visible_chunk_origin
        (
            rel_coords,
            self._visible_tile_keys,
            self._visible_tile_masks,
            self._visible_rel_coord_to_tile_slot,
            self._visible_rel_tile_slot_sizes,
            self._visible_tile_base,
        ) = self._tile_layout_in_view_for_origin(new_origin)

        shifted_visible_coords = False
        if previous_origin is not None and self._visible_chunk_coords:
            visible_coord_set = self._visible_chunk_coord_set
            visible_coords = self._visible_chunk_coords
            if visible_coord_set and len(visible_coord_set) == len(visible_coords):
                shifted_visible_coords = self._apply_visible_chunk_coord_delta(new_origin)
        self._visible_chunk_origin = new_origin

        if shifted_visible_coords:
            visible_coords = self._visible_chunk_coords
            visible_coord_set = self._visible_chunk_coord_set
        else:
            origin_x = int(new_origin[0])
            origin_y = int(new_origin[1])
            origin_z = int(new_origin[2])
            visible_coords = [(origin_x + dx, origin_y + dy, origin_z + dz) for dx, dy, dz in rel_coords]
            self._visible_chunk_coords = visible_coords
            visible_coord_set = set(visible_coords)
            self._visible_chunk_coord_set = visible_coord_set

        self._visible_tile_coords = {}
        self._visible_tile_key_set = set(self._visible_tile_keys)
        # Keep visible slot arrays lazy. The direct render path only needs per-tile
        # active meshes, so eagerly allocating and filling slot arrays here just
        # duplicates bookkeeping on every origin hop.
        self._visible_tile_mesh_slots = {}
        self._visible_tile_active_meshes = {}
        self._visible_active_tile_key_set = set()
        self._visible_tile_slot_index_cache = {}

        chunk_cache = self.chunk_cache
        displayed_coords = visible_coord_set.intersection(chunk_cache.keys())
        chunk_cache_getitem = chunk_cache.__getitem__

        tile_size = int(MERGED_TILE_SIZE_CHUNKS)
        visible_tile_active_meshes: dict[tuple[int, int, int], list[ChunkMesh]] = {}

        for coord in displayed_coords:
            mesh = chunk_cache_getitem(coord)
            if mesh.vertex_count <= 0:
                continue
            tile_key_value = (int(coord[0]) // tile_size, int(coord[1]), int(coord[2]) // tile_size)
            active_list = visible_tile_active_meshes.get(tile_key_value)
            if active_list is None:
                active_list = []
                visible_tile_active_meshes[tile_key_value] = active_list
            active_list.append(mesh)

        self._visible_tile_active_meshes = visible_tile_active_meshes
        self._visible_active_tile_key_set = set(visible_tile_active_meshes)
        self._visible_active_tile_keys = [tile_key_value for tile_key_value in self._visible_tile_keys if tile_key_value in visible_tile_active_meshes]

        self._visible_prefetched_displayed_coords = displayed_coords
        self._visible_displayed_coords = set(displayed_coords)
        self._visible_display_state_dirty = False
        self._visible_tile_dirty_keys = {
            key for key in self._tile_dirty_keys if key in self._visible_tile_key_set
        }
        self._visible_layout_version += 1
        self._visible_tile_mutation_version += 1
        self._cached_tile_draw_batches.clear()
        self._cached_visible_render_batches.clear()

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

    def _ensure_postprocess_targets(self) -> None:
        width, height = self.canvas.get_physical_size()
        target_size = (max(1, int(width)), max(1, int(height)))
        if (
            target_size == self._postprocess_size
            and self.scene_color_view is not None
            and self.composite_color_view is not None
            and self.occlusion_mask_view is not None
            and self.volumetric_shaft_view is not None
            and self.radial_blur_bind_group is not None
            and self.composite_bind_group is not None
            and self.final_scene_bind_group is not None
            and self.final_composite_bind_group is not None
            and self.postprocess_depth_view is not None
            and (self.postprocess_msaa_sample_count <= 1 or self.scene_color_msaa_view is not None)
            and (self.postprocess_msaa_sample_count <= 1 or self.occlusion_mask_msaa_view is not None)
        ):
            return
        self._postprocess_size = target_size
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
        self.composite_color_texture = self.device.create_texture(
            size=(target_size[0], target_size[1], 1),
            format=POSTPROCESS_COMPOSITE_FORMAT,
            usage=texture_usage,
        )
        self.composite_color_view = self.composite_color_texture.create_view()
        self.occlusion_mask_texture = self.device.create_texture(
            size=(target_size[0], target_size[1], 1),
            format=POSTPROCESS_OCCLUSION_FORMAT,
            usage=texture_usage,
        )
        self.occlusion_mask_view = self.occlusion_mask_texture.create_view()
        if self.postprocess_msaa_sample_count > 1:
            self.occlusion_mask_msaa_texture = self.device.create_texture(
                size=(target_size[0], target_size[1], 1),
                format=POSTPROCESS_OCCLUSION_FORMAT,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                sample_count=self.postprocess_msaa_sample_count,
            )
            self.occlusion_mask_msaa_view = self.occlusion_mask_msaa_texture.create_view()
        else:
            self.occlusion_mask_msaa_texture = None
            self.occlusion_mask_msaa_view = self.occlusion_mask_view
        self.volumetric_shaft_texture = self.device.create_texture(
            size=(target_size[0], target_size[1], 1),
            format=POSTPROCESS_SHAFT_FORMAT,
            usage=texture_usage,
        )
        self.volumetric_shaft_view = self.volumetric_shaft_texture.create_view()
        self.postprocess_depth_texture = self.device.create_texture(
            size=(target_size[0], target_size[1], 1),
            format=DEPTH_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.postprocess_msaa_sample_count,
        )
        self.postprocess_depth_view = self.postprocess_depth_texture.create_view()

        self._write_final_present_params(target_size[0], target_size[1])

        self.radial_blur_bind_group = self.device.create_bind_group(
            layout=self.radial_blur_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.occlusion_mask_view},
                {"binding": 1, "resource": self.postprocess_sampler},
                {"binding": 2, "resource": {"buffer": self.radial_blur_params_buffer, "offset": 0, "size": 32}},
            ],
        )
        self.composite_bind_group = self.device.create_bind_group(
            layout=self.composite_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.scene_color_view},
                {"binding": 1, "resource": self.volumetric_shaft_view},
                {"binding": 2, "resource": self.postprocess_sampler},
                {"binding": 3, "resource": self.occlusion_mask_view},
                {"binding": 4, "resource": {"buffer": self.radial_blur_params_buffer, "offset": 0, "size": 32}},
            ],
        )
        self.final_scene_bind_group = self.device.create_bind_group(
            layout=self.final_present_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.scene_color_view},
                {"binding": 1, "resource": self.postprocess_sampler},
                {"binding": 2, "resource": {"buffer": self.final_present_params_buffer, "offset": 0, "size": 32}},
            ],
        )
        self.final_composite_bind_group = self.device.create_bind_group(
            layout=self.final_present_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.composite_color_view},
                {"binding": 1, "resource": self.postprocess_sampler},
                {"binding": 2, "resource": {"buffer": self.final_present_params_buffer, "offset": 0, "size": 32}},
            ],
        )

    @profile
    def _project_directional_light_to_screen(self, right, up, forward) -> tuple[tuple[float, float], float]:
        light_dir = normalize3(tuple(float(v) for v in LIGHT_DIRECTION))
        view_x = dot3(light_dir, right)
        view_y = dot3(light_dir, up)
        view_z = dot3(light_dir, forward)
        if view_z <= 0.001:
            return (0.5, 0.5), 0.0
        width, height = self.canvas.get_physical_size()
        aspect = max(1.0, float(width) / max(1.0, float(height)))
        focal = 1.0 / math.tan(math.radians(90.0) * 0.5)
        ndc_x = view_x * focal / (aspect * view_z)
        ndc_y = view_y * focal / view_z
        uv_x = ndc_x * 0.5 + 0.5
        uv_y = 0.5 - ndc_y * 0.5
        edge = max(abs(ndc_x), abs(ndc_y))
        edge_fade = clamp(1.0 - max(0.0, edge - 1.05) / 0.95, 0.0, 1.0)
        forward_strength = clamp(view_z * 1.5, 0.0, 1.0)
        strength = max(forward_strength * edge_fade, 0.42 if edge_fade > 0.0 else 0.0)
        return (uv_x, uv_y), strength

    @profile
    def _draw_visible_batches_to_pass(self, render_pass, visible_batches, use_gpu_visibility: bool, use_indirect: bool) -> None:
        current_vertex_buffer = None
        current_binding_offset = None
        set_vertex_buffer = render_pass.set_vertex_buffer
        if use_gpu_visibility or use_indirect:
            indirect_buffer = self._mesh_draw_indirect_buffer
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

    @profile
    def _submit_render(self, meshes=None):
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
                max(0.02, 0.1 * BLOCK_SIZE),
                max(128.0 * BLOCK_SIZE, DEFAULT_RENDER_DISTANCE_WORLD * 1.25),
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

        use_postprocess = bool(self.volumetric_lighting_enabled or self.final_present_enabled)
        if use_postprocess:
            self._ensure_postprocess_targets()
            assert self.scene_color_view is not None
            assert self.scene_render_pipeline is not None

            scene_color_attachment_view = self.scene_color_msaa_view if self.postprocess_msaa_sample_count > 1 else self.scene_color_view
            scene_resolve_target = self.scene_color_view if self.postprocess_msaa_sample_count > 1 else None
            scene_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": scene_color_attachment_view,
                        "resolve_target": scene_resolve_target,
                        "clear_value": (0.60, 0.80, 0.98, 1.0),
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
            self._draw_visible_batches_to_pass(scene_pass, visible_batches, use_gpu_visibility, use_indirect)
            scene_pass.end()

            if self.volumetric_lighting_enabled:
                assert self.occlusion_mask_view is not None
                assert self.volumetric_shaft_view is not None
                assert self.occlusion_mask_pipeline is not None
                assert self.radial_blur_pipeline is not None
                assert self.composite_pipeline is not None
                assert self.radial_blur_bind_group is not None
                assert self.composite_bind_group is not None

                mask_color_attachment_view = self.occlusion_mask_msaa_view if self.postprocess_msaa_sample_count > 1 else self.occlusion_mask_view
                mask_resolve_target = self.occlusion_mask_view if self.postprocess_msaa_sample_count > 1 else None
                mask_pass = encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": mask_color_attachment_view,
                            "resolve_target": mask_resolve_target,
                            "clear_value": (1.0, 1.0, 1.0, 1.0),
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
                mask_pass.set_pipeline(self.occlusion_mask_pipeline)
                mask_pass.set_bind_group(0, self.camera_bind_group)
                self._draw_visible_batches_to_pass(mask_pass, visible_batches, use_gpu_visibility, use_indirect)
                mask_pass.end()

                light_uv, light_strength = self._project_directional_light_to_screen(right, up, forward)
                radial_params = np.array([
                    light_uv[0],
                    light_uv[1],
                    float(VOLUMETRIC_LIGHTING_DENSITY),
                    float(VOLUMETRIC_LIGHTING_WEIGHT),
                    float(VOLUMETRIC_LIGHTING_DECAY),
                    float(VOLUMETRIC_LIGHTING_EXPOSURE),
                    float(VOLUMETRIC_LIGHTING_STRENGTH) * float(light_strength),
                    float(VOLUMETRIC_LIGHTING_SAMPLES),
                ], dtype=np.float32)
                self.device.queue.write_buffer(self.radial_blur_params_buffer, 0, radial_params.tobytes())

                blur_pass = encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": self.volumetric_shaft_view,
                            "resolve_target": None,
                            "clear_value": (0.0, 0.0, 0.0, 1.0),
                            "load_op": wgpu.LoadOp.clear,
                            "store_op": wgpu.StoreOp.store,
                        }
                    ],
                )
                blur_pass.set_pipeline(self.radial_blur_pipeline)
                blur_pass.set_bind_group(0, self.radial_blur_bind_group)
                blur_pass.draw(3, 1, 0, 0)
                blur_pass.end()

                composite_target_view = self.composite_color_view if self.final_present_enabled else color_view
                composite_pass = encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": composite_target_view,
                            "resolve_target": None,
                            "clear_value": (0.0, 0.0, 0.0, 1.0),
                            "load_op": wgpu.LoadOp.clear,
                            "store_op": wgpu.StoreOp.store,
                        }
                    ],
                )
                composite_pass.set_pipeline(self.composite_pipeline)
                composite_pass.set_bind_group(0, self.composite_bind_group)
                composite_pass.draw(3, 1, 0, 0)
                composite_pass.end()

            if self.final_present_enabled:
                source_bind_group = self.final_composite_bind_group if self.volumetric_lighting_enabled else self.final_scene_bind_group
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
            self._draw_visible_batches_to_pass(render_pass, visible_batches, use_gpu_visibility, use_indirect)
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

    def encode_render_meshes(self, meshes):
        """Build render commands from an explicit mesh iterable instead of chunk-cache visibility."""
        return self._submit_render(meshes=meshes)

    def submit_render_meshes(self, meshes):
        return self.encode_render_meshes(meshes)

    def _is_device_lost_error(self, exc: Exception) -> bool:
        text = str(exc)
        if "Parent device is lost" in text or "device is lost" in text.lower():
            return True
        return exc.__class__.__name__ == "GPUValidationError" and "lost" in text.lower()

    def _pending_chunk_work_count(self) -> int:
        return (
            len(self._pending_chunk_coords)
            + len(self._pending_voxel_mesh_results)
            + len(self._pending_surface_gpu_batches)
            + int(self._pending_surface_gpu_batches_chunk_total)
            + len(self._pending_gpu_mesh_batches)
        )

    def _view_ready_for_auto_exit(self) -> bool:
        if self._device_lost:
            return False
        target_coords = self._visible_chunk_coord_set if self._visible_chunk_coord_set else set(self._visible_chunk_coords)
        if not target_coords:
            return False
        if self._visible_missing_coords:
            return False
        if self._pending_chunk_work_count() > 0:
            return False
        chunk_cache = self.chunk_cache
        visible_nonempty_chunks = 0
        for coord in target_coords:
            mesh = chunk_cache.get(coord)
            if mesh is None:
                return False
            if int(getattr(mesh, "vertex_count", 0)) > 0:
                visible_nonempty_chunks += 1
        if visible_nonempty_chunks <= 0:
            return False
        if int(self._last_frame_draw_calls) <= 0:
            return False
        if int(self._last_frame_visible_vertices) <= 0:
            return False
        return True

    def _request_auto_exit(self) -> None:
        if self._auto_exit_requested:
            return
        self._auto_exit_requested = True
        target_count = len(self._visible_chunk_coord_set) if self._visible_chunk_coord_set else len(self._visible_chunk_coords)
        print(
            f"Info: fixed-size render complete; loaded {target_count} target chunks. Closing window.",
            file=sys.stderr,
        )
        try:
            self.canvas.close()
        except Exception:
            pass
        stop_loop = getattr(loop, "stop", None)
        if callable(stop_loop):
            try:
                stop_loop()
            except Exception:
                pass
        raise SystemExit(0)

    def _service_auto_exit(self) -> None:
        if not self.exit_when_view_ready or self._auto_exit_requested:
            return
        if self._view_ready_for_auto_exit():
            self._auto_exit_frame_count += 1
        else:
            self._auto_exit_frame_count = 0
        if self._auto_exit_frame_count >= 2:
            self._request_auto_exit()

    @profile
    def draw_frame(self) -> None:
        frame_start = time.perf_counter()
        now = frame_start
        dt = min(0.05, now - self.last_frame_time)
        self.last_frame_time = now
        profile_started_at = hud_profile.profile_begin_frame(self)
        try:
            update_start = time.perf_counter()
            if not self.freeze_camera:
                self._update_camera(dt)
            world_update_ms = (time.perf_counter() - update_start) * 1000.0

            visibility_lookup_ms = self.pipeline.refresh_visibility()
            encoder, color_view, render_stats = self._submit_render()
            visibility_lookup_ms += render_stats["visibility_lookup_ms"]
            camera_upload_ms = render_stats["camera_upload_ms"]
            swapchain_acquire_ms = render_stats["swapchain_acquire_ms"]
            render_encode_ms = render_stats["render_encode_ms"]
            draw_calls = int(render_stats["draw_calls"])
            merged_chunks = int(render_stats["merged_chunks"])
            visible_chunks = int(render_stats["visible_chunks"])
            visible_vertices = int(render_stats["visible_vertices"])
            self._last_frame_draw_calls = draw_calls
            self._last_frame_merged_batches = merged_chunks
            self._last_frame_visible_chunks = visible_chunks
            self._last_frame_visible_vertices = visible_vertices

            hud_profile.draw_profile_hud(self, encoder, color_view)
            hud_profile.draw_frame_breakdown_hud(self, encoder, color_view)

            command_finish_start = time.perf_counter()
            command_buffer = encoder.finish()
            command_finish_ms = (time.perf_counter() - command_finish_start) * 1000.0

            queue_submit_start = time.perf_counter()
            self.device.queue.submit([command_buffer])
            queue_submit_ms = (time.perf_counter() - queue_submit_start) * 1000.0

            self.pipeline.service_background_gpu_work()
            _, chunk_stream_ms = self.pipeline.prepare_chunks(dt)

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

            hud_profile.refresh_frame_breakdown_summary(self, frame_start)
        except Exception as exc:
            if self._is_device_lost_error(exc):
                self._device_lost = True
                print(f"Fatal: render device lost; stopping draw loop ({exc!s})", file=sys.stderr)
                if profile_started_at is not None:
                    hud_profile.profile_end_frame(self, profile_started_at, 0.0)
                return
            try:
                self.pipeline.service_background_gpu_work()
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
        self._service_auto_exit()
        if not self._auto_exit_requested:
            self.canvas.request_draw(self.draw_frame)
