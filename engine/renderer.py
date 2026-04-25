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
        self.radiance_cascades_enabled = bool(RADIANCE_CASCADES_ENABLED)
        self.final_present_enabled = True
        self._postprocess_size = (0, 0)
        self.postprocess_msaa_sample_count = 4 if int(POSTPROCESS_MSAA_SAMPLE_COUNT) > 1 else 1
        self.scene_color_texture = None
        self.scene_color_view = None
        self.scene_color_msaa_texture = None
        self.scene_color_msaa_view = None
        self.scene_gbuffer_texture = None
        self.scene_gbuffer_view = None
        self.scene_gbuffer_msaa_texture = None
        self.scene_gbuffer_msaa_view = None
        self.gi_color_texture = None
        self.gi_color_view = None
        self.gi_cascade_textures = []
        self.gi_cascade_views = []
        self.gi_cascade_param_buffers = []
        self.gi_cascade_bind_groups = []
        self.worldspace_rc_textures = []
        self.worldspace_rc_views = []
        self.worldspace_rc_visibility_textures = []
        self.worldspace_rc_visibility_views = []
        self.worldspace_rc_volume_params_buffer = None
        self._worldspace_rc_signature = None
        self._worldspace_rc_active_signatures = [None, None, None, None]
        self._worldspace_rc_active_mins = [(0.0, 0.0, 0.0) for _ in range(4)]
        self._worldspace_rc_active_inv_extents = [(0.0, 0.0, 0.0) for _ in range(4)]
        self._worldspace_rc_cpu_radiance_volumes = [None, None, None, None]
        self._worldspace_rc_cpu_visibility_volumes = [None, None, None, None]
        self._worldspace_rc_last_update_frame = [-1000000, -1000000, -1000000, -1000000]
        self._worldspace_rc_update_cursor = 0
        self._worldspace_rc_frame_index = 0
        self._worldspace_rc_material_lru: OrderedDict[tuple[int, int, int], int] = OrderedDict()
        self.postprocess_depth_texture = None
        self.postprocess_depth_view = None
        self.postprocess_sampler = None
        self.gi_params_buffer = None
        self.final_present_params_buffer = None
        self.gi_bind_group_layout = None
        self.gi_pipeline = None
        self.gi_bind_group = None
        self.gi_compose_bind_group_layout = None
        self.gi_compose_pipeline = None
        self.gi_compose_bind_group = None
        self.final_present_bind_group_layout = None
        self.final_present_pipeline = None
        self.final_scene_bind_group = None
        self.final_gi_bind_group = None
        self.scene_render_pipeline = None
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
        self._visible_rel_y_bounds: tuple[int, int] = (0, 0)
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
        self.gi_params_buffer = self.device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.worldspace_rc_volume_params_buffer = self.device.create_buffer(
            size=256,
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
                "module": self.device.create_shader_module(code=GI_GBUFFER_SHADER),
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
        self.scene_gbuffer_texture = None
        self.scene_gbuffer_view = None
        self.scene_gbuffer_msaa_texture = None
        self.scene_gbuffer_msaa_view = None
        self.gi_color_texture = None
        self.gi_color_view = None
        self.gi_cascade_textures = []
        self.gi_cascade_views = []
        self.gi_cascade_param_buffers = []
        self.gi_cascade_bind_groups = []
        self.worldspace_rc_visibility_textures = []
        self.worldspace_rc_visibility_views = []
        self._worldspace_rc_signature = None
        self.postprocess_depth_texture = None
        self.postprocess_depth_view = None
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
        if is_new_press and key == "g":
            self.radiance_cascades_enabled = not self.radiance_cascades_enabled
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

    def _visible_rel_y_bounds_for_origin_y(self, origin_y: int) -> tuple[int, int]:
        if not VERTICAL_CHUNK_STACK_ENABLED:
            return (0, 0)
        requested_min_rel_y = -int(self._view_extent_neg_y)
        requested_max_rel_y = int(self._view_extent_pos_y)
        min_rel_y = max(requested_min_rel_y, -int(origin_y))
        max_rel_y = min(requested_max_rel_y, int(VERTICAL_CHUNK_COUNT - 1 - int(origin_y)))
        if min_rel_y > max_rel_y:
            return (0, 0)
        return (int(min_rel_y), int(max_rel_y))

    @profile
    def _build_visible_layout_template(
        self,
        origin_mod_x: int,
        origin_mod_z: int,
        origin_y: int,
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
        min_rel_y, max_rel_y = self._visible_rel_y_bounds_for_origin_y(int(origin_y))
        template_key = (
            neg_x,
            pos_x,
            min_rel_y,
            max_rel_y,
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
            chunk_y,
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

        new_rel_y_bounds = self._visible_rel_y_bounds_for_origin_y(int(new_origin[1]))
        shifted_visible_coords = False
        if previous_origin is not None and self._visible_chunk_coords:
            visible_coord_set = self._visible_chunk_coord_set
            visible_coords = self._visible_chunk_coords
            if (
                visible_coord_set
                and len(visible_coord_set) == len(visible_coords)
                and self._visible_rel_y_bounds == new_rel_y_bounds
            ):
                shifted_visible_coords = self._apply_visible_chunk_coord_delta(new_origin)
        self._visible_chunk_origin = new_origin
        self._visible_rel_y_bounds = new_rel_y_bounds

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
        cascade_count = max(1, int(RADIANCE_CASCADES_CASCADE_COUNT))
        if (
            target_size == self._postprocess_size
            and self.scene_color_view is not None
            and self.scene_gbuffer_view is not None
            and self.gi_color_view is not None
            and len(self.gi_cascade_views) == cascade_count
            and len(self.gi_cascade_bind_groups) == cascade_count
            and len(self.worldspace_rc_views) == 4
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
        worldspace_resolution = max(4, int(WORLDSPACE_RC_GRID_RESOLUTION))
        self.worldspace_rc_textures = []
        self.worldspace_rc_views = []
        self.worldspace_rc_visibility_textures = []
        self.worldspace_rc_visibility_views = []
        for _ in range(4):
            rc_texture = self.device.create_texture(
                size=(worldspace_resolution, worldspace_resolution, worldspace_resolution),
                dimension="3d",
                format=POSTPROCESS_GI_FORMAT,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            vis_texture = self.device.create_texture(
                size=(worldspace_resolution, worldspace_resolution, worldspace_resolution),
                dimension="3d",
                format=POSTPROCESS_GI_FORMAT,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self.worldspace_rc_textures.append(rc_texture)
            self.worldspace_rc_views.append(rc_texture.create_view(dimension="3d"))
            self.worldspace_rc_visibility_textures.append(vis_texture)
            self.worldspace_rc_visibility_views.append(vis_texture.create_view(dimension="3d"))
        self._worldspace_rc_signature = None
        self._worldspace_rc_active_signatures = [None, None, None, None]
        self._worldspace_rc_active_mins = [(0.0, 0.0, 0.0) for _ in range(4)]
        self._worldspace_rc_active_inv_extents = [(0.0, 0.0, 0.0) for _ in range(4)]
        self._worldspace_rc_cpu_radiance_volumes = [None, None, None, None]
        self._worldspace_rc_cpu_visibility_volumes = [None, None, None, None]
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

        self._write_final_present_params(target_size[0], target_size[1])

        gi_params = np.array([
            float(RADIANCE_CASCADES_STRENGTH),
            float(RADIANCE_CASCADES_BIAS),
            float(RADIANCE_CASCADES_SKY_STRENGTH),
            float(RADIANCE_CASCADES_HIT_THICKNESS),
            float(RADIANCE_CASCADES_MERGE_OVERLAP),
            float(RADIANCE_CASCADES_MERGE_STRENGTH),
            float(RADIANCE_CASCADES_CASCADE_COUNT),
            float(WORLDSPACE_RC_GRID_RESOLUTION),
        ], dtype=np.float32)
        self.device.queue.write_buffer(self.gi_params_buffer, 0, gi_params.tobytes())

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
                        {"binding": 4, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 80}},
                        {"binding": 5, "resource": {"buffer": self.gi_params_buffer, "offset": 0, "size": 32}},
                        {"binding": 6, "resource": {"buffer": cascade_param_buffer, "offset": 0, "size": 32}},
                    ],
                )
            )
        self.gi_bind_group = self.gi_cascade_bind_groups[0] if self.gi_cascade_bind_groups else None
        worldspace_rc_params = np.zeros((8, 4), dtype=np.float32)
        self.device.queue.write_buffer(self.worldspace_rc_volume_params_buffer, 0, worldspace_rc_params.tobytes())
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
                {"binding": 11, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 80}},
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
            ],
        )
        self.final_gi_bind_group = self.device.create_bind_group(
            layout=self.final_present_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.gi_color_view},
                {"binding": 1, "resource": self.postprocess_sampler},
                {"binding": 2, "resource": {"buffer": self.final_present_params_buffer, "offset": 0, "size": 32}},
            ],
        )

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
    def _worldspace_rc_material_rgb(self, material: int) -> tuple[float, float, float]:
        if material == 1:
            return 0.24, 0.22, 0.20
        if material == 2:
            return 0.42, 0.40, 0.38
        if material == 3:
            return 0.47, 0.31, 0.18
        if material == 4:
            return 0.31, 0.68, 0.24
        if material == 5:
            return 0.78, 0.71, 0.49
        if material == 6:
            return 0.95, 0.97, 0.98
        return 0.60, 0.80, 0.98

    @profile
    def _worldspace_rc_trace_directions(self) -> tuple[tuple[float, float, float], ...]:
        cached = getattr(self, "_worldspace_rc_trace_dirs", None)
        target_count = max(6, int(WORLDSPACE_RC_TRACE_DIRECTIONS))
        if cached is not None and len(cached) == target_count:
            return cached
        directions: list[tuple[float, float, float]] = []
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(target_count):
            y = 1.0 - (2.0 * (float(i) + 0.5) / float(target_count))
            radius = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden_angle * float(i)
            directions.append((math.cos(theta) * radius, y, math.sin(theta) * radius))
        self._worldspace_rc_trace_dirs = tuple(directions)
        return self._worldspace_rc_trace_dirs

    @profile
    def _worldspace_rc_solid_at(
        self,
        bx: int,
        by: int,
        bz: int,
        material_cache: dict[tuple[int, int, int], int],
    ) -> bool:
        key = (int(bx), int(by), int(bz))
        material = material_cache.get(key)
        if material is None:
            # The previous cross-frame OrderedDict LRU thrashed heavily during
            # flying RC updates: the benchmark saw tens of millions of evictions
            # and only a tiny hit rate. The per-update material_cache already
            # catches locality inside one cascade refresh, so avoid the global LRU
            # bookkeeping on this hot path.
            material = int(self.world.block_at(key[0], key[1], key[2])) if key[1] >= 0 else 1
            material_cache[key] = material
        return material != int(AIR)

    @profile
    def _worldspace_rc_estimate_hit_normal(
        self,
        bx: int,
        by: int,
        bz: int,
        ray_dx: float,
        ray_dy: float,
        ray_dz: float,
        material_cache: dict[tuple[int, int, int], int],
        normal_cache: dict[tuple[int, int, int], tuple[float, float, float]] | None = None,
    ) -> tuple[float, float, float]:
        cache_key = (int(bx), int(by), int(bz))
        if normal_cache is not None:
            cached = normal_cache.get(cache_key)
            if cached is not None:
                return cached
        sx0 = 1.0 if self._worldspace_rc_solid_at(bx - 1, by, bz, material_cache) else 0.0
        sx1 = 1.0 if self._worldspace_rc_solid_at(bx + 1, by, bz, material_cache) else 0.0
        sy0 = 1.0 if self._worldspace_rc_solid_at(bx, by - 1, bz, material_cache) else 0.0
        sy1 = 1.0 if self._worldspace_rc_solid_at(bx, by + 1, bz, material_cache) else 0.0
        sz0 = 1.0 if self._worldspace_rc_solid_at(bx, by, bz - 1, material_cache) else 0.0
        sz1 = 1.0 if self._worldspace_rc_solid_at(bx, by, bz + 1, material_cache) else 0.0
        nx = sx0 - sx1
        ny = sy0 - sy1
        nz = sz0 - sz1
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length <= 0.0001:
            nx = -float(ray_dx)
            ny = -float(ray_dy)
            nz = -float(ray_dz)
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            inv_length = 1.0 / max(length, 0.0001)
            return nx * inv_length, ny * inv_length, nz * inv_length
        inv_length = 1.0 / max(length, 0.0001)
        normal = (nx * inv_length, ny * inv_length, nz * inv_length)
        if normal_cache is not None:
            normal_cache[cache_key] = normal
        return normal

    @profile
    def _worldspace_rc_hit_sky_visibility(
        self,
        bx: int,
        by: int,
        bz: int,
        material_cache: dict[tuple[int, int, int], int],
        sky_visibility_cache: dict[tuple[int, int, int], float] | None = None,
        surface_height_cache: dict[tuple[int, int], int] | None = None,
    ) -> float:
        cache_key = (int(bx), int(by), int(bz))
        if sky_visibility_cache is not None:
            cached = sky_visibility_cache.get(cache_key)
            if cached is not None:
                return cached

        sample_count = max(1, int(WORLDSPACE_RC_SKY_VISIBILITY_STEPS))
        step_blocks = max(1, int(WORLDSPACE_RC_SKY_VISIBILITY_STEP_BLOCKS))
        aperture_radius = max(1, int(WORLDSPACE_RC_SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS))
        aperture_power = max(0.25, float(WORLDSPACE_RC_SKY_VISIBILITY_APERTURE_POWER))
        min_aperture = max(0.0, min(0.25, float(WORLDSPACE_RC_SKY_VISIBILITY_MIN_APERTURE)))
        side_weight = max(0.0, min(1.0, float(WORLDSPACE_RC_SKY_VISIBILITY_SIDE_WEIGHT)))
        world_height = int(getattr(self.world, "height", 0))

        def surface_height_at_cached(cx: int, cz: int) -> int:
            column_key = (int(cx), int(cz))
            if surface_height_cache is not None:
                cached_height = surface_height_cache.get(column_key)
                if cached_height is not None:
                    return int(cached_height)
            try:
                height = int(self.world.surface_height_at(column_key[0], column_key[1]))
            except Exception:
                height = world_height
            if surface_height_cache is not None:
                surface_height_cache[column_key] = height
            return height

        def column_access(sx: int, sz: int) -> float:
            column_x = cache_key[0] + int(sx)
            column_z = cache_key[2] + int(sz)
            surface_height = surface_height_at_cached(column_x, column_z)

            # Terrain blocks are guaranteed air at y >= surface_height. Once the
            # upward probe reaches that point, the remaining samples are open
            # without any block_at/material lookup.
            if cache_key[1] >= surface_height - 1:
                return 1.0

            open_count = 0
            for step in range(1, sample_count + 1):
                sample_y = cache_key[1] + step * step_blocks
                if sample_y >= surface_height:
                    open_count += sample_count - step + 1
                    break
                if self._worldspace_rc_solid_at(column_x, sample_y, column_z, material_cache):
                    break
                open_count += 1
            return float(open_count) / float(sample_count)

        if cache_key[1] < 0:
            visibility = 0.0
        else:
            center_access = column_access(0, 0)
            if center_access <= 0.0:
                visibility = 0.0
            elif center_access >= 0.999 or side_weight <= 0.0001:
                visibility = center_access
            else:
                # The default side weight is low, so use four cardinal side
                # columns instead of eight. Diagonals are reserved for high side
                # weights where the extra aperture detail is explicitly requested.
                side_offsets = [
                    (aperture_radius, 0),
                    (-aperture_radius, 0),
                    (0, aperture_radius),
                    (0, -aperture_radius),
                ]
                if side_weight >= 0.5:
                    side_offsets.extend(
                        [
                            (aperture_radius, aperture_radius),
                            (aperture_radius, -aperture_radius),
                            (-aperture_radius, aperture_radius),
                            (-aperture_radius, -aperture_radius),
                        ]
                    )

                side_access = 0.0
                for sx, sz in side_offsets:
                    side_access += column_access(sx, sz)
                side_access /= float(len(side_offsets))
                aperture = max(min_aperture, side_access ** aperture_power)
                visibility = center_access * aperture

        if sky_visibility_cache is not None:
            sky_visibility_cache[cache_key] = visibility
        return visibility

    @profile
    def _trace_worldspace_rc_probe(
        self,
        origin: tuple[float, float, float],
        max_distance: float,
        cascade_index: int,
        material_cache: dict[tuple[int, int, int], int],
        sky_visibility_cache: dict[tuple[int, int, int], float],
        normal_cache: dict[tuple[int, int, int], tuple[float, float, float]],
        probe_trace_cache: dict[tuple[int, int, int, int], tuple[tuple[float, float, float], float, float, float, float, float]],
        surface_height_cache: dict[tuple[int, int], int],
    ) -> tuple[tuple[float, float, float], float, float, float, float, float]:
        directions = self._worldspace_rc_trace_directions()
        sky_rgb = (
            0.60 * float(RADIANCE_CASCADES_SKY_STRENGTH),
            0.80 * float(RADIANCE_CASCADES_SKY_STRENGTH),
            0.98 * float(RADIANCE_CASCADES_SKY_STRENGTH),
        )
        sun_dx, sun_dy, sun_dz = normalize3(tuple(float(v) for v in LIGHT_DIRECTION))
        direct_sun_strength = max(0.0, float(WORLDSPACE_RC_DIRECT_SUN_STRENGTH))
        indirect_floor = max(0.0, float(WORLDSPACE_RC_INDIRECT_FLOOR))
        accum = np.zeros(3, dtype=np.float32)
        hit_count = 0.0
        sky_count = 0.0
        hit_sky_visibility_accum = 0.0
        distance_accum = 0.0
        distance_sq_accum = 0.0
        ox, oy, oz = origin
        origin_bx = int(math.floor(ox / BLOCK_SIZE))
        origin_by = int(math.floor(oy / BLOCK_SIZE))
        origin_bz = int(math.floor(oz / BLOCK_SIZE))
        if self._worldspace_rc_solid_at(origin_bx, origin_by, origin_bz, material_cache):
            blocked_distance = float(max_distance)
            return (0.0, 0.0, 0.0), 0.0, blocked_distance, blocked_distance * blocked_distance, 0.0, 0.0
        trace_cache_key = (origin_bx, origin_by, origin_bz, int(cascade_index))
        cached_trace = probe_trace_cache.get(trace_cache_key)
        if cached_trace is not None:
            return cached_trace
        origin_sky_visibility = self._worldspace_rc_hit_sky_visibility(origin_bx, origin_by, origin_bz, material_cache, sky_visibility_cache, surface_height_cache)
        effective_directions = directions
        if cascade_index >= 2 and len(effective_directions) > 8:
            effective_directions = effective_directions[::2]
        elif cascade_index >= 1 and origin_sky_visibility <= 0.04 and len(effective_directions) > 12:
            effective_directions = effective_directions[::2]
        effective_step_count = max(6, int(WORLDSPACE_RC_TRACE_MAX_STEPS) - max(0, int(cascade_index)) * 2)
        if origin_sky_visibility <= 0.04:
            effective_step_count = max(6, effective_step_count - 2)
        step_size = max(float(BLOCK_SIZE), float(max_distance) / float(effective_step_count))
        for dx, dy, dz in effective_directions:
            dist = step_size * 0.5
            hit = False
            hit_distance = float(max_distance)
            while dist <= max_distance:
                wx = ox + dx * dist
                wy = oy + dy * dist
                wz = oz + dz * dist
                bx = int(math.floor(wx / BLOCK_SIZE))
                by = int(math.floor(wy / BLOCK_SIZE))
                bz = int(math.floor(wz / BLOCK_SIZE))
                key = (bx, by, bz)
                material = material_cache.get(key)
                if material is None:
                    material = int(self.world.block_at(bx, by, bz)) if by >= 0 else 1
                    material_cache[key] = material
                if material != int(AIR):
                    color = self._worldspace_rc_material_rgb(material)
                    nx, ny, nz = self._worldspace_rc_estimate_hit_normal(bx, by, bz, dx, dy, dz, material_cache, normal_cache)
                    facing = max(0.0, min(1.0, -(nx * dx + ny * dy + nz * dz)))
                    sun_term = max(0.0, nx * sun_dx + ny * sun_dy + nz * sun_dz)
                    sky_visibility = self._worldspace_rc_hit_sky_visibility(bx, by, bz, material_cache, sky_visibility_cache, surface_height_cache)
                    cave_gate = sky_visibility ** 3.0
                    open_hemi = max(0.0, ny) ** 1.5
                    ambient_sky = cave_gate * (0.10 + 0.90 * open_hemi) * 0.38
                    direct_sun = cave_gate * sun_term * direct_sun_strength * 1.10
                    falloff = 1.0 - min(1.0, dist / max(max_distance, 1e-4))
                    range_term = 0.16 + 0.84 * falloff
                    facing_term = 0.20 + 0.80 * facing
                    bounce_scale = (indirect_floor + ambient_sky + direct_sun) * range_term * facing_term
                    accum[0] += color[0] * bounce_scale
                    accum[1] += color[1] * bounce_scale
                    accum[2] += color[2] * bounce_scale
                    hit_sky_visibility_accum += sky_visibility * (0.35 + 0.65 * facing)
                    hit_count += 1.0
                    hit = True
                    hit_distance = float(dist)
                    break
                dist += step_size
            if not hit:
                sky_axis = max(0.0, float(dy))
                sky_ray_weight = (0.03 + 0.97 * (sky_axis ** 2.0)) * (max(0.0, origin_sky_visibility) ** 1.85)
                accum[0] += sky_rgb[0] * sky_ray_weight
                accum[1] += sky_rgb[1] * sky_ray_weight
                accum[2] += sky_rgb[2] * sky_ray_weight
                sky_count += sky_ray_weight
            distance_accum += hit_distance
            distance_sq_accum += hit_distance * hit_distance
        ray_count = float(max(1, len(effective_directions)))
        accum /= ray_count
        hit_fraction = hit_count / ray_count
        sky_fraction = min(1.0, sky_count / ray_count)
        valid_fraction = min(1.0, max(0.0, hit_fraction + 0.75 * sky_fraction))
        observed_sky_access = 0.0
        if hit_count > 0.0:
            observed_sky_access = hit_sky_visibility_accum / hit_count
        observed_sky_access = max(observed_sky_access, sky_fraction)
        probe_sky_access = min(1.0, max(0.0, 0.78 * origin_sky_visibility + 0.22 * observed_sky_access))
        alpha = valid_fraction
        mean_distance = distance_accum / ray_count
        mean_distance_sq = distance_sq_accum / ray_count
        result = ((float(accum[0]), float(accum[1]), float(accum[2])), float(alpha), float(mean_distance), float(mean_distance_sq), float(valid_fraction), float(probe_sky_access))
        probe_trace_cache[trace_cache_key] = result
        return result

    @profile
    def _worldspace_rc_filter_volume(self, volume: np.ndarray) -> np.ndarray:
        passes = max(0, int(WORLDSPACE_RC_SPATIAL_FILTER_PASSES))
        if passes <= 0:
            return volume.astype(np.float16, copy=False)
        filtered = volume.astype(np.float32, copy=True)
        for _ in range(passes):
            src = filtered
            dst = src * 0.40
            weights = np.full(src.shape[:3] + (1,), 0.40, dtype=np.float32)
            for axis in range(3):
                src_front = [slice(None)] * 4
                dst_front = [slice(None)] * 4
                src_front[axis] = slice(0, -1)
                dst_front[axis] = slice(1, None)
                dst[tuple(dst_front)] += src[tuple(src_front)] * 0.10
                weights[tuple(dst_front[:3] + [slice(None)])] += 0.10
                src_back = [slice(None)] * 4
                dst_back = [slice(None)] * 4
                src_back[axis] = slice(1, None)
                dst_back[axis] = slice(0, -1)
                dst[tuple(dst_back)] += src[tuple(src_back)] * 0.10
                weights[tuple(dst_back[:3] + [slice(None)])] += 0.10
            filtered = dst / np.maximum(weights, 1e-6)
        return filtered.astype(np.float16)

    @profile
    def _worldspace_rc_sparse_sample_coords(self, resolution: int, stride: int) -> tuple[int, ...]:
        stride = max(1, int(stride))
        coords = list(range(0, max(1, int(resolution)), stride))
        last = max(0, int(resolution) - 1)
        if not coords or coords[-1] != last:
            coords.append(last)
        return tuple(sorted(set(int(v) for v in coords)))

    @profile
    def _worldspace_rc_fill_sparse_probe_volume(
        self,
        volume: np.ndarray,
        visibility_volume: np.ndarray,
        solid_probe_mask: np.ndarray,
        sample_coords: tuple[int, ...],
    ) -> None:
        resolution = int(volume.shape[0])
        if resolution <= 1:
            return

        sample_set = set(int(v) for v in sample_coords)
        valid_samples: list[tuple[int, int, int]] = []
        for sz in sample_coords:
            for sy in sample_coords:
                for sx in sample_coords:
                    z = int(sz)
                    y = int(sy)
                    x = int(sx)
                    if bool(solid_probe_mask[z, y, x]):
                        continue
                    if (
                        float(volume[z, y, x, 3]) > 0.0001
                        or float(visibility_volume[z, y, x, 2]) > 0.0001
                        or float(visibility_volume[z, y, x, 3]) > 0.0001
                    ):
                        valid_samples.append((x, y, z))
        if not valid_samples:
            return

        fill_invalid = bool(WORLDSPACE_RC_DILATE_INVALID_PROBES)
        max_solid_dist = max(1, int(WORLDSPACE_RC_INVALID_PROBE_DILATE_MAX_GRID_DISTANCE))
        max_solid_dist_sq = max_solid_dist * max_solid_dist
        solid_radiance_scale = max(0.0, min(1.0, float(WORLDSPACE_RC_INVALID_PROBE_DILATE_RADIANCE_SCALE)))

        # Previous code found the nearest valid sample by scanning every valid
        # sample for every cell. That created hundreds of millions of distance
        # checks in the 4096-chunk fly-forward profile. A small 3D multi-source
        # BFS gives each cell a nearby source in O(resolution^3) time.
        nearest_x = np.full((resolution, resolution, resolution), -1, dtype=np.int16)
        nearest_y = np.full((resolution, resolution, resolution), -1, dtype=np.int16)
        nearest_z = np.full((resolution, resolution, resolution), -1, dtype=np.int16)
        nearest_dist = np.full((resolution, resolution, resolution), 32767, dtype=np.int16)

        queue: deque[tuple[int, int, int]] = deque()
        for x, y, z in valid_samples:
            nearest_x[z, y, x] = np.int16(x)
            nearest_y[z, y, x] = np.int16(y)
            nearest_z[z, y, x] = np.int16(z)
            nearest_dist[z, y, x] = np.int16(0)
            queue.append((x, y, z))

        neighbor_offsets = (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
        while queue:
            x, y, z = queue.popleft()
            next_dist = int(nearest_dist[z, y, x]) + 1
            for dx, dy, dz in neighbor_offsets:
                nx = x + dx
                ny = y + dy
                nz = z + dz
                if nx < 0 or nx >= resolution or ny < 0 or ny >= resolution or nz < 0 or nz >= resolution:
                    continue
                if next_dist >= int(nearest_dist[nz, ny, nx]):
                    continue
                nearest_dist[nz, ny, nx] = np.int16(next_dist)
                nearest_x[nz, ny, nx] = nearest_x[z, y, x]
                nearest_y[nz, ny, nx] = nearest_y[z, y, x]
                nearest_z[nz, ny, nx] = nearest_z[z, y, x]
                queue.append((nx, ny, nz))

        for z in range(resolution):
            z_sampled = z in sample_set
            for y in range(resolution):
                yz_sampled = z_sampled and y in sample_set
                for x in range(resolution):
                    sampled = yz_sampled and x in sample_set
                    solid = bool(solid_probe_mask[z, y, x])
                    valid = (
                        (not solid)
                        and (
                            float(volume[z, y, x, 3]) > 0.0001
                            or float(visibility_volume[z, y, x, 2]) > 0.0001
                            or float(visibility_volume[z, y, x, 3]) > 0.0001
                        )
                    )

                    # Keep already-valid air cells. This makes the post-filter and
                    # post-temporal fill calls cheap instead of re-interpolating the
                    # whole volume after it has already been populated.
                    if valid:
                        continue
                    if solid and not fill_invalid:
                        continue

                    sx = int(nearest_x[z, y, x])
                    sy = int(nearest_y[z, y, x])
                    sz = int(nearest_z[z, y, x])
                    if sx < 0 or sy < 0 or sz < 0:
                        continue

                    dist_sq = (sx - x) * (sx - x) + (sy - y) * (sy - y) + (sz - z) * (sz - z)
                    if solid and dist_sq > max_solid_dist_sq:
                        continue

                    volume[z, y, x, :] = volume[sz, sy, sx, :]
                    visibility_volume[z, y, x, :] = visibility_volume[sz, sy, sx, :]
                    if solid:
                        volume[z, y, x, 0:3] = (volume[z, y, x, 0:3].astype(np.float32) * solid_radiance_scale).astype(np.float16)
                        volume[z, y, x, 3] = np.float16(float(volume[z, y, x, 3]) * solid_radiance_scale)
                        visibility_volume[z, y, x, 2] = np.float16(float(visibility_volume[z, y, x, 2]) * solid_radiance_scale)

    @profile
    def _worldspace_rc_temporal_blend(self, cascade_index: int, new_volume: np.ndarray, new_visibility_volume: np.ndarray, min_corner: tuple[float, float, float], full_extent: float) -> tuple[np.ndarray, np.ndarray]:
        prev_volume = self._worldspace_rc_cpu_radiance_volumes[cascade_index]
        prev_visibility = self._worldspace_rc_cpu_visibility_volumes[cascade_index]
        prev_min_corner = self._worldspace_rc_active_mins[cascade_index]
        resolution = max(2, int(WORLDSPACE_RC_GRID_RESOLUTION))
        cell_world = full_extent / float(max(1, resolution - 1))
        if prev_volume is None or prev_visibility is None:
            return new_volume, new_visibility_volume
        shift = max(abs(float(min_corner[i]) - float(prev_min_corner[i])) for i in range(3))
        if shift > cell_world * 2.5:
            return new_volume, new_visibility_volume
        blend_alpha = float(WORLDSPACE_RC_TEMPORAL_BLEND_ALPHA) / (1.0 + 0.15 * float(cascade_index))
        blend_alpha = min(0.80, max(0.10, blend_alpha))
        prev_volume_f = prev_volume.astype(np.float32)
        prev_visibility_f = prev_visibility.astype(np.float32)
        new_volume_f = new_volume.astype(np.float32)
        new_visibility_f = new_visibility_volume.astype(np.float32)
        blended_volume = prev_volume_f * (1.0 - blend_alpha) + new_volume_f * blend_alpha
        blended_visibility = prev_visibility_f * (1.0 - blend_alpha) + new_visibility_f * blend_alpha
        return blended_volume.astype(np.float16), blended_visibility.astype(np.float16)

    @profile
    def _update_worldspace_radiance_cascades(self) -> None:
        if len(self.worldspace_rc_textures) != 4 or len(self.worldspace_rc_visibility_textures) != 4 or self.worldspace_rc_volume_params_buffer is None:
            return
        self._worldspace_rc_frame_index += 1
        resolution = max(4, int(WORLDSPACE_RC_GRID_RESOLUTION))
        base_half_extent = max(float(BLOCK_SIZE) * 4.0, float(WORLDSPACE_RC_BASE_HALF_EXTENT_WORLD))
        camera_pos = tuple(float(v) for v in self.camera.position)
        target_signatures: list[tuple[float, ...]] = []
        target_mins: list[tuple[float, float, float]] = []
        target_inv_extents: list[tuple[float, float, float]] = []
        for cascade_index in range(4):
            half_extent = base_half_extent * (float(WORLDSPACE_RC_INTERVAL_SCALE) ** float(cascade_index))
            full_extent = half_extent * 2.0
            snap = max(float(WORLDSPACE_RC_UPDATE_QUANTIZE_WORLD), full_extent / max(1.0, float(resolution - 1)))
            center_x = math.floor(camera_pos[0] / snap) * snap
            center_y = math.floor(camera_pos[1] / snap) * snap
            center_z = math.floor(camera_pos[2] / snap) * snap
            min_corner = (center_x - half_extent, center_y - half_extent, center_z - half_extent)
            inv_extent = (1.0 / full_extent, 1.0 / full_extent, 1.0 / full_extent)
            target_signatures.append((round(min_corner[0], 4), round(min_corner[1], 4), round(min_corner[2], 4), round(full_extent, 4)))
            target_mins.append(min_corner)
            target_inv_extents.append(inv_extent)

        dirty_indices = [i for i in range(4) if target_signatures[i] != self._worldspace_rc_active_signatures[i]]
        if any(sig is None for sig in self._worldspace_rc_active_signatures):
            max_updates = max(1, int(WORLDSPACE_RC_INITIAL_MAX_CASCADES_PER_FRAME))
        else:
            max_updates = max(1, int(WORLDSPACE_RC_UPDATE_MAX_CASCADES_PER_FRAME))

        updated_any = False
        material_cache: dict[tuple[int, int, int], int] = {}
        sky_visibility_cache: dict[tuple[int, int, int], float] = {}
        surface_height_cache: dict[tuple[int, int], int] = {}
        normal_cache: dict[tuple[int, int, int], tuple[float, float, float]] = {}
        probe_trace_cache: dict[tuple[int, int, int, int], tuple[tuple[float, float, float], float, float, float, float, float]] = {}
        updates_done = 0
        search_order = [((self._worldspace_rc_update_cursor + offset) % 4) for offset in range(4)]
        for cascade_index in search_order:
            if cascade_index not in dirty_indices:
                continue
            base_frame_stride = max(1, int(WORLDSPACE_RC_UPDATE_FRAME_INTERVAL))
            frame_stride = base_frame_stride << cascade_index
            if self._worldspace_rc_active_signatures[cascade_index] is not None:
                frames_since = self._worldspace_rc_frame_index - int(self._worldspace_rc_last_update_frame[cascade_index])
                if frames_since < frame_stride:
                    continue
            min_corner = target_mins[cascade_index]
            full_extent = 1.0 / target_inv_extents[cascade_index][0]
            max_distance = full_extent * 0.5
            volume = np.zeros((resolution, resolution, resolution, 4), dtype=np.float16)
            visibility_volume = np.zeros((resolution, resolution, resolution, 4), dtype=np.float16)
            solid_probe_mask = np.zeros((resolution, resolution, resolution), dtype=np.bool_)
            stride_values = tuple(int(v) for v in WORLDSPACE_RC_CASCADE_PROBE_STRIDES) if bool(WORLDSPACE_RC_SPARSE_UPDATE_ENABLED) else (1, 1, 1, 1)
            stride_index = min(cascade_index, max(0, len(stride_values) - 1))
            probe_stride = max(1, int(stride_values[stride_index])) if stride_values else 1
            sample_coords = self._worldspace_rc_sparse_sample_coords(resolution, probe_stride)
            sample_coord_set = set(sample_coords)
            for z in range(resolution):
                fz = z / max(1, resolution - 1)
                wz = min_corner[2] + full_extent * fz
                for y in range(resolution):
                    fy = y / max(1, resolution - 1)
                    wy = min_corner[1] + full_extent * fy
                    for x in range(resolution):
                        fx = x / max(1, resolution - 1)
                        wx = min_corner[0] + full_extent * fx
                        probe_bx = int(math.floor(wx / BLOCK_SIZE))
                        probe_by = int(math.floor(wy / BLOCK_SIZE))
                        probe_bz = int(math.floor(wz / BLOCK_SIZE))
                        probe_inside_solid = self._worldspace_rc_solid_at(probe_bx, probe_by, probe_bz, material_cache)
                        solid_probe_mask[z, y, x] = probe_inside_solid
                        if probe_inside_solid or x not in sample_coord_set or y not in sample_coord_set or z not in sample_coord_set:
                            continue
                        rgb, alpha, mean_distance, mean_distance_sq, hit_fraction, sky_access = self._trace_worldspace_rc_probe(
                            (wx, wy, wz),
                            max_distance,
                            cascade_index,
                            material_cache,
                            sky_visibility_cache,
                            normal_cache,
                            probe_trace_cache,
                            surface_height_cache,
                        )
                        volume[z, y, x, 0] = np.float16(rgb[0])
                        volume[z, y, x, 1] = np.float16(rgb[1])
                        volume[z, y, x, 2] = np.float16(rgb[2])
                        volume[z, y, x, 3] = np.float16(alpha)
                        visibility_volume[z, y, x, 0] = np.float16(mean_distance)
                        visibility_volume[z, y, x, 1] = np.float16(mean_distance_sq)
                        visibility_volume[z, y, x, 2] = np.float16(hit_fraction)
                        visibility_volume[z, y, x, 3] = np.float16(sky_access)
            self._worldspace_rc_fill_sparse_probe_volume(volume, visibility_volume, solid_probe_mask, sample_coords)
            raw_volume = volume
            raw_visibility_volume = visibility_volume
            filtered_volume = self._worldspace_rc_filter_volume(volume)
            filtered_visibility_volume = self._worldspace_rc_filter_volume(visibility_volume)
            sky_filter_gate = np.power(
                np.clip(raw_visibility_volume[..., 3:4].astype(np.float32), 0.0, 1.0),
                max(0.25, float(WORLDSPACE_RC_SPATIAL_FILTER_SKY_POWER)),
            ).astype(np.float32)
            volume = (raw_volume.astype(np.float32) * (1.0 - sky_filter_gate) + filtered_volume.astype(np.float32) * sky_filter_gate).astype(np.float16)
            visibility_volume = filtered_visibility_volume
            visibility_volume[..., 3] = np.minimum(filtered_visibility_volume[..., 3], raw_visibility_volume[..., 3]).astype(np.float16)
            self._worldspace_rc_fill_sparse_probe_volume(volume, visibility_volume, solid_probe_mask, sample_coords)
            volume, visibility_volume = self._worldspace_rc_temporal_blend(cascade_index, volume, visibility_volume, min_corner, full_extent)
            self._worldspace_rc_fill_sparse_probe_volume(volume, visibility_volume, solid_probe_mask, sample_coords)
            self.device.queue.write_texture(
                {"texture": self.worldspace_rc_textures[cascade_index], "mip_level": 0, "origin": (0, 0, 0)},
                volume.tobytes(),
                {"offset": 0, "bytes_per_row": resolution * 8, "rows_per_image": resolution},
                (resolution, resolution, resolution),
            )
            self.device.queue.write_texture(
                {"texture": self.worldspace_rc_visibility_textures[cascade_index], "mip_level": 0, "origin": (0, 0, 0)},
                visibility_volume.tobytes(),
                {"offset": 0, "bytes_per_row": resolution * 8, "rows_per_image": resolution},
                (resolution, resolution, resolution),
            )
            self._worldspace_rc_cpu_radiance_volumes[cascade_index] = volume
            self._worldspace_rc_cpu_visibility_volumes[cascade_index] = visibility_volume
            self._worldspace_rc_active_signatures[cascade_index] = target_signatures[cascade_index]
            self._worldspace_rc_active_mins[cascade_index] = target_mins[cascade_index]
            self._worldspace_rc_active_inv_extents[cascade_index] = target_inv_extents[cascade_index]
            self._worldspace_rc_last_update_frame[cascade_index] = self._worldspace_rc_frame_index
            self._worldspace_rc_update_cursor = (cascade_index + 1) % 4
            updates_done += 1
            updated_any = True
            if updates_done >= max_updates:
                break

        params = np.zeros((8, 4), dtype=np.float32)
        for cascade_index in range(4):
            params[cascade_index, 0:3] = self._worldspace_rc_active_mins[cascade_index]
            params[4 + cascade_index, 0:3] = self._worldspace_rc_active_inv_extents[cascade_index]
        self.device.queue.write_buffer(self.worldspace_rc_volume_params_buffer, 0, params.tobytes())
        if updated_any:
            signature_parts: list[float] = []
            for cascade_index in range(4):
                active_sig = self._worldspace_rc_active_signatures[cascade_index]
                if active_sig is None:
                    signature_parts.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    signature_parts.extend(list(active_sig))
            self._worldspace_rc_signature = tuple(signature_parts)

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

        use_postprocess = bool(self.final_present_enabled)
        if use_postprocess:
            self._ensure_postprocess_targets()
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
            self._draw_visible_batches_to_pass(scene_pass, visible_batches, use_gpu_visibility, use_indirect)
            scene_pass.end()

            if self.radiance_cascades_enabled:
                assert self.gi_color_view is not None
                assert self.gi_compose_pipeline is not None
                assert self.gi_compose_bind_group is not None
                self._update_worldspace_radiance_cascades()
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
