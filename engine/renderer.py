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
from . import auto_exit, render_contract
from .render_shaders import (
    FINAL_BLIT_SHADER,
    GI_CASCADE_SHADER,
    GI_GBUFFER_SHADER,
    GI_POSTPROCESS_SHADER,
    GPU_VISIBILITY_SHADER,
    WORLDSPACE_RC_FILTER_SHADER,
    WORLDSPACE_RC_TRACE_SHADER,
    WORLDSPACE_RC_UPDATE_PARAMS_BYTES,
    HUD_SHADER,
    RENDER_SHADER,
    TILE_MERGE_SHADER,
    VOXEL_MESH_BATCH_SHADER,
    VOXEL_SURFACE_EXPAND_SHADER,
)
from .collision import walk_solver
from .rendering import direct_draw, postprocess_targets, rc_debug_capture, worldspace_rc
from .render_utils import (
    cross3,
    flat_forward_vector,
    forward_vector,
    normalize3,
    pack_camera_uniform,
    right_vector,
)
from .meshing import gpu_mesher as wgpu_mesher
try:
    from .meshing import metal_mesher as metal_mesher
except Exception as exc:  # Metal is optional; WGPU/CPU modes must import without pyobjc.
    metal_mesher = None  # type: ignore[assignment]
    METAL_MESHER_IMPORT_ERROR = exc
else:
    METAL_MESHER_IMPORT_ERROR = None
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

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


@profile
def _engine_mode_uses_gpu_path() -> bool:
    return engine_mode != ENGINE_MODE_CPU


def _allow_metal_fallback() -> bool:
    return render_contract.truthy_env_flag("MINECHUNK_ALLOW_METAL_FALLBACK")


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
        self.render_api_label = render_contract.describe_adapter(self.adapter)
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
        if engine_mode == ENGINE_MODE_METAL and self.world.terrain_backend_label() != "Metal" and not _allow_metal_fallback():
            failure = getattr(self.world, "_gpu_backend_error", None)
            detail = f" ({type(failure).__name__}: {failure!s})" if failure is not None else ""
            raise RuntimeError(
                "ENGINE_MODE_METAL was requested, but the active terrain backend is "
                f"{self.world.terrain_backend_label()!r}{detail}. Refusing CPU/WGPU fallback. "
                "Run `python3 -m pip install -r requirements.txt`, or launch with --allow-metal-fallback."
            )
        self.mesh_backend_label = "CPU"
        self._using_metal_meshing = bool(self.use_gpu_meshing and engine_mode == ENGINE_MODE_METAL and metal_mesher is not None)
        if self.use_gpu_meshing:
            self.mesh_backend_label = "Metal" if self._using_metal_meshing else "Wgpu"
        if self._using_metal_meshing and metal_mesher is not None:
            metal_mesher.prewarm_metal_chunk_mesher(self)
        if self.use_gpu_terrain and self.world.terrain_backend_label() == "Metal":
            print(
                "Info: Metal terrain backend active; native Metal surface handoff will be used when the mesher is Metal.",
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
        self.rc_debug_mode = 0
        self.rc_debug_mode_names = (
            "off",
            "cascade contribution",
            "sky/open signal",
            "normal",
            "cascade confidence",
            "rc radiance",
            "rc light",
            "volume coverage",
        )
        self._pending_rc_debug_capture_request: dict | None = None
        self._pending_rc_debug_readbacks: list[dict] = []
        self._worldspace_rc_last_capture_paths: list[str] = []
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
        self._gi_color_size = (0, 0)
        self.gi_cascade_textures = []
        self.gi_cascade_views = []
        self.gi_cascade_param_buffers = []
        self.gi_cascade_bind_groups = []
        self.worldspace_rc_textures = []
        self.worldspace_rc_views = []
        self.worldspace_rc_visibility_textures = []
        self.worldspace_rc_visibility_views = []
        self.worldspace_rc_volume_params_buffer = None
        self.worldspace_rc_update_param_buffers = []
        self.worldspace_rc_scratch_textures = []
        self.worldspace_rc_scratch_views = []
        self.worldspace_rc_visibility_scratch_textures = []
        self.worldspace_rc_visibility_scratch_views = []
        self.worldspace_rc_trace_bind_groups = []
        self.worldspace_rc_filter_bind_groups = []
        self.worldspace_rc_filter_pingpong_bind_groups = []
        self._worldspace_rc_signature = None
        self._worldspace_rc_active_signatures = [None, None, None, None]
        self._worldspace_rc_active_mins = [(0.0, 0.0, 0.0) for _ in range(4)]
        self._worldspace_rc_active_inv_extents = [(0.0, 0.0, 0.0) for _ in range(4)]
        self._worldspace_rc_last_update_frame = [-1000000, -1000000, -1000000, -1000000]
        self._worldspace_rc_update_cursor = 0
        self._worldspace_rc_convergence_frames_remaining = 0
        self._worldspace_rc_frame_index = 0
        self._worldspace_rc_last_scheduled_updates: list[int] = []
        self._worldspace_rc_last_dirty_indices: list[int] = []
        self._worldspace_rc_last_stable_refresh_indices: list[int] = []
        self._worldspace_rc_last_history_reject_updates: list[int] = []
        self._worldspace_rc_last_update_kind = "init"
        self._worldspace_rc_last_interval_bands = [(0.0, 0.0) for _ in range(4)]
        self._worldspace_rc_last_snapshot_path: str | None = None
        self.worldspace_rc_trace_bind_group_layout = None
        self.worldspace_rc_filter_bind_group_layout = None
        self.worldspace_rc_trace_pipeline = None
        self.worldspace_rc_filter_pipeline = None
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
            max(1, int(render_contract.device_limit(self.device, "min_storage_buffer_offset_alignment", 256))),
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
        self._voxel_surface_expand_bind_group_cache: OrderedDict[tuple[int, int, int, int, int, int], object] = OrderedDict()
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
        if metal_mesher is not None:
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
        self._gi_color_size = (0, 0)
        self.gi_cascade_textures = []
        self.gi_cascade_views = []
        self.gi_cascade_param_buffers = []
        self.gi_cascade_bind_groups = []
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
        self._worldspace_rc_signature = None
        self._worldspace_rc_active_signatures = [None, None, None, None]
        self._worldspace_rc_last_update_frame = [-1000000, -1000000, -1000000, -1000000]
        self._worldspace_rc_last_scheduled_updates = []
        self._worldspace_rc_last_dirty_indices = []
        self._worldspace_rc_last_stable_refresh_indices = []
        self._worldspace_rc_last_history_reject_updates = []
        self._worldspace_rc_last_update_kind = "resize"
        self._worldspace_rc_last_interval_bands = [(0.0, 0.0) for _ in range(4)]
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
        if is_new_press and key == "f6":
            self.rc_debug_mode = (int(self.rc_debug_mode) + 1) % len(self.rc_debug_mode_names)
            postprocess_targets.write_gi_params(self)
            print(f"Info: RC debug mode = {self.rc_debug_mode} ({self.rc_debug_mode_names[self.rc_debug_mode]})")
        if is_new_press and key == "f7":
            rc_debug_capture.dump_diagnostics(self)
            rc_debug_capture.queue_image_dump(self)
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

    def _log_backend_diagnostics(self) -> None:
        terrain_backend = getattr(self.world, "_backend", None)
        terrain_device = getattr(terrain_backend, "device", None)
        mesh_backend_label = getattr(self, "mesh_backend_label", "CPU")
        print(
            "Info: Backend diagnostics: "
            f"engine_mode={engine_mode}, "
            f"use_gpu_terrain={self.use_gpu_terrain}, "
            f"use_gpu_meshing={self.use_gpu_meshing}, "
            f"render_backend={self.render_backend_label}, "
            f"render_device={type(self.device).__name__}, "
            f"terrain_backend={type(terrain_backend).__name__}, "
            f"terrain_device={type(terrain_device).__name__}, "
            f"mesh_backend={mesh_backend_label}",
            file=sys.stderr,
        )

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
            if hasattr(pending, "slot") or hasattr(pending, "surface_release_callbacks"):
                if metal_mesher is not None:
                    try:
                        metal_mesher.destroy_async_voxel_mesh_batch_resources(pending)
                    except Exception:
                        pass
                continue
            readback_buffer = getattr(pending, "readback_buffer", None)
            if readback_buffer is not None and getattr(readback_buffer, "map_state", "unmapped") != "unmapped":
                try:
                    readback_buffer.unmap()
                except Exception:
                    pass
            resources = getattr(pending, "resources", None)
            if resources is not None:
                wgpu_mesher.destroy_async_voxel_mesh_batch_resources(resources)
                continue
            for buffer in (
                getattr(pending, "blocks_buffer", None),
                getattr(pending, "materials_buffer", None),
                getattr(pending, "coords_buffer", None),
                getattr(pending, "column_totals_buffer", None),
                getattr(pending, "chunk_totals_buffer", None),
                getattr(pending, "chunk_offsets_buffer", None),
                getattr(pending, "params_buffer", None),
                readback_buffer,
            ):
                if buffer is None:
                    continue
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
        if metal_mesher is not None:
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
        if engine_mode == ENGINE_MODE_METAL and self.world.terrain_backend_label() != "Metal" and not _allow_metal_fallback():
            failure = getattr(self.world, "_gpu_backend_error", None)
            detail = f" ({type(failure).__name__}: {failure!s})" if failure is not None else ""
            raise RuntimeError(
                "ENGINE_MODE_METAL was requested after reset, but the active terrain backend is "
                f"{self.world.terrain_backend_label()!r}{detail}. Refusing CPU/WGPU fallback."
            )
        self._using_metal_meshing = bool(self.use_gpu_meshing and engine_mode == ENGINE_MODE_METAL and metal_mesher is not None)
        if self.use_gpu_meshing:
            self.mesh_backend_label = "Metal" if self._using_metal_meshing else "Wgpu"
        if self._using_metal_meshing and metal_mesher is not None:
            metal_mesher.prewarm_metal_chunk_mesher(self)
        if self.use_gpu_terrain and self.world.terrain_backend_label() == "Metal":
            print(
                "Info: Metal terrain backend active; native Metal surface handoff will be used when the mesher is Metal.",
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
    def _update_camera(self, dt: float) -> None:
        walk_solver.update_camera(self, dt)

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


    def raycast_from_camera(self, max_distance: float = 8.0):
        """Return the first block in front of the camera using Amanatides-Woo DDA."""

        direction = normalize3(forward_vector(self.camera.yaw, self.camera.pitch))
        return self.world.raycast_blocks(
            tuple(float(v) for v in self.camera.position),
            direction,
            float(max_distance),
            start_distance=float(BLOCK_SIZE) * 1.0e-4,
        )

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

    def encode_render_meshes(self, meshes):
        """Build render commands from an explicit mesh iterable instead of chunk-cache visibility."""
        return self._submit_render(meshes=meshes)

    def submit_render_meshes(self, meshes):
        return self.encode_render_meshes(meshes)

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
            rc_debug_capture.drain_pending_readbacks(self)

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
            if auto_exit.is_device_lost_error(exc):
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
        auto_exit.service_auto_exit(self)
        if not self._auto_exit_requested:
            self.canvas.request_draw(self.draw_frame)
