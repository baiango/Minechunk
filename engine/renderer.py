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
from .cache import mesh_zstd
from .renderer_config import *
from . import auto_exit, input_controller, profiling_runtime, render_contract, world_reset
from .collision import walk_solver
from .rendering import frame_encoder, gpu_resources, postprocess_targets, rc_debug_capture
from .render_utils import (
    cross3,
    forward_vector,
    normalize3,
    pack_camera_uniform,
    right_vector,
)
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


def _normalize_backend_choice(value: str | None, *, default: str, supported: tuple[str, ...]) -> str:
    selected = default if value is None else str(value).strip().lower()
    if selected not in supported:
        raise ValueError(f"backend must be one of: {', '.join(supported)}")
    return selected


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
        terrain_zstd_enabled: bool | None = None,
        mesh_zstd_enabled: bool | None = None,
        tile_merging_enabled: bool | None = None,
        postprocess_enabled: bool | None = None,
        terrain_caves_enabled: bool | None = None,
        renderer_backend: str | None = None,
        terrain_backend: str | None = None,
        meshing_backend: str | None = None,
    ) -> None:
        default_use_gpu = _engine_mode_uses_gpu_path()
        default_gpu_backend = ENGINE_MODE_METAL if engine_mode == ENGINE_MODE_METAL else ENGINE_MODE_WGPU
        if terrain_backend is None:
            terrain_backend = default_gpu_backend if (default_use_gpu if use_gpu_terrain is None else bool(use_gpu_terrain)) else ENGINE_MODE_CPU
        if meshing_backend is None:
            meshing_backend = default_gpu_backend if (default_use_gpu if use_gpu_meshing is None else bool(use_gpu_meshing)) else ENGINE_MODE_CPU
        self.renderer_backend_kind = _normalize_backend_choice(renderer_backend, default=ENGINE_MODE_WGPU, supported=(ENGINE_MODE_WGPU,))
        self.terrain_backend_kind = _normalize_backend_choice(terrain_backend, default=ENGINE_MODE_CPU, supported=(ENGINE_MODE_CPU, ENGINE_MODE_WGPU, ENGINE_MODE_METAL))
        self.meshing_backend_kind = _normalize_backend_choice(meshing_backend, default=ENGINE_MODE_CPU, supported=(ENGINE_MODE_CPU, ENGINE_MODE_WGPU, ENGINE_MODE_METAL))
        self.use_gpu_terrain = self.terrain_backend_kind != ENGINE_MODE_CPU
        self.use_gpu_meshing = self.meshing_backend_kind != ENGINE_MODE_CPU
        self.terrain_zstd_enabled = bool(TERRAIN_ZSTD_ENABLED if terrain_zstd_enabled is None else terrain_zstd_enabled)
        self.terrain_caves_enabled = True if terrain_caves_enabled is None else bool(terrain_caves_enabled)
        self.mesh_zstd_enabled = bool(MESH_ZSTD_ENABLED if mesh_zstd_enabled is None else mesh_zstd_enabled)
        self.tile_merging_enabled = bool(TILE_MERGING_ENABLED if tile_merging_enabled is None else tile_merging_enabled)
        self.tile_zstd_enabled = bool(TILE_ZSTD_ENABLED and self.mesh_zstd_enabled and self.tile_merging_enabled)
        self.terrain_batch_size = max(1, int(terrain_batch_size))
        if mesh_batch_size is None:
            self.mesh_batch_size = max(1, int(self.terrain_batch_size))
        else:
            self.mesh_batch_size = max(1, int(mesh_batch_size))
        self._pending_surface_gpu_batch_target_multiplier = 2 if CHUNK_SIZE >= 64 else 10
        self.base_title = "Minechunk"
        self.engine_mode_label = f"render={self.renderer_backend_kind} terrain={self.terrain_backend_kind} meshing={self.meshing_backend_kind}"
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
            prefer_metal_backend=self.terrain_backend_kind == ENGINE_MODE_METAL,
            terrain_batch_size=self.terrain_batch_size,
            terrain_zstd_enabled=self.terrain_zstd_enabled,
            terrain_caves_enabled=self.terrain_caves_enabled,
        )
        if self.terrain_backend_kind == ENGINE_MODE_METAL and self.world.terrain_backend_label() != "Metal" and not _allow_metal_fallback():
            failure = getattr(self.world, "_gpu_backend_error", None)
            detail = f" ({type(failure).__name__}: {failure!s})" if failure is not None else ""
            raise RuntimeError(
                "Metal terrain backend was requested, but the active terrain backend is "
                f"{self.world.terrain_backend_label()!r}{detail}. Refusing CPU/WGPU fallback. "
                "Run `python3 -m pip install -r requirements.txt`, or launch with --allow-metal-fallback."
            )
        self.mesh_backend_label = "CPU"
        self._using_metal_meshing = bool(self.use_gpu_meshing and self.meshing_backend_kind == ENGINE_MODE_METAL and metal_mesher is not None)
        if self.use_gpu_meshing and self.meshing_backend_kind == ENGINE_MODE_METAL and metal_mesher is None and not _allow_metal_fallback():
            raise RuntimeError("Metal meshing backend was requested, but the optional Metal mesher is unavailable.")
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
        self.final_present_enabled = True if postprocess_enabled is None else bool(postprocess_enabled)
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
        self._visible_display_state_incremental = False
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
        self._mesh_zstd_cache: OrderedDict[tuple[int, int, int], object] = OrderedDict()
        self._pending_mesh_zstd_readbacks: deque = deque()
        self._pending_mesh_zstd_readback_keys: set[tuple[int, int, int]] = set()
        self._tile_zstd_cache: OrderedDict[tuple[int, int, int], object] = OrderedDict()
        self._pending_tile_zstd_readbacks: deque = deque()
        self._pending_tile_zstd_readback_keys: set[tuple[int, int, int]] = set()
        self._mesh_compaction_retired_cleanup_bytes: deque[tuple[int, int]] = deque()
        self._mesh_compaction_last_stats: dict[str, int | bool] = {}
        self._memory_pressure_next_relief_at = 0.0
        self._memory_pressure_last_relief_at = 0.0
        self._memory_pressure_last_relief_bytes = 0
        self._memory_pressure_relief_calls = 0
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
        self.mesh_zstd_cache_limit = int(self.max_cached_chunks)
        self.world.set_terrain_zstd_cache_limit(self.max_cached_chunks)
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
            "chunk_generated": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_displayed_added": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_rendered_added": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "terrain_zstd_stream_entries": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "terrain_zstd_stream_raw_bytes": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "terrain_zstd_stream_compressed_bytes": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
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
        self._terrain_zstd_total_entries = 0
        self._terrain_zstd_total_raw_bytes = 0
        self._terrain_zstd_total_compressed_bytes = 0
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
        self._last_new_rendered_chunks = 0
        self._last_chunk_stream_generated = 0
        self._last_chunk_stream_drained = 0
        self._last_frame_visible_batches = 0
        self._last_displayed_chunk_coords: set[tuple[int, int, int]] = set()
        self._last_rendered_chunk_coords: set[tuple[int, int, int]] = set()
        self._profile_chunks_generated = 0
        self._profile_chunks_rendered = 0
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

        gpu_resources.initialize_gpu_resources(self)

        input_controller.bind_canvas_events(self)

        profiling_runtime.disable(self)
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
            mesh_zstd.clear_mesh_zstd_cache(self)
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

    def _key_active(self, *names: str) -> bool:
        return input_controller.key_active(self, *names)

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
        world_reset.regenerate_world(self, metal_mesher, _allow_metal_fallback)

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
    def _update_camera(self, dt: float) -> None:
        walk_solver.update_camera(self, dt)

    def _ensure_depth_buffer(self, physical_size: tuple[int, int] | None = None) -> None:
        width, height = self.canvas.get_physical_size() if physical_size is None else physical_size
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
            world_layers = max(1, int(VERTICAL_CHUNK_COUNT))
            requested_layers = max(1, int(self._view_extent_neg_y) + int(self._view_extent_pos_y) + 1)
            current_y = max(0, min(world_layers - 1, int(current[1])))
            if requested_layers < world_layers:
                min_origin_y = max(0, int(self._view_extent_neg_y))
                max_origin_y = max(min_origin_y, int(world_layers - 1 - self._view_extent_pos_y))
                current_y = min(max(current_y, min_origin_y), max_origin_y)
            current = (
                int(current[0]),
                current_y,
                int(current[2]),
            )
        if self.freeze_view_origin:
            if self._frozen_view_origin is None:
                self._frozen_view_origin = current
            return self._frozen_view_origin
        return current



























    def encode_render_meshes(self, meshes):
        """Build render commands from an explicit mesh iterable instead of chunk-cache visibility."""
        return self._submit_render(meshes=meshes)

    def submit_render_meshes(self, meshes):
        return self.encode_render_meshes(meshes)

    @profile
    def _submit_render(self, meshes=None):
        """Encode a frame render pass.

        Kept as an overridable compatibility hook for benchmark entrypoints such
        as render_fly_forward_4096_then_exit.py, which count chunks after the
        frame's visible batches have been encoded.
        """
        return frame_encoder.submit_render(self, meshes=meshes)

    @profile
    def _service_auto_exit(self) -> None:
        """Service auto-exit logic.

        Kept as an overridable compatibility hook for benchmark entrypoints that
        provide their own completion criteria and status logging.
        """
        auto_exit.service_auto_exit(self)

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
            rendered_chunk_coords = set(self._visible_displayed_coords)
            self._last_new_rendered_chunks = len(rendered_chunk_coords - self._last_rendered_chunk_coords)
            self._last_rendered_chunk_coords = rendered_chunk_coords

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
            hud_profile.record_frame_breakdown_sample(self, "chunk_generated", float(self._last_chunk_stream_generated))
            hud_profile.record_frame_breakdown_sample(self, "chunk_displayed_added", float(self._last_new_displayed_chunks))
            hud_profile.record_frame_breakdown_sample(self, "chunk_rendered_added", float(self._last_new_rendered_chunks))
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
        self._service_auto_exit()
        if not self._auto_exit_requested:
            self.canvas.request_draw(self.draw_frame)
