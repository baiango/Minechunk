from __future__ import annotations

import sys
import time

from .cache import mesh_allocator as mesh_cache
from .meshing import gpu_mesher as wgpu_mesher
from .renderer_config import *
from .terrain.world import VoxelWorld

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


@profile
def regenerate_world(renderer, metal_mesher, allow_metal_fallback) -> None:
    """Reset world, chunk caches, pending mesh work, and camera spawn state.

    The public TerrainRenderer.regenerate_world() method delegates here so the
    reset/cleanup matrix can be tested and evolved separately from frame drawing.
    """
    self = renderer
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

