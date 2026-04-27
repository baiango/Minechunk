from __future__ import annotations

import struct
from pathlib import Path
from typing import Callable

import Metal
import numpy as np

from ..terrain.types import ChunkSurfaceGpuBatch, ChunkVoxelResult
from .metal_mesher_common import (
    AsyncMetalMeshBatchResources,
    MetalMesherSlot,
    _normalize_chunk_coords,
    _renderer_module,
    profile,
    release_surface_gpu_batch_immediately,
)


class MetalChunkMesher:
    """Metal compute mesher for stacked Minechunk chunks."""

    def __init__(self, device, *, chunk_capacity: int, sample_size: int, height_limit: int, inflight_slots: int = 3):
        if int(sample_size) < 3:
            raise ValueError(f"sample_size must be >= 3, got {sample_size}")
        if int(height_limit) <= 0:
            raise ValueError(f"height_limit must be > 0, got {height_limit}")
        if int(chunk_capacity) <= 0:
            raise ValueError(f"chunk_capacity must be > 0, got {chunk_capacity}")
        if int(inflight_slots) <= 0:
            raise ValueError(f"inflight_slots must be > 0, got {inflight_slots}")

        renderer_module = _renderer_module()
        self.device = device
        self.command_queue = device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal mesher command queue.")
        self.chunk_capacity = int(chunk_capacity)
        self.sample_size = int(sample_size)
        self.height_limit = int(height_limit)
        self.storage_height = self.height_limit + 2
        self.inflight_slots = int(inflight_slots)
        self.max_vertices_per_chunk = int(renderer_module.MAX_VERTICES_PER_CHUNK)
        self.vertex_stride = int(renderer_module.VERTEX_STRIDE)

        self._lock = threading.Lock()
        self._slots: list[MetalMesherSlot] = []
        self._expand_pipeline = self._build_pipeline("expand_surface_to_voxels")
        self._count_pipeline = self._build_pipeline("count_columns_fixed_slice")
        self._scan_serial_pipeline = self._build_pipeline("scan_columns_fixed_slice_serial")
        self._scan_parallel_pipeline = self._build_pipeline("scan_columns_fixed_slice_parallel")
        self._emit_pipeline = self._build_pipeline("emit_columns_fixed_slice")
        self._destroyed = False

        upload_options = int(getattr(Metal, "MTLResourceStorageModeShared")) | int(getattr(Metal, "MTLResourceCPUCacheModeWriteCombined"))
        meta_options = int(getattr(Metal, "MTLResourceStorageModeShared"))
        vertex_options = int(getattr(Metal, "MTLResourceStorageModeShared"))

        plane = self.sample_size * self.sample_size
        blocks_bytes = self.chunk_capacity * self.storage_height * plane * 4
        coords_bytes = self.chunk_capacity * 4 * 4
        meta_bytes = self.chunk_capacity * 4
        column_count_bytes = self.chunk_capacity * (self.sample_size - 2) * (self.sample_size - 2) * 4
        vertex_pool_bytes = self.chunk_capacity * self.max_vertices_per_chunk * self.vertex_stride

        for slot_id in range(self.inflight_slots):
            self._slots.append(
                MetalMesherSlot(
                    slot_id=slot_id,
                    chunk_capacity=self.chunk_capacity,
                    sample_size=self.sample_size,
                    height_limit=self.height_limit,
                    storage_height=self.storage_height,
                    max_vertices_per_chunk=self.max_vertices_per_chunk,
                    vertex_stride=self.vertex_stride,
                    blocks_buffer=device.newBufferWithLength_options_(max(4, blocks_bytes), upload_options),
                    materials_buffer=device.newBufferWithLength_options_(max(4, blocks_bytes), upload_options),
                    coords_buffer=device.newBufferWithLength_options_(max(16, coords_bytes), upload_options),
                    counts_buffer=device.newBufferWithLength_options_(max(4, meta_bytes), meta_options),
                    overflow_buffer=device.newBufferWithLength_options_(max(4, meta_bytes), meta_options),
                    column_counts_buffer=device.newBufferWithLength_options_(max(4, column_count_bytes), meta_options),
                    column_offsets_buffer=device.newBufferWithLength_options_(max(4, column_count_bytes), meta_options),
                    vertex_pool_buffer=device.newBufferWithLength_options_(max(self.vertex_stride, vertex_pool_bytes), vertex_options),
                )
            )

    def destroy(self) -> None:
        with self._lock:
            if self._destroyed:
                return
            self._destroyed = True
            for slot in self._slots:
                slot.in_flight = False
                slot.blocks_buffer = None
                slot.materials_buffer = None
                slot.coords_buffer = None
                slot.counts_buffer = None
                slot.overflow_buffer = None
                slot.column_counts_buffer = None
                slot.column_offsets_buffer = None
                slot.vertex_pool_buffer = None
            self._slots.clear()
            self.command_queue = None
            self.device = None

    def _build_pipeline(self, entry_point: str):
        src = Path(__file__).parents[1].joinpath("metal_voxel_mesher.metal").read_text(encoding="utf-8")
        library, err = self.device.newLibraryWithSource_options_error_(src, None, None)
        if err is not None or library is None:
            raise RuntimeError(f"Metal library compile failed: {err}")
        fn = library.newFunctionWithName_(entry_point)
        if fn is None:
            raise RuntimeError(f"Metal function not found: {entry_point}")
        pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err is not None or pso is None:
            raise RuntimeError(f"Metal pipeline creation failed for {entry_point}: {err}")
        return pso

    def _acquire_slot(self) -> MetalMesherSlot | None:
        with self._lock:
            if self._destroyed:
                raise RuntimeError("MetalChunkMesher is destroyed")
            for slot in self._slots:
                if not slot.in_flight:
                    slot.in_flight = True
                    return slot
        return None

    def _release_slot(self, slot: MetalMesherSlot | None) -> None:
        if slot is None:
            return
        with self._lock:
            slot.in_flight = False

    @staticmethod
    def _u32_view(buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(max(0, count) * 4), dtype=np.uint32, count=count)

    @staticmethod
    def _i32_view(buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(max(0, count) * 4), dtype=np.int32, count=count)

    def _pack_mesher_params(self, renderer, chunk_count: int) -> bytes:
        return struct.pack(
            "<5If2I",
            int(self.sample_size),
            int(self.height_limit),
            int(chunk_count),
            int(renderer.world.chunk_size),
            int(self.max_vertices_per_chunk),
            float(renderer.world.block_size),
            0,
            0,
        )

    def _pack_expand_params(self, renderer, chunk_count: int) -> bytes:
        return struct.pack(
            "<8I",
            int(self.sample_size),
            int(self.height_limit),
            int(chunk_count),
            int(renderer.world.chunk_size),
            int(renderer.world.height),
            int(renderer.world.seed) & 0xFFFFFFFF,
            0,
            0,
        )

    def _zero_metadata(self, slot: MetalMesherSlot, chunk_count: int) -> None:
        columns_per_side = self.sample_size - 2
        column_count = int(chunk_count) * columns_per_side * columns_per_side
        self._u32_view(slot.counts_buffer, int(chunk_count)).fill(0)
        self._u32_view(slot.overflow_buffer, int(chunk_count)).fill(0)
        self._u32_view(slot.column_counts_buffer, column_count).fill(0)
        self._u32_view(slot.column_offsets_buffer, column_count).fill(0)

    def _write_coords(self, slot: MetalMesherSlot, chunk_coords: list[tuple[int, int, int]]) -> None:
        coords_view = self._i32_view(slot.coords_buffer, len(chunk_coords) * 4).reshape((len(chunk_coords), 4))
        for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunk_coords):
            coords_view[index, 0] = int(chunk_x)
            coords_view[index, 1] = int(chunk_y)
            coords_view[index, 2] = int(chunk_z)
            coords_view[index, 3] = 0

    def _encode_mesh_passes(self, encoder, slot: MetalMesherSlot, chunk_count: int, params_bytes: bytes) -> None:
        columns_per_side = self.sample_size - 2

        count_tew = int(self._count_pipeline.threadExecutionWidth())
        count_max_total = int(self._count_pipeline.maxTotalThreadsPerThreadgroup())
        count_tg_width = max(1, count_tew)
        count_tg_height = max(1, count_max_total // count_tg_width)
        count_threads_per_tg = Metal.MTLSizeMake(count_tg_width, count_tg_height, 1)
        count_grid = Metal.MTLSizeMake(columns_per_side, columns_per_side, chunk_count)

        encoder.setComputePipelineState_(self._count_pipeline)
        encoder.setBuffer_offset_atIndex_(slot.blocks_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(slot.column_counts_buffer, 0, 1)
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(count_grid, count_threads_per_tg)

        columns_per_chunk = columns_per_side * columns_per_side
        scan_parallel_max_total = int(self._scan_parallel_pipeline.maxTotalThreadsPerThreadgroup())
        if columns_per_chunk <= 1024 and scan_parallel_max_total >= 1024:
            scan_threads_per_tg = Metal.MTLSizeMake(1024, 1, 1)
            scan_grid = Metal.MTLSizeMake(1024, 1, chunk_count)
            scan_pipeline = self._scan_parallel_pipeline
        else:
            scan_tew = int(self._scan_serial_pipeline.threadExecutionWidth())
            scan_threads_per_tg = Metal.MTLSizeMake(max(1, scan_tew), 1, 1)
            scan_grid = Metal.MTLSizeMake(chunk_count, 1, 1)
            scan_pipeline = self._scan_serial_pipeline

        encoder.setComputePipelineState_(scan_pipeline)
        encoder.setBuffer_offset_atIndex_(slot.column_counts_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(slot.column_offsets_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(slot.counts_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(slot.overflow_buffer, 0, 3)
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 4)
        encoder.dispatchThreads_threadsPerThreadgroup_(scan_grid, scan_threads_per_tg)

        emit_tew = int(self._emit_pipeline.threadExecutionWidth())
        emit_max_total = int(self._emit_pipeline.maxTotalThreadsPerThreadgroup())
        emit_tg_width = max(1, emit_tew)
        emit_tg_height = max(1, emit_max_total // emit_tg_width)
        emit_threads_per_tg = Metal.MTLSizeMake(emit_tg_width, emit_tg_height, 1)
        emit_grid = Metal.MTLSizeMake(columns_per_side, columns_per_side, chunk_count)

        encoder.setComputePipelineState_(self._emit_pipeline)
        encoder.setBuffer_offset_atIndex_(slot.blocks_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(slot.materials_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(slot.coords_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(slot.column_counts_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(slot.column_offsets_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(slot.overflow_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(slot.vertex_pool_buffer, 0, 6)
        encoder.setBytes_length_atIndex_(params_bytes, len(params_bytes), 7)
        encoder.dispatchThreads_threadsPerThreadgroup_(emit_grid, emit_threads_per_tg)

    @profile
    def submit_chunk_mesh_batch(
        self,
        renderer,
        chunk_results: list[ChunkVoxelResult],
        on_complete: Callable[[list[object]], None] | None = None,
    ) -> AsyncMetalMeshBatchResources | None:
        if not chunk_results:
            return AsyncMetalMeshBatchResources(mesher=self, slot=None, on_complete=on_complete, completed_meshes=[])

        chunk_count = len(chunk_results)
        if chunk_count > self.chunk_capacity:
            raise ValueError(f"chunk batch {chunk_count} exceeds slot capacity {self.chunk_capacity}")

        sample_size = int(chunk_results[0].blocks.shape[1])
        height_limit = int(chunk_results[0].blocks.shape[0])
        if sample_size != self.sample_size or height_limit != self.height_limit:
            raise ValueError(
                f"chunk shape mismatch: mesher expects sample={self.sample_size}, height={self.height_limit}, "
                f"got sample={sample_size}, height={height_limit}"
            )

        slot = self._acquire_slot()
        if slot is None:
            return None

        chunk_coords = [(int(r.chunk_x), int(getattr(r, "chunk_y", 0)), int(r.chunk_z)) for r in chunk_results]
        resources = AsyncMetalMeshBatchResources(
            mesher=self,
            slot=slot,
            chunk_results=list(chunk_results),
            chunk_coords=chunk_coords,
            on_complete=on_complete,
            deliver_to_renderer=on_complete is None,
        )

        try:
            plane = self.sample_size * self.sample_size
            total_voxels = chunk_count * self.storage_height * plane
            blocks_view = np.frombuffer(
                slot.blocks_buffer.contents().as_buffer(total_voxels * 4), dtype=np.uint32, count=total_voxels
            ).reshape((chunk_count, self.storage_height, self.sample_size, self.sample_size))
            materials_view = np.frombuffer(
                slot.materials_buffer.contents().as_buffer(total_voxels * 4), dtype=np.uint32, count=total_voxels
            ).reshape((chunk_count, self.storage_height, self.sample_size, self.sample_size))
            blocks_view.fill(0)
            materials_view.fill(0)
            for i, result in enumerate(chunk_results):
                blocks_view[i, 1 : 1 + self.height_limit] = np.ascontiguousarray(result.blocks, dtype=np.uint32)
                materials_view[i, 1 : 1 + self.height_limit] = np.ascontiguousarray(result.materials, dtype=np.uint32)
                bottom = getattr(result, "bottom_boundary", None)
                top = getattr(result, "top_boundary", None)
                if bottom is not None:
                    blocks_view[i, 0] = np.ascontiguousarray(bottom, dtype=np.uint32)
                if top is not None:
                    blocks_view[i, self.height_limit + 1] = np.ascontiguousarray(top, dtype=np.uint32)

            self._write_coords(slot, chunk_coords)
            self._zero_metadata(slot, chunk_count)
            params_bytes = self._pack_mesher_params(renderer, chunk_count)
            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            self._encode_mesh_passes(encoder, slot, chunk_count, params_bytes)
            encoder.endEncoding()
            command_buffer.commit()
            resources.command_buffer = command_buffer
        except Exception:
            self._release_slot(slot)
            raise
        return resources

    @profile
    def submit_surface_gpu_batch(
        self,
        renderer,
        surface_batch: ChunkSurfaceGpuBatch,
        on_complete: Callable[[list[object]], None] | None = None,
    ) -> AsyncMetalMeshBatchResources | None:
        device_kind = str(getattr(surface_batch, "device_kind", "") or "").strip().lower()
        source = str(getattr(surface_batch, "source", "") or "").strip().lower()
        if device_kind and device_kind != "metal":
            raise TypeError(f"Metal mesher cannot bind {device_kind!r} terrain surface buffers; use ChunkVoxelResult fallback.")
        if not device_kind and "wgpu" in source:
            raise TypeError("Metal mesher cannot bind WGPU terrain surface buffers; use ChunkVoxelResult fallback.")
        chunk_coords = _normalize_chunk_coords(surface_batch.chunks if isinstance(surface_batch.chunks, list) else list(surface_batch.chunks))
        if not chunk_coords:
            release_surface_gpu_batch_immediately(surface_batch)
            return None
        chunk_count = len(chunk_coords)
        if chunk_count > self.chunk_capacity:
            raise ValueError(f"surface batch {chunk_count} exceeds slot capacity {self.chunk_capacity}")

        slot = self._acquire_slot()
        if slot is None:
            return None

        callbacks: list[object] = []
        callback = getattr(surface_batch, "_release_callback", None)
        if callable(callback):
            callbacks.append(callback)
            try:
                setattr(surface_batch, "_release_callback", None)
            except Exception:
                pass

        resources = AsyncMetalMeshBatchResources(
            mesher=self,
            slot=slot,
            chunk_results=[],
            chunk_coords=chunk_coords,
            on_complete=on_complete,
            deliver_to_renderer=on_complete is None,
            surface_release_callbacks=callbacks,
        )
        try:
            self._write_coords(slot, chunk_coords)
            self._zero_metadata(slot, chunk_count)
            expand_params = self._pack_expand_params(renderer, chunk_count)
            mesher_params = self._pack_mesher_params(renderer, chunk_count)
            command_buffer = self.command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()

            encoder.setComputePipelineState_(self._expand_pipeline)
            encoder.setBuffer_offset_atIndex_(surface_batch.heights_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(surface_batch.materials_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(slot.blocks_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(slot.materials_buffer, 0, 3)
            encoder.setBytes_length_atIndex_(expand_params, len(expand_params), 4)
            encoder.setBuffer_offset_atIndex_(slot.coords_buffer, 0, 5)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                Metal.MTLSizeMake(self.sample_size, self.sample_size, chunk_count * self.storage_height),
                Metal.MTLSizeMake(8, 8, 1),
            )

            self._encode_mesh_passes(encoder, slot, chunk_count, mesher_params)
            encoder.endEncoding()
            command_buffer.commit()
            resources.command_buffer = command_buffer
        except Exception:
            self._release_slot(slot)
            resources.surface_release_callbacks = callbacks
            for callback in list(callbacks):
                if callable(callback):
                    try:
                        callback()
                    except Exception:
                        pass
            resources.slot = None
            resources.mesher = None
            resources.command_buffer = None
            raise
        return resources




__all__ = ["MetalChunkMesher"]
