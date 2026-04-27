from __future__ import annotations

"""WGPU terrain surface backend public entry point.

The implementation is split by responsibility:

- :mod:`wgpu_terrain_common` owns shared dataclasses, optional imports, and shader loading.
- :mod:`wgpu_terrain_batches` owns async surface-batch scheduling, readbacks, leases, and cleanup.
- :mod:`wgpu_terrain_voxels` owns surface-result to voxel-result conversion.

The public ``WgpuTerrainBackend`` import path is intentionally preserved.
"""

import struct
from collections import deque

import numpy as np

from .wgpu_terrain_batches import WgpuTerrainBatchMixin
from .wgpu_terrain_common import (
    GPU_TERRAIN_SHADER,
    _ChunkGpuBatch,
    _PendingSurfaceReadback,
    _normalize_chunk_coords,
    profile,
    wgpu,
)
from .wgpu_terrain_voxels import WgpuTerrainVoxelMixin


class WgpuTerrainBackend(WgpuTerrainVoxelMixin, WgpuTerrainBatchMixin):
    def __init__(self, device, seed: int, chunk_size: int, height_limit: int, chunks_per_poll: int = 128) -> None:
        if wgpu is None:
            raise RuntimeError("wgpu is unavailable.")
        self.device = device
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.height_limit = int(height_limit)
        self.chunks_per_poll = max(1, int(chunks_per_poll))
        self.sample_size = self.chunk_size + 2
        self.cell_count = self.sample_size * self.sample_size
        self._pending_jobs: deque[list[tuple[int, int, int]]] = deque()
        self._in_flight_batches: deque[_ChunkGpuBatch] = deque()
        self._next_job_id = 1
        self._params_buffer = device.create_buffer(
            size=64,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._single_heights_buffer = device.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._single_materials_buffer = device.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._single_readback_buffer = device.create_buffer(
            size=8,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        self._grid_heights_buffer = device.create_buffer(
            size=self.cell_count * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._grid_materials_buffer = device.create_buffer(
            size=self.cell_count * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._grid_readback_buffer = device.create_buffer(
            size=max(1, self.cell_count * 8),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        self._batch_params_payload = struct.pack(
            "<4I",
            int(self.sample_size),
            int(self.chunk_size),
            int(self.height_limit),
            int(self.seed) & 0xFFFFFFFF,
        )
        self._submit_target_chunks = max(int(self.chunks_per_poll), min(int(self.chunks_per_poll) * 2, 128))
        self._batch_pool_size = 8
        self._max_in_flight_batches = 3
        self._available_batch_slots: deque[_ChunkGpuBatch] = deque()
        self._batch_slots_pending_reuse: deque[_ChunkGpuBatch] = deque()
        self._pending_surface_readbacks: deque[_PendingSurfaceReadback] = deque()
        self._leased_surface_batches: dict[int, _ChunkGpuBatch] = {}
        self._bind_group_layout = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            ]
        )
        self._batch_bind_group_layout = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            ]
        )
        shader_module = device.create_shader_module(code=GPU_TERRAIN_SHADER)
        self._single_bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._single_heights_buffer}},
                {"binding": 1, "resource": {"buffer": self._single_materials_buffer}},
                {"binding": 2, "resource": {"buffer": self._params_buffer, "offset": 0, "size": 64}},
            ],
        )
        self._batch_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._batch_bind_group_layout]),
            compute={"module": shader_module, "entry_point": "fill_chunk_surface_batch_main"},
        )
        self._grid_bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._grid_heights_buffer}},
                {"binding": 1, "resource": {"buffer": self._grid_materials_buffer}},
                {"binding": 2, "resource": {"buffer": self._params_buffer, "offset": 0, "size": 64}},
            ],
        )
        self._single_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bind_group_layout]),
            compute={"module": shader_module, "entry_point": "sample_surface_profile_at_main"},
        )
        self._grid_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bind_group_layout]),
            compute={"module": shader_module, "entry_point": "fill_chunk_surface_grids_main"},
        )
        for _ in range(self._batch_pool_size):
            self._available_batch_slots.append(
                self._allocate_chunk_batch_resources(self._submit_target_chunks)
            )

    def _write_params(self, *, sample_origin_x: float, sample_origin_z: float, height_limit: int, chunk_x: int, chunk_z: int, sample_size: int, seed: int) -> None:
        payload = struct.pack(
            "<4f4i4I",
            float(sample_origin_x),
            float(sample_origin_z),
            0.0,
            float(height_limit),
            int(chunk_x),
            int(chunk_z),
            int(sample_size),
            int(self.chunk_size),
            int(seed) & 0xFFFFFFFF,
            0,
            0,
            0,
        )
        self.device.queue.write_buffer(self._params_buffer, 0, payload)

    def _readback_u32_copy(self, buffer, total_bytes: int) -> np.ndarray:
        if buffer.map_state != "unmapped":
            buffer.unmap()
        buffer.map_sync(wgpu.MapMode.READ, 0, total_bytes)
        try:
            mapped = buffer.read_mapped(0, total_bytes, copy=False)
            return np.frombuffer(mapped, dtype=np.uint32, count=total_bytes // 4).copy()
        finally:
            if buffer.map_state != "unmapped":
                buffer.unmap()

    @profile
    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        self._write_params(
            sample_origin_x=float(x),
            sample_origin_z=float(z),
            height_limit=self.height_limit,
            chunk_x=0,
            chunk_z=0,
            sample_size=1,
            seed=self.seed,
        )
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._single_pipeline)
        compute_pass.set_bind_group(0, self._single_bind_group)
        compute_pass.dispatch_workgroups(1, 1, 1)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(self._single_heights_buffer, 0, self._single_readback_buffer, 0, 4)
        encoder.copy_buffer_to_buffer(self._single_materials_buffer, 0, self._single_readback_buffer, 4, 4)
        self.device.queue.submit([encoder.finish()])
        values = self._readback_u32_copy(self._single_readback_buffer, 8)
        return int(values[0]), int(values[1])

    @profile
    def fill_chunk_surface_grids(self, heights: np.ndarray, materials: np.ndarray, chunk_x: int, chunk_z: int) -> None:
        self._write_params(
            sample_origin_x=0.0,
            sample_origin_z=0.0,
            height_limit=self.height_limit,
            chunk_x=int(chunk_x),
            chunk_z=int(chunk_z),
            sample_size=self.sample_size,
            seed=self.seed,
        )
        total_bytes = self.cell_count * 4
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._grid_pipeline)
        compute_pass.set_bind_group(0, self._grid_bind_group)
        workgroups = (self.sample_size + 7) // 8
        compute_pass.dispatch_workgroups(workgroups, workgroups, 1)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(self._grid_heights_buffer, 0, self._grid_readback_buffer, 0, total_bytes)
        encoder.copy_buffer_to_buffer(self._grid_materials_buffer, 0, self._grid_readback_buffer, total_bytes, total_bytes)
        self.device.queue.submit([encoder.finish()])
        values = self._readback_u32_copy(self._grid_readback_buffer, total_bytes * 2)
        np.copyto(heights.reshape(-1), values[: self.cell_count])
        np.copyto(materials.reshape(-1), values[self.cell_count : self.cell_count * 2])

    @profile
    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        heights = np.empty(self.cell_count, dtype=np.uint32)
        materials = np.empty(self.cell_count, dtype=np.uint32)
        self.fill_chunk_surface_grids(heights, materials, chunk_x, chunk_z)
        return heights, materials

    @profile
    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        if chunks:
            # Push fresh requests to the front so newly-facing terrain wins over stale backlog.
            self._pending_jobs.appendleft(_normalize_chunk_coords(chunks))
            self._submit_next_batch()
        return job_id

    def terrain_backend_label(self) -> str:
        return "Wgpu"
