from __future__ import annotations

"""
This module is the dedicated WGPU terrain surface backend.

It generates surface height/material grids in compute batches. In stacked
chunk mode those GPU surface buffers are leased directly to the WGPU mesher,
which expands them into local voxel chunks before count/scan/emit meshing.
"""

import struct
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..types import ChunkSurfaceGpuBatch, ChunkSurfaceResult, ChunkVoxelResult
from ...terrain.kernels import (
    expand_chunk_surface_to_voxel_grid,
    fill_chunk_surface_grids as cpu_fill_chunk_surface_grids,
    fill_chunk_voxel_grid as cpu_fill_chunk_voxel_grid,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
    surface_profile_at as cpu_surface_profile_at,
)
from ...world_constants import VERTICAL_CHUNK_STACK_ENABLED
from ...shader_loader import load_shader_text

try:
    import wgpu
except Exception:  # pragma: no cover - optional during CPU-only fallback
    wgpu = None  # type: ignore[assignment]

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


GPU_TERRAIN_SHADER = load_shader_text("terrain_surface.wgsl")


@dataclass
class _ChunkGpuBatch:
    chunks: list[tuple[int, int, int]]
    chunk_count: int
    max_chunks: int
    coords_array: np.ndarray
    coords_buffer: object
    params_buffer: object
    heights_buffer: object
    materials_buffer: object
    readback_buffer: object
    bind_group: object


@dataclass
class _PendingSurfaceReadback:
    batch: _ChunkGpuBatch
    total_cells: int
    total_bytes: int
    map_promise: object | None = None


class _LeasedChunkSurfaceGpuBatch(ChunkSurfaceGpuBatch):
    pass


def _normalize_chunk_coord(coord) -> tuple[int, int, int]:
    if len(coord) >= 3:
        return int(coord[0]), int(coord[1]), int(coord[2])
    if len(coord) == 2:
        return int(coord[0]), 0, int(coord[1])
    raise ValueError(f"Invalid chunk coordinate: {coord!r}")


def _normalize_chunk_coords(chunks) -> list[tuple[int, int, int]]:
    return [_normalize_chunk_coord(chunk) for chunk in chunks]


class WgpuTerrainBackend:
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

    @profile
    def chunk_voxel_grid(self, chunk_x: int, chunk_y: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        if VERTICAL_CHUNK_STACK_ENABLED:
            local_height = self.chunk_size
            blocks = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint8)
            voxel_materials = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint32)
            top_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            bottom_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            heights, materials = self.chunk_surface_grids(int(chunk_x), int(chunk_z))
            fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
                blocks,
                voxel_materials,
                top_boundary,
                bottom_boundary,
                heights,
                materials,
                int(chunk_x),
                int(chunk_y),
                int(chunk_z),
                self.chunk_size,
                self.seed,
                self.height_limit,
            )
            return blocks, voxel_materials
        blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
        voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
        cpu_fill_chunk_voxel_grid(blocks, voxel_materials, int(chunk_x), int(chunk_z), self.chunk_size, self.seed, self.height_limit)
        return blocks, voxel_materials

    @profile
    def _allocate_chunk_batch_resources(self, max_chunks: int) -> "_ChunkGpuBatch":
        max_chunks = max(1, int(max_chunks))
        coords_array = np.empty((max_chunks, 4), dtype=np.int32)
        coords_buffer = self.device.create_buffer(
            size=max(1, coords_array.nbytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        params_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        heights_buffer = self.device.create_buffer(
            size=max(1, max_chunks * self.cell_count * 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        materials_buffer = self.device.create_buffer(
            size=max(1, max_chunks * self.cell_count * 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        readback_buffer = self.device.create_buffer(
            size=max(1, max_chunks * self.cell_count * 8),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        bind_group = self.device.create_bind_group(
            layout=self._batch_bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": heights_buffer}},
                {"binding": 1, "resource": {"buffer": materials_buffer}},
                {"binding": 2, "resource": {"buffer": coords_buffer}},
                {"binding": 3, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
            ],
        )
        return _ChunkGpuBatch(
            chunks=[],
            chunk_count=0,
            max_chunks=max_chunks,
            coords_array=coords_array,
            coords_buffer=coords_buffer,
            params_buffer=params_buffer,
            heights_buffer=heights_buffer,
            materials_buffer=materials_buffer,
            readback_buffer=readback_buffer,
            bind_group=bind_group,
        )

    def _reclaim_batch_slots(self) -> None:
        while self._batch_slots_pending_reuse:
            batch = self._batch_slots_pending_reuse.popleft()
            batch.chunk_count = 0
            batch.chunks.clear()
            self._ensure_batch_readback_buffer(batch)
            self._available_batch_slots.append(batch)

    def _release_gpu_surface_batch(self, lease_id: int) -> None:
        batch = self._leased_surface_batches.pop(int(lease_id), None)
        if batch is None:
            return
        self._batch_slots_pending_reuse.append(batch)

    def _ensure_batch_readback_buffer(self, batch: "_ChunkGpuBatch") -> None:
        readback_buffer = getattr(batch, "readback_buffer", None)
        if readback_buffer is not None:
            return
        batch.readback_buffer = self.device.create_buffer(
            size=max(1, int(batch.max_chunks) * self.cell_count * 8),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    def _destroy_batch_readback_buffer(self, batch: "_ChunkGpuBatch") -> None:
        readback_buffer = getattr(batch, "readback_buffer", None)
        if readback_buffer is None:
            return
        batch.readback_buffer = None
        try:
            if getattr(readback_buffer, "map_state", "unmapped") != "unmapped":
                readback_buffer.unmap()
        except Exception:
            pass
        try:
            readback_buffer.destroy()
        except Exception:
            pass

    @profile
    def _create_chunk_batch(self, chunks: list[tuple[int, int, int]]) -> "_ChunkGpuBatch":
        chunk_count = len(chunks)
        target_capacity = max(int(self._submit_target_chunks), chunk_count)

        if self._available_batch_slots:
            batch = self._available_batch_slots.popleft()
            if chunk_count > batch.max_chunks:
                self._available_batch_slots.appendleft(batch)
                batch = self._allocate_chunk_batch_resources(target_capacity)
        else:
            batch = self._allocate_chunk_batch_resources(target_capacity)

        batch.chunk_count = chunk_count
        batch.chunks = list(chunks)
        if chunk_count > 0:
            coords_view = batch.coords_array[:chunk_count]
            for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunks):
                coords_view[index, 0] = int(chunk_x)
                coords_view[index, 1] = int(chunk_y)
                coords_view[index, 2] = int(chunk_z)
                coords_view[index, 3] = 0
        return batch

    @profile
    def _submit_next_batch(self) -> None:
        if not self._pending_jobs:
            return

        available_slots = max(0, self._max_in_flight_batches - len(self._in_flight_batches))
        if available_slots <= 0:
            return

        target_chunks = self._submit_target_chunks

        submitted: list[_ChunkGpuBatch] = []
        while self._pending_jobs and available_slots > 0:
            merged: list[tuple[int, int, int]] = []
            while self._pending_jobs and len(merged) < target_chunks:
                job = self._pending_jobs.popleft()
                take = min(target_chunks - len(merged), len(job))
                if take:
                    merged.extend(job[:take])
                if take < len(job):
                    self._pending_jobs.appendleft(job[take:])
                    break

            if not merged:
                break

            submitted.append(self._create_chunk_batch(merged))
            available_slots -= 1

        if not submitted:
            return

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._batch_pipeline)
        workgroups = (self.sample_size + 7) // 8

        for tasks in submitted:
            coords_view = memoryview(tasks.coords_array[:tasks.chunk_count])
            self.device.queue.write_buffer(tasks.coords_buffer, 0, coords_view)
            self.device.queue.write_buffer(tasks.params_buffer, 0, self._batch_params_payload)
            compute_pass.set_bind_group(0, tasks.bind_group)
            compute_pass.dispatch_workgroups(workgroups, workgroups, tasks.chunk_count)

        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._in_flight_batches.extend(submitted)

    @profile
    def _enqueue_surface_readback(self, batch: _ChunkGpuBatch) -> None:
        chunk_count = int(batch.chunk_count)
        total_cells = chunk_count * self.cell_count
        total_bytes = total_cells * 4
        if total_bytes <= 0:
            self._batch_slots_pending_reuse.append(batch)
            return

        self._ensure_batch_readback_buffer(batch)
        if batch.readback_buffer.map_state != "unmapped":
            batch.readback_buffer.unmap()

        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(batch.heights_buffer, 0, batch.readback_buffer, 0, total_bytes)
        encoder.copy_buffer_to_buffer(batch.materials_buffer, 0, batch.readback_buffer, total_bytes, total_bytes)
        self.device.queue.submit([encoder.finish()])

        map_promise = batch.readback_buffer.map_async(wgpu.MapMode.READ, 0, total_bytes * 2)
        self._pending_surface_readbacks.append(
            _PendingSurfaceReadback(
                batch=batch,
                total_cells=total_cells,
                total_bytes=total_bytes,
                map_promise=map_promise,
            )
        )

    @profile
    def _drain_ready_surface_readbacks(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        if not self._pending_surface_readbacks:
            return ready

        still_pending: deque[_PendingSurfaceReadback] = deque()
        while self._pending_surface_readbacks:
            pending = self._pending_surface_readbacks.popleft()
            batch = pending.batch
            if batch.readback_buffer.map_state != "mapped":
                still_pending.append(pending)
                continue

            try:
                metadata_view = batch.readback_buffer.read_mapped(0, pending.total_bytes * 2, copy=False)
                heights_data = np.frombuffer(metadata_view, dtype=np.uint32, count=pending.total_cells).copy()
                materials_data = np.frombuffer(
                    metadata_view,
                    dtype=np.uint32,
                    count=pending.total_cells,
                    offset=pending.total_bytes,
                ).copy()
                cell_count = self.cell_count
                for index, (chunk_x, chunk_y, chunk_z) in enumerate(batch.chunks):
                    start = index * cell_count
                    end = start + cell_count
                    ready.append(
                        ChunkSurfaceResult(
                            chunk_x=chunk_x,
                            chunk_y=chunk_y,
                            chunk_z=chunk_z,
                            heights=heights_data[start:end],
                            materials=materials_data[start:end],
                            source="gpu",
                        )
                    )
            finally:
                self._destroy_batch_readback_buffer(batch)
                self._batch_slots_pending_reuse.append(batch)

        self._pending_surface_readbacks = still_pending
        return ready

    @profile
    def _drain_ready_surface_gpu_readbacks(self) -> list[ChunkSurfaceGpuBatch]:
        ready: list[ChunkSurfaceGpuBatch] = []
        if not self._pending_surface_readbacks:
            return ready

        still_pending: deque[_PendingSurfaceReadback] = deque()
        while self._pending_surface_readbacks:
            pending = self._pending_surface_readbacks.popleft()
            batch = pending.batch
            if batch.readback_buffer.map_state != "mapped":
                still_pending.append(pending)
                continue

            lease_id = id(batch)
            try:
                if batch.readback_buffer.map_state != "unmapped":
                    batch.readback_buffer.unmap()
                self._destroy_batch_readback_buffer(batch)
                self._leased_surface_batches[lease_id] = batch
                surface_batch = _LeasedChunkSurfaceGpuBatch(
                    chunks=list(batch.chunks),
                    heights_buffer=batch.heights_buffer,
                    materials_buffer=batch.materials_buffer,
                    cell_count=self.cell_count,
                    source="wgpu_gpu_leased",
                    device_kind="wgpu",
                )

                def _release(
                    lease_id: int = lease_id,
                    backend: "WgpuTerrainBackend" = self,
                ) -> None:
                    backend._release_gpu_surface_batch(lease_id)

                setattr(surface_batch, "_release_callback", _release)
                ready.append(surface_batch)
            except Exception:
                self._leased_surface_batches.pop(lease_id, None)
                self._destroy_batch_readback_buffer(batch)
                self._batch_slots_pending_reuse.append(batch)
                raise

        self._pending_surface_readbacks = still_pending
        return ready

    @profile
    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        self._reclaim_batch_slots()

        while self._in_flight_batches:
            completed_batch = self._in_flight_batches.popleft()
            self._enqueue_surface_readback(completed_batch)

        ready.extend(self._drain_ready_surface_readbacks())
        self._submit_next_batch()
        return ready

    @profile
    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        ready: list[ChunkSurfaceGpuBatch] = []
        self._reclaim_batch_slots()
        while self._in_flight_batches:
            completed_batch = self._in_flight_batches.popleft()
            self._enqueue_surface_readback(completed_batch)

        ready.extend(self._drain_ready_surface_gpu_readbacks())
        self._submit_next_batch()
        return ready

    @profile
    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        return self.request_chunk_surface_batch(chunks)

    @profile
    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        for surface_result in self.poll_ready_chunk_surface_batches():
            chunk_x = int(surface_result.chunk_x)
            chunk_y = int(getattr(surface_result, "chunk_y", 0))
            chunk_z = int(surface_result.chunk_z)
            top_boundary = None
            bottom_boundary = None
            is_empty_chunk = False
            if VERTICAL_CHUNK_STACK_ENABLED:
                local_height = self.chunk_size
                origin_y = chunk_y * self.chunk_size
                blocks = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint8)
                voxel_materials = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint32)
                top_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
                bottom_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
                is_empty_chunk = int(np.max(surface_result.heights)) <= origin_y
                if not is_empty_chunk:
                    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
                        blocks,
                        voxel_materials,
                        top_boundary,
                        bottom_boundary,
                        surface_result.heights,
                        surface_result.materials,
                        chunk_x,
                        chunk_y,
                        chunk_z,
                        self.chunk_size,
                        self.seed,
                        self.height_limit,
                    )
            else:
                blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
                voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
                expand_chunk_surface_to_voxel_grid(
                    blocks,
                    voxel_materials,
                    surface_result.heights,
                    surface_result.materials,
                    self.chunk_size,
                    self.height_limit,
                )
            ready.append(
                ChunkVoxelResult(
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    chunk_z=chunk_z,
                    blocks=blocks,
                    materials=voxel_materials,
                    source=surface_result.source,
                    top_boundary=top_boundary,
                    bottom_boundary=bottom_boundary,
                    is_empty=is_empty_chunk,
                )
            )
        return ready

    def has_pending_chunk_surface_batches(self) -> bool:
        return bool(self._in_flight_batches) or bool(self._pending_surface_readbacks) or bool(self._pending_jobs)

    def has_pending_chunk_voxel_batches(self) -> bool:
        return self.has_pending_chunk_surface_batches()

    @profile
    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        while self._in_flight_batches or self._pending_jobs:
            ready.extend(self.poll_ready_chunk_surface_batches())
        while self._pending_surface_readbacks:
            pending = self._pending_surface_readbacks[0]
            if pending.map_promise is not None and hasattr(pending.map_promise, "sync_wait"):
                pending.map_promise.sync_wait()
            ready.extend(self.poll_ready_chunk_surface_batches())
        return ready

    @profile
    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        for surface_result in self.flush_chunk_surface_batches():
            chunk_x = int(surface_result.chunk_x)
            chunk_y = int(getattr(surface_result, "chunk_y", 0))
            chunk_z = int(surface_result.chunk_z)
            top_boundary = None
            bottom_boundary = None
            is_empty_chunk = False
            if VERTICAL_CHUNK_STACK_ENABLED:
                local_height = self.chunk_size
                origin_y = chunk_y * self.chunk_size
                blocks = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint8)
                voxel_materials = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint32)
                top_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
                bottom_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
                is_empty_chunk = int(np.max(surface_result.heights)) <= origin_y
                if not is_empty_chunk:
                    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
                        blocks,
                        voxel_materials,
                        top_boundary,
                        bottom_boundary,
                        surface_result.heights,
                        surface_result.materials,
                        chunk_x,
                        chunk_y,
                        chunk_z,
                        self.chunk_size,
                        self.seed,
                        self.height_limit,
                    )
            else:
                blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
                voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
                expand_chunk_surface_to_voxel_grid(
                    blocks,
                    voxel_materials,
                    surface_result.heights,
                    surface_result.materials,
                    self.chunk_size,
                    self.height_limit,
                )
            ready.append(
                ChunkVoxelResult(
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    chunk_z=chunk_z,
                    blocks=blocks,
                    materials=voxel_materials,
                    source=surface_result.source,
                    top_boundary=top_boundary,
                    bottom_boundary=bottom_boundary,
                    is_empty=is_empty_chunk,
                )
            )
        return ready

    def terrain_backend_label(self) -> str:
        return "Wgpu"

    def destroy(self) -> None:
        buffers: list[object] = [
            self._single_heights_buffer,
            self._single_materials_buffer,
            self._single_readback_buffer,
            self._grid_heights_buffer,
            self._grid_materials_buffer,
            self._grid_readback_buffer,
            self._params_buffer,
        ]

        while self._pending_surface_readbacks:
            pending = self._pending_surface_readbacks.popleft()
            batch = pending.batch
            self._destroy_batch_readback_buffer(batch)
        while self._in_flight_batches:
            self._batch_slots_pending_reuse.append(self._in_flight_batches.popleft())
        while self._available_batch_slots:
            batch = self._available_batch_slots.popleft()
            buffers.extend([
                batch.coords_buffer,
                batch.params_buffer,
                batch.heights_buffer,
                batch.materials_buffer,
            ])
            self._destroy_batch_readback_buffer(batch)
        while self._batch_slots_pending_reuse:
            batch = self._batch_slots_pending_reuse.popleft()
            buffers.extend([
                batch.coords_buffer,
                batch.params_buffer,
                batch.heights_buffer,
                batch.materials_buffer,
            ])
            self._destroy_batch_readback_buffer(batch)
        for batch in list(self._leased_surface_batches.values()):
            buffers.extend([
                batch.coords_buffer,
                batch.params_buffer,
                batch.heights_buffer,
                batch.materials_buffer,
            ])
            self._destroy_batch_readback_buffer(batch)
        self._leased_surface_batches.clear()
        self._pending_jobs.clear()

        for buffer in buffers:
            if buffer is None:
                continue
            try:
                if getattr(buffer, "map_state", "unmapped") != "unmapped":
                    buffer.unmap()
            except Exception:
                pass
            try:
                buffer.destroy()
            except Exception:
                pass
