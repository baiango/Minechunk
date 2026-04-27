from __future__ import annotations

"""Asynchronous Metal terrain surface-batch scheduling and leasing."""

import numpy as np

from ..types import ChunkSurfaceGpuBatch, ChunkSurfaceResult
from .metal_terrain_common import (
    Metal,
    _ChunkMetalBatch,
    _LeasedChunkSurfaceGpuBatch,
    _normalize_chunk_coords,
    profile,
)


class MetalTerrainBatchOps:
    @profile
    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        if chunks:
            self._pending_jobs.appendleft(_normalize_chunk_coords(chunks))
            self._submit_next_batch()
        return job_id

    @profile
    def _allocate_chunk_batch_resources(self, max_chunks: int) -> _ChunkMetalBatch:
        max_chunks = max(1, int(max_chunks))
        coords_array = np.empty((max_chunks, 4), dtype=np.int32)
        return _ChunkMetalBatch(
            [],
            0,
            max_chunks,
            coords_array,
            self._make_buffer(coords_array.nbytes),
            self._make_buffer(16),
            self._make_buffer(max_chunks * self.cell_count * 4),
            self._make_buffer(max_chunks * self.cell_count * 4),
            None,
        )

    def _reclaim_batch_slots(self) -> None:
        while self._batch_slots_pending_reuse:
            batch = self._batch_slots_pending_reuse.popleft()
            batch.chunk_count = 0
            batch.chunks.clear()
            batch.command_buffer = None
            self._available_batch_slots.append(batch)

    def _release_gpu_surface_batch(self, lease_id: int) -> None:
        batch = self._leased_surface_batches.pop(int(lease_id), None)
        if batch is not None:
            self._batch_slots_pending_reuse.append(batch)

    @staticmethod
    def _batch_completed(batch: _ChunkMetalBatch) -> bool:
        cb = batch.command_buffer
        if cb is None:
            return True
        status = int(cb.status())
        if status == int(getattr(Metal, "MTLCommandBufferStatusCompleted", 4)):
            return True
        if status == int(getattr(Metal, "MTLCommandBufferStatusError", 5)):
            raise RuntimeError(f"Metal terrain command buffer failed: {cb.error()}")
        return False

    @profile
    def _create_chunk_batch(self, chunks: list[tuple[int, int, int]]) -> _ChunkMetalBatch:
        chunk_count = len(chunks)
        if self._available_batch_slots:
            batch = self._available_batch_slots.popleft()
            if chunk_count > batch.max_chunks:
                self._available_batch_slots.appendleft(batch)
                batch = self._allocate_chunk_batch_resources(max(self._submit_target_chunks, chunk_count))
        else:
            batch = self._allocate_chunk_batch_resources(max(self._submit_target_chunks, chunk_count))
        batch.chunk_count = chunk_count
        batch.chunks = list(chunks)
        for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunks):
            batch.coords_array[index] = (int(chunk_x), int(chunk_y), int(chunk_z), 0)
        return batch

    @profile
    def _submit_next_batch(self) -> None:
        if not self._pending_jobs:
            return
        available_slots = max(0, self._max_in_flight_batches - len(self._in_flight_batches))
        if available_slots <= 0:
            return

        submitted: list[_ChunkMetalBatch] = []
        while self._pending_jobs and available_slots > 0:
            merged: list[tuple[int, int, int]] = []
            while self._pending_jobs and len(merged) < self._submit_target_chunks:
                job = self._pending_jobs.popleft()
                take = min(self._submit_target_chunks - len(merged), len(job))
                merged.extend(job[:take])
                if take < len(job):
                    self._pending_jobs.appendleft(job[take:])
                    break
            if merged:
                submitted.append(self._create_chunk_batch(merged))
                available_slots -= 1

        for batch in submitted:
            self._write_buffer_array(batch.coords_buffer, batch.coords_array[:batch.chunk_count])
            self._write_buffer_bytes(batch.params_buffer, self._batch_params_payload)
            batch.command_buffer = self._dispatch(
                self._batch_pipeline,
                [
                    (batch.heights_buffer, 0),
                    (batch.materials_buffer, 1),
                    (batch.coords_buffer, 2),
                    (batch.params_buffer, 3),
                ],
                (self.sample_size, self.sample_size, batch.chunk_count),
                (8, 8, 1),
            )
            self._in_flight_batches.append(batch)

    def _wait_for_batch(self, batch: _ChunkMetalBatch) -> None:
        if batch.command_buffer is not None:
            batch.command_buffer.waitUntilCompleted()
            if not self._batch_completed(batch):
                raise RuntimeError(f"Metal terrain command buffer did not complete: {batch.command_buffer.status()}")

    @profile
    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        self._reclaim_batch_slots()
        while self._in_flight_batches:
            batch = self._in_flight_batches[0]
            if not self._batch_completed(batch):
                break
            batch = self._in_flight_batches.popleft()
            total_cells = batch.chunk_count * self.cell_count
            heights_data = self._buffer_uint32_view(batch.heights_buffer, total_cells).copy()
            materials_data = self._buffer_uint32_view(batch.materials_buffer, total_cells).copy()
            for index, (chunk_x, chunk_y, chunk_z) in enumerate(batch.chunks):
                start = index * self.cell_count
                end = start + self.cell_count
                ready.append(
                    ChunkSurfaceResult(
                        chunk_x=chunk_x,
                        chunk_y=chunk_y,
                        chunk_z=chunk_z,
                        heights=heights_data[start:end],
                        materials=materials_data[start:end],
                        source="metal_gpu",
                    )
                )
            self._batch_slots_pending_reuse.append(batch)
        self._submit_next_batch()
        return ready

    @profile
    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        ready: list[ChunkSurfaceGpuBatch] = []
        self._reclaim_batch_slots()
        while self._in_flight_batches:
            batch = self._in_flight_batches[0]
            if not self._batch_completed(batch):
                break
            batch = self._in_flight_batches.popleft()
            lease_id = id(batch)
            self._leased_surface_batches[lease_id] = batch

            def _release(lease_id: int = lease_id, backend=self) -> None:
                backend._release_gpu_surface_batch(lease_id)

            surface_batch = _LeasedChunkSurfaceGpuBatch(
                chunks=list(batch.chunks),
                heights_buffer=batch.heights_buffer,
                materials_buffer=batch.materials_buffer,
                cell_count=self.cell_count,
                source="metal_gpu_leased",
                device_kind="metal",
            )
            setattr(surface_batch, "_release_callback", _release)
            ready.append(surface_batch)
        self._submit_next_batch()
        return ready

    def has_pending_chunk_surface_batches(self) -> bool:
        return bool(self._in_flight_batches) or bool(self._pending_jobs) or bool(self._leased_surface_batches)

    @profile
    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        while self._pending_jobs or self._in_flight_batches:
            if self._in_flight_batches:
                self._wait_for_batch(self._in_flight_batches[0])
            ready.extend(self.poll_ready_chunk_surface_batches())
        return ready

    def destroy(self) -> None:
        while self._in_flight_batches:
            batch = self._in_flight_batches.popleft()
            try:
                self._wait_for_batch(batch)
            except Exception:
                pass
        self._pending_jobs.clear()
        self._leased_surface_batches.clear()
        self._available_batch_slots.clear()
        self._batch_slots_pending_reuse.clear()
        self.command_queue = None
        self.device = None
