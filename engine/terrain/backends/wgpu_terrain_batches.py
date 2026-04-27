from __future__ import annotations

from collections import deque

import numpy as np

from ..types import ChunkSurfaceGpuBatch, ChunkSurfaceResult
from .wgpu_terrain_common import (
    _ChunkGpuBatch,
    _LeasedChunkSurfaceGpuBatch,
    _PendingSurfaceReadback,
    profile,
    wgpu,
)


class WgpuTerrainBatchMixin:
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

    def has_pending_chunk_surface_batches(self) -> bool:
        return bool(self._in_flight_batches) or bool(self._pending_surface_readbacks) or bool(self._pending_jobs)

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
