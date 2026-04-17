from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .terrain_backend import ChunkSurfaceGpuBatch, ChunkSurfaceResult, ChunkVoxelResult
from .terrain_kernels import (
    fill_chunk_surface_grids,
    fill_stacked_chunk_voxel_grid,
    fill_chunk_voxel_grid,
    surface_profile_at as sample_surface_profile_at,
)
from .world_constants import CHUNK_SIZE as DEFAULT_CHUNK_SIZE, VERTICAL_CHUNK_STACK_ENABLED


@dataclass
class _PendingChunkJob:
    job_id: int
    chunk_coords: list[tuple[int, int, int]]
    cursor: int = 0


class CpuTerrainBackend:
    def __init__(
        self,
        seed: int = 1337,
        height: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunks_per_poll: int = 128,
    ) -> None:
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.height = self.chunk_size if height is None else int(height)
        self.chunks_per_poll = max(1, int(chunks_per_poll))
        self._pending_jobs: deque[_PendingChunkJob] = deque()
        self._pending_voxel_jobs: deque[_PendingChunkJob] = deque()
        self._next_job_id = 1
        self._next_voxel_job_id = 1

    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        height, material = sample_surface_profile_at(float(x), float(z), self.seed, self.height)
        return int(height), int(material)

    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        sample_size = self.chunk_size + 2
        cell_count = sample_size * sample_size
        heights = np.empty(cell_count, dtype=np.uint32)
        materials = np.empty(cell_count, dtype=np.uint32)
        fill_chunk_surface_grids(heights, materials, chunk_x, chunk_z, self.chunk_size, self.seed, self.height)
        return heights, materials

    def chunk_voxel_grid(self, chunk_x: int, chunk_y: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        sample_size = self.chunk_size + 2
        if VERTICAL_CHUNK_STACK_ENABLED:
            local_height = self.chunk_size
            blocks = np.zeros((local_height, sample_size, sample_size), dtype=np.uint8)
            materials = np.zeros((local_height, sample_size, sample_size), dtype=np.uint32)
            fill_stacked_chunk_voxel_grid(
                blocks,
                materials,
                int(chunk_x),
                int(chunk_y),
                int(chunk_z),
                self.chunk_size,
                self.seed,
                self.height,
            )
            return blocks, materials
        blocks = np.zeros((self.height, sample_size, sample_size), dtype=np.uint8)
        materials = np.zeros((self.height, sample_size, sample_size), dtype=np.uint32)
        fill_chunk_voxel_grid(blocks, materials, chunk_x, chunk_z, self.chunk_size, self.seed, self.height)
        return blocks, materials

    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        if chunks:
            self._pending_jobs.appendleft(_PendingChunkJob(job_id=job_id, chunk_coords=[(int(x), int(y), int(z)) for x, y, z in chunks]))
        return job_id

    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        job_id = self._next_voxel_job_id
        self._next_voxel_job_id += 1
        if chunks:
            self._pending_voxel_jobs.appendleft(_PendingChunkJob(job_id=job_id, chunk_coords=[(int(x), int(y), int(z)) for x, y, z in chunks]))
        return job_id

    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        budget = self.chunks_per_poll
        while self._pending_jobs and budget > 0:
            job = self._pending_jobs[0]
            while job.cursor < len(job.chunk_coords) and budget > 0:
                chunk_x, chunk_y, chunk_z = job.chunk_coords[job.cursor]
                heights, materials = self.chunk_surface_grids(chunk_x, chunk_z)
                ready.append(
                    ChunkSurfaceResult(
                        chunk_x=chunk_x,
                        chunk_y=chunk_y,
                        chunk_z=chunk_z,
                        heights=heights,
                        materials=materials,
                        source="cpu",
                    )
                )
                job.cursor += 1
                budget -= 1
            if job.cursor >= len(job.chunk_coords):
                self._pending_jobs.popleft()
        return ready

    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        return []

    def has_pending_chunk_surface_batches(self) -> bool:
        return bool(self._pending_jobs)

    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        while self._pending_jobs:
            ready.extend(self.poll_ready_chunk_surface_batches())
        return ready

    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        budget = self.chunks_per_poll
        while self._pending_voxel_jobs and budget > 0:
            job = self._pending_voxel_jobs[0]
            while job.cursor < len(job.chunk_coords) and budget > 0:
                chunk_x, chunk_y, chunk_z = job.chunk_coords[job.cursor]
                blocks, materials = self.chunk_voxel_grid(chunk_x, chunk_y, chunk_z)
                ready.append(
                    ChunkVoxelResult(
                        chunk_x=chunk_x,
                        chunk_y=chunk_y,
                        chunk_z=chunk_z,
                        blocks=blocks,
                        materials=materials,
                        source="cpu",
                    )
                )
                job.cursor += 1
                budget -= 1
            if job.cursor >= len(job.chunk_coords):
                self._pending_voxel_jobs.popleft()
        return ready

    def has_pending_chunk_voxel_batches(self) -> bool:
        return bool(self._pending_voxel_jobs)

    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        while self._pending_voxel_jobs:
            ready.extend(self.poll_ready_chunk_voxel_batches())
        return ready

    def terrain_backend_label(self) -> str:
        return "CPU"
