from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from ..types import ChunkSurfaceGpuBatch, ChunkSurfaceResult, ChunkVoxelResult
from ...terrain.kernels.zig_kernel import (
    fill_chunk_surface_grids,
    fill_chunk_surface_grids_batch,
    fill_stacked_chunk_voxel_grid,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
    fill_chunk_voxel_grid,
    surface_profile_at as sample_surface_profile_at,
    terrain_kernel_label,
)
from ...world_constants import CHUNK_SIZE as DEFAULT_CHUNK_SIZE, VERTICAL_CHUNK_STACK_ENABLED

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


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
        terrain_caves_enabled: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.height = self.chunk_size if height is None else int(height)
        self.chunks_per_poll = max(1, int(chunks_per_poll))
        self.terrain_caves_enabled = bool(terrain_caves_enabled)
        self._terrain_kernel_label = terrain_kernel_label()
        self._pending_jobs: deque[_PendingChunkJob] = deque()
        self._pending_voxel_jobs: deque[_PendingChunkJob] = deque()
        self._next_job_id = 1
        self._next_voxel_job_id = 1
        sample_size = self.chunk_size + 2
        cell_count = sample_size * sample_size
        local_height = self.chunk_size if VERTICAL_CHUNK_STACK_ENABLED else self.height
        self._empty_voxel_blocks = np.zeros((local_height, sample_size, sample_size), dtype=np.uint8)
        self._empty_voxel_materials = np.zeros((local_height, sample_size, sample_size), dtype=np.uint32)
        self._surface_probe_heights = np.empty(cell_count, dtype=np.uint32)
        self._surface_probe_materials = np.empty(cell_count, dtype=np.uint32)
        self._surface_probe_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray, int, int]] = OrderedDict()
        self._surface_probe_cache_limit = 512

    @profile
    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        height, material = sample_surface_profile_at(float(x), float(z), self.seed, self.height)
        return int(height), int(material)

    @profile
    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        sample_size = self.chunk_size + 2
        cell_count = sample_size * sample_size
        heights = np.empty(cell_count, dtype=np.uint32)
        materials = np.empty(cell_count, dtype=np.uint32)
        fill_chunk_surface_grids(heights, materials, chunk_x, chunk_z, self.chunk_size, self.seed, self.height)
        return heights, materials

    def _cached_surface_grids_for_voxel_chunk(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        key = (int(chunk_x), int(chunk_z))
        cached = self._surface_probe_cache.get(key)
        if cached is not None:
            self._surface_probe_cache.move_to_end(key)
            return cached

        sample_size = self.chunk_size + 2
        cell_count = sample_size * sample_size
        heights = np.empty(cell_count, dtype=np.uint32)
        materials = np.empty(cell_count, dtype=np.uint32)
        fill_chunk_surface_grids(heights, materials, key[0], key[1], self.chunk_size, self.seed, self.height)
        cached = (heights, materials, int(heights.max()), int(heights.min()))
        self._surface_probe_cache[key] = cached
        while len(self._surface_probe_cache) > self._surface_probe_cache_limit:
            self._surface_probe_cache.popitem(last=False)
        return cached

    @profile
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
                self.terrain_caves_enabled,
            )
            return blocks, materials
        blocks = np.zeros((self.height, sample_size, sample_size), dtype=np.uint8)
        materials = np.zeros((self.height, sample_size, sample_size), dtype=np.uint32)
        fill_chunk_voxel_grid(
            blocks,
            materials,
            chunk_x,
            chunk_z,
            self.chunk_size,
            self.seed,
            self.height,
            self.terrain_caves_enabled,
        )
        return blocks, materials

    @profile
    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        if chunks:
            self._pending_jobs.appendleft(_PendingChunkJob(job_id=job_id, chunk_coords=[(int(x), int(y), int(z)) for x, y, z in chunks]))
        return job_id

    @profile
    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        job_id = self._next_voxel_job_id
        self._next_voxel_job_id += 1
        if chunks:
            self._pending_voxel_jobs.appendleft(_PendingChunkJob(job_id=job_id, chunk_coords=[(int(x), int(y), int(z)) for x, y, z in chunks]))
        return job_id

    @profile
    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        budget = self.chunks_per_poll
        batch_chunks: list[tuple[int, int, int]] = []
        while self._pending_jobs and budget > 0:
            job = self._pending_jobs[0]
            while job.cursor < len(job.chunk_coords) and budget > 0:
                batch_chunks.append(job.chunk_coords[job.cursor])
                job.cursor += 1
                budget -= 1
            if job.cursor >= len(job.chunk_coords):
                self._pending_jobs.popleft()
        if not batch_chunks:
            return ready

        sample_size = self.chunk_size + 2
        cell_count = sample_size * sample_size
        heights_batch = np.empty((len(batch_chunks), cell_count), dtype=np.uint32)
        materials_batch = np.empty_like(heights_batch)
        chunk_xs = np.array([chunk_x for chunk_x, _chunk_y, _chunk_z in batch_chunks], dtype=np.int32)
        chunk_zs = np.array([chunk_z for _chunk_x, _chunk_y, chunk_z in batch_chunks], dtype=np.int32)
        fill_chunk_surface_grids_batch(
            heights_batch,
            materials_batch,
            chunk_xs,
            chunk_zs,
            self.chunk_size,
            self.seed,
            self.height,
        )
        for index, (chunk_x, chunk_y, chunk_z) in enumerate(batch_chunks):
            ready.append(
                ChunkSurfaceResult(
                    chunk_x=chunk_x,
                    chunk_y=chunk_y,
                    chunk_z=chunk_z,
                    heights=heights_batch[index],
                    materials=materials_batch[index],
                    source="cpu",
                )
            )
        return ready

    @profile
    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        return []

    def has_pending_chunk_surface_batches(self) -> bool:
        return bool(self._pending_jobs)

    @profile
    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        while self._pending_jobs:
            ready.extend(self.poll_ready_chunk_surface_batches())
        return ready

    @profile
    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        budget = self.chunks_per_poll
        sample_size = self.chunk_size + 2
        surface_grid_lookup: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int, int]] = {}
        surface_probe_cache_get = self._surface_probe_cache.get
        while self._pending_voxel_jobs and budget > 0:
            job = self._pending_voxel_jobs[0]
            while job.cursor < len(job.chunk_coords) and budget > 0:
                chunk_x, chunk_y, chunk_z = job.chunk_coords[job.cursor]
                top_boundary = None
                bottom_boundary = None
                is_empty_chunk = False
                is_fully_occluded_chunk = False
                surface_heights = None
                surface_materials = None
                use_surface_mesher = False
                if VERTICAL_CHUNK_STACK_ENABLED:
                    local_height = self.chunk_size
                    origin_y = int(chunk_y) * self.chunk_size
                    top_world_y = origin_y + local_height
                    surface_key = (int(chunk_x), int(chunk_z))
                    surface_grid = surface_grid_lookup.get(surface_key)
                    if surface_grid is None:
                        surface_grid = surface_probe_cache_get(surface_key)
                        if surface_grid is None:
                            surface_grid = self._cached_surface_grids_for_voxel_chunk(chunk_x, chunk_z)
                        surface_grid_lookup[surface_key] = surface_grid
                    probe_heights, probe_materials, max_surface_height, min_surface_height = surface_grid
                    is_empty_chunk = max_surface_height <= origin_y
                    is_fully_occluded_chunk = (
                        not self.terrain_caves_enabled
                        and origin_y > 0
                        and min_surface_height > top_world_y
                    )
                    if is_empty_chunk or is_fully_occluded_chunk:
                        blocks = self._empty_voxel_blocks
                        materials = self._empty_voxel_materials
                    elif not self.terrain_caves_enabled:
                        blocks = self._empty_voxel_blocks
                        materials = self._empty_voxel_materials
                        surface_heights = probe_heights
                        surface_materials = probe_materials
                        use_surface_mesher = True
                    else:
                        blocks = np.zeros((local_height, sample_size, sample_size), dtype=np.uint8)
                        materials = np.zeros((local_height, sample_size, sample_size), dtype=np.uint32)
                        top_boundary = np.zeros((sample_size, sample_size), dtype=np.uint8)
                        bottom_boundary = np.zeros((sample_size, sample_size), dtype=np.uint8)
                        fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
                            blocks,
                            materials,
                            top_boundary,
                            bottom_boundary,
                            probe_heights,
                            probe_materials,
                            int(chunk_x),
                            int(chunk_y),
                            int(chunk_z),
                            self.chunk_size,
                            self.seed,
                            self.height,
                            self.terrain_caves_enabled,
                        )
                else:
                    blocks, materials = self.chunk_voxel_grid(chunk_x, chunk_y, chunk_z)
                ready.append(
                    ChunkVoxelResult(
                        chunk_x=chunk_x,
                        chunk_y=chunk_y,
                        chunk_z=chunk_z,
                        blocks=blocks,
                        materials=materials,
                        source="cpu",
                        top_boundary=top_boundary,
                        bottom_boundary=bottom_boundary,
                        is_empty=is_empty_chunk,
                        is_fully_occluded=is_fully_occluded_chunk,
                        surface_heights=surface_heights,
                        surface_materials=surface_materials,
                        use_surface_mesher=use_surface_mesher,
                    )
                )
                job.cursor += 1
                budget -= 1
            if job.cursor >= len(job.chunk_coords):
                self._pending_voxel_jobs.popleft()
        return ready

    def has_pending_chunk_voxel_batches(self) -> bool:
        return bool(self._pending_voxel_jobs)

    @profile
    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        while self._pending_voxel_jobs:
            ready.extend(self.poll_ready_chunk_voxel_batches())
        return ready

    def terrain_backend_label(self) -> str:
        return f"CPU/{self._terrain_kernel_label}"
