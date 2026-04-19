from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ChunkSurfaceResult:
    chunk_x: int
    chunk_z: int
    chunk_y: int = 0
    heights: np.ndarray | None = None
    materials: np.ndarray | None = None
    source: str = ""


@dataclass(frozen=True)
class ChunkVoxelResult:
    chunk_x: int
    chunk_z: int
    blocks: np.ndarray
    materials: np.ndarray
    chunk_y: int = 0
    source: str = ""
    top_boundary: np.ndarray | None = None
    bottom_boundary: np.ndarray | None = None
    is_empty: bool = False


@dataclass(frozen=True)
class ChunkSurfaceGpuBatch:
    chunks: list[tuple[int, int, int]]
    heights_buffer: object
    materials_buffer: object
    cell_count: int
    source: str


@dataclass(frozen=True)
class TerrainValidationReport:
    chunk_x: int
    chunk_z: int
    backend_label: str
    total_cells: int
    height_mismatches: int
    material_mismatches: int
    first_height_mismatch: tuple[int, int, int, int] | None
    first_material_mismatch: tuple[int, int, int, int] | None
    chunk_y: int = 0

    @property
    def matches(self) -> bool:
        return self.height_mismatches == 0 and self.material_mismatches == 0


class TerrainBackend(Protocol):
    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        ...

    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        ...

    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        ...

    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ...

    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        ...

    def has_pending_chunk_surface_batches(self) -> bool:
        ...

    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ...

    def chunk_voxel_grid(self, chunk_x: int, chunk_y: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        ...

    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        ...

    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ...

    def has_pending_chunk_voxel_batches(self) -> bool:
        ...

    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ...

    def terrain_backend_label(self) -> str:
        ...
