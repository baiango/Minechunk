from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import ChunkSurfaceGpuBatch
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
    readback_buffer: object | None
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
