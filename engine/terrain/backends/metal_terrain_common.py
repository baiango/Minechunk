from __future__ import annotations

"""Shared Metal terrain backend types and optional imports."""

from dataclasses import dataclass

import numpy as np

from ..types import ChunkSurfaceGpuBatch
from ...shader_loader import load_shader_text

try:
    import Metal
except Exception as exc:  # pragma: no cover
    Metal = None  # type: ignore[assignment]
    _METAL_IMPORT_ERROR = exc
else:
    _METAL_IMPORT_ERROR = None

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def profile(func):
        return func


GPU_TERRAIN_SHADER = load_shader_text("terrain_surface.metal")


@dataclass
class _ChunkMetalBatch:
    chunks: list[tuple[int, int, int]]
    chunk_count: int
    max_chunks: int
    coords_array: np.ndarray
    coords_buffer: object
    params_buffer: object
    heights_buffer: object
    materials_buffer: object
    command_buffer: object | None = None


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
