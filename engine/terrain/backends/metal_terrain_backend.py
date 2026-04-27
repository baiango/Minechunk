from __future__ import annotations

"""Metal terrain backend compatibility entry point."""

import struct
from collections import deque

from .metal_terrain_batches import MetalTerrainBatchOps
from .metal_terrain_buffers import MetalTerrainBufferOps
from .metal_terrain_common import (
    GPU_TERRAIN_SHADER,
    Metal,
    _ChunkMetalBatch,
    _LeasedChunkSurfaceGpuBatch,
    _METAL_IMPORT_ERROR,
    _normalize_chunk_coord,
    _normalize_chunk_coords,
)
from .metal_terrain_voxels import MetalTerrainVoxelOps


class MetalTerrainBackend(MetalTerrainBufferOps, MetalTerrainBatchOps, MetalTerrainVoxelOps):
    def __init__(
        self,
        device=None,
        seed: int = 0,
        chunk_size: int = 64,
        height_limit: int | None = None,
        chunks_per_poll: int = 128,
    ) -> None:
        if Metal is None:
            raise RuntimeError(
                "Metal bindings unavailable. Install with `pip install pyobjc-framework-Metal` on macOS."
            ) from _METAL_IMPORT_ERROR
        if device is None or not hasattr(device, "newCommandQueue"):
            device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("Metal device unavailable.")

        self.device = device
        self.command_queue = device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal command queue.")

        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.height_limit = self.chunk_size if height_limit is None else int(height_limit)
        self.chunks_per_poll = max(1, int(chunks_per_poll))
        self.sample_size = self.chunk_size + 2
        self.cell_count = self.sample_size * self.sample_size

        self._pending_jobs: deque[list[tuple[int, int, int]]] = deque()
        self._in_flight_batches: deque[_ChunkMetalBatch] = deque()
        self._next_job_id = 1
        self._submit_target_chunks = max(int(self.chunks_per_poll), min(int(self.chunks_per_poll) * 2, 128))
        self._batch_pool_size = 8
        self._max_in_flight_batches = 3
        self._available_batch_slots: deque[_ChunkMetalBatch] = deque()
        self._batch_slots_pending_reuse: deque[_ChunkMetalBatch] = deque()
        self._leased_surface_batches: dict[int, _ChunkMetalBatch] = {}

        self._resource_options = int(getattr(Metal, "MTLResourceStorageModeShared"))
        self._params_buffer = self._make_buffer(64)
        self._single_heights_buffer = self._make_buffer(4)
        self._single_materials_buffer = self._make_buffer(4)
        self._grid_heights_buffer = self._make_buffer(self.cell_count * 4)
        self._grid_materials_buffer = self._make_buffer(self.cell_count * 4)
        self._batch_params_payload = struct.pack(
            "<4I",
            self.sample_size,
            self.chunk_size,
            self.height_limit,
            self.seed & 0xFFFFFFFF,
        )

        library, err = self.device.newLibraryWithSource_options_error_(GPU_TERRAIN_SHADER, None, None)
        if err is not None or library is None:
            raise RuntimeError(f"Failed to compile Metal terrain shader: {err}")
        self._single_pipeline = self._create_pipeline(library, "sample_surface_profile_at_main")
        self._grid_pipeline = self._create_pipeline(library, "fill_chunk_surface_grids_main")
        self._batch_pipeline = self._create_pipeline(library, "fill_chunk_surface_batch_main")

        for _ in range(self._batch_pool_size):
            self._available_batch_slots.append(self._allocate_chunk_batch_resources(self._submit_target_chunks))

    def terrain_backend_label(self) -> str:
        return "Metal"


__all__ = [
    "MetalTerrainBackend",
    "GPU_TERRAIN_SHADER",
    "Metal",
    "_ChunkMetalBatch",
    "_LeasedChunkSurfaceGpuBatch",
    "_METAL_IMPORT_ERROR",
    "_normalize_chunk_coord",
    "_normalize_chunk_coords",
]
