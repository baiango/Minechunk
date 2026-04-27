from __future__ import annotations

"""Metal terrain surface-result to voxel-grid conversion helpers."""

import numpy as np

from ..types import ChunkSurfaceResult, ChunkVoxelResult
from ...terrain.kernels import (
    expand_chunk_surface_to_voxel_grid,
    fill_chunk_voxel_grid as cpu_fill_chunk_voxel_grid,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
)
from ...world_constants import VERTICAL_CHUNK_STACK_ENABLED
from .metal_terrain_common import profile


class MetalTerrainVoxelOps:
    @profile
    def chunk_voxel_grid(self, chunk_x: int, chunk_y: int = 0, chunk_z: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if chunk_z is None:
            chunk_z = int(chunk_y)
            chunk_y = 0
        chunk_x = int(chunk_x)
        chunk_y = int(chunk_y)
        chunk_z = int(chunk_z)
        if VERTICAL_CHUNK_STACK_ENABLED:
            local_height = self.chunk_size
            blocks = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint8)
            voxel_materials = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint32)
            top_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            bottom_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            heights, materials = self.chunk_surface_grids(chunk_x, chunk_z)
            fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
                blocks,
                voxel_materials,
                top_boundary,
                bottom_boundary,
                heights,
                materials,
                chunk_x,
                chunk_y,
                chunk_z,
                self.chunk_size,
                self.seed,
                self.height_limit,
            )
            return blocks, voxel_materials
        blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
        voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
        cpu_fill_chunk_voxel_grid(blocks, voxel_materials, chunk_x, chunk_z, self.chunk_size, self.seed, self.height_limit)
        return blocks, voxel_materials

    @profile
    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        return self.request_chunk_surface_batch(chunks)

    def _voxel_result_from_surface_result(self, surface_result: ChunkSurfaceResult) -> ChunkVoxelResult:
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
        return ChunkVoxelResult(
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

    @profile
    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return [self._voxel_result_from_surface_result(result) for result in self.poll_ready_chunk_surface_batches()]

    def has_pending_chunk_voxel_batches(self) -> bool:
        return self.has_pending_chunk_surface_batches()

    @profile
    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return [self._voxel_result_from_surface_result(result) for result in self.flush_chunk_surface_batches()]
