from __future__ import annotations

import numpy as np

from ..types import ChunkSurfaceResult, ChunkVoxelResult
from ...terrain.kernels import (
    expand_chunk_surface_to_voxel_grid,
    fill_chunk_voxel_grid as cpu_fill_chunk_voxel_grid,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
)
from ...world_constants import VERTICAL_CHUNK_STACK_ENABLED
from .wgpu_terrain_common import profile


def surface_result_to_voxel_result(
    surface_result: ChunkSurfaceResult,
    *,
    chunk_size: int,
    sample_size: int,
    seed: int,
    height_limit: int,
) -> ChunkVoxelResult:
    chunk_x = int(surface_result.chunk_x)
    chunk_y = int(getattr(surface_result, "chunk_y", 0))
    chunk_z = int(surface_result.chunk_z)
    top_boundary = None
    bottom_boundary = None
    is_empty_chunk = False

    if VERTICAL_CHUNK_STACK_ENABLED:
        local_height = int(chunk_size)
        origin_y = chunk_y * int(chunk_size)
        blocks = np.zeros((local_height, sample_size, sample_size), dtype=np.uint8)
        voxel_materials = np.zeros((local_height, sample_size, sample_size), dtype=np.uint32)
        top_boundary = np.zeros((sample_size, sample_size), dtype=np.uint8)
        bottom_boundary = np.zeros((sample_size, sample_size), dtype=np.uint8)
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
                int(chunk_size),
                int(seed),
                int(height_limit),
            )
    else:
        blocks = np.zeros((height_limit, sample_size, sample_size), dtype=np.uint8)
        voxel_materials = np.zeros((height_limit, sample_size, sample_size), dtype=np.uint32)
        expand_chunk_surface_to_voxel_grid(
            blocks,
            voxel_materials,
            surface_result.heights,
            surface_result.materials,
            int(chunk_size),
            int(height_limit),
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


def surface_results_to_voxel_results(
    surface_results: list[ChunkSurfaceResult],
    *,
    chunk_size: int,
    sample_size: int,
    seed: int,
    height_limit: int,
) -> list[ChunkVoxelResult]:
    return [
        surface_result_to_voxel_result(
            surface_result,
            chunk_size=chunk_size,
            sample_size=sample_size,
            seed=seed,
            height_limit=height_limit,
        )
        for surface_result in surface_results
    ]


class WgpuTerrainVoxelMixin:
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
    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        return self.request_chunk_surface_batch(chunks)

    @profile
    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return surface_results_to_voxel_results(
            self.poll_ready_chunk_surface_batches(),
            chunk_size=self.chunk_size,
            sample_size=self.sample_size,
            seed=self.seed,
            height_limit=self.height_limit,
        )

    def has_pending_chunk_voxel_batches(self) -> bool:
        return self.has_pending_chunk_surface_batches()

    @profile
    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return surface_results_to_voxel_results(
            self.flush_chunk_surface_batches(),
            chunk_size=self.chunk_size,
            sample_size=self.sample_size,
            seed=self.seed,
            height_limit=self.height_limit,
        )
