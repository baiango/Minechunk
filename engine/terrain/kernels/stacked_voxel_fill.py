"""Compatibility split for stacked voxel fill kernels.

This module surfaces the stacked voxel fill kernels from the historical
monolithic ``engine.terrain_kernels`` module so the engine can migrate
call sites incrementally without changing behavior.
"""

from .core import (
    fill_stacked_chunk_voxel_grid,
    fill_stacked_chunk_vertical_neighbor_planes,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
)
