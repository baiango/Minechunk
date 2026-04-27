"""Compatibility split for stacked voxel fill kernels.

This module surfaces the stacked voxel fill kernels from the dedicated voxel
fill module so older call sites can migrate incrementally without changing
behavior.
"""

from .voxel_fill import (
    fill_stacked_chunk_voxel_grid,
    fill_stacked_chunk_vertical_neighbor_planes,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
)
