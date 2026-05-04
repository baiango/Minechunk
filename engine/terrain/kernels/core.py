from __future__ import annotations

"""Compatibility façade for terrain kernel modules.

The terrain kernels used to live in this single file.  They are now split by
responsibility so terrain generation, voxel filling, and meshing can evolve
without turning this module back into a monolith. Existing imports from
``engine.terrain.kernels.core`` are preserved here.
"""

from .materials import (
    AIR,
    BEDROCK,
    DIRT,
    GRASS,
    MAX_FACES_PER_CELL,
    SAND,
    SNOW,
    STONE,
    VERTEX_COMPONENTS,
    VERTICES_PER_FACE,
    _MATERIAL_COLOR_B,
    _MATERIAL_COLOR_G,
    _MATERIAL_COLOR_R,
    _scale_color,
    _terrain_color,
    _voxel_material_color,
)
from .noise import (
    _fade,
    _hash2,
    _hash3,
    _lerp,
    _mix_u32,
    _value_noise_2d,
    _value_noise_3d,
)
from .terrain_profile import (
    CAVE_BEDROCK_CLEARANCE,
    CAVE_ACTIVE_BAND_MIN,
    CAVE_DETAIL_FREQUENCY_MULTIPLIER,
    CAVE_DETAIL_WEIGHT,
    CAVE_DEPTH_BONUS_MAX,
    CAVE_DEPTH_BONUS_SCALE,
    CAVE_FREQUENCY_SCALE,
    CAVE_MODEL_VERSION,
    CAVE_PRIMARY_THRESHOLD,
    CAVE_VERTICAL_BONUS,
    TERRAIN_FREQUENCY_SCALE,
    _clamp01,
    _should_carve_cave,
    _terrain_material_from_surface_profile,
    surface_profile_at,
    terrain_block_material_at,
)
from .voxel_fill import (
    expand_chunk_surface_to_voxel_grid,
    fill_chunk_surface_grids,
    fill_chunk_voxel_grid,
    fill_stacked_chunk_vertical_neighbor_planes,
    fill_stacked_chunk_voxel_grid,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
)
from .surface_mesher import (
    _emit_quad,
    _emit_triangle,
    _write_vertex,
    build_chunk_vertex_array,
)
from .voxel_mesher import (
    FACE_BOTTOM,
    FACE_EAST,
    FACE_NORTH,
    FACE_SOUTH,
    FACE_TOP,
    FACE_WEST,
    _ambient_occlusion_factor,
    _ao_x_from_planes,
    _ao_y_from_plane,
    _ao_z_from_planes,
    _build_chunk_face_masks_with_boundaries,
    _emit_quad_components_ao,
    _emit_voxel_face,
    _solid_at_with_boundaries,
    build_chunk_surface_run_table_from_heightmap_clipped,
    build_chunk_vertex_array_from_voxels,
    build_chunk_vertex_array_from_voxels_with_boundaries,
    build_chunk_surface_vertex_array_from_heightmap_clipped,
    count_chunk_surface_vertices_from_heightmap_clipped,
    count_chunk_voxel_vertices,
    count_chunk_voxel_vertices_with_boundaries,
    emit_chunk_surface_run_table_vertices,
    emit_chunk_surface_vertices_from_heightmap_clipped,
)

__all__ = [name for name in globals() if not name.startswith("__")]
