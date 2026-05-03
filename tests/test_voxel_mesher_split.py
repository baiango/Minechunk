from engine.terrain.kernels import core
from engine.terrain.kernels import voxel_ao, voxel_emit, voxel_faces, voxel_mesher


def test_voxel_mesher_facade_preserves_split_helper_exports():
    assert voxel_mesher._emit_quad_components_ao is voxel_emit._emit_quad_components_ao
    assert voxel_mesher._emit_quad_components_uniform_color is voxel_emit._emit_quad_components_uniform_color
    assert voxel_mesher._emit_voxel_face is voxel_emit._emit_voxel_face
    assert voxel_mesher._ambient_occlusion_factor is voxel_ao._ambient_occlusion_factor
    assert voxel_mesher._ao_y_from_plane is voxel_ao._ao_y_from_plane
    assert voxel_mesher._ao_x_from_planes is voxel_ao._ao_x_from_planes
    assert voxel_mesher._ao_z_from_planes is voxel_ao._ao_z_from_planes
    assert voxel_mesher._build_chunk_face_masks_with_boundaries is voxel_faces._build_chunk_face_masks_with_boundaries


def test_core_compatibility_facade_still_reexports_voxel_mesher_api():
    assert core.build_chunk_vertex_array_from_voxels is voxel_mesher.build_chunk_vertex_array_from_voxels
    assert core.build_chunk_vertex_array_from_voxels_with_boundaries is voxel_mesher.build_chunk_vertex_array_from_voxels_with_boundaries
    assert core.build_chunk_surface_vertex_array_from_heightmap_clipped is voxel_mesher.build_chunk_surface_vertex_array_from_heightmap_clipped
    assert core.build_chunk_surface_run_table_from_heightmap_clipped is voxel_mesher.build_chunk_surface_run_table_from_heightmap_clipped
    assert core.count_chunk_surface_vertices_from_heightmap_clipped is voxel_mesher.count_chunk_surface_vertices_from_heightmap_clipped
    assert core.emit_chunk_surface_run_table_vertices is voxel_mesher.emit_chunk_surface_run_table_vertices
    assert core.emit_chunk_surface_vertices_from_heightmap_clipped is voxel_mesher.emit_chunk_surface_vertices_from_heightmap_clipped
    assert core.count_chunk_voxel_vertices is voxel_mesher.count_chunk_voxel_vertices
    assert core.count_chunk_voxel_vertices_with_boundaries is voxel_mesher.count_chunk_voxel_vertices_with_boundaries
    assert int(core.FACE_TOP) == int(voxel_faces.FACE_TOP) == 1
    assert int(core.FACE_NORTH) == int(voxel_faces.FACE_NORTH) == 32
