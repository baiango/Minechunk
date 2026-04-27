from engine.terrain.kernels import AIR, STONE, surface_profile_at
from engine.terrain.kernels import core
from engine.terrain.kernels import terrain_profile, voxel_fill, voxel_mesher


def test_kernel_facade_preserves_public_exports():
    assert AIR == 0
    assert STONE == 2
    assert surface_profile_at is core.surface_profile_at
    assert core.surface_profile_at is terrain_profile.surface_profile_at
    assert core.fill_chunk_surface_grids is voxel_fill.fill_chunk_surface_grids
    assert core.build_chunk_vertex_array_from_voxels is voxel_mesher.build_chunk_vertex_array_from_voxels


def test_kernel_facade_preserves_internal_compatibility_names():
    assert hasattr(core, "_value_noise_2d")
    assert hasattr(core, "_terrain_material_from_surface_profile")
    assert hasattr(core, "_ambient_occlusion_factor")
    assert int(core.FACE_TOP) == 1
    assert int(core.FACE_NORTH) == 32
