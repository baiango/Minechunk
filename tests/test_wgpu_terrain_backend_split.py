from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "engine" / "terrain" / "backends"


def _read(name: str) -> str:
    return (BACKENDS / name).read_text(encoding="utf-8")


def test_wgpu_terrain_backend_is_split_by_responsibility() -> None:
    backend = _read("wgpu_terrain_backend.py")
    batches = _read("wgpu_terrain_batches.py")
    common = _read("wgpu_terrain_common.py")
    voxels = _read("wgpu_terrain_voxels.py")

    assert len(backend.splitlines()) < 280
    assert "class WgpuTerrainBackend(WgpuTerrainVoxelMixin, WgpuTerrainBatchMixin)" in backend
    assert "class WgpuTerrainBatchMixin" in batches
    assert "class WgpuTerrainVoxelMixin" in voxels
    assert "GPU_TERRAIN_SHADER = load_shader_text(\"terrain_surface.wgsl\")" in common


def test_wgpu_terrain_surface_to_voxel_conversion_has_one_owner() -> None:
    backend = _read("wgpu_terrain_backend.py")
    voxels = _read("wgpu_terrain_voxels.py")

    assert "surface_result_to_voxel_result" not in backend
    assert "def surface_result_to_voxel_result" in voxels
    assert voxels.count("ChunkVoxelResult(") == 1
