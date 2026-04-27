from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "engine" / "terrain" / "backends"


def _read(name: str) -> str:
    return (BACKENDS / name).read_text(encoding="utf-8")


def test_metal_terrain_backend_is_split_by_responsibility() -> None:
    backend = _read("metal_terrain_backend.py")
    common = _read("metal_terrain_common.py")
    buffers = _read("metal_terrain_buffers.py")
    batches = _read("metal_terrain_batches.py")
    voxels = _read("metal_terrain_voxels.py")

    assert len(backend.splitlines()) < 120
    assert "class MetalTerrainBackend(MetalTerrainBufferOps, MetalTerrainBatchOps, MetalTerrainVoxelOps)" in backend
    assert "GPU_TERRAIN_SHADER = load_shader_text(\"terrain_surface.metal\")" in common
    assert "class MetalTerrainBufferOps" in buffers
    assert "class MetalTerrainBatchOps" in batches
    assert "class MetalTerrainVoxelOps" in voxels


def test_metal_surface_batch_and_voxel_conversion_have_single_owners() -> None:
    backend = _read("metal_terrain_backend.py")
    batches = _read("metal_terrain_batches.py")
    voxels = _read("metal_terrain_voxels.py")

    assert "def poll_ready_chunk_surface_gpu_batches" not in backend
    assert "def _voxel_result_from_surface_result" not in backend
    assert "def poll_ready_chunk_surface_gpu_batches" in batches
    assert "def _voxel_result_from_surface_result" in voxels
    assert voxels.count("ChunkVoxelResult(") == 1
