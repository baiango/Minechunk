from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _install_import_stubs():
    # The split/facade test only verifies module boundaries.  The real runtime
    # imports numpy and wgpu, but CI/smoke environments for maintainability
    # checks may not have those optional graphics dependencies installed.
    sys.modules.setdefault(
        "numpy",
        SimpleNamespace(
            dtype=lambda spec: ("dtype", tuple(map(str, spec))),
            float32="float32",
            uint32="uint32",
        ),
    )
    sys.modules.setdefault("wgpu", SimpleNamespace())


def test_gpu_mesher_facade_exports_split_public_api():
    _install_import_stubs()

    from engine.meshing import gpu_mesher
    from engine.meshing import gpu_mesher_batches, gpu_mesher_finalize, gpu_mesher_resources, gpu_surface_batches

    assert gpu_mesher.ensure_voxel_mesh_batch_scratch is gpu_mesher_resources.ensure_voxel_mesh_batch_scratch
    assert gpu_mesher.process_gpu_buffer_cleanup is gpu_mesher_resources.process_gpu_buffer_cleanup
    assert gpu_mesher.make_chunk_mesh_batch_from_gpu_buffers is gpu_mesher_batches.make_chunk_mesh_batch_from_gpu_buffers
    assert gpu_mesher.make_chunk_mesh_batch_from_terrain_results is gpu_mesher_batches.make_chunk_mesh_batch_from_terrain_results
    assert gpu_mesher.get_voxel_surface_expand_bind_group is gpu_surface_batches.get_voxel_surface_expand_bind_group
    assert gpu_mesher.make_chunk_mesh_batches_from_surface_gpu_batches is gpu_surface_batches.make_chunk_mesh_batches_from_surface_gpu_batches
    assert gpu_mesher.finalize_pending_gpu_mesh_batches is gpu_mesher_finalize.finalize_pending_gpu_mesh_batches


def test_gpu_mesher_facade_stays_small():
    source = (ROOT / "engine" / "meshing" / "gpu_mesher.py").read_text(encoding="utf-8")

    assert len(source.splitlines()) < 80
    assert "def make_chunk_mesh_batch_from_gpu_buffers" not in source
    assert "def finalize_pending_gpu_mesh_batches" not in source
    assert "from .gpu_mesher_resources import *" in source
