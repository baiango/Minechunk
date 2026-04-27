from __future__ import annotations

import sys
import types


def _install_runtime_stubs(monkeypatch):
    class _DType:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, value=0):
            return int(value)

        def __repr__(self) -> str:
            return self.name

    np = types.ModuleType("numpy")
    for name in ("float32", "float64", "uint8", "uint32", "uint64", "int32", "int64", "bool_"):
        setattr(np, name, _DType(name))
    np.ndarray = object
    np.dtype = lambda spec: ("dtype", spec)
    np.empty = lambda *args, **kwargs: None
    np.frombuffer = lambda *args, **kwargs: None
    np.ascontiguousarray = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "numpy", np)

    metal = types.ModuleType("Metal")
    metal.MTLCommandBufferStatusCompleted = 4
    metal.MTLCommandBufferStatusError = 5
    metal.MTLResourceStorageModeShared = 1
    metal.MTLResourceCPUCacheModeWriteCombined = 2
    metal.MTLSizeMake = lambda x, y, z: (x, y, z)
    metal.MTLCreateSystemDefaultDevice = lambda: None
    monkeypatch.setitem(sys.modules, "Metal", metal)

    class _Flags:
        def __getattr__(self, name: str) -> int:
            return 1

    wgpu = types.ModuleType("wgpu")
    wgpu.BufferUsage = _Flags()
    wgpu.ShaderStage = _Flags()
    wgpu.TextureUsage = _Flags()
    wgpu.GPUBuffer = object
    monkeypatch.setitem(sys.modules, "wgpu", wgpu)

    for name in list(sys.modules):
        if name == "engine.meshing.metal_mesher" or name.startswith("engine.meshing.metal_mesher_") or name == "engine.meshing.metal_chunk_mesher":
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_metal_mesher_facade_preserves_public_entry_points(monkeypatch):
    _install_runtime_stubs(monkeypatch)

    from engine.meshing import metal_mesher

    required = {
        "MetalChunkMesher",
        "get_metal_chunk_mesher",
        "prewarm_metal_chunk_mesher",
        "submit_chunk_mesh_batch_async",
        "make_chunk_mesh_batch_from_terrain_results",
        "enqueue_surface_gpu_batches_for_meshing",
        "drain_pending_surface_gpu_batches_to_meshing",
        "make_chunk_mesh_batch_from_surface_gpu_batch",
        "make_chunk_mesh_batches_from_surface_gpu_batches",
        "pending_surface_gpu_batches_chunk_count",
        "release_surface_gpu_batch_immediately",
        "finalize_pending_gpu_mesh_batches",
        "destroy_async_voxel_mesh_batch_resources",
        "process_gpu_buffer_cleanup",
        "shutdown_renderer_async_state",
    }
    assert required.issubset(set(metal_mesher.__all__))
    assert all(hasattr(metal_mesher, name) for name in required)


def test_release_surface_gpu_batch_immediately_clears_callback(monkeypatch):
    _install_runtime_stubs(monkeypatch)

    from engine.meshing.metal_mesher_common import release_surface_gpu_batch_immediately

    calls = []

    class SurfaceBatch:
        _release_callback = staticmethod(lambda: calls.append("released"))

    batch = SurfaceBatch()
    release_surface_gpu_batch_immediately(batch)

    assert calls == ["released"]
    assert getattr(batch, "_release_callback") is None
