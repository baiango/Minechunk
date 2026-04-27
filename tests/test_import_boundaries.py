from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LOW_LEVEL_MODULES = [
    ROOT / "engine" / "auto_exit.py",
    ROOT / "engine" / "cache" / "cached_tile_batches.py",
    ROOT / "engine" / "cache" / "direct_render_batches.py",
    ROOT / "engine" / "cache" / "mesh_allocator.py",
    ROOT / "engine" / "cache" / "mesh_output_allocator.py",
    ROOT / "engine" / "cache" / "mesh_visibility.py",
    ROOT / "engine" / "cache" / "tile_cache_constants.py",
    ROOT / "engine" / "cache" / "tile_draw_batches.py",
    ROOT / "engine" / "cache" / "tile_mesh_cache.py",
    ROOT / "engine" / "collision" / "walk_solver.py",
    ROOT / "engine" / "debug_capture.py",
    ROOT / "engine" / "input_controller.py",
    ROOT / "engine" / "profiling_runtime.py",
    ROOT / "engine" / "world_reset.py",
    ROOT / "engine" / "render_contract.py",
    ROOT / "engine" / "shader_loader.py",
    ROOT / "engine" / "meshing" / "cpu_mesher.py",
    ROOT / "engine" / "meshing" / "gpu_mesher.py",
    ROOT / "engine" / "meshing" / "gpu_mesher_batches.py",
    ROOT / "engine" / "meshing" / "gpu_mesher_common.py",
    ROOT / "engine" / "meshing" / "gpu_mesher_finalize.py",
    ROOT / "engine" / "meshing" / "gpu_mesher_resources.py",
    ROOT / "engine" / "meshing" / "gpu_surface_batches.py",
    ROOT / "engine" / "meshing" / "metal_chunk_mesher.py",
    ROOT / "engine" / "meshing" / "metal_mesher.py",
    ROOT / "engine" / "meshing" / "metal_mesher_async.py",
    ROOT / "engine" / "meshing" / "metal_mesher_cache.py",
    ROOT / "engine" / "meshing" / "metal_mesher_common.py",
    ROOT / "engine" / "meshing" / "metal_mesher_finalize.py",
    ROOT / "engine" / "meshing_types.py",
    ROOT / "engine" / "pipelines" / "chunk_pipeline.py",
    ROOT / "engine" / "pipelines" / "hud_font.py",
    ROOT / "engine" / "pipelines" / "hud_overlay.py",
    ROOT / "engine" / "pipelines" / "profiling_stats.py",
    ROOT / "engine" / "pipelines" / "profiling_summary.py",
    ROOT / "engine" / "pipelines" / "profiling.py",
    ROOT / "engine" / "rendering" / "direct_draw.py",
    ROOT / "engine" / "rendering" / "frame_encoder.py",
    ROOT / "engine" / "rendering" / "gpu_resource_buffers.py",
    ROOT / "engine" / "rendering" / "gpu_resource_layouts.py",
    ROOT / "engine" / "rendering" / "gpu_resource_pipelines.py",
    ROOT / "engine" / "rendering" / "gpu_resources.py",
    ROOT / "engine" / "rendering" / "postprocess_targets.py",
    ROOT / "engine" / "rendering" / "rc_debug_capture.py",
    ROOT / "engine" / "rendering" / "worldspace_rc.py",
    ROOT / "engine" / "terrain" / "kernels" / "core.py",
    ROOT / "engine" / "terrain" / "kernels" / "materials.py",
    ROOT / "engine" / "terrain" / "kernels" / "noise.py",
    ROOT / "engine" / "terrain" / "kernels" / "numba_compat.py",
    ROOT / "engine" / "terrain" / "kernels" / "surface_mesher.py",
    ROOT / "engine" / "terrain" / "kernels" / "terrain_profile.py",
    ROOT / "engine" / "terrain" / "kernels" / "voxel_ao.py",
    ROOT / "engine" / "terrain" / "kernels" / "voxel_emit.py",
    ROOT / "engine" / "terrain" / "kernels" / "voxel_faces.py",
    ROOT / "engine" / "terrain" / "kernels" / "voxel_fill.py",
    ROOT / "engine" / "terrain" / "kernels" / "voxel_mesher.py",
    ROOT / "engine" / "terrain" / "backends" / "wgpu_terrain_backend.py",
    ROOT / "engine" / "terrain" / "backends" / "wgpu_terrain_batches.py",
    ROOT / "engine" / "terrain" / "backends" / "wgpu_terrain_common.py",
    ROOT / "engine" / "terrain" / "backends" / "wgpu_terrain_voxels.py",
    ROOT / "engine" / "terrain" / "backends" / "metal_terrain_backend.py",
    ROOT / "engine" / "terrain" / "backends" / "metal_terrain_batches.py",
    ROOT / "engine" / "terrain" / "backends" / "metal_terrain_buffers.py",
    ROOT / "engine" / "terrain" / "backends" / "metal_terrain_common.py",
    ROOT / "engine" / "terrain" / "backends" / "metal_terrain_voxels.py",
    ROOT / "engine" / "visibility" / "coord_manager.py",
    ROOT / "engine" / "visibility" / "tile_layout.py",
]

FORBIDDEN_RENDERER_IMPORTS = (
    "from .. import renderer\n",
    "from . import renderer\n",
    "import engine.renderer",
    "from engine import renderer",
    "from .renderer import",
    "from ..renderer import",
)


def test_low_level_modules_do_not_import_renderer_runtime():
    offenders: list[str] = []
    for path in LOW_LEVEL_MODULES:
        source = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_RENDERER_IMPORTS:
            if marker in source:
                offenders.append(f"{path.relative_to(ROOT)} contains {marker!r}")
    assert not offenders, "\n".join(offenders)


def test_render_contract_exports_shared_renderer_constants():
    from engine import render_contract

    assert render_contract.VERTEX_STRIDE == 48
    assert render_contract.VERTEX_COMPONENTS == 12
    assert render_contract.CHUNK_WORLD_SIZE == render_contract.CHUNK_SIZE * render_contract.BLOCK_SIZE
    assert render_contract.chunk_prep_request_budget_cap >= 1
