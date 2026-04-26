from __future__ import annotations

from pathlib import Path

from engine.shader_loader import load_shader_text


ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "engine" / "shaders"
SHADER_ASSETS = {
    "compute.wgsl",
    "final_blit.wgsl",
    "gi_cascade.wgsl",
    "gi_compose.wgsl",
    "gi_gbuffer.wgsl",
    "gpu_visibility.wgsl",
    "hud.wgsl",
    "render.wgsl",
    "terrain_surface.metal",
    "terrain_surface.wgsl",
    "voxel_mesh_batch.wgsl",
    "voxel_surface_expand.wgsl",
    "worldspace_rc_filter.wgsl",
    "worldspace_rc_trace.wgsl",
}


def test_shader_assets_are_checked_in_and_nonempty() -> None:
    names = {path.name for path in SHADER_DIR.iterdir() if path.is_file()}
    missing = SHADER_ASSETS - names
    assert not missing
    for name in SHADER_ASSETS:
        text = (SHADER_DIR / name).read_text(encoding="utf-8")
        assert text.strip(), name


def test_shader_loader_reads_checked_in_assets() -> None:
    assert "@compute" in load_shader_text("gpu_visibility.wgsl")
    assert "kernel" in load_shader_text("terrain_surface.metal")


def test_render_shader_module_stays_a_loader_not_a_shader_dump() -> None:
    source = (ROOT / "engine" / "render_shaders.py").read_text(encoding="utf-8")
    assert len(source.splitlines()) < 260
    assert "WORLDSPACE_RC_TRACE_SHADER = \"\"\"" not in source
    assert "VOXEL_MESH_BATCH_SHADER = \"\"\"" not in source


def test_terrain_backends_load_external_shader_assets() -> None:
    wgpu_backend = (ROOT / "engine" / "terrain" / "backends" / "wgpu_terrain_backend.py").read_text(encoding="utf-8")
    metal_backend = (ROOT / "engine" / "terrain" / "backends" / "metal_terrain_backend.py").read_text(encoding="utf-8")
    assert 'load_shader_text("terrain_surface.wgsl")' in wgpu_backend
    assert 'load_shader_text("terrain_surface.metal")' in metal_backend
