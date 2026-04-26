from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LOW_LEVEL_MODULES = [
    ROOT / "engine" / "cache" / "mesh_allocator.py",
    ROOT / "engine" / "debug_capture.py",
    ROOT / "engine" / "meshing" / "cpu_mesher.py",
    ROOT / "engine" / "meshing" / "gpu_mesher.py",
    ROOT / "engine" / "meshing" / "metal_mesher.py",
    ROOT / "engine" / "meshing_types.py",
    ROOT / "engine" / "pipelines" / "chunk_pipeline.py",
    ROOT / "engine" / "pipelines" / "profiling.py",
    ROOT / "engine" / "visibility" / "coord_manager.py",
]

FORBIDDEN_RENDERER_IMPORTS = (
    "from .. import renderer",
    "from . import renderer",
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
