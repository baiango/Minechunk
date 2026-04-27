from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERING = ROOT / "engine" / "rendering"


def _read(name: str) -> str:
    return (RENDERING / name).read_text(encoding="utf-8")


def test_gpu_resources_is_a_small_ordering_facade() -> None:
    facade = _read("gpu_resources.py")

    assert len(facade.splitlines()) < 60
    assert "create_gpu_buffers(renderer)" in facade
    assert "create_gpu_bind_group_layouts(renderer)" in facade
    assert "create_gpu_pipelines(renderer)" in facade
    assert facade.index("create_gpu_buffers(renderer)") < facade.index("create_gpu_bind_group_layouts(renderer)")
    assert facade.index("create_gpu_bind_group_layouts(renderer)") < facade.index("create_gpu_pipelines(renderer)")


def test_gpu_resource_split_has_separate_owners() -> None:
    buffers = _read("gpu_resource_buffers.py")
    layouts = _read("gpu_resource_layouts.py")
    pipelines = _read("gpu_resource_pipelines.py")

    assert "def create_gpu_buffers" in buffers
    assert "worldspace_rc_update_param_buffers" in buffers
    assert "def create_gpu_bind_group_layouts" in layouts
    assert "worldspace_rc_trace_bind_group_layout" in layouts
    assert "final_present_bind_group_layout" in layouts
    assert "def create_gpu_pipelines" in pipelines
    assert "worldspace_rc_trace_pipeline" in pipelines
    assert "voxel_mesh_count_pipeline" in pipelines
    assert "create_render_pipeline" in pipelines
