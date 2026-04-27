from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "engine" / "renderer.py"
FLY_FORWARD = ROOT / "render_fly_forward_4096_then_exit.py"


def test_renderer_keeps_overridable_render_and_auto_exit_hooks() -> None:
    source = RENDERER.read_text(encoding="utf-8")

    assert "def _submit_render(self, meshes=None):" in source
    assert "return frame_encoder.submit_render(self, meshes=meshes)" in source
    assert "encoder, color_view, render_stats = self._submit_render()" in source

    assert "def _service_auto_exit(self) -> None:" in source
    assert "auto_exit.service_auto_exit(self)" in source
    assert "self._service_auto_exit()" in source


def test_fly_forward_benchmark_uses_the_renderer_hooks() -> None:
    source = FLY_FORWARD.read_text(encoding="utf-8")

    assert "def _submit_render(self, meshes=None):" in source
    assert "super()._submit_render(meshes=meshes)" in source
    assert "def _service_auto_exit(self) -> None:" in source
    assert "Info: fly-forward benchmark" in source
