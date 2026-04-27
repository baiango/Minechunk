from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "engine" / "renderer.py"
BENCHMARK_RUNTIME = ROOT / "engine" / "benchmark_runtime.py"
BENCHMARK_LAUNCHER = ROOT / "benchmark_launcher.py"
MAIN = ROOT / "main.py"
REMOVED_BENCHMARK_SCRIPTS = [
    ROOT / "render_8x8x8_then_exit.py",
    ROOT / "render_16x16x16_then_exit.py",
    ROOT / "render_fly_forward_4096_then_exit.py",
]


def test_renderer_keeps_overridable_render_and_auto_exit_hooks() -> None:
    source = RENDERER.read_text(encoding="utf-8")

    assert "def _submit_render(self, meshes=None):" in source
    assert "return frame_encoder.submit_render(self, meshes=meshes)" in source
    assert "encoder, color_view, render_stats = self._submit_render()" in source

    assert "def _service_auto_exit(self) -> None:" in source
    assert "auto_exit.service_auto_exit(self)" in source
    assert "self._service_auto_exit()" in source


def test_fly_forward_cli_benchmark_uses_the_renderer_hooks() -> None:
    source = BENCHMARK_RUNTIME.read_text(encoding="utf-8")

    assert "class FlyingForwardBenchmarkRenderer(TerrainRenderer):" in source
    assert "def _submit_render(self, meshes=None):" in source
    assert "super()._submit_render(meshes=meshes)" in source
    assert "def _service_auto_exit(self) -> None:" in source
    assert "Info: fly-forward benchmark" in source


def test_legacy_benchmark_entrypoints_are_removed() -> None:
    existing = [path for path in REMOVED_BENCHMARK_SCRIPTS if path.exists()]
    assert not existing, "legacy benchmark wrappers should be removed: " + ", ".join(path.name for path in existing)


def test_launcher_runs_main_cli_instead_of_importing_renderer() -> None:
    source = BENCHMARK_LAUNCHER.read_text(encoding="utf-8")

    assert "subprocess.Popen" in source
    assert "spawn_entrypoint" in source
    assert "start_new_session" in source or "CREATE_NEW_PROCESS_GROUP" in source
    assert "MAIN_ENTRYPOINT" in source
    assert "main.py" in source
    assert "--rc" in source
    assert "--no-rc" in source
    assert "--wait" in source
    assert "from engine.renderer" not in source
    assert "from engine import renderer\n" not in source
    assert "TerrainRenderer" not in source


def test_main_exposes_cli_benchmark_and_rc_flags() -> None:
    source = MAIN.read_text(encoding="utf-8")

    assert "--benchmark-mode" in source
    assert "--fixed-view" in source
    assert "--rc" in source
    assert "BooleanOptionalAction" in source
    assert "RADIANCE_CASCADES_ENABLED" in source
    assert "RendererLaunchConfig" in source


def test_checked_in_engine_default_is_cpu() -> None:
    source = (ROOT / "engine" / "renderer_config.py").read_text(encoding="utf-8")
    launcher = BENCHMARK_LAUNCHER.read_text(encoding="utf-8")

    assert "_DEFAULT_ENGINE_MODE = ENGINE_MODE_CPU" in source
    assert 'engine: str = "cpu"' in launcher
