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
    assert "--terrain-caves" in source
    assert "--tile-merge" in source
    assert "--postprocess" in source
    assert "BooleanOptionalAction" in source
    assert "RADIANCE_CASCADES_ENABLED" in source
    assert "TILE_MERGING_ENABLED" in source
    assert "RendererLaunchConfig" in source


def test_main_cache_numba_only_exits_before_renderer_creation(monkeypatch) -> None:
    import main as main_module

    warmed = []

    monkeypatch.setattr(main_module, "_warm_numba_cache", lambda: warmed.append(True))
    monkeypatch.setattr(main_module, "_build_renderer_from_args", lambda _args: (_ for _ in ()).throw(AssertionError("renderer should not be created")))

    parser = main_module._build_arg_parser()
    assert parser.parse_args(["--cache-numba-only"]).cache_numba_only is True

    main_module.main(["--cache-numba-only"])

    assert warmed == [True]


def test_checked_in_engine_default_is_cpu() -> None:
    source = (ROOT / "engine" / "renderer_config.py").read_text(encoding="utf-8")
    launcher = BENCHMARK_LAUNCHER.read_text(encoding="utf-8")

    assert "_DEFAULT_ENGINE_MODE = ENGINE_MODE_CPU" in source
    assert 'engine: str = "cpu"' in launcher


def test_rc_and_tile_merging_default_off_with_launcher_toggle() -> None:
    from benchmark_launcher import (
        ENGINE_DEFAULTS,
        LauncherConfig,
        PRESETS,
        _build_arg_parser as build_launcher_arg_parser,
        _config_from_args,
        build_entrypoint_command,
        main as launcher_main,
    )
    from engine import renderer_config as cfg
    from main import _build_arg_parser as build_main_arg_parser

    assert cfg.RADIANCE_CASCADES_ENABLED is False
    assert cfg.TILE_MERGING_ENABLED is False
    assert ENGINE_DEFAULTS.rc_enabled is False
    assert ENGINE_DEFAULTS.tile_merging_enabled is False

    main_parser = build_main_arg_parser()
    assert main_parser.parse_args([]).rc is None
    assert main_parser.parse_args([]).tile_merge is None
    assert main_parser.parse_args(["--tile-merge"]).tile_merge is True
    assert main_parser.parse_args(["--no-tile-merge"]).tile_merge is False
    assert main_parser.parse_args([]).postprocess is None
    assert main_parser.parse_args(["--postprocess"]).postprocess is True
    assert main_parser.parse_args(["--no-postprocess"]).postprocess is False
    assert main_parser.parse_args([]).terrain_caves is None
    assert main_parser.parse_args(["--terrain-caves"]).terrain_caves is True
    assert main_parser.parse_args(["--no-terrain-caves"]).terrain_caves is False

    default_command = build_entrypoint_command(LauncherConfig(name="test", mode="interactive"))
    enabled_command = build_entrypoint_command(
        LauncherConfig(name="test", mode="interactive", tile_merging_enabled=True)
    )

    assert "--no-tile-merge" in default_command
    assert "--tile-merge" in enabled_command
    assert "--rc" not in default_command
    assert "--no-rc" not in default_command

    launcher_parser = build_launcher_arg_parser("normal_window")
    default_launcher_args = launcher_parser.parse_args([])
    default_launcher_config = _config_from_args(PRESETS[default_launcher_args.preset], default_launcher_args)
    assert launcher_main.__defaults__ == ("normal_window",)
    assert default_launcher_args.preset == "normal_window"
    assert default_launcher_config.mode == "interactive"
    assert default_launcher_config.freeze_view_origin is False
    assert default_launcher_config.freeze_camera is False
    assert default_launcher_config.exit_when_view_ready is False

    launcher_args = launcher_parser.parse_args(["--tile-merge"])
    launcher_config = _config_from_args(LauncherConfig(name="test", mode="interactive"), launcher_args)
    assert launcher_config.tile_merging_enabled is True
