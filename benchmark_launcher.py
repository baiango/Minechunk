from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_ENTRYPOINT = PROJECT_ROOT / "main.py"


@dataclass(frozen=True)
class EngineDefaults:
    engine: str = "cpu"
    rc_enabled: bool = False
    terrain_zstd_enabled: bool = False
    mesh_zstd_enabled: bool = False
    tile_merging_enabled: bool = False
    terrain_batch_size: int = 128
    mesh_batch_size: int = 128
    chunk_request_budget_cap: int = 8
    fly_speed_mps: float = 4.5


def _load_engine_defaults() -> EngineDefaults:
    """Read launcher defaults from the engine config without importing the renderer."""
    try:
        from engine import renderer_config as cfg
    except Exception:
        return EngineDefaults()

    terrain_batch = int(getattr(cfg, "DEFAULT_MESH_BATCH_SIZE", EngineDefaults.terrain_batch_size))
    return EngineDefaults(
        engine=str(getattr(cfg, "engine_mode", EngineDefaults.engine)),
        rc_enabled=bool(getattr(cfg, "RADIANCE_CASCADES_ENABLED", EngineDefaults.rc_enabled)),
        terrain_zstd_enabled=bool(getattr(cfg, "TERRAIN_ZSTD_ENABLED", EngineDefaults.terrain_zstd_enabled)),
        mesh_zstd_enabled=bool(getattr(cfg, "MESH_ZSTD_ENABLED", EngineDefaults.mesh_zstd_enabled)),
        tile_merging_enabled=bool(getattr(cfg, "TILE_MERGING_ENABLED", EngineDefaults.tile_merging_enabled)),
        terrain_batch_size=terrain_batch,
        # The renderer defaults mesh_batch_size to terrain_batch_size when the
        # CLI leaves --mesh-batch-size unset. Mirror that default in the launcher.
        mesh_batch_size=terrain_batch,
        chunk_request_budget_cap=int(getattr(cfg, "chunk_prep_request_budget_cap", EngineDefaults.chunk_request_budget_cap)),
        fly_speed_mps=float(getattr(cfg, "BASE_FLY_SPEED", EngineDefaults.fly_speed_mps)),
    )


ENGINE_DEFAULTS = _load_engine_defaults()
ENGINE_CHOICES = ("cpu", "wgpu", "metal")
ENGINE_DEFAULT_LABEL = f"engine default ({ENGINE_DEFAULTS.engine})"
RC_DEFAULT_LABEL = f"engine default ({'on' if ENGINE_DEFAULTS.rc_enabled else 'off'})"
RC_CHOICES = (RC_DEFAULT_LABEL, "on", "off")


@dataclass(frozen=True)
class LauncherConfig:
    name: str
    mode: str
    # None means: leave the corresponding main.py CLI flag unset and let
    # engine/renderer_config.py or the renderer constructor default decide.
    engine: str | None = None
    rc_enabled: bool | None = None
    terrain_zstd_enabled: bool = ENGINE_DEFAULTS.terrain_zstd_enabled
    mesh_zstd_enabled: bool = ENGINE_DEFAULTS.mesh_zstd_enabled
    tile_merging_enabled: bool = ENGINE_DEFAULTS.tile_merging_enabled
    seed: int = 1337
    view_x: int = 16
    view_y: int = 16
    view_z: int = 16
    terrain_batch_size: int | None = None
    mesh_batch_size: int | None = None
    chunk_request_budget_cap: int | None = None
    freeze_view_origin: bool = True
    freeze_camera: bool = True
    exit_when_view_ready: bool = True
    fly_speed_mps: float = ENGINE_DEFAULTS.fly_speed_mps
    target_rendered_chunks: int = 4096
    status_log_interval_s: float = 1.0
    allow_metal_fallback: bool = False
    start_profiling_hud: bool = False

    @property
    def view_dimensions(self) -> tuple[int, int, int]:
        return (int(self.view_x), int(self.view_y), int(self.view_z))


PRESETS: dict[str, LauncherConfig] = {
    "fixed_8x8x8": LauncherConfig(
        name="Fixed view 8×8×8, then exit",
        mode="fixed",
        view_x=8,
        view_y=8,
        view_z=8,
        freeze_view_origin=True,
        freeze_camera=True,
        exit_when_view_ready=True,
    ),
    "fixed_16x16x16": LauncherConfig(
        name="Fixed view 16×16×16, then exit",
        mode="fixed",
        view_x=16,
        view_y=16,
        view_z=16,
        freeze_view_origin=True,
        freeze_camera=True,
        exit_when_view_ready=True,
    ),
    "fly_forward_4096": LauncherConfig(
        name="Fly forward until 4096 unique chunks render",
        mode="fly_forward",
        view_x=16,
        view_y=16,
        view_z=16,
        freeze_view_origin=False,
        freeze_camera=False,
        exit_when_view_ready=False,
        target_rendered_chunks=4096,
        status_log_interval_s=1.0,
    ),
    "interactive_16x16x16": LauncherConfig(
        name="Interactive 16×16×16 stress window",
        mode="fixed",
        view_x=16,
        view_y=16,
        view_z=16,
        freeze_view_origin=False,
        freeze_camera=False,
        exit_when_view_ready=False,
    ),
    "normal_window": LauncherConfig(
        name="Normal Minechunk window",
        mode="interactive",
        freeze_view_origin=False,
        freeze_camera=False,
        exit_when_view_ready=False,
    ),
}

MODE_LABELS = {
    "interactive": "Normal Minechunk window",
    "fixed": "Fixed view box",
    "fly_forward": "Fly forward until target chunks",
}
MODE_CHOICES = tuple(MODE_LABELS)


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _positive_int(value, field_name)


def _non_negative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be 0 or greater")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _normalize_optional_engine(value: str | None) -> str | None:
    if value is None:
        return None
    engine = str(value).strip().lower()
    if not engine or engine.startswith("engine default") or engine == "default":
        return None
    if engine not in ENGINE_CHOICES:
        raise ValueError(f"engine must be one of: {', '.join(ENGINE_CHOICES)}, or engine default")
    return engine


def validate_config(config: LauncherConfig) -> LauncherConfig:
    mode = str(config.mode).strip().lower().replace("-", "_")
    if mode not in MODE_CHOICES:
        raise ValueError(f"mode must be one of: {', '.join(MODE_CHOICES)}")
    engine = _normalize_optional_engine(config.engine)

    return replace(
        config,
        mode=mode,
        engine=engine,
        seed=_non_negative_int(config.seed, "seed"),
        view_x=_positive_int(config.view_x, "view X"),
        view_y=_positive_int(config.view_y, "view Y"),
        view_z=_positive_int(config.view_z, "view Z"),
        terrain_batch_size=_optional_positive_int(config.terrain_batch_size, "terrain batch size"),
        mesh_batch_size=_optional_positive_int(config.mesh_batch_size, "mesh batch size"),
        chunk_request_budget_cap=_optional_positive_int(config.chunk_request_budget_cap, "chunk request budget cap"),
        fly_speed_mps=_positive_float(config.fly_speed_mps, "fly speed"),
        target_rendered_chunks=_positive_int(config.target_rendered_chunks, "target rendered chunks"),
        status_log_interval_s=_positive_float(config.status_log_interval_s, "status log interval"),
    )


def _parse_view_dimensions(value: str) -> tuple[int, int, int]:
    raw = value.strip().lower().replace("×", "x").replace(",", "x")
    parts = [part.strip() for part in raw.split("x") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("view dimensions must look like 16x16x16")
    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("view dimensions must contain integers") from exc
    if any(value <= 0 for value in dims):
        raise argparse.ArgumentTypeError("view dimensions must be greater than 0")
    return dims  # type: ignore[return-value]


def _bool_flag(command: list[str], enabled: bool, flag_name: str) -> None:
    command.append(f"--{flag_name}" if enabled else f"--no-{flag_name}")


def _optional_int_cli(command: list[str], flag_name: str, value: int | None) -> None:
    if value is not None:
        command.extend([f"--{flag_name}", str(int(value))])


def build_entrypoint_command(config: LauncherConfig) -> list[str]:
    """Build the exact CLI command the launcher runs."""
    config = validate_config(config)
    command = [sys.executable, str(MAIN_ENTRYPOINT)]
    if config.engine is not None:
        command.extend(["--engine", config.engine])
    if config.rc_enabled is not None:
        _bool_flag(command, bool(config.rc_enabled), "rc")
    _bool_flag(command, bool(config.terrain_zstd_enabled), "terrain-zstd")
    _bool_flag(command, bool(config.mesh_zstd_enabled), "mesh-zstd")
    _bool_flag(command, bool(config.tile_merging_enabled), "tile-merge")
    command.extend(["--seed", str(config.seed)])
    command.extend(["--benchmark-mode", config.mode])

    if config.mode != "interactive":
        command.extend(["--fixed-view", f"{config.view_x}x{config.view_y}x{config.view_z}"])
        _bool_flag(command, bool(config.freeze_view_origin), "freeze-view-origin")
        _bool_flag(command, bool(config.freeze_camera), "freeze-camera")
        _bool_flag(command, bool(config.exit_when_view_ready), "exit-when-view-ready")
    else:
        command.extend(["--no-freeze-view-origin", "--no-freeze-camera", "--no-exit-when-view-ready"])

    _optional_int_cli(command, "terrain-batch-size", config.terrain_batch_size)
    _optional_int_cli(command, "mesh-batch-size", config.mesh_batch_size)
    _optional_int_cli(command, "chunk-request-budget-cap", config.chunk_request_budget_cap)

    if config.mode == "fly_forward":
        command.extend(["--fly-speed-mps", str(config.fly_speed_mps)])
        command.extend(["--target-rendered-chunks", str(config.target_rendered_chunks)])
        command.extend(["--status-log-interval-s", str(config.status_log_interval_s)])

    if config.allow_metal_fallback:
        command.append("--allow-metal-fallback")
    if config.start_profiling_hud:
        command.append("--start-profiling-hud")
    return command


def command_preview(config: LauncherConfig) -> str:
    return shlex.join(build_entrypoint_command(config))


def _entrypoint_popen_kwargs() -> dict[str, Any]:
    """Return subprocess options that keep main.py independent of the launcher."""
    kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "env": os.environ.copy(),
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if creationflags:
            kwargs["creationflags"] = creationflags
    else:
        # Put the engine in a new process session so closing the launcher does
        # not deliver the launcher's terminal/control signals to the renderer.
        kwargs["start_new_session"] = True
    return kwargs


def spawn_entrypoint(config: LauncherConfig) -> subprocess.Popen[Any]:
    """Spawn main.py without waiting for the renderer/window to exit."""
    command = build_entrypoint_command(config)
    print("Info: spawning Minechunk CLI:", command_preview(config), file=sys.stderr)
    process = subprocess.Popen(command, **_entrypoint_popen_kwargs())
    print(f"Info: Minechunk process started pid={process.pid}; launcher is not waiting.", file=sys.stderr)
    return process


def launch_entrypoint(config: LauncherConfig, *, wait: bool = False) -> int:
    process = spawn_entrypoint(config)
    if not wait:
        return 0
    return int(process.wait())


def _format_optional(value: int | None, default_value: int) -> str:
    return f"default {default_value}" if value is None else str(value)


class LauncherApp:
    def __init__(self, default_config: LauncherConfig) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("Minechunk CLI launcher")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.processes: list[subprocess.Popen[Any]] = []
        self._last_exit_status = "No engine process launched yet."
        self._poll_job_scheduled = False

        self.preset_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.engine_var = tk.StringVar()
        self.rc_mode_var = tk.StringVar()
        self.terrain_zstd_var = tk.BooleanVar()
        self.mesh_zstd_var = tk.BooleanVar()
        self.tile_merging_var = tk.BooleanVar()
        self.seed_var = tk.StringVar()
        self.view_x_var = tk.StringVar()
        self.view_y_var = tk.StringVar()
        self.view_z_var = tk.StringVar()
        self.terrain_batch_var = tk.StringVar()
        self.mesh_batch_var = tk.StringVar()
        self.chunk_budget_var = tk.StringVar()
        self.freeze_origin_var = tk.BooleanVar()
        self.freeze_camera_var = tk.BooleanVar()
        self.exit_ready_var = tk.BooleanVar()
        self.fly_speed_var = tk.StringVar()
        self.target_chunks_var = tk.StringVar()
        self.status_interval_var = tk.StringVar()
        self.allow_metal_fallback_var = tk.BooleanVar()
        self.profiling_hud_var = tk.BooleanVar()
        self.summary_var = tk.StringVar()
        self.command_var = tk.StringVar()
        self.status_var = tk.StringVar(value=self._last_exit_status)

        self._fixed_controls: list[Any] = []
        self._fly_controls: list[Any] = []
        self._benchmark_only_controls: list[Any] = []
        self._build_widgets()
        self._load_config(default_config)

    def _build_widgets(self) -> None:
        ttk = self.ttk
        root = self.root
        pad = {"padx": 8, "pady": 4}

        frame = ttk.Frame(root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        title = ttk.Label(frame, text="Minechunk CLI presets", font=("TkDefaultFont", 13, "bold"))
        title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="Preset").grid(row=1, column=0, sticky="w", **pad)
        preset = ttk.Combobox(
            frame,
            textvariable=self.preset_var,
            values=[f"{key} — {cfg.name}" for key, cfg in PRESETS.items()],
            width=42,
            state="readonly",
        )
        preset.grid(row=1, column=1, columnspan=3, sticky="ew", **pad)
        preset.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Label(frame, text="Mode").grid(row=2, column=0, sticky="w", **pad)
        mode = ttk.Combobox(
            frame,
            textvariable=self.mode_var,
            values=[f"{key} — {label}" for key, label in MODE_LABELS.items()],
            width=28,
            state="readonly",
        )
        mode.grid(row=2, column=1, sticky="ew", **pad)
        mode.bind("<<ComboboxSelected>>", lambda _event: self._sync_mode_state())

        ttk.Label(frame, text="Engine").grid(row=2, column=2, sticky="w", **pad)
        engine = ttk.Combobox(frame, textvariable=self.engine_var, values=(ENGINE_DEFAULT_LABEL, *ENGINE_CHOICES), width=22, state="readonly")
        engine.grid(row=2, column=3, sticky="ew", **pad)
        engine.bind("<<ComboboxSelected>>", lambda _event: self._update_summary())

        ttk.Label(frame, text="Seed").grid(row=3, column=0, sticky="w", **pad)
        self._entry(frame, self.seed_var, 1, 3, width=12)

        ttk.Label(frame, text="Radiance Cascades").grid(row=3, column=2, sticky="w", **pad)
        rc_mode = ttk.Combobox(frame, textvariable=self.rc_mode_var, values=RC_CHOICES, width=22, state="readonly")
        rc_mode.grid(row=3, column=3, sticky="ew", **pad)
        rc_mode.bind("<<ComboboxSelected>>", lambda _event: self._update_summary())

        ttk.Label(frame, text="View chunks X / Y / Z").grid(row=4, column=0, sticky="w", **pad)
        view_row = ttk.Frame(frame)
        view_row.grid(row=4, column=1, columnspan=3, sticky="w", **pad)
        self._benchmark_only_controls.append(self._entry(view_row, self.view_x_var, 0, 0, width=7))
        ttk.Label(view_row, text="×").grid(row=0, column=1, padx=4)
        self._benchmark_only_controls.append(self._entry(view_row, self.view_y_var, 0, 2, width=7))
        ttk.Label(view_row, text="×").grid(row=0, column=3, padx=4)
        self._benchmark_only_controls.append(self._entry(view_row, self.view_z_var, 0, 4, width=7))

        ttk.Label(frame, text="Terrain batch").grid(row=5, column=0, sticky="w", **pad)
        self._entry(frame, self.terrain_batch_var, 1, 5, width=12)
        ttk.Label(frame, text="Mesh batch").grid(row=5, column=2, sticky="w", **pad)
        self._entry(frame, self.mesh_batch_var, 3, 5, width=12)

        ttk.Label(frame, text="Chunk request cap").grid(row=6, column=0, sticky="w", **pad)
        self._entry(frame, self.chunk_budget_var, 1, 6, width=12)
        ttk.Label(
            frame,
            text=(
                "blank = engine defaults: "
                f"terrain {ENGINE_DEFAULTS.terrain_batch_size}, "
                f"mesh {ENGINE_DEFAULTS.mesh_batch_size}, "
                f"cap {ENGINE_DEFAULTS.chunk_request_budget_cap}"
            ),
            foreground="#666",
        ).grid(row=6, column=2, columnspan=2, sticky="w", **pad)

        fixed_box = ttk.LabelFrame(frame, text="Fixed view options", padding=8)
        fixed_box.grid(row=7, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 4))
        self._fixed_controls.append(ttk.Checkbutton(fixed_box, text="Freeze view origin", variable=self.freeze_origin_var, command=self._update_summary))
        self._fixed_controls[-1].grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self._fixed_controls.append(ttk.Checkbutton(fixed_box, text="Freeze camera", variable=self.freeze_camera_var, command=self._update_summary))
        self._fixed_controls[-1].grid(row=0, column=1, sticky="w", padx=6, pady=2)
        self._fixed_controls.append(ttk.Checkbutton(fixed_box, text="Exit when view is ready", variable=self.exit_ready_var, command=self._update_summary))
        self._fixed_controls[-1].grid(row=0, column=2, sticky="w", padx=6, pady=2)

        fly_box = ttk.LabelFrame(frame, text="Fly-forward options", padding=8)
        fly_box.grid(row=8, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
        ttk.Label(fly_box, text="Speed m/s").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self._fly_controls.append(self._entry(fly_box, self.fly_speed_var, 1, 0, width=10))
        ttk.Label(fly_box, text="Target chunks").grid(row=0, column=2, sticky="w", padx=6, pady=2)
        self._fly_controls.append(self._entry(fly_box, self.target_chunks_var, 3, 0, width=10))
        ttk.Label(fly_box, text="Log interval s").grid(row=0, column=4, sticky="w", padx=6, pady=2)
        self._fly_controls.append(self._entry(fly_box, self.status_interval_var, 5, 0, width=10))

        misc_box = ttk.LabelFrame(frame, text="Runtime options", padding=8)
        misc_box.grid(row=9, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
        ttk.Checkbutton(misc_box, text="Allow Metal fallback", variable=self.allow_metal_fallback_var, command=self._update_summary).grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(misc_box, text="Enable profiling HUD at start", variable=self.profiling_hud_var, command=self._update_summary).grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(misc_box, text="Compress terrain chunks with zstd", variable=self.terrain_zstd_var, command=self._update_summary).grid(row=0, column=2, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(misc_box, text="Compress offscreen mesh slabs with zstd", variable=self.mesh_zstd_var, command=self._update_summary).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        ttk.Checkbutton(misc_box, text="Enable merged tile GPU buffers", variable=self.tile_merging_var, command=self._update_summary).grid(row=1, column=2, sticky="w", padx=6, pady=2)

        summary = ttk.Label(frame, textvariable=self.summary_var, foreground="#444")
        summary.grid(row=10, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 4))

        command = ttk.Entry(frame, textvariable=self.command_var, width=92, state="readonly")
        command.grid(row=11, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))

        status = ttk.Label(frame, textvariable=self.status_var, foreground="#555")
        status.grid(row=12, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

        buttons = ttk.Frame(frame)
        buttons.grid(row=13, column=0, columnspan=4, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Start main.py detached", command=self._on_start).grid(row=0, column=0, padx=5)
        ttk.Button(buttons, text="Close launcher", command=self._on_close).grid(row=0, column=1, padx=5)

        for column in range(4):
            frame.columnconfigure(column, weight=1)

    def _entry(self, parent: Any, variable: Any, column: int, row: int, width: int = 12):
        entry = self.ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column, sticky="w", padx=6, pady=2)
        entry.bind("<KeyRelease>", lambda _event: self._update_summary())
        return entry

    def _preset_key_from_var(self) -> str:
        raw = self.preset_var.get().split(" — ", 1)[0].strip()
        return raw if raw in PRESETS else "normal_window"

    def _mode_key_from_var(self) -> str:
        raw = self.mode_var.get().split(" — ", 1)[0].strip().lower()
        return raw if raw in MODE_CHOICES else "fixed"

    def _engine_from_var(self) -> str | None:
        return _normalize_optional_engine(self.engine_var.get())

    def _rc_from_var(self) -> bool | None:
        raw = self.rc_mode_var.get().strip().lower()
        if not raw or raw.startswith("engine default") or raw == "default":
            return None
        if raw == "on":
            return True
        if raw == "off":
            return False
        raise ValueError("Radiance Cascades must be engine default, on, or off")

    @staticmethod
    def _optional_int_from_var(variable: Any, field_name: str) -> int | None:
        text = str(variable.get()).strip()
        if not text:
            return None
        return _positive_int(text, field_name)

    def _on_preset_selected(self, _event: Any = None) -> None:
        self._load_config(PRESETS[self._preset_key_from_var()])

    def _load_config(self, config: LauncherConfig) -> None:
        key = next((key for key, preset in PRESETS.items() if preset == config), None)
        if key is None:
            key = "normal_window"
        self.preset_var.set(f"{key} — {PRESETS[key].name}")
        self.mode_var.set(f"{config.mode} — {MODE_LABELS.get(config.mode, config.mode)}")
        self.engine_var.set(ENGINE_DEFAULT_LABEL if config.engine is None else str(config.engine))
        if config.rc_enabled is None:
            self.rc_mode_var.set(RC_DEFAULT_LABEL)
        else:
            self.rc_mode_var.set("on" if config.rc_enabled else "off")
        self.seed_var.set(str(config.seed))
        self.view_x_var.set(str(config.view_x))
        self.view_y_var.set(str(config.view_y))
        self.view_z_var.set(str(config.view_z))
        self.terrain_batch_var.set("" if config.terrain_batch_size is None else str(config.terrain_batch_size))
        self.mesh_batch_var.set("" if config.mesh_batch_size is None else str(config.mesh_batch_size))
        self.chunk_budget_var.set("" if config.chunk_request_budget_cap is None else str(config.chunk_request_budget_cap))
        self.freeze_origin_var.set(bool(config.freeze_view_origin))
        self.freeze_camera_var.set(bool(config.freeze_camera))
        self.exit_ready_var.set(bool(config.exit_when_view_ready))
        self.fly_speed_var.set(str(config.fly_speed_mps))
        self.target_chunks_var.set(str(config.target_rendered_chunks))
        self.status_interval_var.set(str(config.status_log_interval_s))
        self.allow_metal_fallback_var.set(bool(config.allow_metal_fallback))
        self.profiling_hud_var.set(bool(config.start_profiling_hud))
        self.terrain_zstd_var.set(bool(config.terrain_zstd_enabled))
        self.mesh_zstd_var.set(bool(config.mesh_zstd_enabled))
        self.tile_merging_var.set(bool(config.tile_merging_enabled))
        self._sync_mode_state()
        self._update_summary()

    def _config_from_vars(self) -> LauncherConfig:
        return validate_config(
            LauncherConfig(
                name="Custom GUI launch",
                mode=self._mode_key_from_var(),
                engine=self._engine_from_var(),
                rc_enabled=self._rc_from_var(),
                seed=int(self.seed_var.get()),
                view_x=int(self.view_x_var.get()),
                view_y=int(self.view_y_var.get()),
                view_z=int(self.view_z_var.get()),
                terrain_batch_size=self._optional_int_from_var(self.terrain_batch_var, "terrain batch size"),
                mesh_batch_size=self._optional_int_from_var(self.mesh_batch_var, "mesh batch size"),
                chunk_request_budget_cap=self._optional_int_from_var(self.chunk_budget_var, "chunk request budget cap"),
                freeze_view_origin=bool(self.freeze_origin_var.get()),
                freeze_camera=bool(self.freeze_camera_var.get()),
                exit_when_view_ready=bool(self.exit_ready_var.get()),
                fly_speed_mps=float(self.fly_speed_var.get()),
                target_rendered_chunks=int(self.target_chunks_var.get()),
                status_log_interval_s=float(self.status_interval_var.get()),
                allow_metal_fallback=bool(self.allow_metal_fallback_var.get()),
                start_profiling_hud=bool(self.profiling_hud_var.get()),
                terrain_zstd_enabled=bool(self.terrain_zstd_var.get()),
                mesh_zstd_enabled=bool(self.mesh_zstd_var.get()),
                tile_merging_enabled=bool(self.tile_merging_var.get()),
            )
        )

    def _sync_mode_state(self) -> None:
        mode = self._mode_key_from_var()
        for widget in self._benchmark_only_controls:
            widget.configure(state="normal" if mode != "interactive" else "disabled")
        for widget in self._fixed_controls:
            widget.configure(state="normal" if mode == "fixed" else "disabled")
        for widget in self._fly_controls:
            widget.configure(state="normal" if mode == "fly_forward" else "disabled")
        self._update_summary()

    def _update_summary(self) -> None:
        try:
            config = self._config_from_vars()
            command_text = command_preview(config)
        except Exception:
            self.summary_var.set("Tune the fields, then Start main.py. Invalid fields will be reported before launch.")
            self.command_var.set("")
            return
        dims = "×".join(str(value) for value in config.view_dimensions)
        engine_text = config.engine.upper() if config.engine is not None else f"engine default {ENGINE_DEFAULTS.engine.upper()}"
        rc_text = "RC default " + ("on" if ENGINE_DEFAULTS.rc_enabled else "off") if config.rc_enabled is None else ("RC on" if config.rc_enabled else "RC off")
        terrain_zstd_text = "terrain zstd on" if config.terrain_zstd_enabled else "terrain zstd off"
        mesh_zstd_text = "mesh zstd on" if config.mesh_zstd_enabled else "mesh zstd off"
        tile_merge_text = "tile merge on" if config.tile_merging_enabled else "tile merge off"
        terrain_text = _format_optional(config.terrain_batch_size, ENGINE_DEFAULTS.terrain_batch_size)
        mesh_text = _format_optional(config.mesh_batch_size, ENGINE_DEFAULTS.mesh_batch_size)
        cap_text = _format_optional(config.chunk_request_budget_cap, ENGINE_DEFAULTS.chunk_request_budget_cap)
        if config.mode == "fly_forward":
            text = (
                f"{engine_text} fly CLI run: view {dims}, {config.fly_speed_mps:g} m/s, "
                f"target {config.target_rendered_chunks} chunks, cap {cap_text}, {rc_text}, {terrain_zstd_text}, {mesh_zstd_text}, {tile_merge_text}."
            )
        elif config.mode == "fixed":
            exit_text = "auto-exit" if config.exit_when_view_ready else "interactive"
            text = (
                f"{engine_text} fixed CLI run: view {dims}, terrain batch {terrain_text}, "
                f"mesh batch {mesh_text}, cap {cap_text}, {exit_text}, {rc_text}, {terrain_zstd_text}, {mesh_zstd_text}, {tile_merge_text}."
            )
        else:
            text = f"{engine_text} normal main.py launch, {rc_text}, {terrain_zstd_text}, {mesh_zstd_text}, {tile_merge_text}."
        self.summary_var.set(text)
        self.command_var.set(command_text)

    def _on_start(self) -> None:
        from tkinter import messagebox

        try:
            config = self._config_from_vars()
            process = spawn_entrypoint(config)
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc), parent=self.root)
            return

        self.processes.append(process)
        self.status_var.set(
            f"Started detached engine PID {process.pid}. Launcher stays open and responsive."
        )
        self._schedule_process_poll()

    def _schedule_process_poll(self) -> None:
        if self._poll_job_scheduled:
            return
        self._poll_job_scheduled = True
        self.root.after(750, self._poll_processes)

    def _poll_processes(self) -> None:
        self._poll_job_scheduled = False
        running: list[str] = []
        still_running: list[subprocess.Popen[Any]] = []

        for process in self.processes:
            return_code = process.poll()
            if return_code is None:
                running.append(str(process.pid))
                still_running.append(process)
            else:
                self._last_exit_status = f"Engine PID {process.pid} exited with code {return_code}. Launcher is still open."

        self.processes = still_running
        if running:
            self.status_var.set(f"Running detached engine PID(s): {', '.join(running)}.")
            self._schedule_process_poll()
        else:
            self.status_var.set(self._last_exit_status)

    def _on_close(self) -> None:
        # Do not terminate renderer children here. They were launched in their
        # own process group/session so the launcher can close independently.
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_gui_launcher(default_config: LauncherConfig) -> None:
    app = LauncherApp(default_config)
    app.run()


def _config_from_args(default_config: LauncherConfig, args: argparse.Namespace) -> LauncherConfig:
    config = default_config
    changes: dict[str, Any] = {}
    for arg_name, field_name in (
        ("mode", "mode"),
        ("engine", "engine"),
        ("rc", "rc_enabled"),
        ("terrain_zstd", "terrain_zstd_enabled"),
        ("mesh_zstd", "mesh_zstd_enabled"),
        ("tile_merge", "tile_merging_enabled"),
        ("seed", "seed"),
        ("terrain_batch_size", "terrain_batch_size"),
        ("mesh_batch_size", "mesh_batch_size"),
        ("chunk_request_budget_cap", "chunk_request_budget_cap"),
        ("fly_speed_mps", "fly_speed_mps"),
        ("target_rendered_chunks", "target_rendered_chunks"),
        ("status_log_interval_s", "status_log_interval_s"),
        ("freeze_view_origin", "freeze_view_origin"),
        ("freeze_camera", "freeze_camera"),
        ("exit_when_view_ready", "exit_when_view_ready"),
        ("allow_metal_fallback", "allow_metal_fallback"),
        ("start_profiling_hud", "start_profiling_hud"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            changes[field_name] = value

    if args.view is not None:
        view_x, view_y, view_z = args.view
        changes.update({"view_x": view_x, "view_y": view_y, "view_z": view_z})

    if changes:
        config = replace(config, name="Command line launch", **changes)
    return validate_config(config)


def _build_arg_parser(default_preset: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Minechunk presets through main.py, with GUI or headless overrides.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default=default_preset, help="Preset to load before applying overrides.")
    parser.add_argument("--headless", action="store_true", help="Skip the Tk GUI and spawn the selected main.py command immediately.")
    parser.add_argument("--wait", action="store_true", help="After spawning main.py, wait for it and return its exit code. Omit this to keep the launcher non-blocking.")
    parser.add_argument("--mode", choices=MODE_CHOICES, default=None, help="Run mode override.")
    parser.add_argument("--engine", choices=ENGINE_CHOICES, default=None, help=f"Backend override. Omit for engine default ({ENGINE_DEFAULTS.engine}).")
    parser.add_argument("--rc", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable Radiance Cascades for this run. Use --no-rc to profile without RC. Omit for engine default.")
    parser.add_argument("--terrain-zstd", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable zstd level-1 compression for CPU-side terrain chunk payloads. Default is off.")
    parser.add_argument("--mesh-zstd", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable experimental zstd readback compression for offscreen mesh buffers. Default is off.")
    parser.add_argument("--tile-merge", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable merged visible tile GPU buffers. Default is off to reduce footprint.")
    parser.add_argument("--seed", type=int, default=None, help="World seed.")
    parser.add_argument("--view", type=_parse_view_dimensions, default=None, help="View dimensions, for example 16x16x16.")
    parser.add_argument("--terrain-batch-size", type=int, default=None, help=f"Terrain backend poll/batch size. Omit for engine default ({ENGINE_DEFAULTS.terrain_batch_size}).")
    parser.add_argument("--mesh-batch-size", type=int, default=None, help=f"Mesh drain/finalize batch size. Omit for engine default ({ENGINE_DEFAULTS.mesh_batch_size}).")
    parser.add_argument("--chunk-request-budget-cap", type=int, default=None, help=f"Chunk prep request budget cap. Omit for engine default ({ENGINE_DEFAULTS.chunk_request_budget_cap}).")
    parser.add_argument("--fly-speed-mps", type=float, default=None, help=f"Fly-forward benchmark speed in meters per second. Default uses BASE_FLY_SPEED ({ENGINE_DEFAULTS.fly_speed_mps:g}).")
    parser.add_argument("--target-rendered-chunks", type=int, default=None, help="Unique rendered chunk target for fly-forward mode.")
    parser.add_argument("--status-log-interval-s", type=float, default=None, help="Fly-forward status log interval in seconds.")
    parser.add_argument("--freeze-view-origin", action=argparse.BooleanOptionalAction, default=None, help="Freeze the fixed benchmark chunk origin.")
    parser.add_argument("--freeze-camera", action=argparse.BooleanOptionalAction, default=None, help="Freeze the camera during fixed benchmarks.")
    parser.add_argument("--exit-when-view-ready", action=argparse.BooleanOptionalAction, default=None, help="Close fixed benchmarks after the visible view is ready.")
    parser.add_argument("--allow-metal-fallback", action=argparse.BooleanOptionalAction, default=None, help="Allow Metal mode to fall back instead of failing loudly.")
    parser.add_argument("--start-profiling-hud", action=argparse.BooleanOptionalAction, default=None, help="Enable the profiling HUD immediately after renderer creation.")
    parser.add_argument("--print-command", action="store_true", help="Print the generated main.py command and exit without launching.")
    return parser


def main(default_preset: str = "normal_window") -> None:
    if default_preset not in PRESETS:
        default_preset = "normal_window"
    parser = _build_arg_parser(default_preset)
    args = parser.parse_args()
    config = _config_from_args(PRESETS[args.preset], args)

    if args.print_command:
        print(command_preview(config))
        return

    if not args.headless:
        try:
            run_gui_launcher(config)
            return
        except Exception as exc:
            print(f"Warning: launcher GUI could not start ({exc!s}); spawning selected preset headless.", file=sys.stderr)

    raise SystemExit(launch_entrypoint(config, wait=bool(args.wait)))


if __name__ == "__main__":
    main()
