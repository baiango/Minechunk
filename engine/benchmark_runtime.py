from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

from . import renderer_config as cfg

DEFAULT_BENCHMARK_FLY_SPEED_MPS = float(getattr(cfg, "BASE_FLY_SPEED", 20.0))

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


@dataclass(frozen=True)
class RendererLaunchConfig:
    mode: str = "interactive"
    seed: int = 1337
    fixed_view_dimensions: tuple[int, int, int] | None = None
    terrain_batch_size: int | None = None
    mesh_batch_size: int | None = None
    freeze_view_origin: bool = False
    freeze_camera: bool = False
    exit_when_view_ready: bool = False
    fly_speed_mps: float = DEFAULT_BENCHMARK_FLY_SPEED_MPS
    target_rendered_chunks: int = 4096
    status_log_interval_s: float = 1.0


def _normalized_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    if value == "flyforward":
        value = "fly_forward"
    if value not in ("interactive", "fixed", "fly_forward"):
        raise ValueError("mode must be one of: interactive, fixed, fly_forward")
    return value


def _view_dimensions_or_default(config: RendererLaunchConfig) -> tuple[int, int, int] | None:
    mode = _normalized_mode(config.mode)
    if config.fixed_view_dimensions is None:
        if mode in ("fixed", "fly_forward"):
            return (16, 16, 16)
        return None
    dims = tuple(max(1, int(value)) for value in config.fixed_view_dimensions)
    if len(dims) != 3:
        raise ValueError("fixed_view_dimensions must contain exactly 3 integers")
    return dims  # type: ignore[return-value]


def _window_title_for(config: RendererLaunchConfig) -> str:
    mode = _normalized_mode(config.mode)
    dims = _view_dimensions_or_default(config)
    dim_text = "default view" if dims is None else "×".join(str(value) for value in dims)
    if mode == "fly_forward":
        return f"Minechunk - fly-forward {config.target_rendered_chunks} chunk CLI run ({dim_text})"
    if mode == "fixed":
        suffix = "auto-exit" if config.exit_when_view_ready else "interactive"
        return f"Minechunk - fixed {dim_text} CLI run ({suffix})"
    return "Minechunk"


def _apply_window_title(renderer: Any, config: RendererLaunchConfig) -> None:
    title = _window_title_for(config)
    renderer.base_title = title
    try:
        renderer.canvas.set_title(title)
    except Exception:
        pass


def _renderer_kwargs(config: RendererLaunchConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "seed": int(config.seed),
        "fixed_view_dimensions": _view_dimensions_or_default(config),
        "freeze_view_origin": bool(config.freeze_view_origin),
        "freeze_camera": bool(config.freeze_camera),
        "exit_when_view_ready": bool(config.exit_when_view_ready),
    }
    if config.terrain_batch_size is not None:
        kwargs["terrain_batch_size"] = max(1, int(config.terrain_batch_size))
    if config.mesh_batch_size is not None:
        kwargs["mesh_batch_size"] = max(1, int(config.mesh_batch_size))
    return kwargs


def make_renderer(config: RendererLaunchConfig):
    mode = _normalized_mode(config.mode)
    if mode == "fly_forward":
        return _make_fly_forward_renderer(config)
    return _make_standard_renderer(config)


def _make_standard_renderer(config: RendererLaunchConfig):
    from .renderer import TerrainRenderer

    renderer = TerrainRenderer(**_renderer_kwargs(config))
    _apply_window_title(renderer, config)
    return renderer


def _make_fly_forward_renderer(config: RendererLaunchConfig):
    from rendercanvas.auto import loop

    from .renderer import TerrainRenderer
    from .render_utils import clamp, flat_forward_vector
    from .renderer_config import CAMERA_HEADROOM_METERS, CAMERA_MIN_HEIGHT_METERS
    from .world_constants import BLOCK_SIZE

    class FlyingForwardBenchmarkRenderer(TerrainRenderer):
        """Fly-forward CLI benchmark renderer created by main.py."""

        def __init__(self) -> None:
            self._benchmark_started_at = time.perf_counter()
            self._benchmark_status_log_interval_s = float(config.status_log_interval_s)
            self._benchmark_next_log_at = self._benchmark_started_at + self._benchmark_status_log_interval_s
            self._benchmark_target_rendered_chunks = int(config.target_rendered_chunks)
            self._benchmark_fly_speed_mps = float(config.fly_speed_mps)
            self._benchmark_rendered_chunk_coords: set[tuple[int, int, int]] = set()
            self._benchmark_rendered_nonempty_chunk_coords: set[tuple[int, int, int]] = set()
            kwargs = _renderer_kwargs(config)
            kwargs["freeze_view_origin"] = False
            kwargs["freeze_camera"] = False
            kwargs["exit_when_view_ready"] = False
            super().__init__(**kwargs)
            _apply_window_title(self, config)
            self.walk_mode = False
            self.camera.move_speed = self._benchmark_fly_speed_mps
            self._current_move_speed = self._benchmark_fly_speed_mps

        @profile
        def _update_camera(self, dt: float) -> None:
            """Force deterministic forward flight instead of relying on keyboard state."""
            speed = self._benchmark_fly_speed_mps
            forward = flat_forward_vector(self.camera.yaw)
            self.camera.position[0] += float(forward[0]) * speed * dt
            self.camera.position[2] += float(forward[2]) * speed * dt
            self.camera.position[1] = clamp(
                float(self.camera.position[1]),
                CAMERA_MIN_HEIGHT_METERS,
                float(self.world.height) * BLOCK_SIZE + CAMERA_HEADROOM_METERS,
            )
            self._current_move_speed = speed
            self._walk_velocity[:] = [0.0, 0.0, 0.0]
            self._camera_on_ground = False
            self._jump_queued = False

        @profile
        def _submit_render(self, meshes=None):
            encoder, color_view, stats = super()._submit_render(meshes=meshes)
            visible_displayed = set(getattr(self, "_visible_displayed_coords", set()))
            self._benchmark_rendered_chunk_coords.update(visible_displayed)

            chunk_cache = getattr(self, "chunk_cache", {})
            for coord in visible_displayed:
                mesh = chunk_cache.get(coord)
                if mesh is not None and int(getattr(mesh, "vertex_count", 0)) > 0:
                    self._benchmark_rendered_nonempty_chunk_coords.add(coord)

            return encoder, color_view, stats

        @profile
        def _service_auto_exit(self) -> None:
            if self._auto_exit_requested or self._device_lost:
                return

            now = time.perf_counter()
            rendered_count = len(self._benchmark_rendered_chunk_coords)
            nonempty_count = len(self._benchmark_rendered_nonempty_chunk_coords)

            if now >= self._benchmark_next_log_at:
                elapsed = max(1e-6, now - self._benchmark_started_at)
                origin = self._current_chunk_origin()
                pos = tuple(float(value) for value in self.camera.position)
                visible_ready = len(getattr(self, "_visible_displayed_coords", ()))
                visible_target = len(getattr(self, "_visible_chunk_coords", ()))
                print(
                    "Info: fly-forward benchmark "
                    f"{rendered_count}/{self._benchmark_target_rendered_chunks} unique rendered chunks "
                    f"({nonempty_count} non-empty), visible_ready={visible_ready}/{visible_target}, "
                    f"origin={origin}, pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                    f"elapsed={elapsed:.2f}s",
                    file=sys.stderr,
                )
                self._benchmark_next_log_at = now + self._benchmark_status_log_interval_s

            if rendered_count >= self._benchmark_target_rendered_chunks:
                self._request_auto_exit()

        @profile
        def _request_auto_exit(self) -> None:
            if self._auto_exit_requested:
                return
            self._auto_exit_requested = True

            elapsed = max(1e-6, time.perf_counter() - self._benchmark_started_at)
            rendered_count = len(self._benchmark_rendered_chunk_coords)
            nonempty_count = len(self._benchmark_rendered_nonempty_chunk_coords)
            pos = tuple(float(value) for value in self.camera.position)
            print(
                "Info: fly-forward benchmark complete; "
                f"rendered {rendered_count} unique chunks "
                f"({nonempty_count} non-empty) while flying at "
                f"{self._benchmark_fly_speed_mps:.2f} m/s for {elapsed:.2f}s. "
                f"Final camera pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}). Closing window.",
                file=sys.stderr,
            )

            try:
                self.canvas.close()
            except Exception:
                pass

            stop_loop = getattr(loop, "stop", None)
            if callable(stop_loop):
                try:
                    stop_loop()
                except Exception:
                    pass

            raise SystemExit(0)

    return FlyingForwardBenchmarkRenderer()
