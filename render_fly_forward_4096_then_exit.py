from __future__ import annotations

import sys
import time

from rendercanvas.auto import loop

from engine.renderer import TerrainRenderer
from engine.render_utils import clamp, flat_forward_vector
from engine.renderer_config import CAMERA_HEADROOM_METERS, CAMERA_MIN_HEIGHT_METERS
from engine.world_constants import BLOCK_SIZE

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


TARGET_RENDERED_CHUNKS = 4096
FLY_FORWARD_SPEED_METERS_PER_SECOND = 20.0
VIEW_DIMENSIONS_CHUNKS = (16, 16, 16)
TERRAIN_BATCH_SIZE = 128
MESH_BATCH_SIZE = 32
STATUS_LOG_INTERVAL_SECONDS = 1.0


class FlyingForward4096ExitRenderer(TerrainRenderer):
    """Benchmark entrypoint: fly forward and exit after 4096 unique chunks render."""

    def __init__(self) -> None:
        self._benchmark_started_at = time.perf_counter()
        self._benchmark_next_log_at = self._benchmark_started_at + STATUS_LOG_INTERVAL_SECONDS
        self._benchmark_rendered_chunk_coords: set[tuple[int, int, int]] = set()
        self._benchmark_rendered_nonempty_chunk_coords: set[tuple[int, int, int]] = set()

        super().__init__(
            fixed_view_dimensions=VIEW_DIMENSIONS_CHUNKS,
            freeze_view_origin=False,
            freeze_camera=False,
            exit_when_view_ready=False,
            terrain_batch_size=TERRAIN_BATCH_SIZE,
            mesh_batch_size=MESH_BATCH_SIZE,
        )

        self.base_title = "Minechunk - fly-forward 4096 chunk benchmark"
        try:
            self.canvas.set_title(self.base_title)
        except Exception:
            pass

        self.walk_mode = False
        self.camera.move_speed = float(FLY_FORWARD_SPEED_METERS_PER_SECOND)
        self._current_move_speed = float(FLY_FORWARD_SPEED_METERS_PER_SECOND)

    @profile
    def _update_camera(self, dt: float) -> None:
        """Force deterministic forward flight instead of relying on keyboard state."""
        speed = float(FLY_FORWARD_SPEED_METERS_PER_SECOND)
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
                f"{rendered_count}/{TARGET_RENDERED_CHUNKS} unique rendered chunks "
                f"({nonempty_count} non-empty), visible_ready={visible_ready}/{visible_target}, "
                f"origin={origin}, pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                f"elapsed={elapsed:.2f}s",
                file=sys.stderr,
            )
            self._benchmark_next_log_at = now + STATUS_LOG_INTERVAL_SECONDS

        if rendered_count >= TARGET_RENDERED_CHUNKS:
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
            "Info: fly-forward 4096 benchmark complete; "
            f"rendered {rendered_count} unique chunks "
            f"({nonempty_count} non-empty) while flying at "
            f"{FLY_FORWARD_SPEED_METERS_PER_SECOND:.2f} m/s for {elapsed:.2f}s. "
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


def main() -> None:
    FlyingForward4096ExitRenderer().run()


if __name__ == "__main__":
    main()
