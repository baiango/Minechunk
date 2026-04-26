from __future__ import annotations

"""World-space Radiance Cascades CPU-side scheduling helpers."""

from typing import Any

import math

import numpy as np

from ..renderer_config import *
from ..world_constants import BLOCK_SIZE

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


def interval_band(cascade_index: int) -> tuple[float, float]:
    """Return the canonical world-space distance interval for one RC cascade.

    The RC volume extent still controls spatial coverage, but interval tracing
    should use a clean geometric band: C0 = [0, R0], C1 = [R0, R1], etc.
    A small overlap makes the local/far handoff continuous instead of a hard
    boundary where one interval stops exactly where the next starts.
    """
    ci = max(0, int(cascade_index))
    base_interval = max(float(BLOCK_SIZE) * 4.0, float(WORLDSPACE_RC_INTERVAL_BASE_WORLD))
    scale = max(1.000001, float(WORLDSPACE_RC_INTERVAL_SCALE))
    if ci <= 0:
        start = 0.0
        end = base_interval
    else:
        start = base_interval * (scale ** float(ci - 1))
        end = base_interval * (scale ** float(ci))
    overlap = max(0.0, min(0.45, float(WORLDSPACE_RC_INTERVAL_OVERLAP)))
    if ci > 0:
        length = max(float(BLOCK_SIZE), end - start)
        start = max(0.0, start - length * overlap)
    return float(start), float(max(start + float(BLOCK_SIZE), end))


def active_direction_count(cascade_index: int) -> int:
    direction_count = max(1, int(WORLDSPACE_RC_DIRECTION_COUNT))
    if int(cascade_index) <= 0:
        return min(direction_count, 8)
    if int(cascade_index) == 1:
        return min(direction_count, 12)
    return direction_count


def format_cascade_list(values: Any) -> str:
    vals = [int(v) for v in list(values or [])]
    return "-" if not vals else ",".join(f"C{v}" for v in vals)


def make_update_params(
    renderer: Any,
    cascade_index: int,
    min_corner: tuple[float, float, float],
    full_extent: float,
    resolution: int,
    temporal_history_weight: float,
) -> np.ndarray:
    max_distance = float(full_extent) * 0.5
    interval_start, interval_end = interval_band(cascade_index)
    interval_start = min(float(interval_start), max_distance)
    interval_end = min(max_distance, max(float(interval_end), interval_start + float(BLOCK_SIZE)))
    seed = float(getattr(renderer.world, "seed", 0))
    world_height = float(renderer.world.height)
    light_dir = tuple(float(v) for v in LIGHT_DIRECTION)
    grid_spacing_world = float(full_extent) / max(1.0, float(resolution - 1))
    grid_spacing_blocks = grid_spacing_world / max(float(BLOCK_SIZE), 0.000001)
    # Far-cascade probe cells are much larger than C0. A fixed 3-8 block
    # push leaves many C1/C2/C3 probes buried inside terrain, so those
    # invalid dark texels show up as sharp square dark patches on distant
    # chunks. Scale the solid-probe push radius with cascade cell size while
    # keeping a sane cap so sealed cave probes still classify by hard sky
    # occlusion after they are moved to nearby air.
    max_probe_push_blocks = min(96.0, max(float(WORLDSPACE_RC_INVALID_PROBE_DILATE_MAX_GRID_DISTANCE), grid_spacing_blocks * 1.25))
    return np.array(
        [
            float(min_corner[0]),
            float(min_corner[1]),
            float(min_corner[2]),
            float(full_extent),
            float(resolution),
            float(cascade_index),
            float(max_distance),
            float(BLOCK_SIZE),
            seed,
            world_height,
            float(WORLDSPACE_RC_TRACE_MAX_STEPS),
            float(WORLDSPACE_RC_TRACE_DIRECTIONS),
            float(RADIANCE_CASCADES_SKY_STRENGTH),
            float(WORLDSPACE_RC_DIRECT_SUN_STRENGTH),
            float(WORLDSPACE_RC_INDIRECT_FLOOR),
            float(WORLDSPACE_RC_SPATIAL_FILTER_SKY_POWER),
            light_dir[0],
            light_dir[1],
            light_dir[2],
            float(temporal_history_weight),
            float(max_probe_push_blocks),
            float(WORLDSPACE_RC_INVALID_PROBE_DILATE_RADIANCE_SCALE),
            float(interval_start),
            float(interval_end),
        ],
        dtype=np.float32,
    )


def write_volume_params_for_schedule(
    renderer: Any,
    target_mins: list[tuple[float, float, float]],
    target_inv_extents: list[tuple[float, float, float]],
    scheduled_updates: set[int] | None = None,
) -> None:
    if renderer.worldspace_rc_volume_params_buffer is None:
        return
    scheduled_updates = scheduled_updates or set()
    params = np.zeros((8, 4), dtype=np.float32)
    for cascade_index in range(4):
        if cascade_index in scheduled_updates:
            params[cascade_index, 0:3] = target_mins[cascade_index]
            params[4 + cascade_index, 0:3] = target_inv_extents[cascade_index]
            continue

        if renderer._worldspace_rc_active_signatures[cascade_index] is not None:
            params[cascade_index, 0:3] = renderer._worldspace_rc_active_mins[cascade_index]
            params[4 + cascade_index, 0:3] = renderer._worldspace_rc_active_inv_extents[cascade_index]
        else:
            params[cascade_index, 0:3] = target_mins[cascade_index]
            params[4 + cascade_index, 0:3] = target_inv_extents[cascade_index]
    renderer.device.queue.write_buffer(renderer.worldspace_rc_volume_params_buffer, 0, params.tobytes())


@profile
def update(renderer: Any, encoder: Any) -> None:
    self = renderer
    if (
        len(self.worldspace_rc_textures) != 4
        or len(self.worldspace_rc_visibility_textures) != 4
        or len(self.worldspace_rc_trace_bind_groups) != 4
        or len(self.worldspace_rc_filter_bind_groups) != 4
        or len(self.worldspace_rc_filter_pingpong_bind_groups) != 4
        or self.worldspace_rc_volume_params_buffer is None
        or self.worldspace_rc_trace_pipeline is None
        or self.worldspace_rc_filter_pipeline is None
    ):
        return

    self._worldspace_rc_frame_index += 1
    resolution = max(4, int(WORLDSPACE_RC_GRID_RESOLUTION))
    base_half_extent = max(float(BLOCK_SIZE) * 4.0, float(WORLDSPACE_RC_BASE_HALF_EXTENT_WORLD))
    camera_pos = tuple(float(v) for v in self.camera.position)
    target_signatures: list[tuple[float, ...]] = []
    target_mins: list[tuple[float, float, float]] = []
    target_inv_extents: list[tuple[float, float, float]] = []

    target_interval_bands: list[tuple[float, float]] = []
    for cascade_index in range(4):
        interval_start, interval_end = interval_band(cascade_index)
        # Canonical RC interval end controls the cascade volume radius. This
        # keeps spatial coverage and distance interval math coherent: C0 is
        # the nearest band, C1 begins around C0's end, and so on.
        half_extent = max(base_half_extent * (float(WORLDSPACE_RC_INTERVAL_SCALE) ** float(cascade_index)), interval_end)
        extent_scales = tuple(float(v) for v in WORLDSPACE_RC_CASCADE_EXTENT_SCALES)
        extent_scale = extent_scales[min(cascade_index, len(extent_scales) - 1)] if extent_scales else 1.0
        half_extent *= max(1.0, extent_scale)
        full_extent = half_extent * 2.0
        snap = max(float(WORLDSPACE_RC_UPDATE_QUANTIZE_WORLD), full_extent / max(1.0, float(resolution - 1)))
        center_x = math.floor((camera_pos[0] / snap) + 0.5) * snap
        snapped_camera_y = math.floor((camera_pos[1] / snap) + 0.5) * snap
        center_z = math.floor((camera_pos[2] / snap) + 0.5) * snap
        vertical_down_bias = max(0.0, min(0.65, float(WORLDSPACE_RC_VERTICAL_DOWN_BIAS)))
        cascade_bias = vertical_down_bias * (0.45 + 0.55 * (float(cascade_index) / 3.0))
        center_y = snapped_camera_y - half_extent * cascade_bias
        min_corner = (center_x - half_extent, center_y - half_extent, center_z - half_extent)
        inv_extent = (1.0 / full_extent, 1.0 / full_extent, 1.0 / full_extent)
        target_signatures.append((round(min_corner[0], 4), round(min_corner[1], 4), round(min_corner[2], 4), round(full_extent, 4), round(interval_start, 4), round(interval_end, 4)))
        target_mins.append(min_corner)
        target_inv_extents.append(inv_extent)
        target_interval_bands.append((interval_start, interval_end))
    self._worldspace_rc_last_interval_bands = target_interval_bands

    dirty_indices = [i for i in range(4) if target_signatures[i] != self._worldspace_rc_active_signatures[i]]
    if dirty_indices:
        # After a recenter/move, spend a short burst of stable refresh frames
        # re-converging temporal directional radiance. History is rejected for
        # the dirty update itself, then allowed again once the volume is stable.
        self._worldspace_rc_convergence_frames_remaining = max(
            int(self._worldspace_rc_convergence_frames_remaining),
            max(0, int(WORLDSPACE_RC_CONVERGENCE_BURST_FRAMES)),
        )

    stable_refresh_indices: list[int] = []
    if not dirty_indices and all(sig is not None for sig in self._worldspace_rc_active_signatures):
        # v32 temporal accumulation only helps when the previous field uses
        # the same world-space volume. If the camera has not crossed a snap
        # boundary, refresh stable cascades so the multi-bounce/directional
        # field converges. During a post-move burst, refresh more than one
        # cascade per frame in far-to-near order; outside the burst, keep the
        # old low-cost one-cascade refresh.
        stable_budget = max(1, int(WORLDSPACE_RC_STABLE_REFRESH_CASCADES_PER_FRAME))
        if int(self._worldspace_rc_convergence_frames_remaining) > 0:
            stable_budget = max(stable_budget, int(WORLDSPACE_RC_CONVERGENCE_REFRESH_CASCADES_PER_FRAME))
        stable_budget = min(4, max(1, stable_budget))
        cursor = int(self._worldspace_rc_update_cursor) % 4
        stable_refresh_indices = [((cursor - offset) % 4) for offset in range(stable_budget)]

    if not dirty_indices and not stable_refresh_indices:
        self._worldspace_rc_last_dirty_indices = []
        self._worldspace_rc_last_stable_refresh_indices = []
        self._worldspace_rc_last_scheduled_updates = []
        self._worldspace_rc_last_history_reject_updates = []
        self._worldspace_rc_last_update_kind = "idle"
        write_volume_params_for_schedule(self, target_mins, target_inv_extents, set())
        return

    if dirty_indices and any(sig is None for sig in self._worldspace_rc_active_signatures):
        max_updates = max(1, int(WORLDSPACE_RC_INITIAL_MAX_CASCADES_PER_FRAME))
    elif dirty_indices:
        max_updates = max(1, int(WORLDSPACE_RC_UPDATE_MAX_CASCADES_PER_FRAME))
    else:
        max_updates = min(4, max(1, len(stable_refresh_indices)))

    updates_done = 0
    updated_any = False
    workgroups = (
        (resolution + 3) // 4,
        (resolution + 3) // 4,
        (resolution + 3) // 4,
    )
    update_candidates = dirty_indices if dirty_indices else stable_refresh_indices
    if dirty_indices:
        # Real RC interval merge wants far cascades available before near
        # cascades trace and fork into them. Updating C3 -> C2 -> C1 -> C0
        # prevents C0 from merging stale C1/C2/C3 fields after camera motion.
        search_order = [i for i in (3, 2, 1, 0) if i in update_candidates]
    else:
        # Stable refresh also walks far-to-near over time, so temporal
        # convergence feeds the next closer interval on following frames.
        # v42 can refresh a small pair during post-move convergence bursts.
        search_order = list(update_candidates)
    scheduled_updates = set(search_order[:max_updates])
    history_reject_updates = [
        i for i in sorted(scheduled_updates, reverse=True)
        if self._worldspace_rc_active_signatures[i] != target_signatures[i]
    ]
    self._worldspace_rc_last_dirty_indices = list(dirty_indices)
    self._worldspace_rc_last_stable_refresh_indices = list(stable_refresh_indices)
    self._worldspace_rc_last_scheduled_updates = list(search_order[:max_updates])
    self._worldspace_rc_last_history_reject_updates = history_reject_updates
    self._worldspace_rc_last_update_kind = "dirty" if dirty_indices else "stable"
    write_volume_params_for_schedule(self, target_mins, target_inv_extents, scheduled_updates)

    for cascade_index in search_order:
        base_frame_stride = max(1, int(WORLDSPACE_RC_UPDATE_FRAME_INTERVAL))
        # Keep every cascade spatially current while moving. Staggering farther
        # cascades makes open-sky terrain sample old/out-of-range volumes for
        # several frames, which reads as global darkening during camera motion.
        frame_stride = base_frame_stride
        if self._worldspace_rc_active_signatures[cascade_index] is not None:
            frames_since = self._worldspace_rc_frame_index - int(self._worldspace_rc_last_update_frame[cascade_index])
            if frames_since < frame_stride:
                continue

        min_corner = target_mins[cascade_index]
        full_extent = 1.0 / target_inv_extents[cascade_index][0]
        temporal_history_weight = 0.0 if cascade_index in dirty_indices else 1.0
        update_params = make_update_params(
            self,
            cascade_index,
            min_corner,
            full_extent,
            resolution,
            temporal_history_weight,
        )
        self.device.queue.write_buffer(
            self.worldspace_rc_update_param_buffers[cascade_index],
            0,
            update_params.tobytes(),
        )

        trace_pass = encoder.begin_compute_pass()
        trace_pass.set_pipeline(self.worldspace_rc_trace_pipeline)
        trace_pass.set_bind_group(0, self.worldspace_rc_trace_bind_groups[cascade_index])
        trace_pass.dispatch_workgroups(*workgroups)
        trace_pass.end()

        filter_passes = max(1, int(WORLDSPACE_RC_SPATIAL_FILTER_PASSES))
        # Compose samples the final RC textures, not the scratch textures. If
        # an even number of blur iterations would end in scratch, add one
        # final ping-pong pass back into the sampled volume.
        filter_dispatches = filter_passes if (filter_passes % 2) == 1 else filter_passes + 1
        for filter_index in range(filter_dispatches):
            filter_bind_group = (
                self.worldspace_rc_filter_bind_groups[cascade_index]
                if (filter_index % 2) == 0
                else self.worldspace_rc_filter_pingpong_bind_groups[cascade_index]
            )
            filter_pass = encoder.begin_compute_pass()
            filter_pass.set_pipeline(self.worldspace_rc_filter_pipeline)
            filter_pass.set_bind_group(0, filter_bind_group)
            filter_pass.dispatch_workgroups(*workgroups)
            filter_pass.end()

        self._worldspace_rc_active_signatures[cascade_index] = target_signatures[cascade_index]
        self._worldspace_rc_active_mins[cascade_index] = target_mins[cascade_index]
        self._worldspace_rc_active_inv_extents[cascade_index] = target_inv_extents[cascade_index]
        self._worldspace_rc_last_update_frame[cascade_index] = self._worldspace_rc_frame_index
        self._worldspace_rc_update_cursor = (cascade_index - 1) % 4
        updates_done += 1
        updated_any = True
        if updates_done >= max_updates:
            break

    if not dirty_indices and updated_any and int(self._worldspace_rc_convergence_frames_remaining) > 0:
        self._worldspace_rc_convergence_frames_remaining = max(
            0,
            int(self._worldspace_rc_convergence_frames_remaining) - 1,
        )

    write_volume_params_for_schedule(self, target_mins, target_inv_extents, set())

    if updated_any:
        signature_parts: list[float] = []
        for cascade_index in range(4):
            active_sig = self._worldspace_rc_active_signatures[cascade_index]
            if active_sig is None:
                signature_parts.extend([0.0, 0.0, 0.0, 0.0])
            else:
                signature_parts.extend(list(active_sig))
        self._worldspace_rc_signature = tuple(signature_parts)
