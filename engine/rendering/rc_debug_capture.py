from __future__ import annotations

"""CPU-side F7 world-space RC diagnostics and PNG capture helpers."""

import os
import sys
import time
from typing import Any

import wgpu

from .. import render_contract
from ..debug_capture import readback_to_rgba8, safe_filename_component, write_rgba8_png
from ..renderer_config import *
from . import postprocess_targets, worldspace_rc


def dump_diagnostics(renderer: Any) -> None:
    """Write a compact RC diagnostic snapshot for screenshot triage."""
    self = renderer
    try:
        out_dir = os.path.join(os.getcwd(), "rc_diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frame = int(getattr(self, "_worldspace_rc_frame_index", 0))
        path = os.path.join(out_dir, f"rc_snapshot_{stamp}_frame_{frame}.txt")
        debug_mode = int(getattr(self, "rc_debug_mode", 0))
        debug_names = tuple(getattr(self, "rc_debug_mode_names", ("off",)))
        debug_name = debug_names[debug_mode] if 0 <= debug_mode < len(debug_names) else "unknown"
        last_updates = list(getattr(self, "_worldspace_rc_last_update_frame", []))
        ages: list[str] = []
        for idx in range(4):
            try:
                last_update = int(last_updates[idx])
            except Exception:
                last_update = -1000000
            ages.append("--" if last_update < -999000 else str(max(0, frame - last_update)))

        lines: list[str] = []
        lines.append("Minechunk RC Diagnostic Snapshot")
        lines.append(f"timestamp={stamp}")
        lines.append(f"frame={frame}")
        lines.append(f"engine_mode={engine_mode}")
        lines.append(f"render_backend={self.render_backend_label}")
        lines.append(f"terrain_backend={self.world.terrain_backend_label()}")
        lines.append(f"mesh_backend={self.mesh_backend_label}")
        lines.append(f"rc_enabled={bool(self.radiance_cascades_enabled)}")
        lines.append(f"debug_mode={debug_mode}:{debug_name}")
        lines.append("")
        lines.append("field:")
        lines.append(f"  resolution={int(WORLDSPACE_RC_GRID_RESOLUTION)}")
        lines.append(f"  direction_count={int(WORLDSPACE_RC_DIRECTION_COUNT)}")
        lines.append(f"  active_direction_counts={[worldspace_rc.active_direction_count(i) for i in range(4)]}")
        lines.append(f"  vertical_down_bias={float(WORLDSPACE_RC_VERTICAL_DOWN_BIAS):.4f}")
        lines.append(f"  cascade_extent_scales={list(WORLDSPACE_RC_CASCADE_EXTENT_SCALES)}")
        lines.append(f"  filter_passes={int(WORLDSPACE_RC_SPATIAL_FILTER_PASSES)}")
        lines.append(f"  temporal_alpha={float(WORLDSPACE_RC_TEMPORAL_BLEND_ALPHA):.4f}")
        lines.append(f"  bounce_feedback={float(WORLDSPACE_RC_BOUNCE_FEEDBACK_STRENGTH):.4f}")
        lines.append(f"  base_interval_world={float(WORLDSPACE_RC_INTERVAL_BASE_WORLD):.6f}")
        lines.append(f"  interval_scale={float(WORLDSPACE_RC_INTERVAL_SCALE):.6f}")
        lines.append(f"  interval_overlap={float(WORLDSPACE_RC_INTERVAL_OVERLAP):.6f}")
        lines.append("")
        lines.append("scheduler:")
        lines.append(f"  update_kind={getattr(self, '_worldspace_rc_last_update_kind', 'unknown')}")
        lines.append(f"  scheduled={worldspace_rc.format_cascade_list(getattr(self, '_worldspace_rc_last_scheduled_updates', []))}")
        lines.append(f"  dirty={worldspace_rc.format_cascade_list(getattr(self, '_worldspace_rc_last_dirty_indices', []))}")
        lines.append(f"  stable_refresh={worldspace_rc.format_cascade_list(getattr(self, '_worldspace_rc_last_stable_refresh_indices', []))}")
        lines.append(f"  history_reject={worldspace_rc.format_cascade_list(getattr(self, '_worldspace_rc_last_history_reject_updates', []))}")
        lines.append(f"  convergence_burst_frames_left={int(getattr(self, '_worldspace_rc_convergence_frames_remaining', 0))}")
        lines.append(f"  update_cursor={int(getattr(self, '_worldspace_rc_update_cursor', 0))}")
        lines.append(f"  cascade_age_frames=C0:{ages[0]} C1:{ages[1]} C2:{ages[2]} C3:{ages[3]}")
        lines.append("")
        lines.append("intervals:")
        for idx in range(4):
            start, end = worldspace_rc.interval_band(idx)
            lines.append(f"  C{idx}: start={start:.6f} end={end:.6f} length={(end - start):.6f}")
        lines.append("")
        lines.append("active_signatures:")
        for idx, sig in enumerate(getattr(self, '_worldspace_rc_active_signatures', [])):
            lines.append(f"  C{idx}: {sig}")
        lines.append("")
        lines.append("active_volume_min_inv_extent:")
        for idx in range(4):
            mins = getattr(self, '_worldspace_rc_active_mins', [(0.0, 0.0, 0.0)] * 4)[idx]
            inv = getattr(self, '_worldspace_rc_active_inv_extents', [(0.0, 0.0, 0.0)] * 4)[idx]
            lines.append(f"  C{idx}: min={mins} inv_extent={inv}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self._worldspace_rc_last_snapshot_path = path
        print(f"Info: RC diagnostics snapshot written to {path}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: failed to write RC diagnostics snapshot: {exc!r}", file=sys.stderr)


def queue_image_dump(renderer: Any) -> None:
    """Queue one-shot PNG captures for every RC compose debug mode."""
    self = renderer
    try:
        out_dir = os.path.join(os.getcwd(), "rc_diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frame = int(getattr(self, "_worldspace_rc_frame_index", 0))
        mode_count = max(1, len(tuple(getattr(self, "rc_debug_mode_names", ("off",)))))
        self._pending_rc_debug_capture_request = {
            "out_dir": out_dir,
            "stamp": stamp,
            "frame": frame,
            "modes": list(range(mode_count)),
        }
        self._worldspace_rc_last_snapshot_path = os.path.join(out_dir, f"rc_debug_images_{stamp}_frame_{frame}")
        print(
            f"Info: queued RC debug PNG capture for {mode_count} mode(s) into {out_dir}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"Warning: failed to queue RC debug PNG capture: {exc!r}", file=sys.stderr)


def align_copy_bytes_per_row(unpadded_bytes_per_row: int) -> int:
    return render_contract.align_up(max(1, int(unpadded_bytes_per_row)), 256)


def encode_pending_captures(renderer: Any, encoder: Any) -> None:
    self = renderer
    request = self._pending_rc_debug_capture_request
    if request is None:
        return
    self._pending_rc_debug_capture_request = None
    if not self.radiance_cascades_enabled:
        print("Warning: F7 RC debug PNG capture skipped because radiance cascades are disabled.", file=sys.stderr)
        return
    if (
        self.gi_compose_pipeline is None
        or self.gi_compose_bind_group_layout is None
        or self.scene_color_view is None
        or self.scene_gbuffer_view is None
        or self.postprocess_sampler is None
        or self.camera_buffer is None
        or self.worldspace_rc_volume_params_buffer is None
        or len(self.worldspace_rc_views) < 4
        or len(self.worldspace_rc_visibility_views) < 4
    ):
        print("Warning: F7 RC debug PNG capture skipped because RC compose resources are not ready.", file=sys.stderr)
        return

    width, height = (int(self._gi_color_size[0]), int(self._gi_color_size[1]))
    if width <= 0 or height <= 0:
        print("Warning: F7 RC debug PNG capture skipped because the GI target has no size.", file=sys.stderr)
        return

    mode_names = tuple(getattr(self, "rc_debug_mode_names", ("off",)))
    modes = [int(mode) for mode in request.get("modes", [])]
    out_dir = str(request.get("out_dir", os.path.join(os.getcwd(), "rc_diagnostics")))
    stamp = str(request.get("stamp", time.strftime("%Y%m%d_%H%M%S")))
    frame = int(request.get("frame", getattr(self, "_worldspace_rc_frame_index", 0)))
    bytes_per_pixel = 8 if POSTPROCESS_GI_FORMAT == "rgba16float" else 4
    unpadded_bpr = width * bytes_per_pixel
    padded_bpr = align_copy_bytes_per_row(unpadded_bpr)
    readback_size = padded_bpr * height
    encoded_count = 0

    for mode in modes:
        if mode < 0 or mode >= len(mode_names):
            continue
        mode_name = mode_names[mode]
        mode_slug = safe_filename_component(mode_name)
        path = os.path.join(out_dir, f"rc_debug_{stamp}_frame_{frame:06d}_{mode:02d}_{mode_slug}.png")
        params_buffer = self.device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(params_buffer, 0, postprocess_targets.make_gi_params(self, mode).tobytes())
        capture_texture = self.device.create_texture(
            size=(width, height, 1),
            format=POSTPROCESS_GI_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        capture_view = capture_texture.create_view()
        capture_bind_group = self.device.create_bind_group(
            layout=self.gi_compose_bind_group_layout,
            entries=[
                {"binding": 0, "resource": self.scene_color_view},
                {"binding": 1, "resource": self.scene_gbuffer_view},
                {"binding": 2, "resource": self.worldspace_rc_views[0]},
                {"binding": 3, "resource": self.worldspace_rc_views[1]},
                {"binding": 4, "resource": self.worldspace_rc_views[2]},
                {"binding": 5, "resource": self.worldspace_rc_views[3]},
                {"binding": 6, "resource": self.worldspace_rc_visibility_views[0]},
                {"binding": 7, "resource": self.worldspace_rc_visibility_views[1]},
                {"binding": 8, "resource": self.worldspace_rc_visibility_views[2]},
                {"binding": 9, "resource": self.worldspace_rc_visibility_views[3]},
                {"binding": 10, "resource": self.postprocess_sampler},
                {"binding": 11, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": CAMERA_UNIFORM_BYTES}},
                {"binding": 12, "resource": {"buffer": params_buffer, "offset": 0, "size": 32}},
                {"binding": 13, "resource": {"buffer": self.worldspace_rc_volume_params_buffer, "offset": 0, "size": 256}},
            ],
        )
        capture_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": capture_view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )
        capture_pass.set_pipeline(self.gi_compose_pipeline)
        capture_pass.set_bind_group(0, capture_bind_group)
        capture_pass.draw(3, 1, 0, 0)
        capture_pass.end()

        readback_buffer = self.device.create_buffer(
            size=readback_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder.copy_texture_to_buffer(
            {"texture": capture_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": readback_buffer, "offset": 0, "bytes_per_row": padded_bpr, "rows_per_image": height},
            (width, height, 1),
        )
        self._pending_rc_debug_readbacks.append(
            {
                "path": path,
                "width": width,
                "height": height,
                "format": POSTPROCESS_GI_FORMAT,
                "bytes_per_pixel": bytes_per_pixel,
                "unpadded_bpr": unpadded_bpr,
                "padded_bpr": padded_bpr,
                "size": readback_size,
                "buffer": readback_buffer,
                "texture": capture_texture,
                "params_buffer": params_buffer,
                "bind_group": capture_bind_group,
            }
        )
        encoded_count += 1

    if encoded_count > 0:
        print(f"Info: encoded F7 RC debug PNG captures for {encoded_count} mode(s).", file=sys.stderr)


def drain_pending_readbacks(renderer: Any) -> None:
    self = renderer
    if not self._pending_rc_debug_readbacks:
        return
    pending = self._pending_rc_debug_readbacks
    self._pending_rc_debug_readbacks = []
    saved_paths: list[str] = []
    for item in pending:
        buffer = item.get("buffer")
        path = str(item.get("path", ""))
        try:
            size = int(item.get("size", 0))
            if getattr(buffer, "map_state", "unmapped") != "unmapped":
                buffer.unmap()
            buffer.map_sync(wgpu.MapMode.READ, 0, size)
            try:
                mapped = buffer.read_mapped(0, size, copy=False)
                rgba = readback_to_rgba8(
                    mapped,
                    width=int(item["width"]),
                    height=int(item["height"]),
                    texture_format=str(item["format"]),
                    padded_bpr=int(item["padded_bpr"]),
                )
            finally:
                if getattr(buffer, "map_state", "unmapped") != "unmapped":
                    buffer.unmap()
            write_rgba8_png(path, rgba)
            saved_paths.append(path)
        except Exception as exc:
            print(f"Warning: failed to save F7 RC debug PNG {path!r}: {exc!r}", file=sys.stderr)
        finally:
            for resource_name in ("buffer", "texture", "params_buffer"):
                resource = item.get(resource_name)
                destroy = getattr(resource, "destroy", None)
                if callable(destroy):
                    try:
                        destroy()
                    except Exception:
                        pass
    if saved_paths:
        self._worldspace_rc_last_capture_paths = saved_paths
        self._worldspace_rc_last_snapshot_path = os.path.dirname(saved_paths[0])
        print(f"Info: saved {len(saved_paths)} F7 RC debug PNG(s) into {os.path.dirname(saved_paths[0])}", file=sys.stderr)
