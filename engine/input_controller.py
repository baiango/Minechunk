from __future__ import annotations

from typing import Any

from . import profiling_runtime
from .rendering import postprocess_targets, rc_debug_capture


def normalize_key(event: dict) -> str:
    key = str(event.get("key", "")).strip().lower()
    if key in {" ", "spacebar"}:
        return "space"
    if key in {"controlleft", "controlright", "ctrl"}:
        return "control"
    if key == "shiftleft":
        return "shiftleft"
    if key == "shiftright":
        return "shiftright"
    return key


def key_active(renderer: Any, *names: str) -> bool:
    for name in names:
        if name in renderer.keys_down:
            return True
    return False


def handle_key_down(renderer: Any, event: dict) -> None:
    key = normalize_key(event)
    is_new_press = key not in renderer.keys_down
    renderer.keys_down.add(key)
    if is_new_press and key == "f3":
        profiling_runtime.toggle(renderer)
    if is_new_press and key in {"r"}:
        renderer.regenerate_world()
    if is_new_press and key == "v":
        renderer.walk_mode = not renderer.walk_mode
        renderer._walk_velocity[:] = [0.0, 0.0, 0.0]
        renderer._jump_queued = False
    if is_new_press and key == "g":
        renderer.radiance_cascades_enabled = not renderer.radiance_cascades_enabled
    if is_new_press and key == "f6":
        renderer.rc_debug_mode = (int(renderer.rc_debug_mode) + 1) % len(renderer.rc_debug_mode_names)
        postprocess_targets.write_gi_params(renderer)
        print(f"Info: RC debug mode = {renderer.rc_debug_mode} ({renderer.rc_debug_mode_names[renderer.rc_debug_mode]})")
    if is_new_press and key == "f7":
        rc_debug_capture.dump_diagnostics(renderer)
        rc_debug_capture.queue_image_dump(renderer)
    if is_new_press and key == "space":
        renderer._jump_queued = True


def handle_key_up(renderer: Any, event: dict) -> None:
    renderer.keys_down.discard(normalize_key(event))


def handle_pointer_down(renderer: Any, event: dict) -> None:
    if int(event.get("button", 0)) == 1:
        renderer.dragging = True
        renderer.last_pointer = (float(event.get("x", 0.0)), float(event.get("y", 0.0)))


def handle_pointer_move(renderer: Any, event: dict) -> None:
    if not renderer.dragging:
        return
    x = float(event.get("x", 0.0))
    y = float(event.get("y", 0.0))
    if renderer.last_pointer is None:
        renderer.last_pointer = (x, y)
        return
    last_x, last_y = renderer.last_pointer
    dx = x - last_x
    dy = y - last_y
    renderer.last_pointer = (x, y)
    renderer.camera.yaw -= dx * renderer.camera.look_speed
    renderer.camera.pitch -= dy * renderer.camera.look_speed
    renderer.camera.clamp_pitch()


def handle_pointer_up(renderer: Any, event: dict) -> None:
    if int(event.get("button", 0)) == 1:
        renderer.dragging = False
        renderer.last_pointer = None


def bind_canvas_events(renderer: Any) -> None:
    canvas = renderer.canvas
    canvas.add_event_handler(lambda event: handle_key_down(renderer, event), "key_down")
    canvas.add_event_handler(lambda event: handle_key_up(renderer, event), "key_up")
    canvas.add_event_handler(lambda event: handle_pointer_down(renderer, event), "pointer_down")
    canvas.add_event_handler(lambda event: handle_pointer_move(renderer, event), "pointer_move")
    canvas.add_event_handler(lambda event: handle_pointer_up(renderer, event), "pointer_up")
    canvas.add_event_handler(renderer._handle_resize, "resize")
