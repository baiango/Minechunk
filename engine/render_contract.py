from __future__ import annotations

"""Stable import boundary for low-level engine systems.

Low-level modules such as meshing, cache allocation, visibility, and profiling
must not import ``engine.renderer`` just to read constants.  Importing the
renderer from those modules couples them to canvas/device setup and reintroduces
circular dependencies.

This module intentionally exposes the constants and runtime knobs that used to
be read indirectly through ``engine.renderer``.
"""

from . import render_constants as _constants
from . import renderer_config as _config
from .render_constants import *  # noqa: F401,F403

# Runtime/config knobs historically read through ``engine.renderer``.
engine_mode = _config.engine_mode
chunk_prep_request_budget_cap = _config.chunk_prep_request_budget_cap
chunk_prep_bootstrap_displayed_ratio_threshold = _config.chunk_prep_bootstrap_displayed_ratio_threshold
chunk_prep_use_screen_border_raycast = _config.chunk_prep_use_screen_border_raycast
chunk_prep_screen_raycast_max_distance_world = _config.chunk_prep_screen_raycast_max_distance_world
chunk_prep_screen_raycast_pixel_stride = _config.chunk_prep_screen_raycast_pixel_stride
chunk_prep_screen_raycast_max_rays = _config.chunk_prep_screen_raycast_max_rays
chunk_prep_screen_raycast_position_quantize_world = _config.chunk_prep_screen_raycast_position_quantize_world
chunk_prep_screen_raycast_angle_quantize_radians = _config.chunk_prep_screen_raycast_angle_quantize_radians


def __getattr__(name: str):
    """Fallback for constants while the codebase is being split apart."""
    if hasattr(_constants, name):
        return getattr(_constants, name)
    if hasattr(_config, name):
        return getattr(_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    names.update(dir(_constants))
    names.update(dir(_config))
    return sorted(names)

def align_up(value: int, alignment: int) -> int:
    """Return ``value`` rounded up to the next ``alignment`` boundary."""
    alignment = int(alignment)
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


def device_limit(device, name: str, default: int) -> int:
    """Read a WGPU device limit without binding callers to renderer internals."""
    limits = getattr(device, "limits", None)
    if limits is None:
        return int(default)
    getter = limits.get if hasattr(limits, "get") else None
    if getter is not None:
        value = getter(name, default)
    else:
        value = getattr(limits, name, default)
    try:
        return int(value)
    except Exception:
        return int(default)


def describe_adapter(adapter) -> str:
    """Build a stable human-readable adapter/backend label."""
    info = getattr(adapter, "info", None)
    summary = getattr(adapter, "summary", "")

    backend = ""
    adapter_type = ""
    description = ""

    if info is not None:
        getter = info.get if hasattr(info, "get") else None
        if getter is not None:
            backend = getter("backend_type", "") or getter("backend", "")
            adapter_type = getter("adapter_type", "") or getter("device_type", "")
            description = getter("description", "") or getter("device", "")
        else:
            backend = getattr(info, "backend_type", "") or getattr(info, "backend", "")
            adapter_type = getattr(info, "adapter_type", "") or getattr(info, "device_type", "")
            description = getattr(info, "description", "") or getattr(info, "device", "")

    if isinstance(summary, str) and summary.strip():
        if not backend and not adapter_type and not description:
            return summary.strip()

    parts = [part for part in (backend, adapter_type, description) if part]
    if parts:
        return " / ".join(parts)
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return "unknown"


def truthy_env_flag(name: str, default: str = "") -> bool:
    """Parse common true-ish environment variable values."""
    import os

    value = os.environ.get(name, default).strip().lower()
    return value in ("1", "true", "yes", "on")

