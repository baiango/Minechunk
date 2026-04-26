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
