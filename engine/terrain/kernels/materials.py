from __future__ import annotations

from .numba_compat import njit

AIR = 0
BEDROCK = 1
STONE = 2
DIRT = 3
GRASS = 4
SAND = 5
SNOW = 6

_MATERIAL_COLOR_R = (0.0, 0.24, 0.42, 0.47, 0.31, 0.78, 0.95)
_MATERIAL_COLOR_G = (0.0, 0.22, 0.40, 0.31, 0.68, 0.71, 0.97)
_MATERIAL_COLOR_B = (0.0, 0.20, 0.38, 0.18, 0.24, 0.49, 0.98)

MAX_FACES_PER_CELL = 5
VERTICES_PER_FACE = 6
VERTEX_COMPONENTS = 9

@njit(cache=True, fastmath=True)
def _terrain_color(height: int) -> tuple[float, float, float]:
    if height <= 14:
        return 0.78, 0.71, 0.49
    if height >= 90:
        return 0.95, 0.97, 0.98
    if height >= 70:
        return 0.60, 0.72, 0.49
    if height >= 40:
        return 0.38, 0.64, 0.31
    return 0.28, 0.54, 0.22


@njit(cache=True, fastmath=True)
def _scale_color(color: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return color[0] * scale, color[1] * scale, color[2] * scale

@njit(cache=True, fastmath=True)
def _voxel_material_color(material: int, height: int) -> tuple[float, float, float]:
    if material == BEDROCK:
        return 0.24, 0.22, 0.20
    if material == STONE:
        return 0.42, 0.40, 0.38
    if material == DIRT:
        return 0.47, 0.31, 0.18
    if material == GRASS:
        return 0.31, 0.68, 0.24
    if material == SAND:
        return 0.78, 0.71, 0.49
    if material == SNOW:
        return 0.95, 0.97, 0.98
    return _terrain_color(height)
