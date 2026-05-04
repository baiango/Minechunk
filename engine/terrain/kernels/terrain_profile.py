from __future__ import annotations

import numpy as np

from .numba_compat import njit
from .materials import AIR, BEDROCK, DIRT, GRASS, SAND, SNOW, STONE
from .noise import _value_noise_2d, _value_noise_3d

TERRAIN_FREQUENCY_SCALE = 0.3
CAVE_FREQUENCY_SCALE = 1.0
CAVE_DETAIL_FREQUENCY_MULTIPLIER = 3.0
CAVE_DETAIL_WEIGHT = 0.18
CAVE_BEDROCK_CLEARANCE = 3
CAVE_ACTIVE_BAND_MIN = 0.58
CAVE_PRIMARY_THRESHOLD = 0.66
CAVE_VERTICAL_BONUS = 0.06
CAVE_DEPTH_BONUS_SCALE = 0.0015
CAVE_DEPTH_BONUS_MAX = 0.06
CAVE_MODEL_VERSION = 14


@njit(cache=True, fastmath=True)
def surface_profile_at(
    x: float,
    z: float,
    seed: int,
    height_limit: int,
) -> tuple[int, int]:
    sample_x = np.float32(x)
    sample_z = np.float32(z)

    broad = _value_noise_2d(sample_x, sample_z, seed + 11, np.float32(0.0009765625) * np.float32(TERRAIN_FREQUENCY_SCALE))
    ridge = _value_noise_2d(sample_x, sample_z, seed + 23, np.float32(0.00390625) * np.float32(TERRAIN_FREQUENCY_SCALE))
    detail = _value_noise_2d(sample_x, sample_z, seed + 47, np.float32(0.010416667) * np.float32(TERRAIN_FREQUENCY_SCALE))
    micro = _value_noise_2d(sample_x, sample_z, seed + 71, np.float32(0.020833334) * np.float32(TERRAIN_FREQUENCY_SCALE))
    nano = _value_noise_2d(sample_x, sample_z, seed + 97, np.float32(0.041666668) * np.float32(TERRAIN_FREQUENCY_SCALE))

    upper_bound = height_limit - 1
    normalized_height = (
        np.float32(24.0)
        + broad * np.float32(11.0)
        + ridge * np.float32(8.0)
        + detail * np.float32(4.5)
        + micro * np.float32(1.75)
        + nano * np.float32(0.75)
    )
    height_scale = np.float32(upper_bound) / np.float32(50.0) if upper_bound > 0 else np.float32(1.0)
    height_i = int(normalized_height * height_scale)
    if height_i < 4:
        height_i = 4
    if height_i > upper_bound:
        height_i = upper_bound

    sand_threshold = max(4, int(np.float32(height_limit) * np.float32(0.18)))
    stone_threshold = max(sand_threshold + 6, int(np.float32(height_limit) * np.float32(0.58)))
    snow_threshold = max(stone_threshold + 6, int(np.float32(height_limit) * np.float32(0.82)))

    if height_i >= snow_threshold:
        return height_i, SNOW
    if height_i <= sand_threshold:
        return height_i, SAND
    if height_i >= stone_threshold and (detail + micro * np.float32(0.5) + nano * np.float32(0.35)) > np.float32(0.10):
        return height_i, STONE
    return height_i, GRASS


@njit(cache=True, fastmath=True, inline="always")
def _clamp01(value: float) -> float:
    value32 = np.float32(value)
    if value32 < np.float32(0.0):
        return np.float32(0.0)
    if value32 > np.float32(1.0):
        return np.float32(1.0)
    return value32


@njit(cache=True, fastmath=True, inline="always")
def _cave_vertical_band(world_y: int, world_height_limit: int) -> float:
    normalized_y = np.float32(world_y) / np.float32(max(1, world_height_limit - 1))
    if normalized_y <= np.float32(0.45):
        return np.float32(1.0)
    return _clamp01(np.float32(1.0) - (normalized_y - np.float32(0.45)) * np.float32(1.6))


@njit(cache=True, fastmath=True, inline="always")
def _should_carve_cave(
    world_x: int,
    world_y: int,
    world_z: int,
    surface_height: int,
    seed: int,
    world_height_limit: int,
) -> bool:
    if world_y <= CAVE_BEDROCK_CLEARANCE:
        return False
    depth_below_surface = surface_height - world_y
    if depth_below_surface <= 0:
        return False
    if world_y >= world_height_limit - 2:
        return False
    cave_model_version = CAVE_MODEL_VERSION
    if cave_model_version < 13:
        return False

    vertical_band = _cave_vertical_band(world_y, world_height_limit)
    if vertical_band <= np.float32(CAVE_ACTIVE_BAND_MIN):
        return False

    xf = np.float32(world_x)
    yf = np.float32(world_y)
    zf = np.float32(world_z)
    cave_frequency = np.float32(0.018) * np.float32(CAVE_FREQUENCY_SCALE)
    cave_primary = _value_noise_3d(xf, yf * np.float32(0.85), zf, seed + 101, cave_frequency)
    cave_detail = _value_noise_3d(
        xf,
        yf * np.float32(0.85),
        zf,
        seed + 157,
        cave_frequency * np.float32(CAVE_DETAIL_FREQUENCY_MULTIPLIER),
    )
    cave_value = cave_primary + cave_detail * np.float32(CAVE_DETAIL_WEIGHT)

    depth_bonus = np.float32(depth_below_surface) * np.float32(CAVE_DEPTH_BONUS_SCALE)
    if depth_bonus > np.float32(CAVE_DEPTH_BONUS_MAX):
        depth_bonus = np.float32(CAVE_DEPTH_BONUS_MAX)

    threshold = np.float32(CAVE_PRIMARY_THRESHOLD) - vertical_band * np.float32(CAVE_VERTICAL_BONUS) - depth_bonus
    return cave_value > threshold


@njit(cache=True, fastmath=True, inline="always")
def _terrain_material_from_surface_profile(
    world_x: int,
    world_y: int,
    world_z: int,
    surface_height: int,
    surface_material: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> int:
    if world_y < 0 or world_y >= world_height_limit:
        return AIR
    if world_y >= surface_height:
        return AIR
    if carve_caves and _should_carve_cave(world_x, world_y, world_z, surface_height, seed, world_height_limit):
        return AIR
    if world_y == 0:
        return BEDROCK
    if world_y < surface_height - 4:
        return STONE
    if world_y < surface_height - 1:
        return DIRT
    return surface_material


@njit(cache=True, fastmath=True)
def terrain_block_material_at(
    world_x: int,
    world_y: int,
    world_z: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> int:
    if world_y < 0 or world_y >= world_height_limit:
        return AIR

    surface_height, surface_material = surface_profile_at(np.float32(world_x), np.float32(world_z), seed, world_height_limit)
    return _terrain_material_from_surface_profile(
        world_x,
        world_y,
        world_z,
        int(surface_height),
        int(surface_material),
        seed,
        world_height_limit,
        carve_caves,
    )
