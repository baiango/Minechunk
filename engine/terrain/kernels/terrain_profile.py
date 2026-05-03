from __future__ import annotations

from .numba_compat import njit
from .materials import AIR, BEDROCK, DIRT, GRASS, SAND, SNOW, STONE
from .noise import _value_noise_2d, _value_noise_3d

TERRAIN_FREQUENCY_SCALE = 0.3
CAVE_FREQUENCY_SCALE = 0.5
CAVE_BEDROCK_CLEARANCE = 3
SURFACE_BREACH_FREQUENCY_SCALE = 1.0


@njit(cache=True, fastmath=True)
def surface_profile_at(
    x: float,
    z: float,
    seed: int,
    height_limit: int,
) -> tuple[int, int]:
    sample_x = x
    sample_z = z

    broad = _value_noise_2d(sample_x, sample_z, seed + 11, 0.0009765625 * TERRAIN_FREQUENCY_SCALE)
    ridge = _value_noise_2d(sample_x, sample_z, seed + 23, 0.00390625 * TERRAIN_FREQUENCY_SCALE)
    detail = _value_noise_2d(sample_x, sample_z, seed + 47, 0.010416667 * TERRAIN_FREQUENCY_SCALE)
    micro = _value_noise_2d(sample_x, sample_z, seed + 71, 0.020833334 * TERRAIN_FREQUENCY_SCALE)
    nano = _value_noise_2d(sample_x, sample_z, seed + 97, 0.041666668 * TERRAIN_FREQUENCY_SCALE)

    upper_bound = height_limit - 1
    normalized_height = 24.0 + broad * 11.0 + ridge * 8.0 + detail * 4.5 + micro * 1.75 + nano * 0.75
    height_scale = float(upper_bound) / 50.0 if upper_bound > 0 else 1.0
    height_i = int(normalized_height * height_scale)
    if height_i < 4:
        height_i = 4
    if height_i > upper_bound:
        height_i = upper_bound

    sand_threshold = max(4, int(height_limit * 0.18))
    stone_threshold = max(sand_threshold + 6, int(height_limit * 0.58))
    snow_threshold = max(stone_threshold + 6, int(height_limit * 0.82))

    if height_i >= snow_threshold:
        return height_i, SNOW
    if height_i <= sand_threshold:
        return height_i, SAND
    if height_i >= stone_threshold and (detail + micro * 0.5 + nano * 0.35) > 0.10:
        return height_i, STONE
    return height_i, GRASS


@njit(cache=True, fastmath=True, inline="always")
def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


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
    if world_y >= world_height_limit - 2:
        return False

    normalized_y = float(world_y) / float(max(1, world_height_limit - 1))
    vertical_band = 1.0 - abs(normalized_y - 0.45) * 1.6
    vertical_band = _clamp01(vertical_band)
    if vertical_band <= 0.0:
        return False

    xf = float(world_x)
    yf = float(world_y)
    zf = float(world_z)
    cave_primary = _value_noise_3d(xf, yf * 0.85, zf, seed + 101, 0.018 * CAVE_FREQUENCY_SCALE)
    cave_detail = _value_noise_3d(xf, yf * 1.15, zf, seed + 149, 0.041666668 * CAVE_FREQUENCY_SCALE)
    cave_shape = _value_noise_3d(xf, yf * 0.35, zf, seed + 173, 0.009765625 * CAVE_FREQUENCY_SCALE)
    density = cave_primary * 0.70 + cave_detail * 0.25 - cave_shape * 0.10

    depth_bonus = float(depth_below_surface) * 0.004
    if depth_bonus > 0.12:
        depth_bonus = 0.12

    shallow_bonus = 0.0
    if depth_below_surface <= 6:
        shallow_bonus = (6.0 - float(depth_below_surface)) * (0.12 / 6.0)

    threshold = 0.62 - vertical_band * 0.08 - depth_bonus - shallow_bonus
    if density > threshold:
        return True

    if depth_below_surface <= 2:
        breach_primary = _value_noise_2d(xf, zf, seed + 211, 0.020833334 * SURFACE_BREACH_FREQUENCY_SCALE)
        breach_detail = _value_noise_3d(xf, yf, zf, seed + 233, 0.03125 * CAVE_FREQUENCY_SCALE)
        breach_density = breach_primary * 0.65 + breach_detail * 0.35
        breach_threshold = 0.78 - vertical_band * 0.06
        return breach_density > breach_threshold

    return False


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

    surface_height, surface_material = surface_profile_at(float(world_x), float(world_z), seed, world_height_limit)
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
