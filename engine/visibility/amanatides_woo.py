from __future__ import annotations

"""Fast 3D voxel traversal using the Amanatides-Woo algorithm.

The traversal works in world-space metres while visiting integer voxel/block
coordinates.  It is intended for block picking, line-of-sight tests, visibility
queries, and CPU-side debugging of the GPU/WGSL DDA used by the renderer.
"""

from dataclasses import dataclass
import math
from typing import Callable, Iterator, Sequence

from ..world_constants import BLOCK_SIZE

Vec3 = tuple[float, float, float]
BlockCoord = tuple[int, int, int]
BlockPredicate = Callable[[int, int, int], bool]
MaterialSampler = Callable[[int, int, int], int]

_EPS = 1.0e-9
_INF = float("inf")


@dataclass(frozen=True, slots=True)
class VoxelRayStep:
    """One voxel visited by an Amanatides-Woo traversal."""

    block: BlockCoord
    t_enter: float
    t_exit: float
    enter_normal: Vec3
    step_index: int


@dataclass(frozen=True, slots=True)
class VoxelRayHit:
    """First solid voxel hit by a ray traversal."""

    block: BlockCoord
    position: Vec3
    distance: float
    normal: Vec3
    material: int | None
    step_index: int


def _as_vec3(value: Sequence[float], *, name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 components")
    return (float(value[0]), float(value[1]), float(value[2]))


def _normalize(value: Sequence[float]) -> Vec3:
    x, y, z = _as_vec3(value, name="direction")
    length_sq = x * x + y * y + z * z
    if length_sq <= _EPS:
        raise ValueError("direction must be non-zero")
    inv = 1.0 / math.sqrt(length_sq)
    return (x * inv, y * inv, z * inv)


def _axis_step(direction_component: float) -> int:
    if direction_component > _EPS:
        return 1
    if direction_component < -_EPS:
        return -1
    return 0


def _axis_t_delta(direction_component: float, block_size: float) -> float:
    if abs(direction_component) <= _EPS:
        return _INF
    return abs(block_size / direction_component)


def _axis_t_max(origin_axis: float, direction_axis: float, block_axis: int, step_axis: int, block_size: float) -> float:
    if step_axis > 0:
        boundary = (block_axis + 1) * block_size
    elif step_axis < 0:
        boundary = block_axis * block_size
    else:
        return _INF
    value = (boundary - origin_axis) / direction_axis
    # Negative zero or tiny negative values happen when the ray starts exactly on
    # a face. Clamp them to zero so the first traversal step is deterministic.
    return max(0.0, float(value))


def _position_at(origin: Vec3, direction: Vec3, distance: float) -> Vec3:
    return (
        origin[0] + direction[0] * distance,
        origin[1] + direction[1] * distance,
        origin[2] + direction[2] * distance,
    )


def block_from_world(position: Sequence[float], block_size: float = BLOCK_SIZE) -> BlockCoord:
    """Return the integer block coordinate containing ``position``."""

    if block_size <= 0.0:
        raise ValueError("block_size must be positive")
    px, py, pz = _as_vec3(position, name="position")
    return (
        math.floor(px / block_size),
        math.floor(py / block_size),
        math.floor(pz / block_size),
    )


def iter_voxels(
    origin: Sequence[float],
    direction: Sequence[float],
    max_distance: float,
    *,
    block_size: float = BLOCK_SIZE,
    start_distance: float = 0.0,
    max_steps: int | None = None,
) -> Iterator[VoxelRayStep]:
    """Yield every block crossed by a ray, in traversal order.

    Distances are measured in world-space metres along the normalized ray.
    ``start_distance`` lets callers skip a small self-intersection bias or start
    at a cascade interval boundary. ``max_steps`` is an optional safety/budget cap.
    """

    if block_size <= 0.0:
        raise ValueError("block_size must be positive")
    if max_distance < 0.0:
        return
    if max_steps is not None and max_steps <= 0:
        return

    ray_origin = _as_vec3(origin, name="origin")
    ray_dir = _normalize(direction)
    start_t = max(0.0, float(start_distance))
    end_t = float(max_distance)
    if start_t > end_t:
        return

    start_pos = _position_at(ray_origin, ray_dir, start_t)
    bx, by, bz = block_from_world(start_pos, block_size)

    step_x = _axis_step(ray_dir[0])
    step_y = _axis_step(ray_dir[1])
    step_z = _axis_step(ray_dir[2])

    t_max_x = start_t + _axis_t_max(start_pos[0], ray_dir[0], bx, step_x, block_size)
    t_max_y = start_t + _axis_t_max(start_pos[1], ray_dir[1], by, step_y, block_size)
    t_max_z = start_t + _axis_t_max(start_pos[2], ray_dir[2], bz, step_z, block_size)

    t_delta_x = _axis_t_delta(ray_dir[0], block_size)
    t_delta_y = _axis_t_delta(ray_dir[1], block_size)
    t_delta_z = _axis_t_delta(ray_dir[2], block_size)

    t_enter = start_t
    normal: Vec3 = (0.0, 0.0, 0.0)
    step_index = 0

    while t_enter <= end_t + _EPS:
        t_exit = min(t_max_x, t_max_y, t_max_z, end_t)
        yield VoxelRayStep(
            block=(int(bx), int(by), int(bz)),
            t_enter=float(t_enter),
            t_exit=float(t_exit),
            enter_normal=normal,
            step_index=step_index,
        )

        step_index += 1
        if max_steps is not None and step_index >= max_steps:
            break
        if t_exit >= end_t:
            break

        # Classic Amanatides-Woo: advance along the nearest grid plane. Ties are
        # resolved deterministically X -> Y -> Z to avoid double-visiting cells.
        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            bx += step_x
            t_enter = t_max_x
            t_max_x += t_delta_x
            normal = (-float(step_x), 0.0, 0.0)
        elif t_max_y <= t_max_z:
            by += step_y
            t_enter = t_max_y
            t_max_y += t_delta_y
            normal = (0.0, -float(step_y), 0.0)
        else:
            bz += step_z
            t_enter = t_max_z
            t_max_z += t_delta_z
            normal = (0.0, 0.0, -float(step_z))


def first_hit(
    origin: Sequence[float],
    direction: Sequence[float],
    is_solid: BlockPredicate,
    max_distance: float,
    *,
    block_size: float = BLOCK_SIZE,
    start_distance: float = 0.0,
    max_steps: int | None = None,
    material_at: MaterialSampler | None = None,
) -> VoxelRayHit | None:
    """Return the first solid block hit by a ray, or ``None`` for a miss."""

    ray_origin = _as_vec3(origin, name="origin")
    ray_dir = _normalize(direction)
    for step in iter_voxels(
        ray_origin,
        ray_dir,
        max_distance,
        block_size=block_size,
        start_distance=start_distance,
        max_steps=max_steps,
    ):
        bx, by, bz = step.block
        if not is_solid(bx, by, bz):
            continue
        distance = max(0.0, step.t_enter)
        return VoxelRayHit(
            block=step.block,
            position=_position_at(ray_origin, ray_dir, distance),
            distance=distance,
            normal=step.enter_normal,
            material=None if material_at is None else int(material_at(bx, by, bz)),
            step_index=step.step_index,
        )
    return None


def line_of_sight(
    origin: Sequence[float],
    target: Sequence[float],
    is_solid: BlockPredicate,
    *,
    block_size: float = BLOCK_SIZE,
    start_bias: float | None = None,
    end_bias: float | None = None,
    max_steps: int | None = None,
) -> bool:
    """Return ``True`` if no solid block lies between ``origin`` and ``target``."""

    ox, oy, oz = _as_vec3(origin, name="origin")
    tx, ty, tz = _as_vec3(target, name="target")
    direction = (tx - ox, ty - oy, tz - oz)
    distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if distance <= _EPS:
        return True
    sb = block_size * 1.0e-4 if start_bias is None else max(0.0, float(start_bias))
    eb = block_size * 1.0e-4 if end_bias is None else max(0.0, float(end_bias))
    max_dist = max(0.0, distance - eb)
    return first_hit(
        (ox, oy, oz),
        direction,
        is_solid,
        max_dist,
        block_size=block_size,
        start_distance=sb,
        max_steps=max_steps,
    ) is None
