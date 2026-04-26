from __future__ import annotations

import math
from typing import Any

from ..render_utils import flat_forward_vector, right_vector
from ..renderer_config import (
    CAMERA_HEADROOM_METERS,
    CAMERA_MIN_HEIGHT_METERS,
    PLAYER_COLLIDER_HALF_WIDTH_METERS,
    PLAYER_COLLIDER_HEIGHT_METERS,
    PLAYER_EYE_OFFSET_METERS,
    PLAYER_GRAVITY_METERS,
    PLAYER_GROUND_SNAP_METERS,
    PLAYER_JUMP_SPEED_METERS,
    PLAYER_STEP_HEIGHT_METERS,
    SPRINT_FLY_SPEED,
    WALK_SPRINT_SPEED,
)
from ..terrain.kernels import AIR
from ..world_constants import BLOCK_SIZE
from ..render_utils import clamp

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


def player_extents() -> tuple[float, float, float]:
    return (
        float(PLAYER_COLLIDER_HALF_WIDTH_METERS),
        float(PLAYER_EYE_OFFSET_METERS),
        float(max(0.0, PLAYER_COLLIDER_HEIGHT_METERS - PLAYER_EYE_OFFSET_METERS)),
    )


def player_aabb(position: list[float] | tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    half_width, eye_down, eye_up = player_extents()
    return (
        float(position[0]) - half_width,
        float(position[1]) - eye_down,
        float(position[2]) - half_width,
        float(position[0]) + half_width,
        float(position[1]) + eye_up,
        float(position[2]) + half_width,
    )


def is_block_solid(renderer: Any, bx: int, by: int, bz: int) -> bool:
    if by < 0:
        return True
    key = (int(bx), int(by), int(bz))
    cached = renderer._solid_block_cache.get(key)
    if cached is not None:
        return bool(cached)
    is_solid = int(renderer.world.block_at(key[0], key[1], key[2])) != int(AIR)
    renderer._solid_block_cache[key] = bool(is_solid)
    return bool(is_solid)


def resolve_small_downward_snap(renderer: Any, position: list[float], delta: float) -> bool:
    if delta >= 0.0 or delta < -BLOCK_SIZE * 1.01:
        return False
    eps = 1e-6
    half_width, eye_down, _ = player_extents()
    min_x = float(position[0]) - half_width
    max_x = float(position[0]) + half_width
    min_z = float(position[2]) - half_width
    max_z = float(position[2]) + half_width
    target_min_y = float(position[1]) + float(delta) - eye_down
    probe_by = int(math.floor(target_min_y / BLOCK_SIZE))
    min_bx = int(math.floor(min_x / BLOCK_SIZE))
    max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
    min_bz = int(math.floor(min_z / BLOCK_SIZE))
    max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
    for bz in range(min_bz, max_bz + 1):
        for bx in range(min_bx, max_bx + 1):
            if is_block_solid(renderer, bx, probe_by, bz):
                position[1] = (float(probe_by) + 1.0) * BLOCK_SIZE + eye_down + eps
                return True
    return False


def resolve_collision_axis(renderer: Any, position: list[float], axis: int, delta: float) -> bool:
    if abs(delta) <= 1e-9:
        return False

    eps = 1e-6
    old_position = [float(position[0]), float(position[1]), float(position[2])]
    old_min_x, old_min_y, old_min_z, old_max_x, old_max_y, old_max_z = player_aabb(old_position)
    position[axis] += float(delta)
    min_x, min_y, min_z, max_x, max_y, max_z = player_aabb(position)
    half_width, eye_down, eye_up = player_extents()

    if axis == 0:
        min_by = int(math.floor(min_y / BLOCK_SIZE))
        max_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
        min_bz = int(math.floor(min_z / BLOCK_SIZE))
        max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
        if delta > 0.0:
            start_bx = int(math.floor((old_max_x - eps) / BLOCK_SIZE)) + 1
            end_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
            for bx in range(start_bx, end_bx + 1):
                for by in range(min_by, max_by + 1):
                    for bz in range(min_bz, max_bz + 1):
                        if is_block_solid(renderer, bx, by, bz):
                            position[0] = float(bx) * BLOCK_SIZE - half_width - eps
                            return True
        else:
            start_bx = int(math.floor(old_min_x / BLOCK_SIZE)) - 1
            end_bx = int(math.floor(min_x / BLOCK_SIZE))
            for bx in range(start_bx, end_bx - 1, -1):
                for by in range(min_by, max_by + 1):
                    for bz in range(min_bz, max_bz + 1):
                        if is_block_solid(renderer, bx, by, bz):
                            position[0] = (float(bx) + 1.0) * BLOCK_SIZE + half_width + eps
                            return True
        return False

    if axis == 1:
        if delta < 0.0:
            snap_probe = [float(old_position[0]), float(old_position[1]), float(old_position[2])]
            if resolve_small_downward_snap(renderer, snap_probe, delta):
                position[1] = snap_probe[1]
                return True
        min_bx = int(math.floor(min_x / BLOCK_SIZE))
        max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
        min_bz = int(math.floor(min_z / BLOCK_SIZE))
        max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
        if delta > 0.0:
            start_by = int(math.floor((old_max_y - eps) / BLOCK_SIZE)) + 1
            end_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
            for by in range(start_by, end_by + 1):
                for bz in range(min_bz, max_bz + 1):
                    for bx in range(min_bx, max_bx + 1):
                        if is_block_solid(renderer, bx, by, bz):
                            position[1] = float(by) * BLOCK_SIZE - eye_up - eps
                            return True
        else:
            start_by = int(math.floor(old_min_y / BLOCK_SIZE)) - 1
            end_by = int(math.floor(min_y / BLOCK_SIZE))
            for by in range(start_by, end_by - 1, -1):
                for bz in range(min_bz, max_bz + 1):
                    for bx in range(min_bx, max_bx + 1):
                        if is_block_solid(renderer, bx, by, bz):
                            position[1] = (float(by) + 1.0) * BLOCK_SIZE + eye_down + eps
                            return True
        return False

    min_bx = int(math.floor(min_x / BLOCK_SIZE))
    max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
    min_by = int(math.floor(min_y / BLOCK_SIZE))
    max_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
    if delta > 0.0:
        start_bz = int(math.floor((old_max_z - eps) / BLOCK_SIZE)) + 1
        end_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
        for bz in range(start_bz, end_bz + 1):
            for by in range(min_by, max_by + 1):
                for bx in range(min_bx, max_bx + 1):
                    if is_block_solid(renderer, bx, by, bz):
                        position[2] = float(bz) * BLOCK_SIZE - half_width - eps
                        return True
    else:
        start_bz = int(math.floor(old_min_z / BLOCK_SIZE)) - 1
        end_bz = int(math.floor(min_z / BLOCK_SIZE))
        for bz in range(start_bz, end_bz - 1, -1):
            for by in range(min_by, max_by + 1):
                for bx in range(min_bx, max_bx + 1):
                    if is_block_solid(renderer, bx, by, bz):
                        position[2] = (float(bz) + 1.0) * BLOCK_SIZE + half_width + eps
                        return True
    return False


@profile
def position_is_clear(renderer: Any, position: list[float]) -> bool:
    min_x, min_y, min_z, max_x, max_y, max_z = player_aabb(position)
    eps = 1e-6
    min_bx = int(math.floor(min_x / BLOCK_SIZE))
    max_bx = int(math.floor((max_x - eps) / BLOCK_SIZE))
    min_by = int(math.floor(min_y / BLOCK_SIZE))
    max_by = int(math.floor((max_y - eps) / BLOCK_SIZE))
    min_bz = int(math.floor(min_z / BLOCK_SIZE))
    max_bz = int(math.floor((max_z - eps) / BLOCK_SIZE))
    for by in range(min_by, max_by + 1):
        for bz in range(min_bz, max_bz + 1):
            for bx in range(min_bx, max_bx + 1):
                if is_block_solid(renderer, bx, by, bz):
                    return False
    return True


@profile
def move_horizontal_with_step(renderer: Any, position: list[float], axis: int, delta: float) -> bool:
    if abs(delta) <= 1e-9:
        return False
    trial = [float(position[0]), float(position[1]), float(position[2])]
    collided = resolve_collision_axis(renderer, trial, axis, delta)
    if not collided:
        position[:] = trial
        return False
    if not renderer.walk_mode or not renderer._camera_on_ground:
        position[:] = trial
        return True
    stepped = [float(position[0]), float(position[1]), float(position[2])]
    resolve_collision_axis(renderer, stepped, 1, float(PLAYER_STEP_HEIGHT_METERS))
    if stepped[1] <= position[1] + 1e-5 or not position_is_clear(renderer, stepped):
        position[:] = trial
        return True
    step_trial = [float(stepped[0]), float(stepped[1]), float(stepped[2])]
    if resolve_collision_axis(renderer, step_trial, axis, delta):
        position[:] = trial
        return True
    resolve_collision_axis(renderer, step_trial, 1, -float(PLAYER_GROUND_SNAP_METERS))
    position[:] = step_trial
    return False


def snap_to_ground(renderer: Any, position: list[float]) -> bool:
    probe = [float(position[0]), float(position[1]), float(position[2])]
    collided = resolve_collision_axis(renderer, probe, 1, -float(PLAYER_GROUND_SNAP_METERS))
    if collided:
        position[:] = probe
        return True
    return False


@profile
def update_camera_walk(renderer: Any, dt: float) -> None:
    renderer._solid_block_cache.clear()
    sprinting = renderer._key_active("shift", "shiftleft", "shiftright")
    speed = WALK_SPRINT_SPEED if sprinting else renderer.camera.move_speed
    renderer._current_move_speed = float(speed)

    move_x = 0.0
    move_z = 0.0
    forward = flat_forward_vector(renderer.camera.yaw)
    right = right_vector(renderer.camera.yaw)
    if renderer._key_active("w", "arrowup"):
        move_x += forward[0]
        move_z += forward[2]
    if renderer._key_active("s", "arrowdown"):
        move_x -= forward[0]
        move_z -= forward[2]
    if renderer._key_active("d", "arrowright"):
        move_x += right[0]
        move_z += right[2]
    if renderer._key_active("a", "arrowleft"):
        move_x -= right[0]
        move_z -= right[2]

    move_len = math.sqrt(move_x * move_x + move_z * move_z)
    if move_len > 0.0:
        move_x /= move_len
        move_z /= move_len
    desired_dx = move_x * speed * dt
    desired_dz = move_z * speed * dt

    if renderer._camera_on_ground and renderer._jump_queued:
        renderer._walk_velocity[1] = float(PLAYER_JUMP_SPEED_METERS)
        renderer._camera_on_ground = False
    renderer._jump_queued = False

    if renderer._camera_on_ground and desired_dx == 0.0 and desired_dz == 0.0 and renderer._walk_velocity[1] <= 0.0:
        position = [float(renderer.camera.position[0]), float(renderer.camera.position[1]), float(renderer.camera.position[2])]
        if resolve_small_downward_snap(renderer, position, -float(PLAYER_GROUND_SNAP_METERS)):
            renderer._walk_velocity[1] = 0.0
            renderer.camera.position[:] = position
            return

    renderer._walk_velocity[1] -= float(PLAYER_GRAVITY_METERS) * dt

    position = [float(renderer.camera.position[0]), float(renderer.camera.position[1]), float(renderer.camera.position[2])]
    if desired_dx != 0.0:
        move_horizontal_with_step(renderer, position, 0, desired_dx)
    if desired_dz != 0.0:
        move_horizontal_with_step(renderer, position, 2, desired_dz)

    vertical_delta = renderer._walk_velocity[1] * dt
    if vertical_delta <= 0.0:
        snap_delta = min(vertical_delta, -float(PLAYER_GROUND_SNAP_METERS))
        collided_y = resolve_small_downward_snap(renderer, position, snap_delta)
        if not collided_y:
            collided_y = resolve_collision_axis(renderer, position, 1, snap_delta)
        if collided_y:
            renderer._camera_on_ground = True
            renderer._walk_velocity[1] = 0.0
        else:
            renderer._camera_on_ground = False
    else:
        collided_y = resolve_collision_axis(renderer, position, 1, vertical_delta)
        if collided_y:
            renderer._walk_velocity[1] = 0.0
        else:
            renderer._camera_on_ground = False

    renderer.camera.position[:] = position


@profile
def update_camera(renderer: Any, dt: float) -> None:
    if renderer.walk_mode:
        update_camera_walk(renderer, dt)
    else:
        renderer._update_camera_fly(dt)
        renderer._walk_velocity[:] = [0.0, 0.0, 0.0]
        renderer._camera_on_ground = False
        renderer._jump_queued = False
    renderer.camera.position[1] = clamp(
        renderer.camera.position[1],
        CAMERA_MIN_HEIGHT_METERS,
        renderer.world.height * BLOCK_SIZE + CAMERA_HEADROOM_METERS,
    )
