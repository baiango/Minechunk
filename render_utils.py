from __future__ import annotations

import math
import struct


_CHUNK_RADIUS_CACHE: dict[int, tuple[float, float]] = {}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalize3(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        return 0.0, 0.0, 0.0
    return x / length, y / length, z / length


def dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def pack_camera_uniform(
    position: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    forward: tuple[float, float, float],
    focal: float,
    aspect: float,
    near: float,
    far: float,
    light_dir: tuple[float, float, float],
) -> bytes:
    return struct.pack(
        "<20f",
        position[0],
        position[1],
        position[2],
        0.0,
        right[0],
        right[1],
        right[2],
        0.0,
        up[0],
        up[1],
        up[2],
        0.0,
        forward[0],
        forward[1],
        forward[2],
        0.0,
        focal,
        aspect,
        near,
        far,
    )


def pack_vertex(
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    color: tuple[float, float, float],
    alpha: float = 1.0,
) -> bytes:
    return struct.pack(
        "<4f4f4f",
        position[0],
        position[1],
        position[2],
        1.0,
        normal[0],
        normal[1],
        normal[2],
        0.0,
        color[0],
        color[1],
        color[2],
        alpha,
    )


def screen_to_ndc(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    return (x / width) * 2.0 - 1.0, 1.0 - (y / height) * 2.0


def forward_vector(yaw: float, pitch: float) -> tuple[float, float, float]:
    cp = math.cos(pitch)
    return math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp


def flat_forward_vector(yaw: float) -> tuple[float, float, float]:
    return math.sin(yaw), 0.0, math.cos(yaw)


def right_vector(yaw: float) -> tuple[float, float, float]:
    return -math.cos(yaw), 0.0, math.sin(yaw)


def chunk_height_center_and_radius(max_height: int, chunk_half: float) -> tuple[float, float]:
    cached = _CHUNK_RADIUS_CACHE.get(int(max_height))
    if cached is not None:
        return cached
    half_height = float(max_height) * 0.5
    radius = float(math.sqrt(chunk_half * chunk_half * 2.0 + half_height * half_height))
    cached = (half_height, radius)
    _CHUNK_RADIUS_CACHE[int(max_height)] = cached
    return cached
