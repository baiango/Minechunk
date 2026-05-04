from __future__ import annotations

import math

import numpy as np

from .numba_compat import njit

@njit(cache=True, fastmath=True, inline="always")
def _mix_u32(value: int) -> int:
    value = value & 0xFFFFFFFF
    value = (value ^ (value >> 16)) & 0xFFFFFFFF
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value = (value ^ (value >> 15)) & 0xFFFFFFFF
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value = (value ^ (value >> 16)) & 0xFFFFFFFF
    return value


@njit(cache=True, fastmath=True, inline="always")
def _hash2(ix: int, iy: int, seed: int) -> float:
    h = ((ix & 0xFFFFFFFF) * 0x9E3779B9) & 0xFFFFFFFF
    h = (h ^ (((iy & 0xFFFFFFFF) * 0x85EBCA6B) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = (h ^ (((seed & 0xFFFFFFFF) * 0xC2B2AE35) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = _mix_u32(h)
    return np.float32(h & 0x00FFFFFF) / np.float32(16777215.0)


@njit(cache=True, fastmath=True, inline="always")
def _hash3(ix: int, iy: int, iz: int, seed: int) -> float:
    h = ((ix & 0xFFFFFFFF) * 0x9E3779B9) & 0xFFFFFFFF
    h = (h ^ (((iy & 0xFFFFFFFF) * 0x85EBCA6B) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = (h ^ (((iz & 0xFFFFFFFF) * 0xC2B2AE35) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = (h ^ (((seed & 0xFFFFFFFF) * 0x27D4EB2F) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = _mix_u32(h)
    return np.float32(h & 0x00FFFFFF) / np.float32(16777215.0)


@njit(cache=True, fastmath=True, inline="always")
def _fade(t: float) -> float:
    t32 = np.float32(t)
    return t32 * t32 * t32 * (t32 * (t32 * np.float32(6.0) - np.float32(15.0)) + np.float32(10.0))


@njit(cache=True, fastmath=True, inline="always")
def _lerp(a: float, b: float, t: float) -> float:
    a32 = np.float32(a)
    b32 = np.float32(b)
    t32 = np.float32(t)
    return a32 + (b32 - a32) * t32


@njit(cache=True, fastmath=True, inline="always")
def _value_noise_2d(x: float, y: float, seed: int, frequency: float) -> float:
    x = np.float32(x) * np.float32(frequency)
    y = np.float32(y) * np.float32(frequency)

    x0 = math.floor(x)
    y0 = math.floor(y)
    xf = x - x0
    yf = y - y0

    ix0 = int(x0)
    iy0 = int(y0)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    v00 = _hash2(ix0, iy0, seed)
    v10 = _hash2(ix1, iy0, seed)
    v01 = _hash2(ix0, iy1, seed)
    v11 = _hash2(ix1, iy1, seed)

    u = _fade(xf)
    v = _fade(yf)
    nx0 = _lerp(v00, v10, u)
    nx1 = _lerp(v01, v11, u)
    return _lerp(nx0, nx1, v) * np.float32(2.0) - np.float32(1.0)


@njit(cache=True, fastmath=True, inline="always")
def _value_noise_3d(x: float, y: float, z: float, seed: int, frequency: float) -> float:
    x = np.float32(x) * np.float32(frequency)
    y = np.float32(y) * np.float32(frequency)
    z = np.float32(z) * np.float32(frequency)

    x0 = math.floor(x)
    y0 = math.floor(y)
    z0 = math.floor(z)
    xf = x - x0
    yf = y - y0
    zf = z - z0

    ix0 = int(x0)
    iy0 = int(y0)
    iz0 = int(z0)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    c000 = _hash3(ix0, iy0, iz0, seed)
    c100 = _hash3(ix1, iy0, iz0, seed)
    c010 = _hash3(ix0, iy1, iz0, seed)
    c110 = _hash3(ix1, iy1, iz0, seed)
    c001 = _hash3(ix0, iy0, iz1, seed)
    c101 = _hash3(ix1, iy0, iz1, seed)
    c011 = _hash3(ix0, iy1, iz1, seed)
    c111 = _hash3(ix1, iy1, iz1, seed)

    x00 = _lerp(c000, c100, u)
    x10 = _lerp(c010, c110, u)
    x01 = _lerp(c001, c101, u)
    x11 = _lerp(c011, c111, u)
    y0v = _lerp(x00, x10, v)
    y1v = _lerp(x01, x11, v)
    return _lerp(y0v, y1v, w) * np.float32(2.0) - np.float32(1.0)
