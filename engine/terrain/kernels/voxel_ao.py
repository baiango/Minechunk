from __future__ import annotations

import numpy as np

from .numba_compat import njit


@njit(cache=True, fastmath=True, inline="always")
def _solid_at_with_boundaries(
    blocks: np.ndarray,
    local_x: int,
    local_z: int,
    sample_y: int,
    height_limit: int,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
) -> int:
    if sample_y < 0:
        return int(bottom_boundary[local_z, local_x] != 0)
    if sample_y >= height_limit:
        return int(top_boundary[local_z, local_x] != 0)
    return int(blocks[sample_y, local_z, local_x] != 0)


@njit(cache=True, fastmath=True, inline="always")
def _ambient_occlusion_factor(side1: int, side2: int, corner: int) -> float:
    if side1 != 0 and side2 != 0:
        occlusion = 3
    else:
        occlusion = side1 + side2 + corner
    if occlusion <= 0:
        return 1.0
    if occlusion == 1:
        return 0.82
    if occlusion == 2:
        return 0.68
    return 0.54


@njit(cache=True, fastmath=True, inline="always")
def _ao_y_from_plane(
    plane: np.ndarray,
    local_x: int,
    local_z: int,
    dx: int,
    dz: int,
) -> float:
    side1 = int(plane[local_z, local_x + dx] != 0)
    side2 = int(plane[local_z + dz, local_x] != 0)
    corner = int(plane[local_z + dz, local_x + dx] != 0)
    return _ambient_occlusion_factor(side1, side2, corner)


@njit(cache=True, fastmath=True, inline="always")
def _ao_x_from_planes(
    current_plane: np.ndarray,
    neighbor_y_plane: np.ndarray,
    sample_x: int,
    local_z: int,
    dz: int,
) -> float:
    side1 = int(neighbor_y_plane[local_z, sample_x] != 0)
    side2 = int(current_plane[local_z + dz, sample_x] != 0)
    corner = int(neighbor_y_plane[local_z + dz, sample_x] != 0)
    return _ambient_occlusion_factor(side1, side2, corner)


@njit(cache=True, fastmath=True, inline="always")
def _ao_z_from_planes(
    current_plane: np.ndarray,
    neighbor_y_plane: np.ndarray,
    local_x: int,
    sample_z: int,
    dx: int,
) -> float:
    side1 = int(current_plane[sample_z, local_x + dx] != 0)
    side2 = int(neighbor_y_plane[sample_z, local_x] != 0)
    corner = int(neighbor_y_plane[sample_z, local_x + dx] != 0)
    return _ambient_occlusion_factor(side1, side2, corner)
