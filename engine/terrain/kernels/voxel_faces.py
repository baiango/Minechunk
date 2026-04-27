from __future__ import annotations

import numpy as np

from .numba_compat import njit
from .materials import VERTICES_PER_FACE


FACE_TOP = np.uint8(1)
FACE_BOTTOM = np.uint8(2)
FACE_EAST = np.uint8(4)
FACE_WEST = np.uint8(8)
FACE_SOUTH = np.uint8(16)
FACE_NORTH = np.uint8(32)


@njit(cache=True, fastmath=True)
def _build_chunk_face_masks_with_boundaries(
    blocks: np.ndarray,
    chunk_size: int,
    height_limit: int,
    top_boundary: np.ndarray,
    bottom_boundary: np.ndarray,
) -> tuple[np.ndarray, int]:
    sample_size = chunk_size + 2
    end = sample_size - 1
    last_y = height_limit - 1
    face_masks = np.zeros((height_limit, sample_size, sample_size), dtype=np.uint8)
    vertex_count = 0

    for y in range(height_limit):
        plane = blocks[y]
        plane_above = blocks[y + 1] if y < last_y else top_boundary
        plane_below = blocks[y - 1] if y > 0 else bottom_boundary
        for local_z in range(1, end):
            row = plane[local_z]
            row_north = plane[local_z - 1]
            row_south = plane[local_z + 1]
            row_above = plane_above[local_z]
            row_below = plane_below[local_z]
            mask_row = face_masks[y, local_z]
            for local_x in range(1, end):
                if row[local_x] == 0:
                    continue

                mask = np.uint8(0)
                if row_above[local_x] == 0:
                    mask |= FACE_TOP
                    vertex_count += VERTICES_PER_FACE
                if row_below[local_x] == 0:
                    mask |= FACE_BOTTOM
                    vertex_count += VERTICES_PER_FACE
                if row[local_x + 1] == 0:
                    mask |= FACE_EAST
                    vertex_count += VERTICES_PER_FACE
                if row[local_x - 1] == 0:
                    mask |= FACE_WEST
                    vertex_count += VERTICES_PER_FACE
                if row_south[local_x] == 0:
                    mask |= FACE_SOUTH
                    vertex_count += VERTICES_PER_FACE
                if row_north[local_x] == 0:
                    mask |= FACE_NORTH
                    vertex_count += VERTICES_PER_FACE
                mask_row[local_x] = mask

    return face_masks, vertex_count
