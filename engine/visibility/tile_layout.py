from __future__ import annotations

"""Visible chunk/tile layout helpers.

This module owns the geometry for chunk visibility windows and merged-tile slot
assignment.  It deliberately mutates only the small renderer caches passed to it;
GPU resource ownership stays outside this module.
"""

from typing import Any

from ..renderer_config import MERGED_TILE_SIZE_CHUNKS
from ..world_constants import VERTICAL_CHUNK_COUNT, VERTICAL_CHUNK_STACK_ENABLED

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


def visible_rel_y_bounds_for_origin_y(renderer: Any, origin_y: int) -> tuple[int, int]:
    if not VERTICAL_CHUNK_STACK_ENABLED:
        return (0, 0)
    requested_min_rel_y = -int(renderer._view_extent_neg_y)
    requested_max_rel_y = int(renderer._view_extent_pos_y)
    min_rel_y = max(requested_min_rel_y, -int(origin_y))
    max_rel_y = min(requested_max_rel_y, int(VERTICAL_CHUNK_COUNT - 1 - int(origin_y)))
    if min_rel_y > max_rel_y:
        return (0, 0)
    return (int(min_rel_y), int(max_rel_y))


@profile
def build_visible_layout_template(
    renderer: Any,
    origin_mod_x: int,
    origin_mod_z: int,
    origin_y: int,
) -> tuple[
    tuple[tuple[int, int, int], ...],
    tuple[tuple[int, int, int], ...],
    tuple[int, ...],
    dict[tuple[int, int, int], tuple[tuple[int, int, int], int]],
    dict[tuple[int, int, int], int],
]:
    tile_size = int(MERGED_TILE_SIZE_CHUNKS)
    neg_x = int(renderer._view_extent_neg_x)
    pos_x = int(renderer._view_extent_pos_x)
    neg_z = int(renderer._view_extent_neg_z)
    pos_z = int(renderer._view_extent_pos_z)
    min_rel_y, max_rel_y = visible_rel_y_bounds_for_origin_y(renderer, int(origin_y))
    template_key = (
        neg_x,
        pos_x,
        min_rel_y,
        max_rel_y,
        neg_z,
        pos_z,
        tile_size,
        int(origin_mod_x),
        int(origin_mod_z),
        1 if VERTICAL_CHUNK_STACK_ENABLED else 0,
        1 if renderer.fixed_view_box_mode else 0,
    )
    cached = renderer._visible_layout_template_cache.get(template_key)
    if cached is not None:
        return cached

    if VERTICAL_CHUNK_STACK_ENABLED:
        cy_order: list[int] = [0]
        max_offset = max(-min_rel_y, max_rel_y)
        for offset in range(1, max_offset + 1):
            up = offset
            down = -offset
            if up <= max_rel_y:
                cy_order.append(up)
            if down >= min_rel_y:
                cy_order.append(down)
    else:
        cy_order = [0]

    rel_coords: list[tuple[int, int, int]] = []
    rel_tile_keys: list[tuple[int, int, int]] = []
    rel_tile_masks: dict[tuple[int, int, int], int] = {}
    rel_coord_to_tile_slot: dict[tuple[int, int, int], tuple[tuple[int, int, int], int]] = {}
    rel_tile_slot_sizes: dict[tuple[int, int, int], int] = {}
    base_tile_x = int(origin_mod_x) // tile_size
    base_tile_z = int(origin_mod_z) // tile_size

    rel_xz_cache_key = (neg_x, pos_x, neg_z, pos_z, 1 if renderer.fixed_view_box_mode else 0)
    rel_xz_order = renderer._visible_rel_xz_order_cache.get(rel_xz_cache_key)
    if rel_xz_order is None:
        rel_xz = [
            (dx, dz)
            for dz in range(-neg_z, pos_z + 1)
            for dx in range(-neg_x, pos_x + 1)
        ]
        rel_xz.sort(key=lambda delta: (delta[0] * delta[0] + delta[1] * delta[1], abs(delta[1]), abs(delta[0]), delta[1], delta[0]))
        rel_xz_order = tuple(rel_xz)
        renderer._visible_rel_xz_order_cache[rel_xz_cache_key] = rel_xz_order

    for rel_y in cy_order:
        for dx, dz in rel_xz_order:
            rel_coord = (dx, rel_y, dz)
            rel_coords.append(rel_coord)
            abs_x = int(origin_mod_x) + dx
            abs_z = int(origin_mod_z) + dz
            tile_x = abs_x // tile_size
            tile_z = abs_z // tile_size
            local_x = abs_x - tile_x * tile_size
            local_z = abs_z - tile_z * tile_size
            rel_tile_key = (tile_x - base_tile_x, rel_y, tile_z - base_tile_z)
            slot_index = int(rel_tile_slot_sizes.get(rel_tile_key, 0))
            if slot_index == 0:
                rel_tile_keys.append(rel_tile_key)
            rel_coord_to_tile_slot[rel_coord] = (rel_tile_key, slot_index)
            rel_tile_slot_sizes[rel_tile_key] = slot_index + 1
            if 0 <= local_x < tile_size and 0 <= local_z < tile_size:
                rel_tile_masks[rel_tile_key] = int(rel_tile_masks.get(rel_tile_key, 0)) | int(1 << (local_z * tile_size + local_x))

    rel_tile_mask_values = tuple(int(rel_tile_masks.get(rel_tile_key, 0)) for rel_tile_key in rel_tile_keys)
    template = (
        tuple(rel_coords),
        tuple(rel_tile_keys),
        rel_tile_mask_values,
        rel_coord_to_tile_slot,
        rel_tile_slot_sizes,
    )
    renderer._visible_layout_template_cache[template_key] = template
    return template


@profile
def tile_layout_in_view_for_origin(
    renderer: Any,
    origin: tuple[int, int, int],
) -> tuple[
    tuple[tuple[int, int, int], ...],
    list[tuple[int, int, int]],
    dict[tuple[int, int, int], int],
    dict[tuple[int, int, int], tuple[tuple[int, int, int], int]],
    dict[tuple[int, int, int], int],
    tuple[int, int, int],
]:
    chunk_x = int(origin[0])
    chunk_y = int(origin[1])
    chunk_z = int(origin[2])
    tile_size = int(MERGED_TILE_SIZE_CHUNKS)
    rel_coords, rel_tile_keys, rel_tile_mask_values, rel_coord_to_tile_slot, rel_tile_slot_sizes = build_visible_layout_template(
        renderer,
        chunk_x % tile_size,
        chunk_z % tile_size,
        chunk_y,
    )
    base_tile_x = chunk_x // tile_size
    base_tile_z = chunk_z // tile_size
    tile_base = (base_tile_x, chunk_y, base_tile_z)
    visible_tile_keys: list[tuple[int, int, int]] = []
    visible_tile_masks: dict[tuple[int, int, int], int] = {}
    append_visible_tile_key = visible_tile_keys.append
    for index, (tx, ty, tz) in enumerate(rel_tile_keys):
        tile_key_value = (base_tile_x + tx, chunk_y + ty, base_tile_z + tz)
        append_visible_tile_key(tile_key_value)
        mask_value = int(rel_tile_mask_values[index])
        if mask_value != 0:
            visible_tile_masks[tile_key_value] = mask_value
    return rel_coords, visible_tile_keys, visible_tile_masks, rel_coord_to_tile_slot, rel_tile_slot_sizes, tile_base


@profile
def chunk_coords_and_tile_layout_in_view_for_origin(
    renderer: Any,
    origin: tuple[int, int, int],
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    dict[tuple[int, int, int], int],
    dict[tuple[int, int, int], tuple[tuple[int, int, int], int]],
    dict[tuple[int, int, int], int],
    tuple[int, int, int],
]:
    rel_coords, visible_tile_keys, visible_tile_masks, rel_coord_to_tile_slot, rel_tile_slot_sizes, tile_base = tile_layout_in_view_for_origin(renderer, origin)
    chunk_x = int(origin[0])
    chunk_y = int(origin[1])
    chunk_z = int(origin[2])
    coords = [(chunk_x + dx, chunk_y + dy, chunk_z + dz) for dx, dy, dz in rel_coords]
    return coords, visible_tile_keys, visible_tile_masks, rel_coord_to_tile_slot, rel_tile_slot_sizes, tile_base


def chunk_coords_in_view_for_origin(renderer: Any, origin: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    coords, _, _, _, _, _ = chunk_coords_and_tile_layout_in_view_for_origin(renderer, origin)
    return coords


def chunk_coords_in_view(renderer: Any) -> list[tuple[int, int, int]]:
    origin = renderer._current_chunk_origin()
    return chunk_coords_in_view_for_origin(renderer, origin)


def tile_key_for_chunk(chunk_x: int, chunk_z: int, chunk_y: int = 0) -> tuple[int, int, int]:
    tile_size = int(MERGED_TILE_SIZE_CHUNKS)
    return (int(chunk_x) // tile_size, int(chunk_y), int(chunk_z) // tile_size)


def tile_bit_for_chunk(chunk_x: int, chunk_z: int, chunk_y: int = 0) -> tuple[tuple[int, int, int], int]:
    tile_size = int(MERGED_TILE_SIZE_CHUNKS)
    tile_key_value = tile_key_for_chunk(int(chunk_x), int(chunk_z), int(chunk_y))
    local_x = int(chunk_x) - tile_key_value[0] * tile_size
    local_z = int(chunk_z) - tile_key_value[2] * tile_size
    if 0 <= local_x < tile_size and 0 <= local_z < tile_size:
        return tile_key_value, 1 << (local_z * tile_size + local_x)
    return tile_key_value, 0


def visible_tile_slot_info_for_coord(renderer: Any, coord: tuple[int, int, int]):
    origin = renderer._visible_chunk_origin
    if origin is None:
        return None
    rel_coord = (
        int(coord[0]) - int(origin[0]),
        int(coord[1]) - int(origin[1]),
        int(coord[2]) - int(origin[2]),
    )
    rel_slot = renderer._visible_rel_coord_to_tile_slot.get(rel_coord)
    if rel_slot is None:
        return None
    rel_tile_key, slot_index = rel_slot
    tile_base = renderer._visible_tile_base
    tile_key_value = (
        int(tile_base[0]) + int(rel_tile_key[0]),
        int(tile_base[1]) + int(rel_tile_key[1]),
        int(tile_base[2]) + int(rel_tile_key[2]),
    )
    slots = renderer._visible_tile_mesh_slots.get(tile_key_value)
    if slots is None:
        slot_count = int(renderer._visible_rel_tile_slot_sizes.get(rel_tile_key, 0))
        if slot_count <= 0:
            return None
        slots = [None] * slot_count
        renderer._visible_tile_mesh_slots[tile_key_value] = slots
    return tile_key_value, int(slot_index), slots
