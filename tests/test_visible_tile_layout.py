from __future__ import annotations

from types import SimpleNamespace

from engine.visibility import tile_layout


def _fake_layout_renderer(**overrides):
    defaults = dict(
        _view_extent_neg_x=1,
        _view_extent_pos_x=1,
        _view_extent_neg_z=1,
        _view_extent_pos_z=1,
        _view_extent_neg_y=0,
        _view_extent_pos_y=0,
        fixed_view_box_mode=False,
        _visible_layout_template_cache={},
        _visible_rel_xz_order_cache={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_tile_layout_builds_center_first_visible_window():
    renderer = _fake_layout_renderer()
    rel_coords, visible_tile_keys, visible_tile_masks, rel_slot, rel_sizes, tile_base = tile_layout.tile_layout_in_view_for_origin(renderer, (0, 0, 0))

    assert rel_coords[0] == (0, 0, 0)
    assert len(rel_coords) == 9
    assert all(rel_y == 0 for _, rel_y, _ in rel_coords)
    assert tile_base == (0, 0, 0)
    assert visible_tile_keys
    assert visible_tile_masks
    assert rel_slot[(0, 0, 0)][1] == 0
    assert sum(rel_sizes.values()) == len(rel_coords)


def test_tile_bit_for_chunk_uses_merged_tile_local_slot():
    tile_key, bit = tile_layout.tile_bit_for_chunk(5, 6, 2)

    assert tile_key == (1, 2, 1)
    assert bit == 1 << ((6 - 4) * 4 + (5 - 4))


def test_vertical_bounds_shift_to_world_floor_when_view_covers_height():
    renderer = _fake_layout_renderer(_view_extent_neg_y=16, _view_extent_pos_y=16)

    assert tile_layout.visible_rel_y_bounds_for_origin_y(renderer, 20) == (-20, 11)


def test_visible_layout_includes_world_floor_when_vertical_view_covers_height():
    renderer = _fake_layout_renderer(_view_extent_neg_y=16, _view_extent_pos_y=16)

    rel_coords, *_ = tile_layout.tile_layout_in_view_for_origin(renderer, (0, 20, 0))
    visible_y_layers = {20 + rel_y for _, rel_y, _ in rel_coords}

    assert visible_y_layers == set(range(32))
    assert len(rel_coords) == 32 * 9


def test_visible_layout_clips_oversized_fixed_view_down_to_world_floor():
    renderer = _fake_layout_renderer(_view_extent_neg_y=32, _view_extent_pos_y=31, fixed_view_box_mode=True)

    rel_coords, *_ = tile_layout.tile_layout_in_view_for_origin(renderer, (0, 32, 0))
    visible_y_layers = {32 + rel_y for _, rel_y, _ in rel_coords}

    assert visible_y_layers == set(range(32))
    assert len(rel_coords) == 32 * 9


def test_vertical_bounds_keep_requested_layer_count_near_world_ceiling():
    renderer = _fake_layout_renderer(_view_extent_neg_y=4, _view_extent_pos_y=4)

    min_rel_y, max_rel_y = tile_layout.visible_rel_y_bounds_for_origin_y(renderer, 30)

    assert (min_rel_y, max_rel_y) == (-7, 1)
    assert max_rel_y - min_rel_y + 1 == 9
