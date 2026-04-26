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
