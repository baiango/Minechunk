from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from engine.rendering import worldspace_rc


def test_interval_bands_are_ordered_and_nonempty():
    bands = [worldspace_rc.interval_band(i) for i in range(4)]

    assert bands[0][0] == 0.0
    for start, end in bands:
        assert end > start
    assert bands[3][1] > bands[2][1] > bands[1][1] > bands[0][1]


def test_make_update_params_shape_and_key_fields():
    renderer = SimpleNamespace(world=SimpleNamespace(seed=1234, height=128))

    params = worldspace_rc.make_update_params(
        renderer,
        cascade_index=2,
        min_corner=(1.0, 2.0, 3.0),
        full_extent=64.0,
        resolution=16,
        temporal_history_weight=0.75,
    )

    assert params.dtype == np.float32
    assert params.shape == (24,)
    assert params[0:4].tolist() == [1.0, 2.0, 3.0, 64.0]
    assert params[5] == 2.0
    assert params[8] == 1234.0
    assert params[9] == 128.0
    assert params[19] == np.float32(0.75)


def test_format_cascade_list_matches_hud_snapshot_text():
    assert worldspace_rc.format_cascade_list([]) == "-"
    assert worldspace_rc.format_cascade_list([3, 1, 0]) == "C3,C1,C0"
