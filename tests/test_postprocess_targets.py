from types import SimpleNamespace

from engine.rendering import postprocess_targets
from engine.renderer_config import RADIANCE_CASCADES_CASCADE_COUNT, WORLDSPACE_RC_GRID_RESOLUTION


def test_make_gi_params_uses_renderer_debug_mode_and_override():
    renderer = SimpleNamespace(rc_debug_mode=6)
    default_params = postprocess_targets.make_gi_params(renderer)
    override_params = postprocess_targets.make_gi_params(renderer, rc_debug_mode=2)

    assert default_params.shape == (8,)
    assert float(default_params[4]) == 6.0
    assert float(override_params[4]) == 2.0
    assert float(default_params[6]) == float(RADIANCE_CASCADES_CASCADE_COUNT)
    assert float(default_params[7]) == float(WORLDSPACE_RC_GRID_RESOLUTION)
