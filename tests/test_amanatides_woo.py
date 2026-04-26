from engine.visibility.amanatides_woo import first_hit, iter_voxels, line_of_sight


def test_iter_voxels_x_axis_order():
    visited = [step.block for step in iter_voxels((0.05, 0.05, 0.05), (1.0, 0.0, 0.0), 0.35, block_size=0.1)]
    assert visited == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]


def test_first_hit_returns_enter_face_normal():
    hit = first_hit(
        (0.05, 0.05, 0.05),
        (1.0, 0.0, 0.0),
        lambda bx, by, bz: bx == 2 and by == 0 and bz == 0,
        1.0,
        block_size=0.1,
    )
    assert hit is not None
    assert hit.block == (2, 0, 0)
    assert hit.normal == (-1.0, 0.0, 0.0)


def test_line_of_sight_blocks_solid_between_points():
    solid = lambda bx, by, bz: bx == 2 and by == 0 and bz == 0
    assert line_of_sight((0.05, 0.05, 0.05), (0.15, 0.05, 0.05), solid, block_size=0.1)
    assert not line_of_sight((0.05, 0.05, 0.05), (0.35, 0.05, 0.05), solid, block_size=0.1)
