from types import SimpleNamespace

from engine.collision import walk_solver
from engine.terrain.kernels import AIR, STONE
from engine.world_constants import BLOCK_SIZE


class FakeWorld:
    height = 32

    def __init__(self, solids=()):
        self.solids = {tuple(coord) for coord in solids}

    def block_at(self, bx, by, bz):
        return STONE if (int(bx), int(by), int(bz)) in self.solids else AIR


def fake_renderer(solids=()):
    return SimpleNamespace(world=FakeWorld(solids), _solid_block_cache={})


def test_player_aabb_uses_configured_eye_center_extents():
    min_x, min_y, min_z, max_x, max_y, max_z = walk_solver.player_aabb([10.0, 20.0, 30.0])
    assert min_x < 10.0 < max_x
    assert min_y < 20.0 < max_y
    assert min_z < 30.0 < max_z


def test_is_block_solid_treats_below_world_as_solid_and_caches_world_reads():
    renderer = fake_renderer(solids={(1, 2, 3)})
    assert walk_solver.is_block_solid(renderer, 0, -1, 0) is True
    assert walk_solver.is_block_solid(renderer, 1, 2, 3) is True
    assert renderer._solid_block_cache[(1, 2, 3)] is True
    assert walk_solver.is_block_solid(renderer, 9, 9, 9) is False
    assert renderer._solid_block_cache[(9, 9, 9)] is False


def test_small_downward_snap_lands_on_block_top():
    renderer = fake_renderer(solids={(0, 0, 0)})
    position = [0.5 * BLOCK_SIZE, 1.5 * BLOCK_SIZE, 0.5 * BLOCK_SIZE]
    assert walk_solver.resolve_small_downward_snap(renderer, position, -0.75 * BLOCK_SIZE) is True
    assert position[1] > BLOCK_SIZE
