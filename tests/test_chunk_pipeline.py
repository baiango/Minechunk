from collections import deque
from collections import OrderedDict
from types import SimpleNamespace


def test_rebuild_chunk_request_queue_keeps_nearest_persistent_coords() -> None:
    from engine.pipelines.chunk_pipeline import chunk_prep_priority, rebuild_chunk_request_queue

    missing_coords = {
        (x, y, z)
        for x in range(-6, 7)
        for y in range(-3, 4)
        for z in range(-6, 7)
    }
    renderer = SimpleNamespace(
        _visible_missing_coords=missing_coords,
        _chunk_request_queue=deque(),
        _chunk_request_queue_origin=None,
        _chunk_request_queue_view_signature=None,
        _chunk_request_queue_dirty=True,
        _chunk_request_view_stride=1,
        mesh_batch_size=1,
        terrain_batch_size=1,
        world=SimpleNamespace(_backend=None),
    )

    rebuild_chunk_request_queue(renderer, 0, 0, 0)

    expected = sorted(
        missing_coords,
        key=lambda coord: chunk_prep_priority(renderer, coord[0], coord[1], coord[2], 0, 0, 0),
    )[: len(renderer._chunk_request_queue)]
    assert list(renderer._chunk_request_queue) == expected
    assert renderer._chunk_request_target_coords == set(expected)
    assert renderer._chunk_request_queue_origin == (0, 0, 0)
    assert renderer._chunk_request_queue_dirty is False


def test_visible_missing_rebuild_preserves_request_queue_for_display_updates() -> None:
    from engine.visibility.coord_manager import rebuild_visible_missing_tracking

    queued = deque([(2, 0, 0), (3, 0, 0)])
    renderer = SimpleNamespace(
        _visible_chunk_coord_set={(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)},
        _visible_chunk_coords=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
        chunk_cache=OrderedDict({(0, 0, 0): object()}),
        max_cached_chunks=16,
        _visible_missing_coords=set(),
        _pending_chunk_coords={(1, 0, 0)},
        _visible_displayed_coords=set(),
        _visible_display_state_dirty=True,
        _visible_prefetched_displayed_coords=None,
        _chunk_request_target_coords={(2, 0, 0), (3, 0, 0)},
        _chunk_request_queue=queued,
        _chunk_request_queue_origin=(0, 0, 0),
        _chunk_request_queue_view_signature=(0, 0, 0),
        _chunk_request_queue_dirty=False,
    )

    rebuild_visible_missing_tracking(renderer)

    assert renderer._visible_missing_coords == {(2, 0, 0), (3, 0, 0)}
    assert renderer._visible_displayed_coords == {(0, 0, 0)}
    assert renderer._chunk_request_queue is queued
    assert list(renderer._chunk_request_queue) == [(2, 0, 0), (3, 0, 0)]
    assert renderer._chunk_request_queue_dirty is False


def test_refresh_visible_chunk_set_accepts_incremental_display_state() -> None:
    from engine.visibility.coord_manager import refresh_visible_chunk_set

    class NoMoveOrderedDict(OrderedDict):
        def move_to_end(self, key, last=True):  # noqa: ANN001, ANN202 - test sentinel
            raise AssertionError("incremental display refresh should not reorder the full cache")

    coord = (0, 0, 0)
    renderer = SimpleNamespace(
        _current_chunk_origin=lambda: (0, 0, 0),
        _visible_chunk_origin=(0, 0, 0),
        _visible_chunk_coords=[coord],
        _visible_chunk_coord_set={coord},
        chunk_cache=NoMoveOrderedDict({coord: object()}),
        max_cached_chunks=16,
        _cache_capacity_warned=False,
        _visible_missing_coords=set(),
        _visible_displayed_coords={coord},
        _visible_display_state_dirty=True,
        _visible_display_state_incremental=True,
    )

    refresh_visible_chunk_set(renderer)

    assert renderer._visible_displayed_coords == {coord}
    assert renderer._visible_missing_coords == set()
    assert renderer._visible_display_state_dirty is False
    assert renderer._visible_display_state_incremental is False
