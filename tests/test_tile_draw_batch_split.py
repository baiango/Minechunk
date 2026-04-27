from __future__ import annotations

import sys
from types import SimpleNamespace


def _install_import_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "wgpu", SimpleNamespace(GPUBuffer=object))
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(
            ndarray=object,
            dtype=lambda spec: ("dtype", spec),
            float32="float32",
            uint32="uint32",
        ),
    )
    for name in list(sys.modules):
        if name == "engine.cache" or name.startswith("engine.cache.") or name == "engine.meshing_types":
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_tile_draw_batch_module_delegates_split_helpers(monkeypatch):
    _install_import_stubs(monkeypatch)

    from engine.cache import cached_tile_batches, direct_render_batches, tile_draw_batches

    assert tile_draw_batches._draw_batches_to_render_batches is direct_render_batches._draw_batches_to_render_batches
    assert tile_draw_batches._group_render_batches is direct_render_batches._group_render_batches
    assert tile_draw_batches._cached_tile_batch_stats is cached_tile_batches._cached_tile_batch_stats
    assert tile_draw_batches._store_cached_tile_render_batch is cached_tile_batches._store_cached_tile_render_batch


def test_direct_render_batch_normalization_merges_contiguous_ranges(monkeypatch):
    _install_import_stubs(monkeypatch)

    from engine.cache.direct_render_batches import _normalize_direct_render_batches

    vertex_buffer = object()
    other_buffer = object()
    batches = [
        (vertex_buffer, 0, 4, 8),
        (vertex_buffer, 0, 4, 0),
        (vertex_buffer, 0, 4, 4),
        (other_buffer, 0, 3, 0),
    ]

    assert _normalize_direct_render_batches(batches) == [
        (vertex_buffer, 0, 12, 0),
        (other_buffer, 0, 3, 0),
    ]
