from types import SimpleNamespace


def test_mesh_allocator_facade_exports_split_public_api():
    from engine.cache import mesh_allocator
    from engine.cache import mesh_output_allocator, mesh_visibility, tile_draw_batches, tile_mesh_cache

    assert mesh_allocator.allocate_mesh_output_range is mesh_output_allocator.allocate_mesh_output_range
    assert mesh_allocator.release_chunk_mesh_storage is mesh_output_allocator.release_chunk_mesh_storage
    assert mesh_allocator.store_chunk_meshes is tile_mesh_cache.store_chunk_meshes
    assert mesh_allocator.merge_tile_meshes is tile_mesh_cache.merge_tile_meshes
    assert mesh_allocator.build_tile_draw_batches is tile_draw_batches.build_tile_draw_batches
    assert mesh_allocator.visible_render_batches is mesh_visibility.visible_render_batches
    assert mesh_allocator.build_gpu_visibility_records is mesh_visibility.build_gpu_visibility_records


def test_mesh_output_free_range_coalescing():
    from engine.cache.mesh_output_allocator import coalesce_mesh_output_free_ranges

    assert coalesce_mesh_output_free_ranges([]) == []
    assert coalesce_mesh_output_free_ranges([(16, 8), (0, 8), (8, 8), (40, 0), (32, 4)]) == [
        (0, 24),
        (32, 4),
    ]


def test_tile_visible_mask_uses_tile_local_chunk_positions():
    from engine.cache.tile_mesh_cache import tile_visible_mask

    meshes = [
        SimpleNamespace(chunk_x=8, chunk_z=12),
        SimpleNamespace(chunk_x=9, chunk_z=12),
        SimpleNamespace(chunk_x=11, chunk_z=15),
        SimpleNamespace(chunk_x=12, chunk_z=12),  # outside tile x range
    ]
    # Default MERGED_TILE_SIZE_CHUNKS is 4. Tile (2, 0, 3) covers x=8..11, z=12..15.
    assert tile_visible_mask(None, (2, 0, 3), meshes) == (1 << 0) | (1 << 1) | (1 << 15)


def test_chunk_cache_memory_bytes_uses_render_contract_stride_without_renderer_module():
    from engine.cache.mesh_output_allocator import chunk_cache_memory_bytes
    from engine.render_contract import VERTEX_STRIDE

    shared_buffer = object()
    renderer = SimpleNamespace(
        chunk_cache={
            "a": SimpleNamespace(allocation_id=None, vertex_buffer=shared_buffer, vertex_offset=0, vertex_count=3),
            "b": SimpleNamespace(allocation_id=None, vertex_buffer=shared_buffer, vertex_offset=2 * VERTEX_STRIDE, vertex_count=4),
        },
        _mesh_allocations={},
    )

    assert chunk_cache_memory_bytes(renderer) == 6 * VERTEX_STRIDE
