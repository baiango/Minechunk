from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_profiling_facade_exports_split_modules():
    from engine.pipelines import profiling
    from engine.pipelines import hud_font, hud_overlay, profiling_stats

    assert profiling.HUD_FONT_FALLBACK is hud_font.HUD_FONT_FALLBACK
    assert profiling.hud_glyph_rows is hud_font.hud_glyph_rows
    assert profiling.get_hud_font is hud_font.get_hud_font
    assert profiling.build_hud_vertices is hud_overlay.build_hud_vertices
    assert profiling.draw_profile_hud is hud_overlay.draw_profile_hud
    assert profiling.profile_begin_frame is profiling_stats.profile_begin_frame
    assert profiling.record_frame_breakdown_sample is profiling_stats.record_frame_breakdown_sample
    assert callable(profiling.refresh_profile_summary)
    assert callable(profiling.refresh_frame_breakdown_summary)
    assert callable(profiling.terrain_zstd_runtime_stats)


def test_profiling_facade_stays_small():
    source = (ROOT / "engine" / "pipelines" / "profiling.py").read_text(encoding="utf-8")
    assert len(source.splitlines()) < 80
    assert "class FT_FaceRec" not in source
    assert "FRAME BREAKDOWN @ DIMENSION" not in source
    assert "from .profiling_summary import refresh_frame_breakdown_summary" not in source


def test_hud_font_and_summary_live_in_separate_files():
    hud_font_source = (ROOT / "engine" / "pipelines" / "hud_font.py").read_text(encoding="utf-8")
    summary_source = (ROOT / "engine" / "pipelines" / "profiling_summary.py").read_text(encoding="utf-8")

    assert "HUD_FONT_FALLBACK" in hud_font_source
    assert "_build_hud_font_from_freetype" in hud_font_source
    assert "def get_hud_font" in hud_font_source
    assert "refresh_frame_breakdown_summary" in summary_source
    assert "RC INTERVALS" in summary_source
    assert "MEM CPU: TRACK" in summary_source


def test_process_memory_stats_reports_rss() -> None:
    from engine.pipelines.profiling_summary import process_memory_stats

    stats = process_memory_stats()
    assert int(stats["rss_bytes"]) > 0
    assert int(stats["footprint_bytes"]) > 0
    assert int(stats["peak_rss_bytes"]) >= 0


def test_engine_memory_breakdown_separates_cpu_and_gpu_estimates() -> None:
    from collections import OrderedDict, deque
    from types import SimpleNamespace

    import numpy as np

    from engine.pipelines.profiling_summary import engine_memory_breakdown_stats
    from engine.terrain.types import ChunkVoxelResult

    blocks = np.zeros((2, 2, 2), dtype=np.uint8)
    materials = np.zeros((2, 2, 2), dtype=np.uint32)
    collision_blocks = np.zeros((5,), dtype=np.uint8)
    backend_probe = np.zeros((3,), dtype=np.uint32)
    pending = ChunkVoxelResult(chunk_x=0, chunk_y=0, chunk_z=0, blocks=blocks, materials=materials)
    renderer = SimpleNamespace(
        world=SimpleNamespace(
            _collision_block_chunk_cache=OrderedDict({(0, 0, 0): collision_blocks}),
            _collision_block_chunk_cache_limit=4,
            _backend=SimpleNamespace(_probe=backend_probe),
        ),
        _pending_voxel_mesh_results=deque([pending]),
        _pending_gpu_mesh_batches=deque(),
        _async_voxel_mesh_batch_pool=deque(),
        _pending_surface_gpu_batches=deque(),
        _voxel_mesh_scratch_capacity=0,
        _voxel_mesh_scratch_sample_size=0,
        _voxel_mesh_scratch_height_limit=0,
        _mesh_draw_indirect_capacity=2,
        _mesh_visibility_record_capacity=3,
        _mesh_visibility_record_array=np.empty(0, dtype=np.uint32),
        _mesh_draw_indirect_array=np.empty((0, 4), dtype=np.uint32),
        _tile_render_batches={
            (0, 0, 0): SimpleNamespace(
                owns_vertex_buffer=True,
                vertex_buffer=object(),
                owned_vertex_buffer_capacity_bytes=128,
                vertex_count=99,
            )
        },
        _merged_tile_buffer_pool={64: [object(), object()]},
        _merged_tile_buffer_reuse_queue=[[(object(), 32)]],
        chunk_cache={},
        _mesh_allocations={},
    )
    terrain_zstd = {
        "raw_bytes": 1000,
        "compressed_bytes": 100,
        "cache_compressed_bytes": 100,
        "queue_compressed_bytes": 0,
    }
    process_memory = {"rss_bytes": 2000, "footprint_bytes": 3000, "peak_rss_bytes": 2500, "current_supported": True}
    stats = engine_memory_breakdown_stats(
        renderer,
        process_memory=process_memory,
        terrain_zstd=terrain_zstd,
        slab_stats=(1, 400, 250, 150, 100, 2),
    )

    raw_queue_bytes = blocks.nbytes + materials.nbytes
    assert stats["terrain_payload_cpu_bytes"] == 100 + raw_queue_bytes
    assert stats["collision_cpu_bytes"] == collision_blocks.nbytes
    assert stats["scratch_numpy_cpu_bytes"] == backend_probe.nbytes
    assert stats["tracked_cpu_bytes"] == 100 + raw_queue_bytes + collision_blocks.nbytes + backend_probe.nbytes
    tile_gpu_bytes = 128 + 64 * 2 + 32
    assert stats["tile_render_gpu_bytes"] == tile_gpu_bytes
    assert stats["gpu_estimated_bytes"] == 400 + tile_gpu_bytes + 2 * 16 + 3 * np.dtype(np.uint32).itemsize
    assert stats["other_rss_bytes"] == 2000 - stats["tracked_cpu_bytes"]
    assert stats["other_footprint_bytes"] == 3000 - stats["tracked_cpu_bytes"]


def test_hud_overlay_does_not_call_removed_renderer_module_helper():
    source = (ROOT / "engine" / "pipelines" / "hud_overlay.py").read_text(encoding="utf-8")
    assert "_renderer_module()" not in source
