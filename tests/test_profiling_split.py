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


def test_hud_overlay_does_not_call_removed_renderer_module_helper():
    source = (ROOT / "engine" / "pipelines" / "hud_overlay.py").read_text(encoding="utf-8")
    assert "_renderer_module()" not in source
