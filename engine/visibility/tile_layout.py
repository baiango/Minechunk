"""Tile-layout façade for incremental migration away from renderer.py."""

def build_visible_layout_template(renderer, *args, **kwargs):
    return renderer._build_visible_layout_template(*args, **kwargs)

def tile_layout_in_view_for_origin(renderer, origin):
    return renderer._tile_layout_in_view_for_origin(origin)
