"""Render-stage entry points."""

from ..rendering.tile_batcher import build_tile_draw_batches
from ..rendering.direct_render import build_gpu_visibility_records, visible_render_batches, visible_render_batches_indirect
from ..rendering.merge_pipeline import merge_chunk_bounds, merge_tile_meshes
