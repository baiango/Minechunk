from __future__ import annotations

# Compatibility façade.  Existing renderer/mesher code imports
# ``engine.cache.mesh_allocator``; the implementation is now split by concern.
from .mesh_output_allocator import *
from .tile_mesh_cache import *
from .tile_draw_batches import *
from .mesh_visibility import *

# ``tile_mesh_cache`` imports the GPU mesher facade, which imports this module
# during startup.  Bind these explicitly after the facade imports finish so
# callers still see the merged-tile helpers even if the star import ran during
# that circular import window.
from . import tile_mesh_cache as _tile_mesh_cache

merge_chunk_bounds = _tile_mesh_cache.merge_chunk_bounds
merge_tile_meshes = _tile_mesh_cache.merge_tile_meshes
