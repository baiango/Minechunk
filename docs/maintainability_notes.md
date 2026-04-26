# Minechunk maintainability notes

This patch starts the renderer split without changing render behavior.

## What changed

- Added `engine/render_contract.py` as the stable import boundary for low-level systems.
- Added `VERTEX_COMPONENTS` to `engine/render_constants.py` so CPU meshing no longer has to rely on a magic fallback value.
- Stopped cache, meshing, profiling, chunk-pipeline, and mesh type modules from importing `engine.renderer` just to read constants.
- Removed a dead renderer-module lookup in `engine/visibility/coord_manager.py`.
- Added `tests/test_import_boundaries.py` to prevent these renderer reverse-imports from creeping back in.

## Why this matters

`engine.renderer` owns canvas/device setup and a very large runtime state surface. Importing it from meshing or cache code creates circular dependencies and makes small modules harder to test independently. The new contract module keeps those low-level modules connected only to constants/configuration, not to the renderer god object.

## Recommended next splits

1. Extract world-space radiance cascades into `engine/rendering/worldspace_rc.py`.
2. Extract postprocess target lifetime into a `FrameTargets` / `PostprocessTargets` object.
3. Move WGSL strings into individual `.wgsl` files or at least shader-specific Python modules.
4. Split `build_tile_draw_batches()` into validation, batch collection, allocation, upload, and draw-record construction steps.
5. Add CPU-only regression tests for terrain, meshing, allocator store/release, and visibility updates.
