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

## Patch 2: Debug capture helpers

The F7 / world-space RC PNG capture path no longer stores PNG encoding and WGPU readback conversion helpers as private `TerrainRenderer` methods.  Those pure helpers now live in `engine/debug_capture.py`:

- `safe_filename_component()` for stable debug output filenames.
- `readback_to_rgba8()` for converting mapped WGPU readback buffers into RGBA8 arrays.
- `write_rgba8_png()` for writing minimal PNG files without adding a Pillow dependency.

This is intentionally behavior-preserving.  The renderer still owns GPU command encoding and resource lifetime, but the CPU-side image conversion can now be tested without creating a canvas, adapter, device, or renderer instance.

## Patch 3: shared renderer contract utilities

The renderer no longer owns generic adapter/limit/alignment helper logic.
`engine.render_contract` now exposes `align_up`, `device_limit`, and
`describe_adapter`, so cache/meshing code can share allocator math and WGPU
limit probing without calling private `TerrainRenderer` methods. This removes
another small piece of renderer-as-service coupling before splitting the large
resource systems.


## One-pass split: collision, postprocess targets, and auto-exit

This pass moves three more renderer-adjacent responsibilities out of `TerrainRenderer` without changing the draw algorithm:

- `engine/collision/walk_solver.py` now owns player AABB math, block-solid collision queries, ground snapping, walking movement, and the walking camera update. The renderer keeps fly-mode movement and delegates walk-mode/camera clamping to the solver.
- `engine/rendering/postprocess_targets.py` now owns GI parameter packing and the lifetime/rebuild logic for postprocess, GI cascade, and world-space RC textures/bind groups.
- `engine/auto_exit.py` now owns fixed-view auto-exit readiness checks and device-lost error classification.

The goal is to reduce `TerrainRenderer` from a catch-all engine service into a coordinator. The extracted modules still operate on the renderer state object for compatibility, but the logic is now isolated enough to test and later replace with explicit state objects.
