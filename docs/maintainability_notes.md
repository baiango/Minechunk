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

## One-pass split 2: visibility layout and RC debug/update ownership

This pass makes two previously-facade modules real and moves more renderer-owned logic out of `TerrainRenderer`:

- `engine/visibility/tile_layout.py` now owns visible chunk window geometry, merged-tile key/mask computation, relative slot assignment, and tile-bit helpers.
- `engine/visibility/coord_manager.py` now owns visible-coordinate refresh, incremental origin delta shifting, visible missing tracking, and cache-capacity warnings.
- `engine/rendering/rc_debug_capture.py` now owns F7 RC diagnostic text snapshots, queued RC debug image captures, GPU readback draining, and debug-copy row alignment.
- `engine/rendering/worldspace_rc.py` now owns RC interval bands, active direction counts, update parameter packing, volume parameter writes, and the world-space RC update/dispatch scheduler.

The renderer still owns GPU device creation, pipeline creation, and final render submission, but the visibility and world-space RC subsystems are no longer private method islands inside the renderer class.  The new tests cover the tile-layout math and RC parameter helpers, and the import-boundary test now includes the new modules.
- `engine/rendering/direct_draw.py` now owns direct/indirect visible-batch draw submission, including the optional native multi-draw fast path.

## One-pass split 3: renderer setup and frame encoding ownership

This pass targets the remaining high-risk renderer concentration: setup code and frame-command encoding.

- `engine/rendering/gpu_resources.py` now owns long-lived GPU buffer creation, bind group layout creation, and render/compute pipeline creation. `TerrainRenderer.__init__` delegates to it instead of embedding hundreds of lines of WGPU setup.
- `engine/rendering/frame_encoder.py` now owns the per-frame render command encoding path: camera uniform upload, visible batch selection, optional GPU visibility compute dispatch, scene/gbuffer pass, world-space RC composition pass, debug capture copies, and final blit.
- `engine/input_controller.py` now owns key normalization, keyboard toggles, pointer-drag camera rotation, and canvas event binding.
- `engine/world_reset.py` now owns the reset path for chunk caches, pending mesh jobs, deferred GPU resource cleanup, world recreation, backend fallback checks, and camera respawn.
- `engine/profiling_runtime.py` now owns F3 profiling enable/disable state initialization.
- `engine/collision/walk_solver.py` now also owns free-fly camera motion, so camera update behavior is fully outside the renderer.

`TerrainRenderer` is now mostly an orchestration shell: device/world bootstrap, runtime state fields, resize invalidation, backend diagnostics, small camera helpers, and the draw-frame loop. The extracted modules still mutate renderer state for compatibility, but the biggest remaining monolith—the renderer class—is now small enough to audit by eye.
