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

## One-pass split 4: shader assets instead of Python shader dumps

This pass removes the largest non-runtime blob from Python source. `engine/render_shaders.py` is now a small shader loader and token-substitution module instead of a 3.5k-line pile of embedded WGSL strings.

- `engine/shader_loader.py` owns cached loading of checked-in shader assets.
- `engine/shaders/*.wgsl` now stores render, GI, world-space RC, voxel meshing, visibility, HUD, blit, and postprocess shader source.
- `engine/shaders/terrain_surface.wgsl` and `engine/shaders/terrain_surface.metal` now store the WGPU and Metal terrain surface compute shaders.
- `engine/render_shaders.py` still owns compile-time token replacement for RC/GI constants, so existing renderer code can continue importing the same shader constants.

A hash check against the previous embedded-string module confirmed the exported shader constants are byte-for-byte identical after loading and token replacement. This should make shader diffs, editor highlighting, copy/paste into shader tools, and RC shader debugging much easier without changing the graphics path.

## One-pass split 5: mesh cache and allocator ownership

This pass splits the previous `engine/cache/mesh_allocator.py` monolith by responsibility while keeping `engine.cache.mesh_allocator` as a compatibility façade for existing imports.

- `engine/cache/mesh_output_allocator.py` owns mesh-output slab allocation, free-range coalescing, deferred frees, allocation stats, and chunk-mesh storage retain/release.
- `engine/cache/tile_mesh_cache.py` owns chunk-cache mutation, visible tile active-mesh bookkeeping, merged-tile buffer pooling, and tile mesh merging.
- `engine/cache/tile_draw_batches.py` owns visible tile draw-batch construction, cached tile render batches, direct-batch grouping, and merge-age gating.
- `engine/cache/mesh_visibility.py` owns GPU visibility scratch buffers, visibility records, indirect draw command packing, and visible render-batch query entry points.
- `engine/cache/mesh_allocator.py` now re-exports the split modules so older call sites do not need to change yet.

This keeps the allocator/cache path behavior-preserving, but the individual files now map to engine subsystems instead of one 2k-line cache god-module. The next cleanup should turn the renderer-state mutation in these helpers into explicit small state structs.

## One-pass split 6: profiling and HUD ownership

This pass splits the old `engine/pipelines/profiling.py` HUD/profiling pile into smaller modules while keeping `engine.pipelines.profiling` as a compatibility façade.

- `engine/pipelines/hud_font.py` owns fallback glyphs, FreeType lookup, and HUD font rasterization.
- `engine/pipelines/hud_overlay.py` owns HUD geometry packing, HUD vertex-buffer uploads, and HUD overlay draw submission.
- `engine/pipelines/profiling_stats.py` owns frame-timing windows, percentile/FPS helpers, and rolling breakdown samples.
- `engine/pipelines/profiling_summary.py` owns the profile HUD text and F3 frame-breakdown text generation.
- `engine/pipelines/profiling.py` now re-exports the split modules so existing renderer and pipeline imports do not need to change.

This makes the F3 HUD path easier to audit: text generation, font loading, geometry building, and timing state are now separate concerns instead of one 700-line profiling module.

## One-pass split 7: terrain kernel ownership

This pass splits the old `engine/terrain/kernels/core.py` monolith into smaller kernel modules while keeping `engine.terrain.kernels.core` as a compatibility façade.

- `engine/terrain/kernels/numba_compat.py` owns the optional Numba fallback shim.
- `engine/terrain/kernels/materials.py` owns material IDs, color tables, vertex-layout constants, and material-color helpers.
- `engine/terrain/kernels/noise.py` owns the hash/value-noise primitives.
- `engine/terrain/kernels/terrain_profile.py` owns surface-height sampling, cave carving, and block material lookup.
- `engine/terrain/kernels/voxel_fill.py` owns surface-grid filling, voxel-grid filling, stacked-chunk neighbor planes, and surface-to-voxel expansion.
- `engine/terrain/kernels/surface_mesher.py` owns the legacy heightfield surface mesher.
- `engine/terrain/kernels/voxel_mesher.py` owns voxel face masks, AO factors, voxel vertex emission, and voxel vertex counting.

The public import surface is intentionally preserved: existing `from engine.terrain.kernels import ...` and `from engine.terrain.kernels.core import ...` imports still work. This makes terrain generation and CPU meshing easier to optimize separately without editing a 1.3k-line kitchen-sink kernel file.

## One-pass split 8: WGPU voxel mesher ownership

This pass splits the old `engine/meshing/gpu_mesher.py` WGPU voxel-meshing blob while keeping `engine.meshing.gpu_mesher` as a compatibility façade.

- `engine/meshing/gpu_mesher_common.py` owns shared constants, alignment helpers, chunk-coordinate normalization, and the profiling decorator fallback.
- `engine/meshing/gpu_mesher_resources.py` owns scratch-buffer creation, async batch-resource pooling, deferred GPU-buffer cleanup, and async resource release queues.
- `engine/meshing/gpu_mesher_batches.py` owns CPU-voxel-result upload, count/scan dispatch, synchronous metadata readback, and immediate mesh-batch creation from WGPU buffers.
- `engine/meshing/gpu_surface_batches.py` owns WGPU terrain-surface expansion, surface-batch queue draining, and surface-batch release callbacks.
- `engine/meshing/gpu_mesher_finalize.py` owns async metadata finalization, shared output allocation, emit dispatch, and final `ChunkMesh` construction/storage.
- `engine/meshing/gpu_mesher.py` now re-exports the split modules so existing imports do not need to change.

This reduces the WGPU mesher façade from a 1.2k-line runtime module into a small import boundary. The next useful cleanup is to apply the same treatment to `metal_mesher.py`, then replace renderer-state mutation in both GPU meshers with explicit small state objects.

## One-pass split 9: Metal voxel mesher ownership

This pass applies the same compatibility-façade treatment to `engine/meshing/metal_mesher.py` that the WGPU mesher already received.

- `engine/meshing/metal_mesher_common.py` owns Metal mesher dataclasses, shared renderer-state helpers, device resolution, command-buffer status checks, chunk-coordinate normalization, and surface-batch release callbacks.
- `engine/meshing/metal_chunk_mesher.py` owns `MetalChunkMesher`: slot allocation, Metal pipeline creation, CPU voxel upload, native Metal surface-buffer expansion, and count/scan/emit command encoding.
- `engine/meshing/metal_mesher_cache.py` owns renderer-local Metal mesher cache keys, capacity selection, mesher construction, and prewarming.
- `engine/meshing/metal_mesher_async.py` owns async queue entry points, surface-batch enqueue/drain behavior, synchronous batch fallback, and renderer async-state shutdown.
- `engine/meshing/metal_mesher_finalize.py` owns completed Metal batch materialization, overflow CPU fallback, mesh-output allocation/upload, renderer delivery, and async resource cleanup.
- `engine/meshing/metal_mesher.py` now re-exports the split modules so existing renderer and pipeline imports do not need to change.

This reduces the Metal mesher public module from a 900-line backend blob into a stable façade. The WGPU and Metal meshers now have parallel internal shapes, which should make backend-specific bug fixes and future Vulkan/Metal ownership cleanup easier to reason about.

## One-pass split 10: tile draw-batch ownership

This pass splits the remaining oversized `engine/cache/tile_draw_batches.py` tile-batching module while preserving the existing public entry point.

- `engine/cache/direct_render_batches.py` now owns direct render-batch grouping, contiguous range merging, grouped-batch finalization, and draw-to-render batch conversion.
- `engine/cache/cached_tile_batches.py` now owns cached tile-batch stats, visible-tile iteration expansion, and tile render-batch storage/update bookkeeping.
- `engine/cache/tile_cache_constants.py` now owns shared cache constants that need to be imported without pulling in the heavier tile-mesh cache module.
- `engine/cache/tile_draw_batches.py` remains the owner of `build_tile_draw_batches()`, but is now mostly orchestration around tile cache lookup, merge-age gating, and merged-buffer rebuild decisions.

The split also removes a fragile import-time cycle: lightweight tile draw-batch helpers no longer need to import `tile_mesh_cache` at module import time. Heavier tile mesh/cache functions are imported lazily only when `build_tile_draw_batches()` actually runs.

## One-pass split 11: WGPU terrain backend ownership

This pass splits the previous `engine/terrain/backends/wgpu_terrain_backend.py` backend blob while preserving the public `WgpuTerrainBackend` import path.

- `engine/terrain/backends/wgpu_terrain_common.py` owns the optional WGPU import, profiling decorator fallback, WGPU terrain shader loading, chunk-batch dataclasses, leased surface-batch type, and chunk-coordinate normalization.
- `engine/terrain/backends/wgpu_terrain_batches.py` owns surface-batch resource allocation, pending-job merging, compute dispatch, async readback queuing, CPU readback materialization, GPU surface-batch leasing, batch-slot reuse, flushing, and cleanup.
- `engine/terrain/backends/wgpu_terrain_voxels.py` owns surface-result to voxel-result conversion, stacked-chunk boundary plane allocation, CPU voxel-grid fallback, and voxel-batch polling/flushing.
- `engine/terrain/backends/wgpu_terrain_backend.py` now owns the public backend class, one-shot surface sampling, one-shot surface-grid readback, job submission entry points, and the stable backend label.

The behavior is intended to remain the same, but the terrain backend no longer mixes GPU resource setup, async batch/readback lifecycle, and surface-to-voxel expansion in one 800-line file. This makes the terrain path closer to the WGPU/Metal mesher split: public import path stays stable while backend internals become easier to audit.

## One-pass split 12: GPU resource initialization ownership

This pass splits the renderer GPU-resource setup blob while preserving `engine.rendering.gpu_resources.initialize_gpu_resources()` as the public entry point used by `TerrainRenderer`.

- `engine/rendering/gpu_resource_buffers.py` owns long-lived uniform/storage buffers plus the shared postprocess sampler.
- `engine/rendering/gpu_resource_layouts.py` owns bind group layouts for voxel meshing, mesh visibility, tile merging, scene rendering, GI composition, world-space RC, and final presentation.
- `engine/rendering/gpu_resource_pipelines.py` owns compute/render pipeline creation and the backend fallback behavior when GPU tile merge, visibility, or WGPU meshing pipelines fail to initialize.
- `engine/rendering/gpu_resources.py` is now a small façade that runs the three setup phases in order.

This keeps `TerrainRenderer` from depending on a single 700-line GPU setup function, and it makes future shader-layout/pipeline edits less risky because buffer creation, layout declaration, and pipeline creation now have separate owners.

## One-pass split 13: Metal terrain backend ownership

This pass applies the same backend split pattern to `engine/terrain/backends/metal_terrain_backend.py` that the WGPU terrain backend already received, while preserving the public `MetalTerrainBackend` import path.

- `engine/terrain/backends/metal_terrain_common.py` owns the optional Metal import, profiling fallback, Metal terrain shader loading, chunk-batch dataclass, leased surface-batch type, and chunk-coordinate normalization.
- `engine/terrain/backends/metal_terrain_buffers.py` owns Metal buffer allocation, buffer writes/views, compute dispatch, parameter packing, one-shot surface sampling, and one-shot surface-grid readback.
- `engine/terrain/backends/metal_terrain_batches.py` owns pending-job merging, batch-slot reuse, async command-buffer polling, CPU surface-result materialization, native Metal surface-batch leasing, flushing, and cleanup.
- `engine/terrain/backends/metal_terrain_voxels.py` owns direct voxel-grid generation, surface-result to voxel-result conversion, stacked-chunk boundary plane allocation, and voxel-batch polling/flushing.
- `engine/terrain/backends/metal_terrain_backend.py` now owns only construction, pipeline setup orchestration, and the stable backend label.

The behavior is intended to stay the same, but Metal terrain no longer mixes Metal buffer primitives, async surface scheduling, leased GPU-surface handoff, and CPU voxel conversion in one file. This makes the WGPU and Metal terrain backends structurally parallel, which should make future backend-specific fixes much easier to compare.


## One-pass split 14: voxel mesher helper ownership

This pass splits the remaining voxel mesher helper pile without changing the public terrain-kernel façade.

- `engine/terrain/kernels/voxel_emit.py` owns low-level voxel quad/face vertex emission.
- `engine/terrain/kernels/voxel_ao.py` owns boundary-aware solid sampling and ambient-occlusion factors.
- `engine/terrain/kernels/voxel_faces.py` owns face-mask constants and face-mask construction.
- `engine/terrain/kernels/voxel_mesher.py` now owns the high-level voxel mesh build/count entry points and re-exports the helper symbols needed by `engine.terrain.kernels.core`.

This keeps existing imports from `engine.terrain.kernels.core` and `engine.terrain.kernels.voxel_mesher` working, while making AO, face-mask generation, and vertex emission independently testable.
