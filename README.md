# Minechunk

Minechunk is an experimental voxel terrain engine written in Python. This branch is focused on making chunk streaming, terrain generation, meshing, residency, and frame cost visible enough to study instead of hiding the work behind heavy simplification.

This checked-in branch is the **stacked vertical chunk prototype**, with CPU as the safe default and WGPU/Metal selectable for GPU terrain/meshing experiments.

Current defaults:

- chunk size: `64 × 64 × 64`
- block size: `0.1 m`
- world height: `2000` blocks (`200 m`)
- horizontal render radius: `8` chunks, circular in XZ
- vertical render radius: `16` chunk layers above and below the camera
- caves: enabled with gated 3D noise
- default movement mode: AABB walk mode with gravity and collision
- renderer backend: WGPU
- terrain backend: CPU
- meshing backend: CPU
- Radiance Cascades: off by default
- tile merging: off by default
- terrain, offscreen mesh, and tile zstd compression: on by default

The renderer presents through `wgpu`, while terrain generation and meshing can be selected independently from the launcher or with `--terrain-backend` and `--meshing-backend`.

## What This Branch Is For

This is not a gameplay-first project. It is a systems project.

The branch exists to answer questions like:

- can a Python voxel engine stream a real visible scene without hiding the cost?
- what changes when the engine moves from 2D chunk columns to a true `(chunk_x, chunk_y, chunk_z)` stack?
- how much memory, chunk churn, and frame time show up when the world is tall and cave voxel fill is part of the terrain path?
- how far can WGPU terrain and meshing be pushed before Python orchestration becomes the next bottleneck?

## World Contract

### Geometry

- `CHUNK_SIZE = 64`
- `BLOCK_SIZE = 0.1`
- `WORLD_HEIGHT_BLOCKS = 2000`
- chunk world span: `6.4 m` per axis
- total world height: `200 m`
- CPU meshing uses run-length quads along the X/Z axes to reduce vertex output while keeping voxel occupancy exact

### Visibility

- horizontal render radius: `8` chunks
- horizontal visibility shape: circular
- horizontal visible chunk count at full radius: `197`
- vertical chunk radius: `16`
- vertical streamed layer count: up to `33`
- cache capacity: conservatively sized from the enclosing horizontal box × streamed vertical layers

At full radius, the circular visible set is roughly `6,501` chunk slots. The conservative cache target is sized from the enclosing `17 × 17 × 33` box, or `9,537` chunk slots.

## Current Runtime Path

### Active by default

- stacked vertical chunk coordinates: `(chunk_x, chunk_y, chunk_z)`
- CPU terrain generation
- CPU voxel meshing
- WGPU presentation
- AABB walk collision
- profiling HUD
- camera position in HUD
- cave carving with gated 3D noise
- 3D caves that can naturally punch through the terrain skin
- final-present screen-space crease shadow from the G-buffer
- terrain zstd cache for CPU-side terrain payloads
- experimental zstd readback caches for offscreen mesh and tile payloads
- aggressive mesh slab compaction after offscreen mesh compression

### Still CPU-side or conservative

- collision and direct block queries still use the CPU terrain function path
- surface GPU batches use a tiny readback fence before being leased to the mesher
- GPU visibility culling remains disabled in stacked mode
- indirect GPU-driven visibility remains disabled in stacked mode
- CPU terrain and CPU meshing are retained as fallback paths
- active visible meshes stay uncompressed GPU buffers; zstd applies to CPU terrain payloads and offscreen/readback caches

## Rendering Notes

The default lighting path is intentionally simple. Radiance Cascades are available behind `--rc`, but the default launcher/runtime path keeps RC off for profiling.

The final present pass applies a **screen-space crease shadow** from the G-buffer normal/depth texture. This is not physically correct ambient occlusion. It emphasizes screen-space depth and normal discontinuities, so it reads like darkened voxel creases or mesh-edge contrast rather than soft world-space contact AO.

## WGPU Port Notes

The stacked chunk WGPU path is not the old 2D-column path simply switched back on. The old GPU mesher assumed a whole-world height buffer, which is not viable for `WORLD_HEIGHT_BLOCKS = 2000`.

This branch ports the stream around local vertical chunks instead:

1. The WGPU terrain backend generates batched surface height/material grids for `(chunk_x, chunk_y, chunk_z)` requests.
2. The portable path reads those surface grids back and expands them through the shared CPU voxel-fill code, so cave carving follows the same Numba/Zig terrain model.
3. Same-API GPU terrain plus GPU meshing can still use a zero-copy surface-buffer handoff.
4. In that zero-copy path, `VOXEL_SURFACE_EXPAND_SHADER` expands each surface grid into a local `64`-block-high voxel chunk on the GPU using the same 3D cave formula, then `VOXEL_MESH_BATCH_SHADER` counts faces, scans per-chunk offsets, and emits vertices.
5. Final mesh bounds and cache keys remain vertical-stack aware.

The CPU fallback path still exists because collision, debug queries, and safe recovery are easier to keep deterministic while the WGPU stream is being validated.

## Terrain Model

Terrain starts from layered 2D height noise, then fills solid voxels below the sampled surface. Caves are carved afterward with a gated 3D noise model.

Current checked-in frequency tuning:

- `TERRAIN_FREQUENCY_SCALE = 0.3`
- `CAVE_FREQUENCY_SCALE = 1.0`
- `CAVE_DETAIL_FREQUENCY_MULTIPLIER = 3.0`
- `CAVE_DETAIL_WEIGHT = 0.18`

Broadly, that means:

- hills come from layered 2D value noise
- caves come from a primary 3D value-noise field plus a light high-frequency detail octave
- the 3D cave field can carve from just above bedrock up to the terrain skin, so visible mouths stay connected to the same cave model
- cave checks are skipped above the upper active vertical/depth range
- material bands are derived from surface height and world height

The important point is that the final mesh is built from **voxel occupancy**, not from a pure heightfield shell. In the portable terrain path, WGPU/Metal surface output is converted to voxel occupancy through the shared CPU fill so caves match the Numba/Zig model. In same-API zero-copy GPU meshing, the WGPU/Metal expansion shaders carry the same 3D cave constants.

## Movement And Collision

Default movement is **walk mode** with a player AABB.

### Walk mode

- collider width: `0.6 m`
- collider height: `1.8 m`
- eye height: `1.62 m`
- step height: `0.45 m`
- jump speed: `6.5 m/s`
- gravity: `24.0 m/s²`
- sprint speed: `9.0 m/s`

### Fly mode

- base fly speed: `4.5 m/s`
- sprint fly speed: `200.0 m/s`

The walk solver resolves motion axis-by-axis against solid voxels and supports stepping up small ledges.

## Controls

- `WASD` or arrow keys: move
- left mouse drag: look
- `Space`: jump in walk mode
- `Shift`: sprint
- `V`: toggle walk mode / noclip fly mode
- `X`: move up in fly mode
- `Z`: move down in fly mode
- `F3`: toggle profiling HUD
- `R`: regenerate world with a new seed

## HUD

The HUD is part of the project, not an afterthought. It exists to expose system behavior while the engine runs.

Current HUD includes:

- frame timings and percentiles
- backend diagnostics
- chunk dimensions
- mesh slab allocator stats
- process memory: macOS footprint, RSS, and peak RSS
- CPU tracked memory split: terrain payloads, collision cache, scratch arrays, and untracked remainder
- GPU memory estimate split: mesh slabs, tile buffers, and transient buffers
- terrain / mesh / tile zstd cache entries, raw bytes, compressed bytes, ratios, and pending readbacks
- mesh slab compaction and pending retired slab bytes
- visible / pending chunk counts
- draw call count
- camera position in meters
- camera position in block units

Because the HUD is built every frame, it adds real overhead. Treat on-screen numbers as instrumented runtime numbers, not absolute maximum throughput.

## Why The Engine Looks Weird Compared To Game Engines

This branch still avoids most usual escape hatches. It does **not** rely on:

- LOD
- geometry decimation
- far-field impostors
- fake benchmark scenes with low residency pressure

The CPU mesher now uses local X/Z run-length quads to reduce obvious vertex waste, but the engine still renders real nearby voxel terrain rather than swapping to far-field proxies. That makes the visuals harsher and the numbers more painful, but it also makes failures easier to diagnose.

## Known Limits In This Branch

This branch is a prototype, not a polished engine build.

Known limits:

- WGPU meshing currently supports local chunk heights up to `128`, so tall worlds must stay stacked
- the WGPU terrain surface stage still uses a readback fence before handing GPU buffers to the mesher
- collision still evaluates terrain/block queries on CPU
- performance on large fixed views such as `16 × 16 × 16` can still be brutal because residency pressure is intentionally exposed
- terrain generation is deterministic but not yet designed for gameplay-quality biome variety
- there is no save/load, block editing gameplay loop, lighting system, or gameplay content stack yet

## Build And Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Optional Zig terrain kernel:

```bash
python3 tools/build_zig_terrain.py
```

The build writes `libminechunk_terrain` into `engine/terrain/kernels/native/`. The CPU terrain backend loads it automatically through `ctypes` when present, otherwise it falls back to the existing Python/Numba kernels. CPU surface polling uses the Zig batch entry point so multiple chunks can be filled in one native call. Both CPU terrain kernels use f32 terrain noise/profile math; the Zig surface-grid path uses four-lane `@Vector(4, f32)` SIMD for adjacent X columns, and the Zig batch voxel path uses active cave-range gating plus four-lane SIMD cave checks. Use `MINECHUNK_TERRAIN_ZIG_LIB=/path/to/libminechunk_terrain` to point at a custom build, or `MINECHUNK_DISABLE_ZIG_TERRAIN=1` to force the fallback path.

The launcher and `main.py` also expose the CPU terrain kernel choice directly:

```bash
python3 main.py --terrain-kernel auto   # Zig when built, otherwise Numba
python3 main.py --terrain-kernel numba  # force Python/Numba kernels
python3 main.py --terrain-kernel zig    # require the Zig shared library
```

Terrain kernel microbenchmark:

```bash
python3 tools/benchmark_terrain_kernels.py --chunks 128
```

Recent local reference on the default 128-chunk benchmark:

- Numba f32: surface batch about `11-16 ms`, caves-off voxel fill about `16-24 ms`, caves-on voxel fill about `140-180 ms`
- Zig f32/SIMD: surface batch about `1.3-2.1 ms`, caves-off voxel fill about `6.8-11.5 ms`, caves-on voxel fill about `60-75 ms`

These numbers are hardware and thermal-state dependent; use the script above for the current machine.

Run:

```bash
python3 main.py
```

Graphical CLI launcher with tunable presets:

```bash
python3 benchmark_launcher.py
```

The legacy benchmark wrapper scripts were removed. The launcher now builds and spawns a `main.py` command instead of importing the renderer directly. The GUI stays open after launch, remains responsive while the engine window runs, and does not kill the engine when the launcher is closed. Use `--print-command` to inspect the exact CLI command without launching the engine:

```bash
python3 benchmark_launcher.py --preset fixed_16x16x16 --print-command
python3 benchmark_launcher.py --preset fly_forward_4096 --print-command
```

For automated profiling runs, skip the GUI with `--headless` and override individual knobs. Headless launches are non-blocking by default; add `--wait` only when a script should block until `main.py` exits and return its exit code:

```bash
python3 benchmark_launcher.py --headless --preset fixed_16x16x16 --terrain-backend wgpu --meshing-backend wgpu --view 16x16x16 --terrain-batch-size 128 --mesh-batch-size 32 --no-rc
python3 benchmark_launcher.py --headless --wait --preset fly_forward_4096 --target-rendered-chunks 4096 --fly-speed-mps 20 --rc
python3 benchmark_launcher.py --headless --preset fixed_16x16x16 --no-terrain-zstd --no-mesh-zstd --no-tile-merge --print-command
```

The same options are available directly through `main.py`:

```bash
python3 main.py --benchmark-mode fixed --fixed-view 16x16x16 --exit-when-view-ready --no-rc
python3 main.py --benchmark-mode fly_forward --fixed-view 16x16x16 --target-rendered-chunks 4096 --rc
python3 main.py --benchmark-mode fixed --fixed-view 16x16x16 --no-terrain-zstd --no-mesh-zstd --no-tile-merge
```

Useful runtime flags:

- `--renderer-backend wgpu`: renderer/presentation backend
- `--terrain-backend cpu|wgpu|metal`: terrain generation backend
- `--meshing-backend cpu|wgpu|metal`: voxel meshing backend
- `--terrain-zstd / --no-terrain-zstd`: CPU-side terrain payload compression
- `--mesh-zstd / --no-mesh-zstd`: experimental offscreen mesh readback compression
- `--tile-merge / --no-tile-merge`: merged visible tile GPU buffers; default is off to reduce footprint
- `--rc / --no-rc`: Radiance Cascades; default is off

## Backend Notes

Backend selection is exposed by the launcher and `main.py`.

Available renderer backend:

- `wgpu`

Available terrain and meshing backends:

- `cpu`
- `wgpu`
- `metal`

The checked-in defaults are CPU terrain and CPU meshing. With stacked chunks enabled, that means:

- presentation: WGPU
- terrain generation: CPU
- voxel meshing: CPU
- collision/block queries: CPU

## Repository Layout

- `main.py` — entry point
- `engine/world_constants.py` — chunk size, block size, world height, vertical chunk stack settings
- `engine/renderer.py` — main runtime, camera update, collision, visibility selection, frame loop
- `engine/renderer_config.py` — backend and runtime tuning knobs
- `engine/render_constants.py` — render-time constants and cache sizing
- `engine/terrain/world.py` — terrain world facade and backend selection
- `engine/terrain/backends/cpu_terrain_backend.py` — stacked CPU fallback terrain path
- `engine/terrain/backends/wgpu_terrain_backend.py` — WGPU terrain surface backend
- `engine/terrain/kernels/noise.py` — shared Numba value-noise helpers
- `engine/terrain/kernels/terrain_profile.py` — f32 surface profile and cave model
- `engine/terrain/kernels/voxel_fill.py` — Numba stacked voxel fill kernels
- `engine/terrain/kernels/zig_kernel.py` — ctypes wrapper for the optional Zig terrain kernel
- `engine/terrain/kernels/native/terrain_kernel.zig` — Zig terrain kernel source
- `tools/build_zig_terrain.py` — Zig shared-library build helper
- `tools/benchmark_terrain_kernels.py` — Numba/Zig terrain microbenchmark
- `engine/meshing/gpu_mesher.py` — WGPU surface expansion and voxel mesh batching
- `engine/render_shaders.py` — Python shader loader/token substitution for checked-in shader assets
- `engine/shaders/` — WGSL/MSL shader source files for render, terrain, meshing, RC, visibility, HUD, and postprocess passes
- `engine/pipelines/chunk_pipeline.py` — chunk request queueing, meshing handoff, cache insertion
- `engine/pipelines/profiling.py` — HUD generation and profiling text
- `docs/` — screenshots
- `res/` — HUD font asset

## Summary

Minechunk, in this branch, is an experimental tall-world voxel engine with:

- true stacked vertical chunks
- 10 cm blocks
- 200 m world height
- caves carved with upper-gated 3D noise that can reach the surface and continue down near bedrock
- walk collision with an AABB player body
- CPU-default terrain generation and X/Z run-length meshing, with WGPU/Metal experiments still available
- screen-space crease shadow in the final present pass
- terrain/mesh zstd compression experiments and memory-focused HUD accounting
- a profiling-first runtime that favors debuggability over illusion

If the branch feels rough, that is normal. The point right now is to make the architecture honest enough that backend bottlenecks can be fixed instead of hidden.
