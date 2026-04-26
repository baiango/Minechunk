# Minechunk

Minechunk is an experimental voxel terrain engine written in Python. This branch is focused on making chunk streaming, terrain generation, meshing, residency, and frame cost visible enough to study instead of hiding the work behind heavy simplification.

This checked-in branch is the **stacked vertical chunk + WGPU compute prototype**.

Current defaults:

- chunk size: `64 × 64 × 64`
- block size: `0.1 m`
- world height: `2000` blocks (`200 m`)
- horizontal render radius: `16` chunks, circular in XZ
- vertical render radius: `16` chunk layers above and below the camera
- caves: enabled with 3D noise
- default movement mode: AABB walk mode with gravity and collision
- backend mode: `ENGINE_MODE_WGPU`

The renderer presents through `wgpu`, and the stacked chunk stream now uses WGPU compute for the main terrain/meshing path.

## What This Branch Is For

This is not a gameplay-first project. It is a systems project.

The branch exists to answer questions like:

- can a Python voxel engine stream a real visible scene without hiding the cost?
- what changes when the engine moves from 2D chunk columns to a true `(chunk_x, chunk_y, chunk_z)` stack?
- how much memory, chunk churn, and frame time show up when the world is tall and nearby caves can breach the surface?
- how far can WGPU terrain and meshing be pushed before Python orchestration becomes the next bottleneck?

## World Contract

### Geometry

- `CHUNK_SIZE = 64`
- `BLOCK_SIZE = 0.1`
- `WORLD_HEIGHT_BLOCKS = 2000`
- chunk world span: `6.4 m` per axis
- total world height: `200 m`

### Visibility

- horizontal render radius: `16` chunks
- horizontal visibility shape: circular
- horizontal visible chunk count at full radius: `797`
- vertical chunk radius: `16`
- vertical streamed layer count: up to `33`
- cache capacity: sized from horizontal visible chunks × streamed vertical layers

At full radius, the current cache target is built for roughly `26,301` visible chunk slots.

## Current Runtime Path

### Active by default

- stacked vertical chunk coordinates: `(chunk_x, chunk_y, chunk_z)`
- WGPU terrain surface generation
- WGPU surface-to-local-voxel expansion
- WGPU voxel mesh count / scan / emit
- WGPU presentation
- AABB walk collision
- profiling HUD
- camera position in HUD
- cave carving with 3D noise
- surface hole / cave mouth support

### Still CPU-side or conservative

- collision and direct block queries still use the CPU terrain function path
- surface GPU batches use a tiny readback fence before being leased to the mesher
- GPU visibility culling remains disabled in stacked mode
- indirect GPU-driven visibility remains disabled in stacked mode
- CPU terrain and CPU meshing are retained as fallback paths

## WGPU Port Notes

The stacked chunk WGPU path is not the old 2D-column path simply switched back on. The old GPU mesher assumed a whole-world height buffer, which is not viable for `WORLD_HEIGHT_BLOCKS = 2000`.

This branch ports the stream around local vertical chunks instead:

1. The WGPU terrain backend generates batched surface height/material grids for `(chunk_x, chunk_y, chunk_z)` requests.
2. `VOXEL_SURFACE_EXPAND_SHADER` expands each surface grid into a local `64`-block-high voxel chunk on the GPU.
3. The expansion shader writes two extra ghost Y layers, so top/bottom faces can be culled correctly across vertical chunk boundaries.
4. `VOXEL_MESH_BATCH_SHADER` counts faces, scans per-chunk offsets, and emits final vertices from that local chunk storage.
5. Final mesh bounds and cache keys remain vertical-stack aware.

The CPU fallback path still exists because collision, debug queries, and safe recovery are easier to keep deterministic while the WGPU stream is being validated.

## Terrain Model

Terrain starts from layered 2D height noise, then fills solid voxels below the sampled surface. Caves are carved afterward with 3D noise.

Current checked-in frequency tuning:

- `TERRAIN_FREQUENCY_SCALE = 0.3`
- `CAVE_FREQUENCY_SCALE = 0.5`

Broadly, that means:

- hills come from layered 2D value noise
- caves come from layered 3D value noise
- caves are allowed to breach the surface in this branch
- material bands are derived from surface height and world height

The important point is that the final mesh is built from **voxel occupancy**, not from a pure heightfield shell. If a cave removes voxels, the WGPU expansion/meshing path sees that occupancy directly.

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
- visible / pending chunk counts
- draw call count
- camera position in meters
- camera position in block units

Because the HUD is built every frame, it adds real overhead. Treat on-screen numbers as instrumented runtime numbers, not absolute maximum throughput.

## Why The Engine Looks Weird Compared To Game Engines

Minechunk is intentionally not using the usual escape hatches.

This branch does **not** rely on:

- greedy meshing
- LOD
- geometry decimation
- far-field impostors
- fake benchmark scenes with low residency pressure

That makes the visuals harsher and the numbers more painful, but it also makes failures easier to diagnose.

## Known Limits In This Branch

This branch is a prototype, not a polished engine build.

Known limits:

- WGPU meshing currently supports local chunk heights up to `128`, so tall worlds must stay stacked
- the WGPU terrain surface stage still uses a readback fence before handing GPU buffers to the mesher
- collision still evaluates terrain/block queries on CPU
- performance at full `16 × 16 × 16` streaming can still be brutal because residency pressure is intentionally exposed
- terrain generation is deterministic but not yet designed for gameplay-quality biome variety
- there is no save/load, block editing gameplay loop, lighting system, or gameplay content stack yet

## Build And Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run:

```bash
python3 main.py
```

Optional fixed-size stress entries:

```bash
python3 render_8x8x8_then_exit.py
python3 render_16x16x16_then_exit.py
```

## Backend Notes

Renderer backend selection lives in `engine/renderer_config.py`.

Available modes:

- `ENGINE_MODE_CPU`
- `ENGINE_MODE_WGPU`
- `ENGINE_MODE_METAL`

The checked-in config now uses `ENGINE_MODE_WGPU`. With stacked chunks enabled, that means:

- presentation: WGPU
- terrain surface batches: WGPU compute
- surface-to-voxel expansion: WGPU compute
- voxel meshing: WGPU compute
- collision/block queries: CPU fallback

## Repository Layout

- `main.py` — entry point
- `engine/world_constants.py` — chunk size, block size, world height, vertical chunk stack settings
- `engine/renderer.py` — main runtime, camera update, collision, visibility selection, frame loop
- `engine/renderer_config.py` — backend and runtime tuning knobs
- `engine/render_constants.py` — render-time constants and cache sizing
- `engine/terrain/world.py` — terrain world facade and backend selection
- `engine/terrain/backends/cpu_terrain_backend.py` — stacked CPU fallback terrain path
- `engine/terrain/backends/wgpu_terrain_backend.py` — WGPU terrain surface backend
- `engine/terrain/kernels/core.py` — CPU terrain noise, cave carving, and fallback voxel fill kernels
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
- caves carved with 3D noise
- walk collision with an AABB player body
- WGPU terrain expansion and meshing for the active chunk stream
- a profiling-first runtime that favors debuggability over illusion

If the branch feels rough, that is normal. The point right now is to make the architecture honest enough that backend bottlenecks can be fixed instead of hidden.
