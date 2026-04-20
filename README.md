# Minechunk

Minechunk is an experimental voxel terrain engine written in Python. This branch is focused on making chunk streaming, terrain generation, meshing, residency, and frame cost visible enough to study instead of hiding the work behind heavy simplification.

This checked-in branch is the **stacked vertical chunk prototype**.

Current defaults:

- chunk size: `64 × 64 × 64`
- block size: `0.1 m`
- world height: `2000` blocks (`200 m`)
- horizontal render radius: `16` chunks, circular in XZ
- vertical render radius: `16` chunk layers above and below the camera
- caves: enabled with 3D noise
- default movement mode: AABB walk mode with gravity and collision

The renderer still presents through `wgpu`. In stacked-chunk mode, terrain generation and meshing are currently forced onto the **CPU** for correctness while the 3D chunk architecture is being stabilized.

## What This Branch Is For

This is not a gameplay-first project. It is a systems project.

The branch exists to answer questions like:

- can a Python voxel engine stream a real visible scene without hiding the cost?
- what changes when the engine moves from 2D chunk columns to a true `(chunk_x, chunk_y, chunk_z)` stack?
- how much memory, chunk churn, and frame time show up when the world is tall and nearby caves can breach the surface?
- what needs to stay on CPU first before GPU terrain and GPU meshing can safely come back?

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

This branch is intentionally conservative.

### Active by default

- stacked vertical chunk coordinates: `(chunk_x, chunk_y, chunk_z)`
- CPU terrain generation
- CPU meshing
- WGPU presentation
- AABB walk collision
- profiling HUD
- camera position in HUD
- cave carving with 3D noise
- surface hole / cave mouth support

### Disabled in stacked mode for now

- GPU terrain generation
- GPU meshing
- GPU visibility culling
- indirect GPU-driven visibility path
- merged GPU visibility records

Those systems still exist in the repository, but this branch keeps them out of the active path while the 3D chunk stack is validated.

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

The important point is that the final mesh is built from **voxel occupancy**, not from a pure heightfield shell. If a cave removes voxels, the CPU mesher sees that occupancy directly.

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

- stacked chunk mode currently runs CPU terrain + CPU meshing only
- GPU terrain / GPU meshing paths are not yet ported back to the stacked-chunk runtime
- performance is much worse at full `16 × 16 × 16` streaming than the old 2D column world
- chunk residency and meshing costs are intentionally exposed instead of hidden
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

Optional macOS Metal bindings:

```bash
python3 -m pip install pyobjc-framework-Metal
```

Even with PyObjC Metal installed, stacked chunk mode currently stays on CPU terrain + CPU meshing for correctness.

## Backend Notes

Renderer backend selection lives in `engine/renderer_config.py`.

Available modes:

- `ENGINE_MODE_CPU`
- `ENGINE_MODE_WGPU`
- `ENGINE_MODE_METAL`

The checked-in config still prefers `ENGINE_MODE_METAL`, but with vertical chunk stacking enabled the terrain world currently selects the CPU backend and the renderer disables the GPU visibility path.

So for this branch, the practical runtime is:

- presentation: `wgpu`
- terrain: CPU
- meshing: CPU

That is expected.

## Repository Layout

- `main.py` — entry point
- `engine/world_constants.py` — chunk size, block size, world height, vertical chunk stack settings
- `engine/renderer.py` — main runtime, camera update, collision, visibility selection, frame loop
- `engine/renderer_config.py` — backend and runtime tuning knobs
- `engine/render_constants.py` — render-time constants and cache sizing
- `engine/voxel_world.py` — terrain world facade and solid-block queries
- `engine/cpu_terrain_backend.py` — stacked CPU terrain generation path
- `engine/terrain/kernels/core.py` — terrain noise, cave carving, and CPU meshing kernels
- `engine/pipelines/chunk_pipeline.py` — chunk request queueing, meshing handoff, cache insertion
- `engine/pipelines/profiling.py` — HUD generation and profiling text
- `engine/wgpu_terrain_backend.py` — non-stacked GPU terrain path kept for future reintegration
- `engine/metal_terrain_backend.py` — non-stacked Metal terrain path kept for future reintegration
- `engine/wgpu_chunk_mesher.py` — GPU meshing path kept for future reintegration
- `engine/metal_chunk_mesher.py` — Metal meshing path kept for future reintegration
- `docs/` — screenshots
- `res/` — HUD font asset

## Summary

Minechunk, in its current branch, is an experimental tall-world voxel engine with:

- true stacked vertical chunks
- 10 cm blocks
- 200 m world height
- caves carved with 3D noise
- walk collision with an AABB player body
- a profiling-first runtime that favors debuggability over illusion

If the branch feels rough, that is normal. The point right now is to make the architecture honest enough that the next backend pass can fix real bottlenecks instead of hiding them.
