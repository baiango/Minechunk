# Minechunk

Minechunk is a measurement-first voxel terrain engine written in Python.

It streams a full `65 × 65` visible chunk field, keeps explicit chunk residency, builds chunk meshes on CPU or GPU, performs GPU visibility culling, and submits terrain through indirect draw commands. The project is intentionally **not** built around geometry-reduction tricks. It exists to measure raw terrain generation, meshing, streaming, and presentation cost under a real visible scene.

The checked-in default is currently:

- `engine_mode = ENGINE_MODE_METAL`

That means Minechunk will try to use the native Metal terrain backend on macOS first. If Metal is unavailable, it falls back to the `wgpu` terrain backend, and then to the CPU terrain backend.

## Current State

What exists today:

- deterministic terrain synthesis
- chunk streaming and explicit residency tracking
- CPU terrain backend
- `wgpu` terrain backend
- native Metal terrain backend for macOS
- CPU meshing
- `wgpu` GPU meshing
- Metal GPU meshing
- GPU visibility culling
- indirect draw submission
- merged render tiles
- built-in profiling HUD and frame-breakdown HUD

What does **not** exist yet:

- native Metal renderer path
- greedy meshing
- level of detail
- geometry simplification

Rendering still goes through the `wgpu` presentation path. Metal is already used for terrain generation and meshing when the Metal backend is active.

## Engine Contract

- Chunk size: `32 × 128 × 32`
- Surface sampling footprint per chunk: `34 × 34`
- Default render distance: `32` chunks (`1024` blocks)
- Default visible square: `65 × 65` chunks (`4225` chunk slots)
- Default cache capacity: `4225` chunks
- Vertex stride: `48` bytes
- Default mesh batch size: `128`
- Minimum mesh output slab: `64 MiB`
- GPU visibility workgroup size: `64`
- Swapchain cap: `320 FPS`
- VSync default: `off`

The cache is intentionally sized to hold the full default visible square.

## Why This Engine Is Weird On Purpose

Minechunk deliberately keeps expensive things visible instead of hiding them behind simplification.

It does **not** use:

- greedy meshing
- LOD
- decimated far terrain
- “fake” benchmark scenes that avoid full residency pressure

That makes the numbers harsher, but also more honest. The project is meant to answer questions like:

- how much chunk throughput survives under a real visible scene?
- what is the actual p99 frame cost of sustained streaming?
- where do CPU↔GPU synchronization costs start to dominate?
- how stable is chunk residency under continuous movement?

## Pipeline Overview

`TerrainRenderer.draw_frame()` drives the frame.

1. Camera input updates yaw, pitch, and free-flight motion.
2. The visible chunk square is recomputed when the camera crosses chunk boundaries.
3. Visible render batches are prepared from the chunk cache and merged tile cache.
4. If enabled, GPU visibility culling writes indirect draw commands.
5. Terrain, HUD, and frame-breakdown overlays are encoded into one command buffer.
6. The command buffer is submitted through `wgpu`.
7. Deferred GPU resources and pending GPU mesh batches are finalized.
8. Ready terrain results are drained, meshed, and inserted into the ordered chunk cache.
9. Missing visible chunks are reprioritized and new terrain requests are issued.
10. Per-frame breakdown metrics are sampled for the HUD.

The runtime uses slab-backed mesh output buffers plus explicit suballocation metadata instead of per-chunk buffer churn. That keeps residency pressure visible and makes allocator behavior measurable.

## Terrain Model

Terrain is deterministic and heightfield-driven.

For each `(x, z)` surface position, layered 2D value noise produces:

- surface height
- top material

The engine then expands that surface into a full voxel column.

Material layering is simple and explicit:

- `BEDROCK` at `y = 0`
- `STONE` below the upper soil band
- `DIRT` near the top
- surface material at the top voxel
- `AIR` above the sampled height

This keeps terrain generation predictable enough for backend validation while still producing large streamed worlds.

## Backends

### CPU

`CpuTerrainBackend` is the reference path. It is the simplest backend and the fallback when GPU terrain cannot be created.

### Wgpu

`WgpuTerrainBackend` uses compute passes through `wgpu-py`. This path shares the renderer’s graphics API and is portable across supported `wgpu` platforms.

### Metal

`MetalTerrainBackend` is a native macOS backend implemented through PyObjC + Metal. When Metal terrain is active, Minechunk can also route chunk meshing through the Metal mesher.

Fallback order when `ENGINE_MODE_METAL` is selected:

1. Metal terrain backend
2. `wgpu` terrain backend
3. CPU terrain backend

## Measured Performance

The checked-in measurements below describe a profiled Apple M4 run using the **CPU backend**.

These are **profiled** numbers. The built-in HUD and profiling instrumentation add real overhead, so they should not be treated as absolute ceilings.

| Scenario | Motion / Load | Result |
| --- | --- | --- |
| End-to-end chunk streaming, meshing, and rendering | Flying at approximately `1.5k blocks/s` | approximately `550 chunks/s`, `P99 4.7 ms` |
| Saturated visible set | Standing in a fully loaded `65 × 65` chunk field | `P99 36.2 ms` |

Those figures describe the CPU path under a real visible scene, not a terrain-only benchmark.

## Build And Run

Install base dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run:

```bash
python3 main.py
```

Optional macOS Metal support:

```bash
python3 -m pip install pyobjc-framework-Metal
```

## Backend Selection

Backend selection lives in `engine/renderer_config.py`:

- `ENGINE_MODE_CPU`
- `ENGINE_MODE_WGPU`
- `ENGINE_MODE_METAL`

Checked-in default:

```python
engine_mode = ENGINE_MODE_METAL
```

If the preferred backend cannot be created, Minechunk prints a warning and falls back to an available backend.

## Controls

- `WASD` or arrow keys: horizontal movement
- `X`: move up
- `Z`: move down
- `Shift`: sprint / fast flight
- left mouse drag: look around
- `F3`: toggle profiling HUD
- `R`: regenerate world with a new seed

## Repository Layout

- `main.py` — entry point
- `engine/renderer.py` — main runtime, render loop, chunk streaming, visibility, submission
- `engine/voxel_world.py` — terrain facade and backend selection
- `engine/cpu_terrain_backend.py` — CPU terrain generation path
- `engine/wgpu_terrain_backend.py` — `wgpu` terrain compute backend
- `engine/metal_terrain_backend.py` — native Metal terrain backend
- `engine/wgpu_chunk_mesher.py` — `wgpu` GPU meshing path
- `engine/metal_chunk_mesher.py` — Metal GPU meshing path
- `engine/terrain_kernels.py` — terrain and mesh-generation kernels
- `engine/mesh_cache_helpers.py` — visible-batch and tile-cache helpers
- `engine/hud_profile_helpers.py` — profiling HUD generation
- `docs/` — captured demo screenshots
- `res/` — HUD font asset

## Screenshots

| CPU | Wgpu | Metal |
| --- | --- | --- |
| ![CPU mode screenshot](docs/m4-cpu-demo-screenshot.png) | ![Wgpu mode screenshot](docs/m4-wgpu-demo-screenshot.png) | ![Metal mode screenshot](docs/m4-metal-demo-screenshot.png) |

## Notes

Minechunk is a renderer-and-systems project, not a content pipeline or gameplay project.

The point is not “Minecraft clone in Python.” The point is building a chunk-streamed voxel engine where backend behavior, synchronization cost, residency pressure, and frame-time composition stay visible enough to study.
