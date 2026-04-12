# Infinite Procedural World

A real-time voxel terrain renderer built in Python on top of `wgpu-py`, with deterministic procedural generation, pluggable CPU/GPU terrain backends, GPU meshing, GPU-driven visibility culling, indirect draw submission, and in-engine profiling HUDs.

## Screenshots

| CPU backend | GPU backend |
| --- | --- |
| ![CPU mode screenshot](docs/m4-cpu-demo-screenshot.png) | ![GPU mode screenshot](docs/m4-gpu-demo-screenshot.png) |

## Overview

This project is a renderer-first voxel terrain tech demo focused on frame-time decomposition, chunk streaming, and render-path experimentation.

The world is chunked in the X/Z plane, generated deterministically from `(seed, chunk_x, chunk_z)`, and streamed on demand as the camera moves. The renderer supports both CPU and GPU execution paths for terrain generation and meshing, and includes internal tooling to benchmark chunk throughput, batch sizing, frame breakdown, and render capacity.

## Performance

This is not a synthetic offline terrain test. It is a live renderer.

In local testing, the CPU terrain backend can generate roughly **~500 real chunks/second on a single core while maintaining a live, viewable scene**. At that point, terrain generation is no longer the dominant ceiling in that configuration; render cost, mesh density, visibility, draw submission, and GPU work start to matter more.

That is the important threshold for a voxel engine: once chunk generation becomes fast enough, the problem shifts from “can the world keep up?” to “can the renderer keep up?”

### Why that matters

- Real chunk dimensions: `32 x 128 x 32`
- Single-core CPU terrain generation
- Live scene on screen, not an offline bake
- Benchmark path measures real chunk drain rate in chunks/second
- Renderer includes profiling overlays and frame breakdown instrumentation

## Core technical features

- Infinite procedural terrain with deterministic sampling
- Chunk size `32 x 128 x 32`
- CPU terrain backend
- GPU terrain backend slot using compute-based terrain generation
- CPU voxel meshing path
- GPU voxel meshing path
- GPU visibility culling using per-mesh bounding spheres
- GPU indirect draw command generation
- Persistent mesh slab allocator for chunk vertex storage
- Chunk cache with eviction and reuse
- Optional merged render batches for distant terrain
- Profiling HUD and frame breakdown overlay
- Benchmark harness for terrain throughput, validation, and render scaling

## Runtime model

### World representation

The terrain is infinite in X/Z and vertically capped at `128` blocks. Surface generation is deterministic and driven from a compact procedural height/material sampler. Chunk requests are resolved lazily and cached once meshed.

### Terrain backends

`VoxelWorld` is a façade over a swappable terrain backend:

- `CpuTerrainBackend`: computes chunk surface grids and voxel grids on CPU
- `MetalTerrainBackend`: dedicated GPU backend slot using the current compute path as a placeholder for a future native Metal-oriented implementation

Both backends expose the same chunk-surface and chunk-voxel interfaces, so renderer-side scheduling does not depend on the generation path.

### Meshing paths

Two meshing strategies exist:

- **CPU meshing** using Numba-accelerated kernels
- **GPU meshing** using compute shaders for voxel-face counting, prefix/scan, and vertex emission

The GPU path builds mesh data asynchronously and finalizes chunk meshes into renderer-managed vertex storage.

### Render path

The render loop uses `wgpu-py` and a custom WGSL pipeline:

1. Update camera and visible chunk set
2. Stream/generate missing chunks
3. Build or finalize chunk meshes
4. Upload camera uniform
5. Build visibility records for resident meshes
6. Run GPU visibility culling
7. Emit indirect draw commands
8. Submit render pass
9. Overlay profiling HUDs

## Renderer architecture

### Chunk streaming

The renderer tracks visible chunk coordinates around the camera and schedules chunk preparation with bounded request budgets. Missing chunks are prioritized with a forward-cone heuristic so camera-facing terrain arrives first.

### Chunk cache

Chunk meshes are stored in an ordered cache keyed by `(chunk_x, chunk_z)`. Cached meshes include:

- vertex count
- GPU vertex buffer handle
- chunk-space bounds
- allocation metadata
- creation timestamp

### Mesh storage allocator

Chunk mesh output is stored in persistent GPU slabs and suballocated into aligned regions rather than always allocating one standalone GPU buffer per chunk. This reduces allocation churn and makes chunk mesh residency more stable under heavy streaming.

Tracked allocator state includes:

- slab count
- used bytes
- free bytes
- largest free range
- live allocation count

### GPU visibility culling

Resident chunk meshes carry bounding spheres derived from chunk center and max height. These bounds are uploaded into a visibility-record buffer, and a compute pass tests them against the camera frustum. Surviving meshes write indirect draw commands; rejected meshes write zero-vertex commands.

This keeps visibility classification on GPU and reduces CPU-side draw filtering overhead when many chunk meshes are resident.

### Indirect draw path

The renderer supports indirect draw command buffers, allowing GPU-generated visibility results to flow directly into submission. This reduces CPU work when scene residency grows.

### Profiling overlays

Two HUDs are built into the engine:

- **Profiler HUD**: frame stats, CPU hotspots, backend labels, allocator state
- **Frame breakdown HUD**: world update, visibility lookup, chunk streaming, camera upload, render encode, command finish, queue submit, wall-frame timing, draw calls, visible vertices, pending chunk requests, and chunk memory

## Procedural generation

Terrain sampling is based on layered 2D value noise with broad, ridge, and detail components. Surface material is derived from sampled height bands and local detail thresholds.

Current material palette includes:

- bedrock
- stone
- dirt
- grass
- sand
- snow

For voxel expansion, the surface layer is converted into full chunk voxel occupancy/material grids, then meshed either on CPU or GPU.

## Shader pipeline

The project includes several WGSL stages:

- terrain surface generation compute shader
- voxel surface expansion compute shader
- voxel mesh count / scan / emit compute shaders
- mesh visibility compute shader
- terrain render shader
- HUD render shader

The main terrain render shader performs camera-relative projection in shader code and shades fragments using a directional light.

## Benchmarks and validation

`benchmark_chunk_generation.py` includes tooling for:

- surface-grid throughput measurement
- chunk mesh build latency measurement
- terrain batch-size sweeps
- frame-mode isolation tests
- render-capacity search by chunk radius
- terrain backend validation
- GPU timestamp-assisted render timing where supported

This makes the repo useful both as a renderer demo and as a profiling / optimization sandbox.

## Repo layout

### `main.py`
Minimal entry point that creates `TerrainRenderer` and starts the render loop.

### `renderer.py`
Main engine loop, input, camera, chunk streaming, cache management, GPU meshing orchestration, visibility culling, indirect rendering, slab allocation, and profiling HUDs.

### `voxel_world.py`
World façade exposing deterministic terrain queries and backend routing.

### `cpu_terrain_backend.py`
CPU implementation of surface-grid and voxel-grid generation, with batched request / poll interfaces.

### `metal_terrain_backend.py`
GPU terrain backend slot using compute-driven chunk generation through `wgpu-py`.

### `terrain_kernels.py`
Numba-accelerated terrain kernels for noise sampling, surface generation, voxel expansion, and CPU mesh construction.

### `terrain_backend.py`
Backend protocol and shared result dataclasses.

### `benchmark_chunk_generation.py`
Microbenchmark and validation harness for terrain, meshing, and render throughput studies.

## Build and run

```bash
pip install -r requirements.txt
python3 main.py
```

## Dependencies

* `wgpu`
* `rendercanvas`
* `glfw`
* `numpy`
* `numba`

## Controls

* `WASD` or arrow keys — move horizontally
* `X` — move up
* `Z` — move down
* `Right Shift` — sprint
* Left mouse drag — look around
* `F3` — toggle profiling HUD
* `R` — regenerate world with a new seed

## Why this repo exists

This project is mainly a sandbox for pushing a voxel engine until the bottleneck moves.

Useful questions here are not only:

* How fast is terrain generation?
* How many chunks per second can the backend sustain?

But also:

* When does chunk generation stop being the bottleneck?
* When do draw calls dominate?
* How much does GPU meshing shift the ceiling?
* How expensive is visibility?
* How much chunk residency can the renderer carry before frame time collapses?
* When does the engine become primarily render-bound?

## Notes

* Default engine mode is configured in code and currently targets the GPU path
* GPU terrain is isolated behind a backend interface and falls back to CPU if setup fails
* GPU meshing falls back to CPU meshing if required compute pipelines cannot be created
* The project is designed for iteration and profiling, so instrumentation is part of the runtime design rather than an external afterthought
