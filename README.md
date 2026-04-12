# Minechunk

Minechunk is a live voxel terrain playground built in Python on top of `wgpu-py`. It is designed to show how far a procedural world can be pushed when terrain generation, meshing, visibility, and rendering all stay on the critical path in real time.

It combines deterministic world generation, swappable CPU/GPU terrain backends, GPU meshing, GPU-driven visibility culling, indirect draw submission, and in-engine profiling overlays.

## Screenshots

| CPU backend | GPU backend |
| --- | --- |
| ![CPU mode screenshot](docs/m4-cpu-demo-screenshot.png) | ![GPU mode screenshot](docs/m4-gpu-demo-screenshot.png) |

## What Makes It Interesting

Minechunk is renderer-first by design. Instead of baking terrain ahead of time, it streams chunked terrain on demand, builds meshes as the camera moves, and keeps enough instrumentation in the loop to show where the frame time is really going.

That makes it useful as more than a demo:

- a practical testbed for procedural terrain ideas
- a benchmark harness for chunk throughput and mesh generation
- a profiling sandbox for CPU/GPU tradeoffs
- a render-path experiment for visibility, batching, and indirect submission

## Performance Snapshot

This is a live renderer, not an offline terrain generator.

In local testing, the CPU terrain backend can generate roughly **~500 real chunks/second on a single core** while still maintaining a viewable scene. Once terrain generation reaches that level, the bottleneck shifts: render cost, mesh density, visibility, draw submission, and GPU work start to matter more.

That is the interesting part of a voxel engine. When the world keeps up, the renderer becomes the challenge.

### Why that matters

- Real chunk dimensions: `32 x 128 x 32`
- Single-core CPU terrain generation
- Live scene on screen, not a static bake
- Benchmark path measures real chunk drain rate in chunks/second
- Built-in profiling overlays and frame breakdown instrumentation

## Core Highlights

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

## Runtime Model

### World Representation

Terrain extends infinitely in X/Z and is vertically capped at `128` blocks. Surface generation is deterministic and driven from a compact procedural height/material sampler. Chunk requests are resolved lazily and cached once meshed.

### Terrain Backends

`VoxelWorld` is a façade over a swappable terrain backend:

- `CpuTerrainBackend`: computes chunk surface grids and voxel grids on CPU
- `MetalTerrainBackend`: dedicated GPU backend slot that currently uses the project’s compute path as a placeholder for a future native Metal-oriented implementation

Both backends expose the same chunk-surface and chunk-voxel interfaces, so renderer-side scheduling does not depend on the generation path.

### Meshing Paths

Two meshing strategies are available:

- CPU meshing using Numba-accelerated kernels
- GPU meshing using compute shaders for voxel-face counting, prefix/scan, and vertex emission

The GPU path builds mesh data asynchronously and finalizes chunk meshes into renderer-managed vertex storage.

### Render Path

The render loop uses `wgpu-py` and a custom WGSL pipeline:

1. Update the camera and visible chunk set
2. Stream or generate missing chunks
3. Build or finalize chunk meshes
4. Upload the camera uniform
5. Build visibility records for resident meshes
6. Run GPU visibility culling
7. Emit indirect draw commands
8. Submit the render pass
9. Overlay profiling HUDs

## Rendering Architecture

### Chunk Streaming

The renderer tracks visible chunk coordinates around the camera and schedules chunk preparation with bounded request budgets. Missing chunks are prioritized with a forward-cone heuristic so camera-facing terrain arrives first.

### Chunk Cache

Chunk meshes are stored in an ordered cache keyed by `(chunk_x, chunk_z)`. Cached meshes include:

- vertex count
- GPU vertex buffer handle
- chunk-space bounds
- allocation metadata
- creation timestamp

### Mesh Storage Allocator

Chunk mesh output is stored in persistent GPU slabs and suballocated into aligned regions rather than always allocating one standalone GPU buffer per chunk. This reduces allocation churn and keeps chunk residency steadier under heavy streaming.

Tracked allocator state includes:

- slab count
- used bytes
- free bytes
- largest free range
- live allocation count

### GPU Visibility Culling

Resident chunk meshes carry bounding spheres derived from chunk center and max height. These bounds are uploaded into a visibility-record buffer, and a compute pass tests them against the camera frustum. Surviving meshes write indirect draw commands; rejected meshes write zero-vertex commands.

This keeps visibility classification on GPU and reduces CPU-side draw filtering overhead when many chunk meshes are resident.

### Indirect Draw Path

The renderer supports indirect draw command buffers, allowing GPU-generated visibility results to flow directly into submission. That keeps CPU overhead lower as the scene grows denser.

### Profiling Overlays

Two HUDs are built into the engine:

- Profiler HUD: frame stats, CPU hotspots, backend labels, allocator state
- Frame breakdown HUD: world update, visibility lookup, chunk streaming, camera upload, render encode, command finish, queue submit, wall-frame timing, draw calls, visible vertices, pending chunk requests, and chunk memory

## Procedural Generation

Terrain sampling is based on layered 2D value noise with broad, ridge, and detail components. Surface material is derived from sampled height bands and local detail thresholds.

Current material palette:

- bedrock
- stone
- dirt
- grass
- sand
- snow

For voxel expansion, the surface layer is converted into full chunk voxel occupancy/material grids, then meshed either on CPU or GPU.

## Shader Pipeline

The project includes several WGSL stages:

- terrain surface generation compute shader
- voxel surface expansion compute shader
- voxel mesh count / scan / emit compute shaders
- mesh visibility compute shader
- terrain render shader
- HUD render shader

The main terrain render shader performs camera-relative projection in shader code and shades fragments using a directional light.

## Benchmarks and Validation

`benchmark_chunk_generation.py` includes tooling for:

- surface-grid throughput measurement
- chunk mesh build latency measurement
- terrain batch-size sweeps
- frame-mode isolation tests
- render-capacity search by chunk radius
- terrain backend validation
- GPU timestamp-assisted render timing where supported

This makes the repo useful both as a renderer demo and as a profiling/optimization sandbox.

## Repo Layout

- `main.py`: minimal entry point that creates `TerrainRenderer` and starts the render loop
- `renderer.py`: main engine loop, input, camera, chunk streaming, cache management, GPU meshing orchestration, visibility culling, indirect rendering, slab allocation, and profiling HUDs
- `voxel_world.py`: world façade exposing deterministic terrain queries and backend routing
- `cpu_terrain_backend.py`: CPU implementation of surface-grid and voxel-grid generation, with batched request/poll interfaces
- `metal_terrain_backend.py`: GPU terrain backend slot using compute-driven chunk generation through `wgpu-py`
- `terrain_kernels.py`: Numba-accelerated terrain kernels for noise sampling, surface generation, voxel expansion, and CPU mesh construction
- `terrain_backend.py`: backend protocol and shared result dataclasses
- `benchmark_chunk_generation.py`: microbenchmark and validation harness for terrain, meshing, and render throughput studies

## Build and Run

```bash
pip install -r requirements.txt
python3 main.py
```

## Dependencies

- `wgpu`
- `rendercanvas`
- `glfw`
- `numpy`
- `numba`

## Controls

- `WASD` or arrow keys: move horizontally
- `X`: move up
- `Z`: move down
- `Right Shift`: sprint
- Left mouse drag: look around
- `F3`: toggle profiling HUD
- `R`: regenerate the world with a new seed
