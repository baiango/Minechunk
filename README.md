# Minechunk

Minechunk is a chunk-streamed voxel terrain engine implemented in Python on top of `wgpu-py`. The codebase is organized around deterministic terrain synthesis, explicit chunk residency, meshing, visibility culling, and indirect draw submission. It is a measurement-oriented engine, not a simplification showcase.

The checked-in configuration defaults to the CPU terrain backend. Rendering still goes through the GPU-backed `wgpu` presentation path, and optional terrain backends exist for `wgpu` and native Metal on macOS.

## Engine Specification

- Chunk dimensions: `32 x 128 x 32`
- Chunk surface sample footprint: `34 x 34`
- Default render radius: `32` chunks (`1,024` blocks)
- Default visible square: `65 x 65` chunks (`4,225` chunk slots)
- Default mesh cache capacity: `4,225` chunks
- Terrain model: deterministic heightfield with voxel column expansion
- Render submission: indirect draws with GPU visibility culling
- Profiling: built-in HUD plus frame breakdown overlay
- Movement: free-flight camera with sprint scaling

The default residency envelope is a `65 x 65` chunk square. The cache capacity is sized to hold that envelope.

## Measured Performance

All figures below were recorded on an Apple M4 system using the CPU backend with profiling enabled.

Profiling overhead is substantial. The profiler materially slows the engine and lowers the observed frame-time and chunk-throughput numbers. Treat the values below as profiled measurements, not unprofiled ceilings, and do not compare them directly against unprofiled runs.

| Scenario | Motion / Load | Result |
| --- | --- | --- |
| End-to-end chunk streaming, meshing, and rendering | Flying at approximately `1.5k blocks/s` | Approximately `550 chunks/s`, `P99 4.7 ms` |
| Saturated visible set | Standing in a fully loaded `65 x 65` chunk field | `P99 36.2 ms` |

These numbers describe the CPU backend path, not a GPU-terrain-only configuration.

## Pipeline

The runtime is split into distinct stages so the cost of each stage stays visible:

1. Camera input and movement integration.
2. Visible chunk set recomputation around the camera.
3. Chunk request scheduling and backlog draining.
4. Terrain generation through the selected backend.
5. Mesh construction and mesh residency management.
6. Visibility classification and indirect draw preparation.
7. Render pass execution and HUD/profiler overlays.

The implementation uses shared mesh slabs and chunk allocation metadata instead of per-chunk buffer churn. That keeps residency stable under sustained streaming pressure.

## Terrain Backends

`VoxelWorld` exposes the terrain source through a backend facade:

- `CpuTerrainBackend` is the reference implementation and the default checked-in path.
- `WgpuTerrainBackend` generates terrain data via `wgpu` compute passes.
- `MetalTerrainBackend` is a native Metal backend for macOS via PyObjC.

Terrain is deterministic and heightfield-driven. Surface height and top material are sampled from layered 2D value noise, then expanded into voxel columns. This engine does not attempt to hide terrain cost behind aggressive simplification.

## Explicit Non-Goals

The following are intentionally not implemented:

- Greedy meshing
- Level of detail
- Geometry simplification

These are excluded by design to keep the visual output honest and to preserve raw geometry-generation and presentation cost as measured quantities.

## Repository Layout

- `main.py`: minimal entry point that starts `TerrainRenderer`
- `renderer.py`: camera, chunk residency, resource setup, draw submission, HUD, profiler
- `renderer_config.py`: engine mode selection and top-level renderer tuning
- `voxel_world.py`: world facade plus backend selection and fallback logic
- `cpu_terrain_backend.py`: CPU terrain generation backend
- `wgpu_terrain_backend.py`: `wgpu` compute terrain backend
- `metal_terrain_backend.py`: native Metal terrain backend for macOS
- `chunk_generation_helpers.py`: visible-set refresh, chunk request scheduling, backlog draining
- `wgpu_chunk_mesher.py`: GPU voxel meshing, batch finalization, buffer lifecycle helpers
- `mesh_cache_helpers.py`: mesh cache, slab allocator, tile merges, visibility batch building
- `terrain_kernels.py`: Numba-backed terrain sampling, voxel expansion, and CPU meshing kernels
- `render_shaders.py`: WGSL shaders for terrain rendering, meshing, visibility, and HUDs
- `benchmark_chunk_generation.py`: terrain validation and benchmark harness

## Build And Run

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

To enable the optional Metal terrain backend on macOS:

```bash
python3 -m pip install pyobjc-framework-Metal
```

Backend selection lives in `renderer_config.py` via `engine_mode`:

- `ENGINE_MODE_CPU`
- `ENGINE_MODE_WGPU`
- `ENGINE_MODE_METAL`

If the preferred backend cannot be created, the terrain facade falls back to an available backend and prints a warning.

## Benchmark Harness

The benchmark script measures terrain throughput, validates backend parity, sweeps batch sizes, and collects frame-time statistics when timestamp queries are available.

```bash
python3 benchmark_chunk_generation.py
```

The harness is intended for controlled comparisons between backend choices, cache sizes, batch sizes, and render-radius settings.

## Controls

- `WASD` or arrow keys: move horizontally
- `X`: move up
- `Z`: move down
- `Shift`: sprint and fly faster
- Left mouse drag: look around
- `F3`: toggle profiling HUDs
- `R`: regenerate the world with a new seed

## Captures

| CPU | Wgpu | Metal |
| --- | --- | --- |
| ![CPU mode screenshot](docs/m4-cpu-demo-screenshot.png) | ![Wgpu mode screenshot](docs/m4-wgpu-demo-screenshot.png) | ![Metal mode screenshot](docs/m4-metal-demo-screenshot.png) |
