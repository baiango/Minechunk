# Minechunk

Minechunk is a live voxel terrain renderer and profiling playground written in Python on top of `wgpu-py`.

The current engine is built around chunk streaming, batched terrain generation, GPU-assisted meshing, persistent mesh residency, tile merging, and GPU visibility/indirect draw submission. The older README described an earlier design; this one reflects how the engine works today.

## Screenshots

| CPU | Wgpu | Metal |
| --- | --- | --- |
| ![CPU mode screenshot](docs/m4-cpu-demo-screenshot.png) | ![Wgpu mode screenshot](docs/m4-wgpu-demo-screenshot.png) | ![Metal mode screenshot](docs/m4-metal-demo-screenshot.png) |

## Current Engine At A Glance

- Infinite X/Z terrain with chunk dimensions `32 x 128 x 32`
- Three engine modes: `cpu`, `wgpu`, and `metal`
- `VoxelWorld` facade with swappable CPU, `Wgpu`, and `Metal` terrain backends
- Batched chunk requests and ordered chunk residency cache
- CPU meshing fallback plus GPU voxel meshing via WGSL compute passes
- Shared mesh slab allocator instead of one GPU vertex buffer per chunk
- Optional merged render batches for mature `4 x 4` chunk tiles
- GPU visibility culling from per-batch bounds into indirect draw commands
- Built-in profiling HUD and frame breakdown overlay
- Benchmark and validation harness for terrain throughput and render-path studies

The default checked-in renderer config currently uses `ENGINE_MODE_METAL` in `renderer_config.py`.

## Runtime Model

### World And Chunk Scheduling

The renderer tracks a square chunk radius around the camera and keeps an explicit set of visible, missing, pending, and displayed chunks. Missing chunks are queued in bounded batches so the renderer can keep drawing while terrain and meshing continue in the background.

Chunk meshes live in an ordered cache keyed by `(chunk_x, chunk_z)`. When the cache exceeds capacity, older meshes are evicted and their mesh storage is released back to the allocator.

### Terrain Backends

`VoxelWorld` hides the terrain source behind one interface:

- `CpuTerrainBackend` samples terrain and expands chunk voxel grids on CPU.
- `WgpuTerrainBackend` batches chunk surface generation on the GPU through `wgpu`.
- `MetalTerrainBackend` is a real Metal compute backend on macOS via PyObjC, not a placeholder stub.

The GPU backends currently batch surface generation on GPU, then produce chunk voxel results for the meshing stage. Rendering and meshing still go through the `wgpu` renderer.

### Terrain Model

Terrain is deterministic and heightfield-driven. Surface height and top material come from layered 2D value noise, then each surface sample is expanded into a voxel column with:

- `BEDROCK` at the bottom
- `STONE` below the surface band
- `DIRT` just under the top layer
- `GRASS`, `SAND`, or `SNOW` as the surface material depending on height/detail

This means the current world is a surface terrain engine expanded into voxels, rather than a full volumetric cave system.

### Meshing

Two mesh paths are implemented:

- CPU meshing through Numba-accelerated terrain kernels
- GPU meshing through compute passes that count visible faces, scan per-chunk offsets, and emit final vertices

Finished meshes are packed into shared GPU slabs and tracked with allocation metadata. That keeps chunk residency steadier and avoids constant per-chunk buffer creation/destruction.

### Render Path

Per frame, the renderer:

1. Updates camera movement/input.
2. Refreshes the visible chunk set around the camera.
3. Uploads the camera uniform.
4. Builds visible draw batches from resident chunk meshes.
5. Optionally merges mature chunk groups into `4 x 4` tile batches.
6. Builds indirect draw metadata and optionally runs GPU visibility culling.
7. Submits the terrain render pass and profiling HUDs.
8. Services background GPU mesh work and drains more chunk prep work.

The preferred path uses per-batch bounds, a compute visibility pass, and indirect draw buffers so the GPU can reject hidden batches before draw submission.

## Current Scope

- The terrain is heightfield-based and then expanded into voxels.
- Greedy meshing is not implemented.
- LOD and geometry simplification are not implemented.
- The project is focused on live chunk throughput, render residency, and profiling rather than offline world baking.

## Repo Layout

- `main.py`: minimal entry point that starts `TerrainRenderer`
- `renderer.py`: render loop, input, camera, resource setup, draw submission, HUDs
- `renderer_config.py`: engine mode selection and top-level renderer tuning
- `voxel_world.py`: world facade plus backend selection/fallback logic
- `cpu_terrain_backend.py`: CPU terrain generation backend
- `wgpu_terrain_backend.py`: `wgpu` compute terrain backend
- `metal_terrain_backend.py`: native Metal terrain backend for macOS
- `chunk_generation_helpers.py`: visible-set refresh, chunk request scheduling, chunk prep drain
- `wgpu_chunk_mesher.py`: GPU voxel meshing, async batch finalization, GPU buffer lifecycle helpers
- `mesh_cache_helpers.py`: mesh cache, slab allocator, tile merges, visibility batch building
- `terrain_kernels.py`: Numba kernels for terrain sampling, voxel expansion, and CPU meshing
- `render_shaders.py`: WGSL shaders for terrain rendering, tile merging, meshing, visibility, and HUDs
- `benchmark_chunk_generation.py`: terrain validation and benchmark harness

## Build And Run

```bash
pip install -r requirements.txt
python3 main.py
```

For the native Metal terrain backend on macOS, also install:

```bash
pip install pyobjc-framework-Metal
```

To switch engine modes, edit `engine_mode` in `renderer_config.py`:

- `ENGINE_MODE_CPU`
- `ENGINE_MODE_WGPU`
- `ENGINE_MODE_METAL`

If the preferred GPU terrain backend cannot be created, the world facade falls back to another available backend and prints a warning.

## Benchmarks

The benchmark harness can validate terrain parity and measure terrain/render throughput:

```bash
python3 benchmark_chunk_generation.py --terrain-batch-size 128 --mesh-batch-size 64
```

It includes terrain validation, terrain batch-size sweeps, render-capacity searches, isolation passes, and GPU timestamp timing when the adapter supports timestamp queries.

## Dependencies

- `wgpu`
- `rendercanvas`
- `numpy`
- `numba` for optional CPU kernel acceleration
- `pyobjc-framework-Metal` for the optional macOS Metal backend

## Controls

- `WASD` or arrow keys: move horizontally
- `X`: move up
- `Z`: move down
- `Shift`: sprint/fly faster
- Left mouse drag: look around
- `F3`: toggle profiling HUDs
- `R`: regenerate the world with a new seed
