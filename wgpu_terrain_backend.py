from __future__ import annotations

"""
This module is the dedicated wgpu backend slot.

The current implementation uses the project’s existing GPU compute path
as a placeholder. It is kept separate from the CPU backend so a future
backend-specific implementation can be swapped in without touching the
renderer or world façade.
"""

import struct
from collections import deque
from dataclasses import dataclass

import numpy as np

from terrain_backend import ChunkSurfaceGpuBatch, ChunkSurfaceResult, ChunkVoxelResult
from terrain_kernels import expand_chunk_surface_to_voxel_grid

try:
    import wgpu
except Exception:  # pragma: no cover - optional during CPU-only fallback
    wgpu = None  # type: ignore[assignment]

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


GPU_TERRAIN_SHADER = """
struct TerrainParams {
    sample_origin: vec4f,
    chunk_and_sample: vec4i,
    seed_and_pad: vec4u,
}

struct TerrainBatchParams {
    sample_size: u32,
    chunk_size: u32,
    height_limit: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> heights: array<u32>;
@group(0) @binding(1) var<storage, read_write> materials: array<u32>;
@group(0) @binding(2) var<uniform> params: TerrainParams;

fn hash2(ix: i32, iy: i32, seed: u32) -> f32 {
    let value = sin(
        f32(ix) * 127.1 +
        f32(iy) * 311.7 +
        f32(seed) * 74.7
    ) * 43758.5453123;
    return fract(value);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn value_noise_2d(x: f32, y: f32, seed: u32, frequency: f32) -> f32 {
    let px = x * frequency;
    let py = y * frequency;
    let x0 = floor(px);
    let y0 = floor(py);
    let xf = px - x0;
    let yf = py - y0;

    let ix0 = i32(x0);
    let iy0 = i32(y0);
    let ix1 = ix0 + 1;
    let iy1 = iy0 + 1;

    let v00 = hash2(ix0, iy0, seed);
    let v10 = hash2(ix1, iy0, seed);
    let v01 = hash2(ix0, iy1, seed);
    let v11 = hash2(ix1, iy1, seed);

    let u = fade(xf);
    let v = fade(yf);
    let nx0 = lerp(v00, v10, u);
    let nx1 = lerp(v01, v11, u);
    return lerp(nx0, nx1, v) * 2.0 - 1.0;
}

fn terrain_sample(x: f32, z: f32, seed: u32, height_limit: u32) -> vec2u {
    let broad = value_noise_2d(x, z, seed + 11u, 0.0009765625);
    let ridge = value_noise_2d(x, z, seed + 23u, 0.00390625);
    let detail = value_noise_2d(x, z, seed + 47u, 0.010416667);

    var height_f = 26.0 + broad * 18.0 + ridge * 14.0 + detail * 8.0;
    if (height_f < 4.0) {
        height_f = 4.0;
    }

    let upper_bound = height_limit - 1u;
    let upper_bound_f = f32(upper_bound);
    if (height_f > upper_bound_f) {
        height_f = upper_bound_f;
    }
    let height_i = u32(height_f);

    var material = 4u;
    if (height_i >= 90u) {
        material = 6u;
    } else if (height_i <= 14u) {
        material = 5u;
    } else if (height_i >= 70u && detail > 0.12) {
        material = 2u;
    }
    return vec2u(height_i, material);
}

@compute @workgroup_size(1, 1, 1)
fn sample_surface_profile_at_main() {
    let result = terrain_sample(
        params.sample_origin.x,
        params.sample_origin.y,
        params.seed_and_pad.x,
        u32(params.sample_origin.w),
    );
    heights[0u] = result.x;
    materials[0u] = result.y;
}

@compute @workgroup_size(8, 8, 1)
fn fill_chunk_surface_grids_main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_size = u32(params.chunk_and_sample.z);
    if (gid.x >= sample_size || gid.y >= sample_size) {
        return;
    }

    let chunk_x = params.chunk_and_sample.x;
    let chunk_z = params.chunk_and_sample.y;
    let chunk_size = 32i;
    let origin_x = chunk_x * chunk_size - 1i;
    let origin_z = chunk_z * chunk_size - 1i;
    let world_x = f32(origin_x + i32(gid.x));
    let world_z = f32(origin_z + i32(gid.y));
    let result = terrain_sample(world_x, world_z, params.seed_and_pad.x, u32(params.sample_origin.w));

    let cell_index = gid.y * sample_size + gid.x;
    heights[cell_index] = result.x;
    materials[cell_index] = result.y;
}

@group(0) @binding(0) var<storage, read_write> batch_heights: array<u32>;
@group(0) @binding(1) var<storage, read_write> batch_materials: array<u32>;
@group(0) @binding(2) var<storage, read> batch_coords: array<vec2i>;
@group(0) @binding(3) var<uniform> batch_params: TerrainBatchParams;

@compute @workgroup_size(8, 8, 1)
fn fill_chunk_surface_batch_main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_size = batch_params.sample_size;
    if (gid.x >= sample_size || gid.y >= sample_size) {
        return;
    }

    let chunk_index = gid.z;
    let coord = batch_coords[chunk_index];
    let chunk_x = coord.x;
    let chunk_z = coord.y;
    let chunk_size = i32(batch_params.chunk_size);
    let origin_x = chunk_x * chunk_size - 1i;
    let origin_z = chunk_z * chunk_size - 1i;
    let world_x = f32(origin_x + i32(gid.x));
    let world_z = f32(origin_z + i32(gid.y));
    let result = terrain_sample(world_x, world_z, batch_params.seed, batch_params.height_limit);

    let cell_index = chunk_index * sample_size * sample_size + gid.y * sample_size + gid.x;
    batch_heights[cell_index] = result.x;
    batch_materials[cell_index] = result.y;
}
"""


@dataclass
class _ChunkGpuBatch:
    chunks: list[tuple[int, int]]
    chunk_count: int
    max_chunks: int
    coords_array: np.ndarray
    coords_buffer: object
    params_buffer: object
    heights_buffer: object
    materials_buffer: object
    bind_group: object


class _LeasedChunkSurfaceGpuBatch(ChunkSurfaceGpuBatch):
    pass


class WgpuTerrainBackend:
    def __init__(self, device, seed: int, chunk_size: int, height_limit: int, chunks_per_poll: int = 128) -> None:
        if wgpu is None:
            raise RuntimeError("wgpu is unavailable.")
        self.device = device
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.height_limit = int(height_limit)
        self.chunks_per_poll = max(1, int(chunks_per_poll))
        self.sample_size = self.chunk_size + 2
        self.cell_count = self.sample_size * self.sample_size
        self._pending_jobs: deque[list[tuple[int, int]]] = deque()
        self._in_flight_batches: deque[_ChunkGpuBatch] = deque()
        self._next_job_id = 1
        self._params_buffer = device.create_buffer(
            size=64,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._single_heights_buffer = device.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._single_materials_buffer = device.create_buffer(
            size=4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._grid_heights_buffer = device.create_buffer(
            size=self.cell_count * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._grid_materials_buffer = device.create_buffer(
            size=self.cell_count * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._batch_params_payload = struct.pack(
            "<4I",
            int(self.sample_size),
            int(self.chunk_size),
            int(self.height_limit),
            int(self.seed) & 0xFFFFFFFF,
        )
        self._submit_target_chunks = max(int(self.chunks_per_poll), min(int(self.chunks_per_poll) * 2, 128))
        self._batch_pool_size = 8
        self._max_in_flight_batches = 3
        self._available_batch_slots: deque[_ChunkGpuBatch] = deque()
        self._batch_slots_pending_reuse: deque[_ChunkGpuBatch] = deque()
        self._leased_surface_batches: dict[int, _ChunkGpuBatch] = {}
        self._bind_group_layout = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            ]
        )
        self._batch_bind_group_layout = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            ]
        )
        shader_module = device.create_shader_module(code=GPU_TERRAIN_SHADER)
        self._single_bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._single_heights_buffer}},
                {"binding": 1, "resource": {"buffer": self._single_materials_buffer}},
                {"binding": 2, "resource": {"buffer": self._params_buffer, "offset": 0, "size": 64}},
            ],
        )
        self._batch_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._batch_bind_group_layout]),
            compute={"module": shader_module, "entry_point": "fill_chunk_surface_batch_main"},
        )
        self._grid_bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._grid_heights_buffer}},
                {"binding": 1, "resource": {"buffer": self._grid_materials_buffer}},
                {"binding": 2, "resource": {"buffer": self._params_buffer, "offset": 0, "size": 64}},
            ],
        )
        self._single_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bind_group_layout]),
            compute={"module": shader_module, "entry_point": "sample_surface_profile_at_main"},
        )
        self._grid_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bind_group_layout]),
            compute={"module": shader_module, "entry_point": "fill_chunk_surface_grids_main"},
        )
        for _ in range(self._batch_pool_size):
            self._available_batch_slots.append(
                self._allocate_chunk_batch_resources(self._submit_target_chunks)
            )

    def _write_params(self, *, sample_origin_x: float, sample_origin_z: float, height_limit: int, chunk_x: int, chunk_z: int, sample_size: int, seed: int) -> None:
        payload = struct.pack(
            "<4f4i4I",
            float(sample_origin_x),
            float(sample_origin_z),
            0.0,
            float(height_limit),
            int(chunk_x),
            int(chunk_z),
            int(sample_size),
            0,
            int(seed) & 0xFFFFFFFF,
            0,
            0,
            0,
        )
        self.device.queue.write_buffer(self._params_buffer, 0, payload)

    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        self._write_params(sample_origin_x=float(x), sample_origin_z=float(z), height_limit=self.height_limit, chunk_x=0, chunk_z=0, sample_size=1, seed=self.seed)
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._single_pipeline)
        compute_pass.set_bind_group(0, self._single_bind_group)
        compute_pass.dispatch_workgroups(1, 1, 1)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        height = np.frombuffer(self.device.queue.read_buffer(self._single_heights_buffer, 0, 4), dtype=np.uint32, count=1)[0]
        material = np.frombuffer(self.device.queue.read_buffer(self._single_materials_buffer, 0, 4), dtype=np.uint32, count=1)[0]
        return int(height), int(material)

    def fill_chunk_surface_grids(self, heights: np.ndarray, materials: np.ndarray, chunk_x: int, chunk_z: int) -> None:
        self._write_params(sample_origin_x=0.0, sample_origin_z=0.0, height_limit=self.height_limit, chunk_x=chunk_x, chunk_z=chunk_z, sample_size=self.sample_size, seed=self.seed)
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._grid_pipeline)
        compute_pass.set_bind_group(0, self._grid_bind_group)
        workgroups = (self.sample_size + 7) // 8
        compute_pass.dispatch_workgroups(workgroups, workgroups, 1)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        heights[:] = np.frombuffer(self.device.queue.read_buffer(self._grid_heights_buffer, 0, self.cell_count * 4), dtype=np.uint32, count=self.cell_count)
        materials[:] = np.frombuffer(self.device.queue.read_buffer(self._grid_materials_buffer, 0, self.cell_count * 4), dtype=np.uint32, count=self.cell_count)

    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        sample_size = self.chunk_size + 2
        cell_count = sample_size * sample_size
        heights = np.empty(cell_count, dtype=np.uint32)
        materials = np.empty(cell_count, dtype=np.uint32)
        self.fill_chunk_surface_grids(heights, materials, chunk_x, chunk_z)
        return heights, materials

    def request_chunk_surface_batch(self, chunks: list[tuple[int, int]]) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        if chunks:
            # Push fresh requests to the front so newly-facing terrain wins over stale backlog.
            self._pending_jobs.appendleft(list(chunks))
            self._submit_next_batch()
        return job_id

    def chunk_voxel_grid(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        heights, materials = self.chunk_surface_grids(chunk_x, chunk_z)
        blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
        voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
        expand_chunk_surface_to_voxel_grid(blocks, voxel_materials, heights, materials, self.chunk_size, self.height_limit)
        return blocks, voxel_materials

    def _allocate_chunk_batch_resources(self, max_chunks: int) -> "_ChunkGpuBatch":
        max_chunks = max(1, int(max_chunks))
        coords_array = np.empty((max_chunks, 2), dtype=np.int32)
        coords_buffer = self.device.create_buffer(
            size=max(1, coords_array.nbytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        params_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        heights_buffer = self.device.create_buffer(
            size=max(1, max_chunks * self.cell_count * 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        materials_buffer = self.device.create_buffer(
            size=max(1, max_chunks * self.cell_count * 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        bind_group = self.device.create_bind_group(
            layout=self._batch_bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": heights_buffer}},
                {"binding": 1, "resource": {"buffer": materials_buffer}},
                {"binding": 2, "resource": {"buffer": coords_buffer}},
                {"binding": 3, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
            ],
        )
        return _ChunkGpuBatch(
            chunks=[],
            chunk_count=0,
            max_chunks=max_chunks,
            coords_array=coords_array,
            coords_buffer=coords_buffer,
            params_buffer=params_buffer,
            heights_buffer=heights_buffer,
            materials_buffer=materials_buffer,
            bind_group=bind_group,
        )

    def _reclaim_batch_slots(self) -> None:
        while self._batch_slots_pending_reuse:
            batch = self._batch_slots_pending_reuse.popleft()
            batch.chunk_count = 0
            batch.chunks.clear()
            self._available_batch_slots.append(batch)

    def _release_gpu_surface_batch(self, lease_id: int) -> None:
        batch = self._leased_surface_batches.pop(int(lease_id), None)
        if batch is None:
            return
        self._batch_slots_pending_reuse.append(batch)

    @profile
    def _create_chunk_batch(self, chunks: list[tuple[int, int]]) -> "_ChunkGpuBatch":
        chunk_count = len(chunks)
        target_capacity = max(int(self._submit_target_chunks), chunk_count)

        if self._available_batch_slots:
            batch = self._available_batch_slots.popleft()
            if chunk_count > batch.max_chunks:
                self._available_batch_slots.appendleft(batch)
                batch = self._allocate_chunk_batch_resources(target_capacity)
        else:
            batch = self._allocate_chunk_batch_resources(target_capacity)

        batch.chunk_count = chunk_count
        batch.chunks = list(chunks)
        if chunk_count > 0:
            batch.coords_array[:chunk_count] = np.asarray(chunks, dtype=np.int32).reshape(-1, 2)
        return batch

    @profile
    def _submit_next_batch(self) -> None:
        if not self._pending_jobs:
            return

        available_slots = max(0, self._max_in_flight_batches - len(self._in_flight_batches))
        if available_slots <= 0:
            return

        target_chunks = self._submit_target_chunks

        submitted: list[_ChunkGpuBatch] = []
        while self._pending_jobs and available_slots > 0:
            merged: list[tuple[int, int]] = []
            while self._pending_jobs and len(merged) < target_chunks:
                job = self._pending_jobs.popleft()
                take = min(target_chunks - len(merged), len(job))
                if take:
                    merged.extend(job[:take])
                if take < len(job):
                    self._pending_jobs.appendleft(job[take:])
                    break

            if not merged:
                break

            submitted.append(self._create_chunk_batch(merged))
            available_slots -= 1

        if not submitted:
            return

        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._batch_pipeline)
        workgroups = (self.sample_size + 7) // 8

        for tasks in submitted:
            coords_view = memoryview(tasks.coords_array[:tasks.chunk_count])
            self.device.queue.write_buffer(tasks.coords_buffer, 0, coords_view)
            self.device.queue.write_buffer(tasks.params_buffer, 0, self._batch_params_payload)
            compute_pass.set_bind_group(0, tasks.bind_group)
            compute_pass.dispatch_workgroups(workgroups, workgroups, tasks.chunk_count)

        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._in_flight_batches.extend(submitted)

    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        self._reclaim_batch_slots()
        while self._in_flight_batches:
            completed_batch = self._in_flight_batches.popleft()
            chunk_count = completed_batch.chunk_count
            cell_count = self.cell_count
            total_cells = chunk_count * cell_count
            heights_data = np.frombuffer(
                self.device.queue.read_buffer(completed_batch.heights_buffer, 0, total_cells * 4),
                dtype=np.uint32,
                count=total_cells,
            )
            materials_data = np.frombuffer(
                self.device.queue.read_buffer(completed_batch.materials_buffer, 0, total_cells * 4),
                dtype=np.uint32,
                count=total_cells,
            )
            for index, (chunk_x, chunk_z) in enumerate(completed_batch.chunks):
                start = index * cell_count
                end = start + cell_count
                ready.append(
                    ChunkSurfaceResult(
                        chunk_x=chunk_x,
                        chunk_z=chunk_z,
                        heights=heights_data[start:end].copy(),
                        materials=materials_data[start:end].copy(),
                        source="gpu",
                    )
                )
            self._batch_slots_pending_reuse.append(completed_batch)

        self._submit_next_batch()
        return ready

    @profile
    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        ready: list[ChunkSurfaceGpuBatch] = []
        self._reclaim_batch_slots()
        while self._in_flight_batches:
            completed_batch = self._in_flight_batches.popleft()
            lease_id = id(completed_batch)
            self._leased_surface_batches[lease_id] = completed_batch

            def _release(lease_id: int = lease_id, backend: "WgpuTerrainBackend" = self) -> None:
                backend._release_gpu_surface_batch(lease_id)

            surface_batch = _LeasedChunkSurfaceGpuBatch(
                chunks=list(completed_batch.chunks),
                heights_buffer=completed_batch.heights_buffer,
                materials_buffer=completed_batch.materials_buffer,
                cell_count=self.cell_count,
                source="gpu_leased",
            )
            setattr(surface_batch, "_release_callback", _release)
            ready.append(surface_batch)

        self._submit_next_batch()
        return ready

    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int]]) -> int:
        return self.request_chunk_surface_batch(chunks)

    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        for surface_result in self.poll_ready_chunk_surface_batches():
            blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
            voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
            expand_chunk_surface_to_voxel_grid(
                blocks,
                voxel_materials,
                surface_result.heights,
                surface_result.materials,
                self.chunk_size,
                self.height_limit,
            )
            ready.append(
                ChunkVoxelResult(
                    chunk_x=surface_result.chunk_x,
                    chunk_z=surface_result.chunk_z,
                    blocks=blocks,
                    materials=voxel_materials,
                    source=surface_result.source,
                )
            )
        return ready

    def has_pending_chunk_surface_batches(self) -> bool:
        return bool(self._in_flight_batches) or bool(self._pending_jobs)

    def has_pending_chunk_voxel_batches(self) -> bool:
        return self.has_pending_chunk_surface_batches()

    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        while self.has_pending_chunk_surface_batches():
            ready.extend(self.poll_ready_chunk_surface_batches())
        return ready

    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        ready: list[ChunkVoxelResult] = []
        while self.has_pending_chunk_voxel_batches():
            ready.extend(self.poll_ready_chunk_voxel_batches())
        return ready

    def terrain_backend_label(self) -> str:
        return "Wgpu"
