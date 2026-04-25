from __future__ import annotations

"""Metal terrain surface backend with stacked ``(chunk_x, chunk_y, chunk_z)`` batches."""

import struct
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..types import ChunkSurfaceGpuBatch, ChunkSurfaceResult, ChunkVoxelResult
from ...terrain.kernels import (
    expand_chunk_surface_to_voxel_grid,
    fill_chunk_voxel_grid as cpu_fill_chunk_voxel_grid,
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
)
from ...world_constants import VERTICAL_CHUNK_STACK_ENABLED

try:
    import Metal
except Exception as exc:  # pragma: no cover
    Metal = None  # type: ignore[assignment]
    _METAL_IMPORT_ERROR = exc
else:
    _METAL_IMPORT_ERROR = None

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def profile(func):
        return func


GPU_TERRAIN_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

struct TerrainParams { float4 sample_origin; int4 chunk_and_sample; uint4 seed_and_pad; };
struct TerrainBatchParams { uint sample_size; uint chunk_size; uint height_limit; uint seed; };

inline uint mix_u32(uint value) {
    uint x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

inline float hash2(int ix, int iy, uint seed) {
    uint h = uint(ix) * 0x9e3779b9u;
    h = h ^ (uint(iy) * 0x85ebca6bu);
    h = h ^ (seed * 0xc2b2ae35u);
    h = mix_u32(h);
    return float(h & 0x00ffffffu) / 16777215.0f;
}
inline float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
inline float lerp_f(float a, float b, float t) { return a + (b - a) * t; }
inline float value_noise_2d(float x, float y, uint seed, float frequency) {
    float px = x * frequency; float py = y * frequency;
    float x0 = floor(px); float y0 = floor(py);
    float xf = px - x0; float yf = py - y0;
    int ix0 = int(x0); int iy0 = int(y0); int ix1 = ix0 + 1; int iy1 = iy0 + 1;
    float u = fade(xf); float v = fade(yf);
    float nx0 = lerp_f(hash2(ix0, iy0, seed), hash2(ix1, iy0, seed), u);
    float nx1 = lerp_f(hash2(ix0, iy1, seed), hash2(ix1, iy1, seed), u);
    return lerp_f(nx0, nx1, v) * 2.0f - 1.0f;
}
inline uint2 terrain_sample(float x, float z, uint seed, uint height_limit) {
    constexpr float terrain_frequency_scale = 0.3f;
    float broad = value_noise_2d(x, z, seed + 11u, 0.0009765625f * terrain_frequency_scale);
    float ridge = value_noise_2d(x, z, seed + 23u, 0.00390625f * terrain_frequency_scale);
    float detail = value_noise_2d(x, z, seed + 47u, 0.010416667f * terrain_frequency_scale);
    float micro = value_noise_2d(x, z, seed + 71u, 0.020833334f * terrain_frequency_scale);
    float nano = value_noise_2d(x, z, seed + 97u, 0.041666668f * terrain_frequency_scale);
    uint upper_bound = height_limit - 1u;
    float upper_bound_f = float(upper_bound);
    float normalized_height = 24.0f + broad * 11.0f + ridge * 8.0f + detail * 4.5f + micro * 1.75f + nano * 0.75f;
    float height_scale = upper_bound > 0u ? upper_bound_f / 50.0f : 1.0f;
    float height_f = clamp(normalized_height * height_scale, 4.0f, upper_bound_f);
    uint height_i = uint(height_f);
    uint sand_threshold = max(4u, uint(float(height_limit) * 0.18f));
    uint stone_threshold = max(sand_threshold + 6u, uint(float(height_limit) * 0.58f));
    uint snow_threshold = max(stone_threshold + 6u, uint(float(height_limit) * 0.82f));
    uint material = 4u;
    if (height_i >= snow_threshold) material = 6u;
    else if (height_i <= sand_threshold) material = 5u;
    else if (height_i >= stone_threshold && (detail + micro * 0.5f + nano * 0.35f) > 0.10f) material = 2u;
    return uint2(height_i, material);
}

kernel void sample_surface_profile_at_main(device uint* heights [[buffer(0)]], device uint* materials [[buffer(1)]], constant TerrainParams& params [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x != 0 || gid.y != 0 || gid.z != 0) return;
    uint2 result = terrain_sample(params.sample_origin.x, params.sample_origin.y, params.seed_and_pad.x, uint(params.sample_origin.w));
    heights[0] = result.x; materials[0] = result.y;
}

kernel void fill_chunk_surface_grids_main(device uint* heights [[buffer(0)]], device uint* materials [[buffer(1)]], constant TerrainParams& params [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint sample_size = uint(params.chunk_and_sample.z);
    if (gid.x >= sample_size || gid.y >= sample_size) return;
    int chunk_x = params.chunk_and_sample.x;
    int chunk_z = params.chunk_and_sample.y;
    int chunk_size = int(params.seed_and_pad.y);
    int origin_x = chunk_x * chunk_size - 1;
    int origin_z = chunk_z * chunk_size - 1;
    uint2 result = terrain_sample(float(origin_x + int(gid.x)), float(origin_z + int(gid.y)), params.seed_and_pad.x, uint(params.sample_origin.w));
    uint cell_index = gid.y * sample_size + gid.x;
    heights[cell_index] = result.x; materials[cell_index] = result.y;
}

kernel void fill_chunk_surface_batch_main(device uint* batch_heights [[buffer(0)]], device uint* batch_materials [[buffer(1)]], device const int4* batch_coords [[buffer(2)]], constant TerrainBatchParams& batch_params [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    uint sample_size = batch_params.sample_size;
    if (gid.x >= sample_size || gid.y >= sample_size) return;
    uint chunk_index = gid.z;
    int4 coord = batch_coords[chunk_index];
    int origin_x = coord.x * int(batch_params.chunk_size) - 1;
    int origin_z = coord.z * int(batch_params.chunk_size) - 1;
    uint2 result = terrain_sample(float(origin_x + int(gid.x)), float(origin_z + int(gid.y)), batch_params.seed, batch_params.height_limit);
    uint cell_index = chunk_index * sample_size * sample_size + gid.y * sample_size + gid.x;
    batch_heights[cell_index] = result.x; batch_materials[cell_index] = result.y;
}
"""


@dataclass
class _ChunkMetalBatch:
    chunks: list[tuple[int, int, int]]
    chunk_count: int
    max_chunks: int
    coords_array: np.ndarray
    coords_buffer: object
    params_buffer: object
    heights_buffer: object
    materials_buffer: object
    command_buffer: object | None = None


class _LeasedChunkSurfaceGpuBatch(ChunkSurfaceGpuBatch):
    pass


def _normalize_chunk_coord(coord) -> tuple[int, int, int]:
    if len(coord) >= 3:
        return int(coord[0]), int(coord[1]), int(coord[2])
    if len(coord) == 2:
        return int(coord[0]), 0, int(coord[1])
    raise ValueError(f"Invalid chunk coordinate: {coord!r}")


def _normalize_chunk_coords(chunks) -> list[tuple[int, int, int]]:
    return [_normalize_chunk_coord(chunk) for chunk in chunks]


class MetalTerrainBackend:
    def __init__(self, device=None, seed: int = 0, chunk_size: int = 64, height_limit: int | None = None, chunks_per_poll: int = 128) -> None:
        if Metal is None:
            raise RuntimeError("Metal bindings unavailable. Install with `pip install pyobjc-framework-Metal` on macOS.") from _METAL_IMPORT_ERROR
        if device is None or not hasattr(device, "newCommandQueue"):
            device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("Metal device unavailable.")
        self.device = device
        self.command_queue = device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal command queue.")
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.height_limit = self.chunk_size if height_limit is None else int(height_limit)
        self.chunks_per_poll = max(1, int(chunks_per_poll))
        self.sample_size = self.chunk_size + 2
        self.cell_count = self.sample_size * self.sample_size
        self._pending_jobs: deque[list[tuple[int, int, int]]] = deque()
        self._in_flight_batches: deque[_ChunkMetalBatch] = deque()
        self._next_job_id = 1
        self._submit_target_chunks = max(int(self.chunks_per_poll), min(int(self.chunks_per_poll) * 2, 128))
        self._batch_pool_size = 8
        self._max_in_flight_batches = 3
        self._available_batch_slots: deque[_ChunkMetalBatch] = deque()
        self._batch_slots_pending_reuse: deque[_ChunkMetalBatch] = deque()
        self._leased_surface_batches: dict[int, _ChunkMetalBatch] = {}
        self._resource_options = int(getattr(Metal, "MTLResourceStorageModeShared"))
        self._params_buffer = self._make_buffer(64)
        self._single_heights_buffer = self._make_buffer(4)
        self._single_materials_buffer = self._make_buffer(4)
        self._grid_heights_buffer = self._make_buffer(self.cell_count * 4)
        self._grid_materials_buffer = self._make_buffer(self.cell_count * 4)
        self._batch_params_payload = struct.pack("<4I", self.sample_size, self.chunk_size, self.height_limit, self.seed & 0xFFFFFFFF)
        library, err = self.device.newLibraryWithSource_options_error_(GPU_TERRAIN_SHADER, None, None)
        if err is not None or library is None:
            raise RuntimeError(f"Failed to compile Metal terrain shader: {err}")
        self._single_pipeline = self._create_pipeline(library, "sample_surface_profile_at_main")
        self._grid_pipeline = self._create_pipeline(library, "fill_chunk_surface_grids_main")
        self._batch_pipeline = self._create_pipeline(library, "fill_chunk_surface_batch_main")
        for _ in range(self._batch_pool_size):
            self._available_batch_slots.append(self._allocate_chunk_batch_resources(self._submit_target_chunks))

    def _create_pipeline(self, library, function_name: str):
        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Missing Metal kernel function '{function_name}'.")
        pipeline, err = self.device.newComputePipelineStateWithFunction_error_(function, None)
        if err is not None or pipeline is None:
            raise RuntimeError(f"Failed to create Metal compute pipeline for '{function_name}': {err}")
        return pipeline

    def _make_buffer(self, size: int):
        return self.device.newBufferWithLength_options_(max(4, int(size)), self._resource_options)

    @staticmethod
    def _buffer_memoryview(buffer, size: Optional[int] = None) -> memoryview:
        length = int(buffer.length()) if size is None else int(size)
        return memoryview(buffer.contents().as_buffer(length))

    def _write_buffer_bytes(self, buffer, data: bytes | bytearray | memoryview) -> None:
        payload = memoryview(data).cast("B")
        self._buffer_memoryview(buffer, len(payload))[: len(payload)] = payload

    def _write_buffer_array(self, buffer, array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array)
        payload = memoryview(contiguous).cast("B")
        self._buffer_memoryview(buffer, contiguous.nbytes)[: contiguous.nbytes] = payload

    @staticmethod
    def _buffer_uint32_view(buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(int(count) * 4), dtype=np.uint32, count=int(count))

    def _dispatch(self, pipeline, buffers: list[tuple[object, int]], grid_size: tuple[int, int, int], group_size: tuple[int, int, int]):
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for buffer, index in buffers:
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(*map(int, grid_size)), Metal.MTLSizeMake(*map(int, group_size)))
        encoder.endEncoding()
        command_buffer.commit()
        return command_buffer

    def _write_params(self, *, sample_origin_x: float, sample_origin_z: float, height_limit: int, chunk_x: int, chunk_z: int, sample_size: int, seed: int) -> None:
        payload = struct.pack(
            "<4f4i4I",
            float(sample_origin_x), float(sample_origin_z), 0.0, float(height_limit),
            int(chunk_x), int(chunk_z), int(sample_size), int(self.chunk_size),
            int(seed) & 0xFFFFFFFF, int(self.chunk_size) & 0xFFFFFFFF, 0, 0,
        )
        self._write_buffer_bytes(self._params_buffer, payload)

    @profile
    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        self._write_params(sample_origin_x=float(x), sample_origin_z=float(z), height_limit=self.height_limit, chunk_x=0, chunk_z=0, sample_size=1, seed=self.seed)
        cb = self._dispatch(self._single_pipeline, [(self._single_heights_buffer, 0), (self._single_materials_buffer, 1), (self._params_buffer, 2)], (1, 1, 1), (1, 1, 1))
        cb.waitUntilCompleted()
        return int(self._buffer_uint32_view(self._single_heights_buffer, 1)[0]), int(self._buffer_uint32_view(self._single_materials_buffer, 1)[0])

    @profile
    def fill_chunk_surface_grids(self, heights: np.ndarray, materials: np.ndarray, chunk_x: int, chunk_z: int) -> None:
        self._write_params(sample_origin_x=0.0, sample_origin_z=0.0, height_limit=self.height_limit, chunk_x=int(chunk_x), chunk_z=int(chunk_z), sample_size=self.sample_size, seed=self.seed)
        cb = self._dispatch(self._grid_pipeline, [(self._grid_heights_buffer, 0), (self._grid_materials_buffer, 1), (self._params_buffer, 2)], (self.sample_size, self.sample_size, 1), (8, 8, 1))
        cb.waitUntilCompleted()
        heights[:] = self._buffer_uint32_view(self._grid_heights_buffer, self.cell_count)
        materials[:] = self._buffer_uint32_view(self._grid_materials_buffer, self.cell_count)

    @profile
    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        heights = np.empty(self.cell_count, dtype=np.uint32)
        materials = np.empty(self.cell_count, dtype=np.uint32)
        self.fill_chunk_surface_grids(heights, materials, chunk_x, chunk_z)
        return heights, materials

    @profile
    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        if chunks:
            self._pending_jobs.appendleft(_normalize_chunk_coords(chunks))
            self._submit_next_batch()
        return job_id

    @profile
    def chunk_voxel_grid(self, chunk_x: int, chunk_y: int = 0, chunk_z: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if chunk_z is None:
            chunk_z = int(chunk_y); chunk_y = 0
        chunk_x = int(chunk_x); chunk_y = int(chunk_y); chunk_z = int(chunk_z)
        if VERTICAL_CHUNK_STACK_ENABLED:
            local_height = self.chunk_size
            blocks = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint8)
            voxel_materials = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint32)
            top_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            bottom_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            heights, materials = self.chunk_surface_grids(chunk_x, chunk_z)
            fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(blocks, voxel_materials, top_boundary, bottom_boundary, heights, materials, chunk_x, chunk_y, chunk_z, self.chunk_size, self.seed, self.height_limit)
            return blocks, voxel_materials
        blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
        voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
        cpu_fill_chunk_voxel_grid(blocks, voxel_materials, chunk_x, chunk_z, self.chunk_size, self.seed, self.height_limit)
        return blocks, voxel_materials

    @profile
    def _allocate_chunk_batch_resources(self, max_chunks: int) -> _ChunkMetalBatch:
        max_chunks = max(1, int(max_chunks))
        coords_array = np.empty((max_chunks, 4), dtype=np.int32)
        return _ChunkMetalBatch([], 0, max_chunks, coords_array, self._make_buffer(coords_array.nbytes), self._make_buffer(16), self._make_buffer(max_chunks * self.cell_count * 4), self._make_buffer(max_chunks * self.cell_count * 4), None)

    def _reclaim_batch_slots(self) -> None:
        while self._batch_slots_pending_reuse:
            batch = self._batch_slots_pending_reuse.popleft()
            batch.chunk_count = 0; batch.chunks.clear(); batch.command_buffer = None
            self._available_batch_slots.append(batch)

    def _release_gpu_surface_batch(self, lease_id: int) -> None:
        batch = self._leased_surface_batches.pop(int(lease_id), None)
        if batch is not None:
            self._batch_slots_pending_reuse.append(batch)

    @staticmethod
    def _batch_completed(batch: _ChunkMetalBatch) -> bool:
        cb = batch.command_buffer
        if cb is None:
            return True
        status = int(cb.status())
        if status == int(getattr(Metal, "MTLCommandBufferStatusCompleted", 4)):
            return True
        if status == int(getattr(Metal, "MTLCommandBufferStatusError", 5)):
            raise RuntimeError(f"Metal terrain command buffer failed: {cb.error()}")
        return False

    @profile
    def _create_chunk_batch(self, chunks: list[tuple[int, int, int]]) -> _ChunkMetalBatch:
        chunk_count = len(chunks)
        if self._available_batch_slots:
            batch = self._available_batch_slots.popleft()
            if chunk_count > batch.max_chunks:
                self._available_batch_slots.appendleft(batch)
                batch = self._allocate_chunk_batch_resources(max(self._submit_target_chunks, chunk_count))
        else:
            batch = self._allocate_chunk_batch_resources(max(self._submit_target_chunks, chunk_count))
        batch.chunk_count = chunk_count
        batch.chunks = list(chunks)
        for index, (chunk_x, chunk_y, chunk_z) in enumerate(chunks):
            batch.coords_array[index] = (int(chunk_x), int(chunk_y), int(chunk_z), 0)
        return batch

    @profile
    def _submit_next_batch(self) -> None:
        if not self._pending_jobs:
            return
        available_slots = max(0, self._max_in_flight_batches - len(self._in_flight_batches))
        if available_slots <= 0:
            return
        submitted: list[_ChunkMetalBatch] = []
        while self._pending_jobs and available_slots > 0:
            merged: list[tuple[int, int, int]] = []
            while self._pending_jobs and len(merged) < self._submit_target_chunks:
                job = self._pending_jobs.popleft()
                take = min(self._submit_target_chunks - len(merged), len(job))
                merged.extend(job[:take])
                if take < len(job):
                    self._pending_jobs.appendleft(job[take:]); break
            if merged:
                submitted.append(self._create_chunk_batch(merged)); available_slots -= 1
        for batch in submitted:
            self._write_buffer_array(batch.coords_buffer, batch.coords_array[:batch.chunk_count])
            self._write_buffer_bytes(batch.params_buffer, self._batch_params_payload)
            batch.command_buffer = self._dispatch(self._batch_pipeline, [(batch.heights_buffer, 0), (batch.materials_buffer, 1), (batch.coords_buffer, 2), (batch.params_buffer, 3)], (self.sample_size, self.sample_size, batch.chunk_count), (8, 8, 1))
            self._in_flight_batches.append(batch)

    def _wait_for_batch(self, batch: _ChunkMetalBatch) -> None:
        if batch.command_buffer is not None:
            batch.command_buffer.waitUntilCompleted()
            if not self._batch_completed(batch):
                raise RuntimeError(f"Metal terrain command buffer did not complete: {batch.command_buffer.status()}")

    @profile
    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        self._reclaim_batch_slots()
        while self._in_flight_batches:
            batch = self._in_flight_batches[0]
            if not self._batch_completed(batch):
                break
            batch = self._in_flight_batches.popleft()
            total_cells = batch.chunk_count * self.cell_count
            heights_data = self._buffer_uint32_view(batch.heights_buffer, total_cells).copy()
            materials_data = self._buffer_uint32_view(batch.materials_buffer, total_cells).copy()
            for index, (chunk_x, chunk_y, chunk_z) in enumerate(batch.chunks):
                start = index * self.cell_count; end = start + self.cell_count
                ready.append(ChunkSurfaceResult(chunk_x=chunk_x, chunk_y=chunk_y, chunk_z=chunk_z, heights=heights_data[start:end], materials=materials_data[start:end], source="metal_gpu"))
            self._batch_slots_pending_reuse.append(batch)
        self._submit_next_batch()
        return ready

    @profile
    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        ready: list[ChunkSurfaceGpuBatch] = []
        self._reclaim_batch_slots()
        while self._in_flight_batches:
            batch = self._in_flight_batches[0]
            if not self._batch_completed(batch):
                break
            batch = self._in_flight_batches.popleft()
            lease_id = id(batch)
            self._leased_surface_batches[lease_id] = batch
            def _release(lease_id: int = lease_id, backend: "MetalTerrainBackend" = self) -> None:
                backend._release_gpu_surface_batch(lease_id)
            surface_batch = _LeasedChunkSurfaceGpuBatch(
                chunks=list(batch.chunks),
                heights_buffer=batch.heights_buffer,
                materials_buffer=batch.materials_buffer,
                cell_count=self.cell_count,
                source="metal_gpu_leased",
                device_kind="metal",
            )
            setattr(surface_batch, "_release_callback", _release)
            ready.append(surface_batch)
        self._submit_next_batch()
        return ready

    @profile
    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        return self.request_chunk_surface_batch(chunks)

    def _voxel_result_from_surface_result(self, surface_result: ChunkSurfaceResult) -> ChunkVoxelResult:
        chunk_x = int(surface_result.chunk_x); chunk_y = int(getattr(surface_result, "chunk_y", 0)); chunk_z = int(surface_result.chunk_z)
        top_boundary = None; bottom_boundary = None; is_empty_chunk = False
        if VERTICAL_CHUNK_STACK_ENABLED:
            local_height = self.chunk_size; origin_y = chunk_y * self.chunk_size
            blocks = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint8)
            voxel_materials = np.zeros((local_height, self.sample_size, self.sample_size), dtype=np.uint32)
            top_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            bottom_boundary = np.zeros((self.sample_size, self.sample_size), dtype=np.uint8)
            is_empty_chunk = int(np.max(surface_result.heights)) <= origin_y
            if not is_empty_chunk:
                fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(blocks, voxel_materials, top_boundary, bottom_boundary, surface_result.heights, surface_result.materials, chunk_x, chunk_y, chunk_z, self.chunk_size, self.seed, self.height_limit)
        else:
            blocks = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint8)
            voxel_materials = np.zeros((self.height_limit, self.sample_size, self.sample_size), dtype=np.uint32)
            expand_chunk_surface_to_voxel_grid(blocks, voxel_materials, surface_result.heights, surface_result.materials, self.chunk_size, self.height_limit)
        return ChunkVoxelResult(chunk_x=chunk_x, chunk_y=chunk_y, chunk_z=chunk_z, blocks=blocks, materials=voxel_materials, source=surface_result.source, top_boundary=top_boundary, bottom_boundary=bottom_boundary, is_empty=is_empty_chunk)

    @profile
    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return [self._voxel_result_from_surface_result(result) for result in self.poll_ready_chunk_surface_batches()]

    def has_pending_chunk_surface_batches(self) -> bool:
        return bool(self._in_flight_batches) or bool(self._pending_jobs) or bool(self._leased_surface_batches)
    def has_pending_chunk_voxel_batches(self) -> bool:
        return self.has_pending_chunk_surface_batches()

    @profile
    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        ready: list[ChunkSurfaceResult] = []
        while self._pending_jobs or self._in_flight_batches:
            if self._in_flight_batches:
                self._wait_for_batch(self._in_flight_batches[0])
            ready.extend(self.poll_ready_chunk_surface_batches())
        return ready

    @profile
    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return [self._voxel_result_from_surface_result(result) for result in self.flush_chunk_surface_batches()]

    def terrain_backend_label(self) -> str:
        return "Metal"

    def destroy(self) -> None:
        while self._in_flight_batches:
            batch = self._in_flight_batches.popleft()
            try: self._wait_for_batch(batch)
            except Exception: pass
        self._pending_jobs.clear(); self._leased_surface_batches.clear(); self._available_batch_slots.clear(); self._batch_slots_pending_reuse.clear()
        self.command_queue = None; self.device = None
