from __future__ import annotations

import sys
import struct

import numpy as np

from .cpu_terrain_backend import CpuTerrainBackend
try:
    from .metal_terrain_backend import MetalTerrainBackend
except Exception as exc:  # pragma: no cover - optional on non-mac / CPU-only fallback
    MetalTerrainBackend = None  # type: ignore[assignment]
    METAL_TERRAIN_IMPORT_ERROR = exc
else:
    METAL_TERRAIN_IMPORT_ERROR = None

try:
    from .wgpu_terrain_backend import WgpuTerrainBackend
except Exception:  # pragma: no cover - optional during Metal-only deployments
    WgpuTerrainBackend = None  # type: ignore[assignment]

from .terrain_backend import ChunkSurfaceGpuBatch, ChunkSurfaceResult, ChunkVoxelResult, TerrainValidationReport

from .terrain_kernels import (
    AIR,
    BEDROCK,
    DIRT,
    STONE,
)


WORLD_HEIGHT = 128
CHUNK_SIZE = 32


def _create_preferred_gpu_backend(
    gpu_device,
    seed: int,
    chunk_size: int,
    height_limit: int,
    chunks_per_poll: int,
    *,
    prefer_metal_backend: bool = False,
):
    errors: list[str] = []

    if prefer_metal_backend:
        if MetalTerrainBackend is not None:
            metal_candidates = []
            if gpu_device is not None:
                metal_candidates.append((gpu_device, "provided device"))
            metal_candidates.append((None, "system default device"))

            tried_none = False
            for metal_device, label in metal_candidates:
                if metal_device is None:
                    if tried_none:
                        continue
                    tried_none = True
                try:
                    return MetalTerrainBackend(
                        metal_device,
                        seed,
                        chunk_size,
                        height_limit,
                        chunks_per_poll=chunks_per_poll,
                    )
                except Exception as exc:
                    errors.append(f"Metal terrain backend could not be created via {label} ({exc!s})")
        elif METAL_TERRAIN_IMPORT_ERROR is not None:
            errors.append(
                "Metal terrain backend is unavailable; falling back to wgpu terrain backend "
                f"({METAL_TERRAIN_IMPORT_ERROR!s})"
            )

        if errors:
            print(
                "Warning: Metal terrain backend could not be used; falling back to wgpu terrain. "
                + "; ".join(errors),
                file=sys.stderr,
            )

    if WgpuTerrainBackend is not None and gpu_device is not None:
        try:
            return WgpuTerrainBackend(
                gpu_device,
                seed,
                chunk_size,
                height_limit,
                chunks_per_poll=chunks_per_poll,
            )
        except Exception as exc:
            errors.append(f"wgpu terrain backend could not be created ({exc!s})")

    if not errors:
        if gpu_device is None:
            errors.append("GPU terrain was requested, but no compatible GPU device or backend was available")
        else:
            errors.append("GPU terrain was requested, but no compatible backend accepted the provided device")

    raise RuntimeError("; ".join(errors))


GPU_TERRAIN_SHADER = """
struct TerrainParams {
    sample_origin: vec4f,
    chunk_and_sample: vec4i,
    seed_and_pad: vec4u,
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
"""


class _TerrainGpuBackend:
    def __init__(self, device, chunk_size: int, height_limit: int) -> None:
        if wgpu is None:
            raise RuntimeError("wgpu is unavailable.")
        self.device = device
        self.chunk_size = int(chunk_size)
        self.height_limit = int(height_limit)
        self.sample_size = self.chunk_size + 2
        self.cell_count = self.sample_size * self.sample_size
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
        self._bind_group_layout = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
            ]
        )
        self._single_bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._single_heights_buffer}},
                {"binding": 1, "resource": {"buffer": self._single_materials_buffer}},
                {"binding": 2, "resource": {"buffer": self._params_buffer, "offset": 0, "size": 64}},
            ],
        )
        self._grid_bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._grid_heights_buffer}},
                {"binding": 1, "resource": {"buffer": self._grid_materials_buffer}},
                {"binding": 2, "resource": {"buffer": self._params_buffer, "offset": 0, "size": 64}},
            ],
        )
        shader_module = device.create_shader_module(code=GPU_TERRAIN_SHADER)
        self._single_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bind_group_layout]),
            compute={"module": shader_module, "entry_point": "sample_surface_profile_at_main"},
        )
        self._grid_pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._bind_group_layout]),
            compute={"module": shader_module, "entry_point": "fill_chunk_surface_grids_main"},
        )

    def _write_params(
        self,
        *,
        sample_origin_x: float,
        sample_origin_z: float,
        height_limit: int,
        chunk_x: int,
        chunk_z: int,
        sample_size: int,
        seed: int,
    ) -> None:
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

    def surface_profile_at(self, x: int, z: int, seed: int) -> tuple[int, int]:
        self._write_params(
            sample_origin_x=float(x),
            sample_origin_z=float(z),
            height_limit=self.height_limit,
            chunk_x=0,
            chunk_z=0,
            sample_size=1,
            seed=seed,
        )
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

    def fill_chunk_surface_grids(
        self,
        heights: np.ndarray,
        materials: np.ndarray,
        chunk_x: int,
        chunk_z: int,
        seed: int,
    ) -> None:
        self._write_params(
            sample_origin_x=0.0,
            sample_origin_z=0.0,
            height_limit=self.height_limit,
            chunk_x=chunk_x,
            chunk_z=chunk_z,
            sample_size=self.sample_size,
            seed=seed,
        )
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._grid_pipeline)
        compute_pass.set_bind_group(0, self._grid_bind_group)
        workgroups = (self.sample_size + 7) // 8
        compute_pass.dispatch_workgroups(workgroups, workgroups, 1)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])
        heights[:] = np.frombuffer(
            self.device.queue.read_buffer(self._grid_heights_buffer, 0, self.cell_count * 4),
            dtype=np.uint32,
            count=self.cell_count,
        )
        materials[:] = np.frombuffer(
            self.device.queue.read_buffer(self._grid_materials_buffer, 0, self.cell_count * 4),
            dtype=np.uint32,
            count=self.cell_count,
        )


class VoxelWorld:
    height: int = WORLD_HEIGHT
    chunk_size: int = CHUNK_SIZE

    def __init__(
        self,
        seed: int = 1337,
        *,
        gpu_device=None,
        prefer_gpu_terrain: bool = False,
        prefer_metal_backend: bool = False,
        terrain_batch_size: int = 128,
    ) -> None:
        self.seed = int(seed)
        self.terrain_batch_size = max(1, int(terrain_batch_size))
        self._backend = CpuTerrainBackend(
            self.seed,
            self.height,
            self.chunk_size,
            chunks_per_poll=self.terrain_batch_size,
        )
        if prefer_gpu_terrain:
            try:
                self._backend = _create_preferred_gpu_backend(
                    gpu_device,
                    self.seed,
                    self.chunk_size,
                    self.height,
                    self.terrain_batch_size,
                    prefer_metal_backend=prefer_metal_backend,
                )
            except Exception as exc:
                print(
                    f"Warning: GPU terrain backend could not be created ({exc!s}); using CPU terrain.",
                    file=sys.stderr,
                )
                self._backend = CpuTerrainBackend(
                    self.seed,
                    self.height,
                    self.chunk_size,
                    chunks_per_poll=self.terrain_batch_size,
                )

    def block_at(self, x: int, y: int, z: int) -> int:
        if not (0 <= y < self.height):
            return AIR
        height, material = self.surface_profile_at(x, z)
        if y >= height:
            return AIR
        return self._layer_material(y, height, material)

    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        return self._backend.surface_profile_at(x, z)

    def surface_height_at(self, x: int, z: int) -> int:
        height, _ = self.surface_profile_at(x, z)
        return height

    def surface_material_at(self, x: int, z: int) -> int:
        _, material = self.surface_profile_at(x, z)
        return material

    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        return self._backend.chunk_surface_grids(chunk_x, chunk_z)

    def chunk_voxel_grid(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        return self._backend.chunk_voxel_grid(chunk_x, chunk_z)

    def request_chunk_surface_batch(self, chunks: list[tuple[int, int]]) -> int:
        return self._backend.request_chunk_surface_batch(chunks)

    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int]]) -> int:
        return self._backend.request_chunk_voxel_batch(chunks)

    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        return self._backend.poll_ready_chunk_surface_batches()

    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        method = getattr(self._backend, "poll_ready_chunk_surface_gpu_batches", None)
        if method is None:
            return []
        return method()

    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return self._backend.poll_ready_chunk_voxel_batches()

    def has_pending_chunk_surface_batches(self) -> bool:
        return self._backend.has_pending_chunk_surface_batches()

    def has_pending_chunk_voxel_batches(self) -> bool:
        return self._backend.has_pending_chunk_voxel_batches()

    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        return self._backend.flush_chunk_surface_batches()

    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return self._backend.flush_chunk_voxel_batches()

    def terrain_backend_label(self) -> str:
        return self._backend.terrain_backend_label()

    def validate_chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> TerrainValidationReport:
        cpu_backend = CpuTerrainBackend(self.seed, self.height, self.chunk_size)
        cpu_heights, cpu_materials = cpu_backend.chunk_surface_grids(chunk_x, chunk_z)
        active_heights, active_materials = self.chunk_surface_grids(chunk_x, chunk_z)

        height_diff = np.nonzero(cpu_heights != active_heights)[0]
        material_diff = np.nonzero(cpu_materials != active_materials)[0]

        first_height_mismatch: tuple[int, int, int, int] | None = None
        if height_diff.size:
            index = int(height_diff[0])
            first_height_mismatch = (
                index,
                int(cpu_heights[index]),
                int(active_heights[index]),
                int(active_materials[index]),
            )

        first_material_mismatch: tuple[int, int, int, int] | None = None
        if material_diff.size:
            index = int(material_diff[0])
            first_material_mismatch = (
                index,
                int(cpu_materials[index]),
                int(active_materials[index]),
                int(active_heights[index]),
            )

        return TerrainValidationReport(
            chunk_x=int(chunk_x),
            chunk_z=int(chunk_z),
            backend_label=self.terrain_backend_label(),
            total_cells=int(cpu_heights.size),
            height_mismatches=int(height_diff.size),
            material_mismatches=int(material_diff.size),
            first_height_mismatch=first_height_mismatch,
            first_material_mismatch=first_material_mismatch,
        )

    def _layer_material(self, y: int, surface_height: int, surface_material: int) -> int:
        if y == 0:
            return BEDROCK
        if y < surface_height - 4:
            return STONE
        if y < surface_height - 1:
            return DIRT
        return surface_material
