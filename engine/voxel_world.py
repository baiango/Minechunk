from __future__ import annotations

import sys
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
    terrain_block_material_at,
)
from .world_constants import BLOCK_SIZE, CHUNK_SIZE, CHUNK_WORLD_SIZE, WORLD_HEIGHT_BLOCKS

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


# Legacy inline wgpu terrain backend removed.
# Use engine.wgpu_terrain_backend.WgpuTerrainBackend or engine.metal_terrain_backend.MetalTerrainBackend.


class VoxelWorld:
    height: int = WORLD_HEIGHT_BLOCKS
    chunk_size: int = CHUNK_SIZE
    block_size: float = BLOCK_SIZE

    def __init__(
        self,
        seed: int = 1337,
        *,
        gpu_device=None,
        prefer_gpu_terrain: bool = False,
        prefer_metal_backend: bool = False,
        terrain_batch_size: int = 1 if WORLD_HEIGHT_BLOCKS > 256 else 16,
    ) -> None:
        self.seed = int(seed)
        self.chunk_size = int(CHUNK_SIZE)
        self.height = int(WORLD_HEIGHT_BLOCKS)
        self.block_size = float(BLOCK_SIZE)
        self.terrain_batch_size = max(1, int(terrain_batch_size))
        self._backend = CpuTerrainBackend(
            self.seed,
            self.height,
            self.chunk_size,
            chunks_per_poll=self.terrain_batch_size,
        )
        if prefer_gpu_terrain and not VERTICAL_CHUNK_STACK_ENABLED:
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
        return int(terrain_block_material_at(int(x), int(y), int(z), self.seed, self.height))

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

    def chunk_voxel_grid(self, chunk_x: int, chunk_y: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        return self._backend.chunk_voxel_grid(chunk_x, chunk_y, chunk_z)

    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        return self._backend.request_chunk_surface_batch(chunks)

    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
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

    def destroy(self) -> None:
        destroy = getattr(self._backend, "destroy", None)
        if destroy is None:
            return
        destroy()

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
