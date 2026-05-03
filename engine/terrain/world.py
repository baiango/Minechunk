from __future__ import annotations

import os
import sys
from collections import OrderedDict, deque

import numpy as np

from .backends.cpu_terrain_backend import CpuTerrainBackend
try:
    from .backends.metal_terrain_backend import MetalTerrainBackend
except Exception as exc:  # pragma: no cover - optional on non-mac / CPU-only fallback
    MetalTerrainBackend = None  # type: ignore[assignment]
    METAL_TERRAIN_IMPORT_ERROR = exc
else:
    METAL_TERRAIN_IMPORT_ERROR = None

try:
    from .backends.wgpu_terrain_backend import WgpuTerrainBackend
except Exception as exc:  # pragma: no cover - optional during Metal-only deployments
    WgpuTerrainBackend = None  # type: ignore[assignment]
    WGPU_TERRAIN_IMPORT_ERROR = exc
else:
    WGPU_TERRAIN_IMPORT_ERROR = None

from .types import ChunkSurfaceGpuBatch, ChunkSurfaceResult, ChunkVoxelResult, TerrainValidationReport
from .compression import (
    CompressedChunkVoxelResult,
    compress_chunk_voxel_result,
    decompress_chunk_voxel_result,
)

from .kernels import (
    AIR,
    BEDROCK,
    DIRT,
    STONE,
    terrain_block_material_at,
)
from ..render_constants import MAX_CACHED_CHUNKS, TERRAIN_ZSTD_ENABLED
from ..world_constants import BLOCK_SIZE, CHUNK_SIZE, CHUNK_WORLD_SIZE, WORLD_HEIGHT_BLOCKS, VERTICAL_CHUNK_STACK_ENABLED
from ..visibility.amanatides_woo import VoxelRayHit, first_hit as amanatides_woo_first_hit, line_of_sight as amanatides_woo_line_of_sight


def _allow_gpu_backend_fallback() -> bool:
    value = os.environ.get("MINECHUNK_ALLOW_METAL_FALLBACK", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _format_backend_error(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown error"
    return f"{type(exc).__name__}: {exc!s}"

def _create_preferred_gpu_backend(
    gpu_device,
    seed: int,
    chunk_size: int,
    height_limit: int,
    chunks_per_poll: int,
    *,
    prefer_metal_backend: bool = False,
    terrain_caves_enabled: bool = True,
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
                        terrain_caves_enabled=terrain_caves_enabled,
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
                terrain_caves_enabled=terrain_caves_enabled,
            )
        except Exception as exc:
            errors.append(f"wgpu terrain backend could not be created ({_format_backend_error(exc)})")
    elif WgpuTerrainBackend is None and WGPU_TERRAIN_IMPORT_ERROR is not None:
        errors.append(f"wgpu terrain backend import failed ({_format_backend_error(WGPU_TERRAIN_IMPORT_ERROR)})")

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
        terrain_zstd_enabled: bool = TERRAIN_ZSTD_ENABLED,
        terrain_zstd_cache_limit: int | None = None,
        terrain_caves_enabled: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.chunk_size = int(CHUNK_SIZE)
        self.height = int(WORLD_HEIGHT_BLOCKS)
        self.block_size = float(BLOCK_SIZE)
        self.terrain_batch_size = max(1, int(terrain_batch_size))
        self.terrain_zstd_enabled = bool(terrain_zstd_enabled)
        self.terrain_caves_enabled = bool(terrain_caves_enabled)
        self._terrain_zstd_cache_limit = int(MAX_CACHED_CHUNKS if terrain_zstd_cache_limit is None else max(0, int(terrain_zstd_cache_limit)))
        self._terrain_zstd_cache: OrderedDict[tuple[int, int, int], CompressedChunkVoxelResult] = OrderedDict()
        self._terrain_zstd_cache_raw_bytes = 0
        self._terrain_zstd_cache_compressed_bytes = 0
        self._ready_cached_voxel_results: deque[CompressedChunkVoxelResult] = deque()
        self._collision_block_chunk_cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        try:
            self._collision_block_chunk_cache_limit = max(0, int(os.environ.get("MINECHUNK_COLLISION_CHUNK_CACHE", "256")))
        except Exception:
            self._collision_block_chunk_cache_limit = 256
        self._backend = CpuTerrainBackend(
            self.seed,
            self.height,
            self.chunk_size,
            chunks_per_poll=self.terrain_batch_size,
            terrain_caves_enabled=self.terrain_caves_enabled,
        )
        self._gpu_backend_error: BaseException | None = None
        if prefer_gpu_terrain:
            try:
                self._backend = _create_preferred_gpu_backend(
                    gpu_device,
                    self.seed,
                    self.chunk_size,
                    self.height,
                    self.terrain_batch_size,
                    prefer_metal_backend=prefer_metal_backend,
                    terrain_caves_enabled=self.terrain_caves_enabled,
                )
            except Exception as exc:
                self._gpu_backend_error = exc
                message = f"GPU terrain backend could not be created ({_format_backend_error(exc)})"
                if prefer_metal_backend and not _allow_gpu_backend_fallback():
                    raise RuntimeError(
                        "ENGINE_MODE_METAL was requested, but the Metal terrain backend did not initialize. "
                        + message
                        + "; refusing to silently run the CPU backend. "
                        + "Install/verify pyobjc-framework-Metal or run with --allow-metal-fallback to permit fallback."
                    ) from exc
                print(f"Warning: {message}; using CPU terrain.", file=sys.stderr)
                self._backend = CpuTerrainBackend(
                    self.seed,
                    self.height,
                    self.chunk_size,
                    chunks_per_poll=self.terrain_batch_size,
                    terrain_caves_enabled=self.terrain_caves_enabled,
                )

    def set_terrain_zstd_cache_limit(self, limit: int | None) -> None:
        self._terrain_zstd_cache_limit = int(MAX_CACHED_CHUNKS if limit is None else max(0, int(limit)))
        self._trim_terrain_zstd_cache()

    def clear_terrain_zstd_cache(self) -> None:
        self._terrain_zstd_cache.clear()
        self._terrain_zstd_cache_raw_bytes = 0
        self._terrain_zstd_cache_compressed_bytes = 0
        self._ready_cached_voxel_results.clear()

    def drop_terrain_zstd_cache_entries(self, keys) -> None:
        drop_keys = {self._chunk_key(chunk_x, chunk_y, chunk_z) for chunk_x, chunk_y, chunk_z in keys}
        if not drop_keys:
            return
        for key in drop_keys:
            self._remove_terrain_zstd_cache_entry(key)
        if self._ready_cached_voxel_results:
            self._ready_cached_voxel_results = deque(
                result
                for result in self._ready_cached_voxel_results
                if self._chunk_key(result.chunk_x, result.chunk_y, result.chunk_z) not in drop_keys
            )

    def terrain_zstd_cache_stats(self) -> dict[str, int | bool]:
        return {
            "enabled": bool(self.terrain_zstd_enabled),
            "entries": len(self._terrain_zstd_cache),
            "raw_bytes": int(self._terrain_zstd_cache_raw_bytes),
            "compressed_bytes": int(self._terrain_zstd_cache_compressed_bytes),
            "limit": int(self._terrain_zstd_cache_limit),
        }

    @staticmethod
    def _chunk_key(chunk_x: int, chunk_y: int, chunk_z: int) -> tuple[int, int, int]:
        return int(chunk_x), int(chunk_y), int(chunk_z)

    def _remove_terrain_zstd_cache_entry(self, key: tuple[int, int, int]) -> None:
        cached = self._terrain_zstd_cache.pop(key, None)
        if cached is None:
            return
        self._terrain_zstd_cache_raw_bytes -= int(cached.raw_nbytes)
        self._terrain_zstd_cache_compressed_bytes -= int(cached.compressed_nbytes)

    def _trim_terrain_zstd_cache(self) -> None:
        cache_limit = max(0, int(self._terrain_zstd_cache_limit))
        while len(self._terrain_zstd_cache) > cache_limit:
            _key, cached = self._terrain_zstd_cache.popitem(last=False)
            self._terrain_zstd_cache_raw_bytes -= int(cached.raw_nbytes)
            self._terrain_zstd_cache_compressed_bytes -= int(cached.compressed_nbytes)

    def _store_terrain_zstd_result(self, result: ChunkVoxelResult | CompressedChunkVoxelResult) -> CompressedChunkVoxelResult | None:
        if not self.terrain_zstd_enabled or self._terrain_zstd_cache_limit <= 0:
            return None
        compressed = compress_chunk_voxel_result(result)
        key = self._chunk_key(compressed.chunk_x, compressed.chunk_y, compressed.chunk_z)
        self._remove_terrain_zstd_cache_entry(key)
        self._terrain_zstd_cache[key] = compressed
        self._terrain_zstd_cache_raw_bytes += int(compressed.raw_nbytes)
        self._terrain_zstd_cache_compressed_bytes += int(compressed.compressed_nbytes)
        self._terrain_zstd_cache.move_to_end(key)
        self._trim_terrain_zstd_cache()
        return compressed

    def _terrain_zstd_cache_get_compressed(
        self,
        chunk_x: int,
        chunk_y: int,
        chunk_z: int,
        *,
        require_meshing_boundaries: bool = False,
    ) -> CompressedChunkVoxelResult | None:
        if not self.terrain_zstd_enabled:
            return None
        key = self._chunk_key(chunk_x, chunk_y, chunk_z)
        cached = self._terrain_zstd_cache.get(key)
        if cached is None:
            return None
        if (
            require_meshing_boundaries
            and not bool(cached.is_empty)
            and not bool(getattr(cached, "is_fully_occluded", False))
            and not bool(getattr(cached, "use_surface_mesher", False))
            and (cached.top_boundary is None or cached.bottom_boundary is None)
        ):
            return None
        self._terrain_zstd_cache.move_to_end(key)
        return cached

    def _terrain_zstd_cache_get(self, chunk_x: int, chunk_y: int, chunk_z: int) -> ChunkVoxelResult | None:
        cached = self._terrain_zstd_cache_get_compressed(chunk_x, chunk_y, chunk_z)
        if cached is None:
            return None
        return decompress_chunk_voxel_result(cached)

    def _block_to_stacked_chunk_local(self, x: int, y: int, z: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        chunk_x = int(x) // self.chunk_size
        chunk_y = int(y) // self.chunk_size if VERTICAL_CHUNK_STACK_ENABLED else 0
        chunk_z = int(z) // self.chunk_size
        local_x = int(x) - chunk_x * self.chunk_size
        local_y = int(y) - chunk_y * self.chunk_size if VERTICAL_CHUNK_STACK_ENABLED else int(y)
        local_z = int(z) - chunk_z * self.chunk_size
        return (chunk_x, chunk_y, chunk_z), (local_x, local_y, local_z)

    def clear_collision_block_cache(self) -> None:
        self._collision_block_chunk_cache.clear()

    def _collision_blocks_for_chunk(self, chunk_x: int, chunk_y: int, chunk_z: int) -> np.ndarray:
        key = (int(chunk_x), int(chunk_y), int(chunk_z))
        cached = self._collision_block_chunk_cache.get(key)
        if cached is not None:
            self._collision_block_chunk_cache.move_to_end(key)
            return cached

        blocks, _materials = self.chunk_voxel_grid(key[0], key[1], key[2])
        blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
        cache_limit = int(getattr(self, "_collision_block_chunk_cache_limit", 0))
        if cache_limit > 0:
            self._collision_block_chunk_cache[key] = blocks
            self._collision_block_chunk_cache.move_to_end(key)
            while len(self._collision_block_chunk_cache) > cache_limit:
                self._collision_block_chunk_cache.popitem(last=False)
        return blocks

    def block_at(self, x: int, y: int, z: int) -> int:
        x = int(x)
        y = int(y)
        z = int(z)
        if y < 0 or y >= self.height:
            return int(AIR)

        # In GPU terrain modes, rendering is built from backend-generated chunk
        # voxel payloads, not from the CPU scalar sampler below.  Sampling the
        # active backend's chunk grid keeps collision aligned with the visible
        # mesh even when WGPU/Metal use their own shader math and precision.
        if self.terrain_backend_label() != "CPU":
            chunk_key, local = self._block_to_stacked_chunk_local(x, y, z)
            local_x, local_y, local_z = local
            blocks = self._collision_blocks_for_chunk(*chunk_key)
            sample_x = local_x + 1
            sample_z = local_z + 1
            if (
                local_y < 0
                or local_y >= int(blocks.shape[0])
                or sample_z < 0
                or sample_z >= int(blocks.shape[1])
                or sample_x < 0
                or sample_x >= int(blocks.shape[2])
            ):
                return int(AIR)
            return int(STONE if int(blocks[local_y, sample_z, sample_x]) != 0 else AIR)

        return int(terrain_block_material_at(x, y, z, self.seed, self.height, self.terrain_caves_enabled))


    def raycast_blocks(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float,
        *,
        start_distance: float = 0.0,
        max_steps: int | None = None,
    ) -> VoxelRayHit | None:
        """Raycast through world blocks using Amanatides-Woo 3D DDA.

        The returned distance is in world metres along the ray.  This samples
        ``block_at`` so CPU/GPU/Metal terrain modes stay aligned with collision.
        """

        return amanatides_woo_first_hit(
            origin,
            direction,
            lambda bx, by, bz: int(self.block_at(bx, by, bz)) != int(AIR),
            max_distance,
            block_size=float(self.block_size),
            start_distance=start_distance,
            max_steps=max_steps,
            material_at=lambda bx, by, bz: int(self.block_at(bx, by, bz)),
        )

    def has_clear_line_of_sight(
        self,
        origin: tuple[float, float, float],
        target: tuple[float, float, float],
        *,
        max_steps: int | None = None,
    ) -> bool:
        """Return True when Amanatides-Woo finds no solid block between points."""

        return amanatides_woo_line_of_sight(
            origin,
            target,
            lambda bx, by, bz: int(self.block_at(bx, by, bz)) != int(AIR),
            block_size=float(self.block_size),
            max_steps=max_steps,
        )

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
        cached = self._terrain_zstd_cache_get(chunk_x, chunk_y, chunk_z)
        if cached is not None:
            return cached.blocks, cached.materials
        blocks, materials = self._backend.chunk_voxel_grid(chunk_x, chunk_y, chunk_z)
        if self.terrain_zstd_enabled and not VERTICAL_CHUNK_STACK_ENABLED:
            self._store_terrain_zstd_result(
                ChunkVoxelResult(
                    chunk_x=int(chunk_x),
                    chunk_y=int(chunk_y),
                    chunk_z=int(chunk_z),
                    blocks=blocks,
                    materials=materials,
                    source="sync",
                )
            )
        return blocks, materials

    def request_chunk_surface_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        return self._backend.request_chunk_surface_batch(chunks)

    def request_chunk_voxel_batch(self, chunks: list[tuple[int, int, int]]) -> int:
        normalized = [(int(chunk_x), int(chunk_y), int(chunk_z)) for chunk_x, chunk_y, chunk_z in chunks]
        if not self.terrain_zstd_enabled:
            return self._backend.request_chunk_voxel_batch(normalized)
        missing: list[tuple[int, int, int]] = []
        for chunk_x, chunk_y, chunk_z in normalized:
            cached = self._terrain_zstd_cache_get_compressed(
                chunk_x,
                chunk_y,
                chunk_z,
                require_meshing_boundaries=bool(VERTICAL_CHUNK_STACK_ENABLED),
            )
            if cached is None:
                missing.append((chunk_x, chunk_y, chunk_z))
            else:
                self._ready_cached_voxel_results.append(cached)
        if missing:
            return self._backend.request_chunk_voxel_batch(missing)
        return 0

    def poll_ready_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        return self._backend.poll_ready_chunk_surface_batches()

    def poll_ready_chunk_surface_gpu_batches(self) -> list[ChunkSurfaceGpuBatch]:
        method = getattr(self._backend, "poll_ready_chunk_surface_gpu_batches", None)
        if method is None:
            return []
        return method()

    def poll_ready_chunk_voxel_payloads(self) -> list[ChunkVoxelResult | CompressedChunkVoxelResult]:
        ready: list[ChunkVoxelResult | CompressedChunkVoxelResult] = []
        while self._ready_cached_voxel_results:
            ready.append(self._ready_cached_voxel_results.popleft())
        backend_ready = self._backend.poll_ready_chunk_voxel_batches()
        if self.terrain_zstd_enabled:
            for result in backend_ready:
                compressed = self._store_terrain_zstd_result(result)
                ready.append(compressed if compressed is not None else result)
            return ready
        ready.extend(backend_ready)
        return ready

    def poll_ready_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return [decompress_chunk_voxel_result(result) for result in self.poll_ready_chunk_voxel_payloads()]

    def has_pending_chunk_surface_batches(self) -> bool:
        return self._backend.has_pending_chunk_surface_batches()

    def has_pending_chunk_voxel_batches(self) -> bool:
        return bool(self._ready_cached_voxel_results) or self._backend.has_pending_chunk_voxel_batches()

    def flush_chunk_surface_batches(self) -> list[ChunkSurfaceResult]:
        return self._backend.flush_chunk_surface_batches()

    def flush_chunk_voxel_payloads(self) -> list[ChunkVoxelResult | CompressedChunkVoxelResult]:
        ready: list[ChunkVoxelResult | CompressedChunkVoxelResult] = []
        while self._ready_cached_voxel_results:
            ready.append(self._ready_cached_voxel_results.popleft())
        backend_ready = self._backend.flush_chunk_voxel_batches()
        if self.terrain_zstd_enabled:
            for result in backend_ready:
                compressed = self._store_terrain_zstd_result(result)
                ready.append(compressed if compressed is not None else result)
            return ready
        ready.extend(backend_ready)
        return ready

    def flush_chunk_voxel_batches(self) -> list[ChunkVoxelResult]:
        return [decompress_chunk_voxel_result(result) for result in self.flush_chunk_voxel_payloads()]

    def terrain_backend_label(self) -> str:
        return self._backend.terrain_backend_label()

    def destroy(self) -> None:
        self.clear_terrain_zstd_cache()
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
