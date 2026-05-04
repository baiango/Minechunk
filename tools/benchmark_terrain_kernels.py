from __future__ import annotations

import argparse
import os
import sys
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.terrain.kernels import zig_kernel
from engine.world_constants import CHUNK_SIZE, WORLD_HEIGHT_BLOCKS


def _chunk_coords(count: int) -> np.ndarray:
    side = max(1, int(np.ceil(np.sqrt(count))))
    coords = np.empty((count, 3), dtype=np.int32)
    for index in range(count):
        x = index % side - side // 2
        z = index // side - side // 2
        y = (index * 7) % 8
        coords[index] = (x, y, z)
    return coords


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def _time_call(warmups: int, repeats: int, callback) -> float:
    for _ in range(warmups):
        callback()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
    return _median_ms(samples)


def _set_kernel(mode: str) -> None:
    os.environ["MINECHUNK_TERRAIN_KERNEL"] = mode


def _benchmark_mode(
    mode: str,
    coords: np.ndarray,
    chunk_size: int,
    seed: int,
    world_height: int,
    warmups: int,
    repeats: int,
) -> dict[str, float]:
    _set_kernel(mode)
    sample_size = chunk_size + 2
    cell_count = sample_size * sample_size
    chunk_count = int(coords.shape[0])
    local_height = chunk_size
    heights = np.empty((chunk_count, cell_count), dtype=np.uint32)
    materials = np.empty_like(heights)
    chunk_xs = np.ascontiguousarray(coords[:, 0])
    chunk_zs = np.ascontiguousarray(coords[:, 2])

    def fill_surfaces() -> None:
        zig_kernel.fill_chunk_surface_grids_batch(
            heights,
            materials,
            chunk_xs,
            chunk_zs,
            chunk_size,
            seed,
            world_height,
        )

    surface_ms = _time_call(warmups, repeats, fill_surfaces)
    fill_surfaces()

    blocks = np.zeros((chunk_count, local_height, sample_size, sample_size), dtype=np.uint8)
    voxel_materials = np.zeros((chunk_count, local_height, sample_size, sample_size), dtype=np.uint32)
    top = np.zeros((chunk_count, sample_size, sample_size), dtype=np.uint8)
    bottom = np.zeros_like(top)

    def fill_voxels(carve_caves: bool):
        def _callback() -> None:
            blocks.fill(0)
            voxel_materials.fill(0)
            top.fill(0)
            bottom.fill(0)
            zig_kernel.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch(
                blocks,
                voxel_materials,
                top,
                bottom,
                heights,
                materials,
                coords,
                chunk_size,
                seed,
                world_height,
                carve_caves,
            )

        return _callback

    return {
        "surface_batch_ms": surface_ms,
        "voxel_batch_caves_off_ms": _time_call(warmups, repeats, fill_voxels(False)),
        "voxel_batch_caves_on_ms": _time_call(warmups, repeats, fill_voxels(True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Minechunk Numba and Zig terrain kernels.")
    parser.add_argument("--chunks", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--world-height", type=int, default=WORLD_HEIGHT_BLOCKS)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    coords = _chunk_coords(max(1, args.chunks))
    print(f"chunks={len(coords)} chunk_size={args.chunk_size} world_height={args.world_height}")
    for mode in ("numba", "zig"):
        results = _benchmark_mode(
            mode,
            coords,
            int(args.chunk_size),
            int(args.seed),
            int(args.world_height),
            int(args.warmups),
            int(args.repeats),
        )
        label = zig_kernel.terrain_kernel_label()
        print(
            f"{mode} ({label}): "
            f"surface_batch={results['surface_batch_ms']:.3f} ms, "
            f"voxel_caves_off={results['voxel_batch_caves_off_ms']:.3f} ms, "
            f"voxel_caves_on={results['voxel_batch_caves_on_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
