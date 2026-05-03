#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.render_constants import (
    BLOCK_SIZE,
    CAMERA_HEADROOM_METERS,
    CAMERA_MIN_HEIGHT_METERS,
    CHUNK_WORLD_SIZE,
    PLAYER_EYE_OFFSET_METERS,
)
from engine.terrain.backends.cpu_terrain_backend import CpuTerrainBackend
from engine.world_constants import CHUNK_SIZE, VERTICAL_CHUNK_COUNT, WORLD_HEIGHT_BLOCKS


FORMAT_VERSION = "minechunk-compression-corpus-v1"
DEFAULT_COUNT = 4096
DEFAULT_DIMS = (16, 16, 16)
DEFAULT_COUNT_VARIANTS = (256, 512, 1024, 2048, 4096)
DEFAULT_PAGE_SIZES = (32, 16)


@dataclass(frozen=True)
class ChunkStats:
    non_air_voxels: int
    min_block_id: int
    max_block_id: int
    unique_block_ids: tuple[int, ...]

    @property
    def occupancy(self) -> float:
        return self.non_air_voxels / float(CHUNK_SIZE**3)

    @property
    def stratum(self) -> str:
        if self.non_air_voxels <= 0:
            return "empty_air"
        if self.occupancy < 0.05:
            return "mostly_air"
        if self.occupancy > 0.95:
            return "dense_or_cave"
        return "surface_or_transition"


def parse_dims(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected dimensions like 16x16x16")
    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dimensions must be integers") from exc
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return dims  # type: ignore[return-value]


def parse_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_origin(value: str) -> tuple[int, int, int]:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected origin like 0,22,0")
    try:
        return tuple(int(part.strip()) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("origin values must be integers") from exc


def fixed_box_coords(origin: tuple[int, int, int], dims: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    dim_x, dim_y, dim_z = dims
    neg_x = dim_x // 2
    pos_x = dim_x - neg_x - 1
    neg_y = dim_y // 2
    pos_y = dim_y - neg_y - 1
    neg_z = dim_z // 2
    pos_z = dim_z - neg_z - 1

    rel_y_order = [0]
    for offset in range(1, max(neg_y, pos_y) + 1):
        if offset <= pos_y:
            rel_y_order.append(offset)
        if -offset >= -neg_y:
            rel_y_order.append(-offset)

    rel_xz = [(dx, dz) for dz in range(-neg_z, pos_z + 1) for dx in range(-neg_x, pos_x + 1)]
    rel_xz.sort(key=lambda delta: (delta[0] * delta[0] + delta[1] * delta[1], abs(delta[1]), abs(delta[0]), delta[1], delta[0]))

    origin_x, origin_y, origin_z = origin
    return [(origin_x + dx, origin_y + dy, origin_z + dz) for dy in rel_y_order for dx, dz in rel_xz]


def auto_origin(backend: CpuTerrainBackend, dims: tuple[int, int, int]) -> tuple[int, int, int]:
    surface_height_blocks, _material = backend.surface_profile_at(0, 0)
    spawn_y = max(
        CAMERA_MIN_HEIGHT_METERS,
        min(
            float(WORLD_HEIGHT_BLOCKS) * BLOCK_SIZE + CAMERA_HEADROOM_METERS,
            float(surface_height_blocks) * BLOCK_SIZE + PLAYER_EYE_OFFSET_METERS,
        ),
    )
    camera_origin_y = int(spawn_y // CHUNK_WORLD_SIZE)
    neg_y = dims[1] // 2
    pos_y = dims[1] - neg_y - 1
    min_origin_y = neg_y
    max_origin_y = max(min_origin_y, int(VERTICAL_CHUNK_COUNT - 1 - pos_y))
    return (0, min(max(camera_origin_y, min_origin_y), max_origin_y), 0)


def payload_from_chunk(backend: CpuTerrainBackend, coord: tuple[int, int, int]) -> np.ndarray:
    blocks, materials = backend.chunk_voxel_grid(*coord)
    interior_blocks = blocks[:, 1:-1, 1:-1]
    interior_materials = materials[:, 1:-1, 1:-1]
    payload = np.where(interior_blocks != 0, interior_materials, 0)
    max_id = int(payload.max(initial=0))
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"chunk {coord} contains block id {max_id}, which cannot fit in u16")
    return np.ascontiguousarray(payload, dtype="<u2")


def chunk_stats(payload: np.ndarray) -> ChunkStats:
    unique = tuple(int(value) for value in np.unique(payload))
    non_air = int(np.count_nonzero(payload))
    return ChunkStats(
        non_air_voxels=non_air,
        min_block_id=int(min(unique)) if unique else 0,
        max_block_id=int(max(unique)) if unique else 0,
        unique_block_ids=unique,
    )


def write_page_payloads(payload: np.ndarray, page_size: int, file_handle, digest: "hashlib._Hash") -> None:
    for page_y in range(0, CHUNK_SIZE, page_size):
        for page_z in range(0, CHUNK_SIZE, page_size):
            for page_x in range(0, CHUNK_SIZE, page_size):
                page = np.ascontiguousarray(
                    payload[
                        page_y : page_y + page_size,
                        page_z : page_z + page_size,
                        page_x : page_x + page_size,
                    ],
                    dtype="<u2",
                )
                data = page.tobytes(order="C")
                file_handle.write(data)
                digest.update(data)


def git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def relative_file(path: Path, output_dir: Path) -> str:
    return str(path.relative_to(output_dir))


def build_count_variants(requested: Iterable[int], count: int) -> list[int]:
    variants = sorted({int(value) for value in requested if int(value) <= count})
    if count not in variants:
        variants.append(count)
    return variants


def build_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    coords: list[tuple[int, int, int]],
    count_variants: list[int],
    raw_files: dict[str, dict],
    stats_summary: dict[str, int | float | dict[str, int]],
    elapsed_s: float,
) -> dict:
    count = len(coords)
    chunk_bytes = CHUNK_SIZE**3 * 2
    page_variants = []
    for page_size in args.page_sizes:
        pages_per_axis = CHUNK_SIZE // page_size
        pages_per_chunk = pages_per_axis**3
        page_bytes = page_size**3 * 2
        page_file_key = f"pages{page_size}"
        entry = {
            "name": f"pages{page_size}_yzx_u16le",
            "page_size": page_size,
            "shape_yzx": [page_size, page_size, page_size],
            "bytes_per_page": page_bytes,
            "pages_per_axis": pages_per_axis,
            "pages_per_chunk": pages_per_chunk,
            "page_count": count * pages_per_chunk,
            "source": "derived_from_chunks64_yzx_u16le.raw",
            "local_page_order": ["page_y", "page_z", "page_x"],
            "page_index_formula": "chunk_index * pages_per_chunk + page_y_index * pages_per_axis^2 + page_z_index * pages_per_axis + page_x_index",
        }
        if page_file_key in raw_files:
            entry["raw_file"] = raw_files[page_file_key]["path"]
            entry["sha256"] = raw_files[page_file_key]["sha256"]
            entry["total_bytes"] = raw_files[page_file_key]["bytes"]
        page_variants.append(entry)

    return {
        "format": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": "tools/export_compression_corpus.py",
            "git_commit": git_value(["git", "rev-parse", "HEAD"]),
            "git_status_short": git_value(["git", "status", "--short"]),
            "elapsed_seconds": elapsed_s,
        },
        "world": {
            "seed": args.seed,
            "chunk_size": CHUNK_SIZE,
            "world_height_blocks": WORLD_HEIGHT_BLOCKS,
            "vertical_chunk_count": VERTICAL_CHUNK_COUNT,
            "origin_chunk": list(args.origin),
            "fixed_box_dimensions_chunks": list(args.dims),
            "coordinate_order": "renderer-compatible center-first fixed box prefix",
        },
        "payload": {
            "logical_payload": "u16 block_id material IDs with air encoded as 0",
            "layout": "yzx",
            "endianness": "little",
            "shape_yzx": [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE],
            "element_type": "uint16",
            "bytes_per_chunk": chunk_bytes,
            "chunk_count": count,
            "raw_file": raw_files["chunks64"]["path"],
            "total_bytes": raw_files["chunks64"]["bytes"],
            "sha256": raw_files["chunks64"]["sha256"],
            "excluded": ["x/z ghost borders", "top/bottom neighbor planes", "mesh buffers", "render metadata"],
        },
        "sidecars": {
            "chunk_coordinates_csv": relative_file(output_dir / "chunks64_coords.csv", output_dir),
        },
        "variants": {
            "chunk_count_prefixes": [
                {
                    "name": f"chunks64_count{variant_count}",
                    "chunk_count": variant_count,
                    "raw_file": raw_files["chunks64"]["path"],
                    "offset_bytes": 0,
                    "length_bytes": variant_count * chunk_bytes,
                }
                for variant_count in count_variants
            ],
            "pages": page_variants,
        },
        "stats_summary": stats_summary,
    }


def export_corpus(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output directory is not empty: {output_dir} (use --force to overwrite generated files)")
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = CpuTerrainBackend(seed=args.seed, height=WORLD_HEIGHT_BLOCKS, chunk_size=CHUNK_SIZE)
    if args.origin is None:
        args.origin = auto_origin(backend, args.dims)

    coords = fixed_box_coords(args.origin, args.dims)
    if len(coords) < args.count:
        raise SystemExit(f"dims {args.dims!r} only produce {len(coords)} chunks; need at least {args.count}")
    coords = coords[: args.count]

    for page_size in args.page_sizes:
        if CHUNK_SIZE % page_size != 0:
            raise SystemExit(f"page size {page_size} does not divide chunk size {CHUNK_SIZE}")

    count_variants = build_count_variants(args.count_variants, args.count)
    chunk_path = output_dir / "chunks64_yzx_u16le.raw"
    coord_path = output_dir / "chunks64_coords.csv"
    manifest_path = output_dir / "manifest.json"

    page_paths = {
        page_size: output_dir / f"pages{page_size}_yzx_u16le.raw"
        for page_size in args.page_sizes
        if args.materialize_pages
    }

    tmp_paths = [chunk_path.with_suffix(chunk_path.suffix + ".tmp")]
    tmp_paths.extend(path.with_suffix(path.suffix + ".tmp") for path in page_paths.values())
    if args.force:
        for path in [chunk_path, coord_path, manifest_path, *page_paths.values(), *tmp_paths]:
            if path.exists():
                path.unlink()

    chunk_digest = hashlib.sha256()
    page_digests = {page_size: hashlib.sha256() for page_size in page_paths}
    stats_by_stratum: dict[str, int] = {}
    total_non_air = 0
    min_block_id = None
    max_block_id = 0
    start = time.perf_counter()

    page_handles = {}
    try:
        with chunk_path.with_suffix(chunk_path.suffix + ".tmp").open("wb", buffering=8 * 1024 * 1024) as chunk_file, coord_path.open("w", newline="") as csv_file:
            for page_size, path in page_paths.items():
                page_handles[page_size] = path.with_suffix(path.suffix + ".tmp").open("wb", buffering=8 * 1024 * 1024)

            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "index",
                    "chunk_x",
                    "chunk_y",
                    "chunk_z",
                    "byte_offset",
                    "byte_length",
                    "non_air_voxels",
                    "occupancy",
                    "min_block_id",
                    "max_block_id",
                    "unique_block_ids",
                    "stratum",
                ]
            )

            chunk_bytes = CHUNK_SIZE**3 * 2
            for index, coord in enumerate(coords):
                payload = payload_from_chunk(backend, coord)
                data = payload.tobytes(order="C")
                chunk_file.write(data)
                chunk_digest.update(data)

                for page_size, handle in page_handles.items():
                    write_page_payloads(payload, page_size, handle, page_digests[page_size])

                stats = chunk_stats(payload)
                stats_by_stratum[stats.stratum] = stats_by_stratum.get(stats.stratum, 0) + 1
                total_non_air += stats.non_air_voxels
                min_block_id = stats.min_block_id if min_block_id is None else min(min_block_id, stats.min_block_id)
                max_block_id = max(max_block_id, stats.max_block_id)
                writer.writerow(
                    [
                        index,
                        coord[0],
                        coord[1],
                        coord[2],
                        index * chunk_bytes,
                        chunk_bytes,
                        stats.non_air_voxels,
                        f"{stats.occupancy:.8f}",
                        stats.min_block_id,
                        stats.max_block_id,
                        " ".join(str(value) for value in stats.unique_block_ids),
                        stats.stratum,
                    ]
                )

                completed = index + 1
                if args.progress_interval > 0 and (completed == 1 or completed == args.count or completed % args.progress_interval == 0):
                    elapsed = max(1e-9, time.perf_counter() - start)
                    rate = completed / elapsed
                    print(f"exported {completed}/{args.count} chunks ({rate:.1f} chunks/s)", file=sys.stderr)
    finally:
        for handle in page_handles.values():
            handle.close()

    chunk_tmp = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    chunk_tmp.replace(chunk_path)
    for page_size, path in page_paths.items():
        path.with_suffix(path.suffix + ".tmp").replace(path)

    elapsed_s = time.perf_counter() - start
    raw_files = {
        "chunks64": {
            "path": relative_file(chunk_path, output_dir),
            "bytes": chunk_path.stat().st_size,
            "sha256": chunk_digest.hexdigest(),
        }
    }
    for page_size, path in page_paths.items():
        raw_files[f"pages{page_size}"] = {
            "path": relative_file(path, output_dir),
            "bytes": path.stat().st_size,
            "sha256": page_digests[page_size].hexdigest(),
        }

    stats_summary = {
        "chunk_count": args.count,
        "non_air_voxels": total_non_air,
        "total_voxels": args.count * CHUNK_SIZE**3,
        "overall_occupancy": total_non_air / float(args.count * CHUNK_SIZE**3),
        "min_block_id": 0 if min_block_id is None else int(min_block_id),
        "max_block_id": int(max_block_id),
        "strata": dict(sorted(stats_by_stratum.items())),
    }
    manifest = build_manifest(
        args=args,
        output_dir=output_dir,
        coords=coords,
        count_variants=count_variants,
        raw_files=raw_files,
        stats_summary=stats_summary,
        elapsed_s=elapsed_s,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {manifest_path}", file=sys.stderr)
    print(f"wrote {chunk_path} ({chunk_path.stat().st_size:,} bytes)", file=sys.stderr)
    for path in page_paths.values():
        print(f"wrote {path} ({path.stat().st_size:,} bytes)", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Minechunk u16 terrain corpus for isolated compression benchmarks.")
    parser.add_argument("--output-dir", type=Path, default=Path("compression_corpus/minechunk_seed1337_count4096"), help="Directory for raw payloads and manifest.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of 64^3 chunks to export.")
    parser.add_argument("--count-variants", type=parse_int_list, default=DEFAULT_COUNT_VARIANTS, help="Comma-separated lower count variants stored as prefixes in the manifest.")
    parser.add_argument("--dims", type=parse_dims, default=DEFAULT_DIMS, help="Fixed view box dimensions, for example 16x16x16.")
    parser.add_argument("--origin", type=parse_origin, default=None, help="Origin chunk x,y,z. Defaults to the engine's spawn-adjacent fixed view origin.")
    parser.add_argument("--seed", type=int, default=1337, help="Terrain seed.")
    parser.add_argument("--page-sizes", type=parse_int_list, default=DEFAULT_PAGE_SIZES, help="Comma-separated lower page sizes to describe or materialize.")
    parser.add_argument("--materialize-pages", action="store_true", help="Also write raw page files for each page size.")
    parser.add_argument("--progress-interval", type=int, default=128, help="Progress print interval in chunks; use 0 to disable.")
    parser.add_argument("--force", action="store_true", help="Overwrite generated files in the output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    try:
        export_corpus(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
