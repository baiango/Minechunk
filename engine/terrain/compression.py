from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .types import ChunkVoxelResult

ZSTD_LEVEL = 1

_zstd = None
_compressor = None
_decompressor = None


def _zstd_module():
    global _zstd
    if _zstd is not None:
        return _zstd
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Terrain zstd compression requires the 'zstandard' package. "
            "Install dependencies with `python3 -m pip install -r requirements.txt`."
        ) from exc
    _zstd = zstd
    return zstd


def _zstd_compressor():
    global _compressor
    if _compressor is None:
        _compressor = _zstd_module().ZstdCompressor(level=ZSTD_LEVEL)
    return _compressor


def _zstd_decompressor():
    global _decompressor
    if _decompressor is None:
        _decompressor = _zstd_module().ZstdDecompressor()
    return _decompressor


@dataclass(frozen=True)
class _ArraySegment:
    offset: int
    raw_nbytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CompressedChunkVoxelResult:
    chunk_x: int
    chunk_z: int
    chunk_y: int
    payload: bytes
    blocks: _ArraySegment
    materials: _ArraySegment
    source: str = ""
    top_boundary: _ArraySegment | None = None
    bottom_boundary: _ArraySegment | None = None
    is_empty: bool = False

    @property
    def raw_nbytes(self) -> int:
        total = int(self.blocks.raw_nbytes) + int(self.materials.raw_nbytes)
        if self.top_boundary is not None:
            total += int(self.top_boundary.raw_nbytes)
        if self.bottom_boundary is not None:
            total += int(self.bottom_boundary.raw_nbytes)
        return total

    @property
    def compressed_nbytes(self) -> int:
        return len(self.payload)


def is_compressed_chunk_voxel_result(value) -> bool:
    return isinstance(value, CompressedChunkVoxelResult)


def _write_array(writer, raw_offset: int, array: np.ndarray | None) -> tuple[_ArraySegment | None, int]:
    if array is None:
        return None, int(raw_offset)
    contiguous = np.ascontiguousarray(array)
    writer.write(memoryview(contiguous).cast("B"))
    segment = _ArraySegment(
        offset=int(raw_offset),
        raw_nbytes=int(contiguous.nbytes),
        shape=tuple(int(dim) for dim in contiguous.shape),
        dtype=contiguous.dtype.str,
    )
    return segment, int(raw_offset) + int(contiguous.nbytes)


def _decompress_array(raw_payload: bytes, segment: _ArraySegment, *, copy: bool) -> np.ndarray:
    start = int(segment.offset)
    end = start + int(segment.raw_nbytes)
    array = np.frombuffer(memoryview(raw_payload)[start:end], dtype=np.dtype(segment.dtype)).reshape(segment.shape)
    if copy:
        return array.copy(order="C")
    return array


def compress_chunk_voxel_result(result: ChunkVoxelResult | CompressedChunkVoxelResult) -> CompressedChunkVoxelResult:
    if isinstance(result, CompressedChunkVoxelResult):
        return result
    payload_buffer = io.BytesIO()
    raw_offset = 0
    writer = _zstd_compressor().stream_writer(payload_buffer, closefd=False)
    try:
        blocks_segment, raw_offset = _write_array(writer, raw_offset, result.blocks)
        assert blocks_segment is not None

        materials_segment, raw_offset = _write_array(writer, raw_offset, result.materials)
        assert materials_segment is not None

        top_segment, raw_offset = _write_array(writer, raw_offset, getattr(result, "top_boundary", None))
        bottom_segment, _raw_offset = _write_array(writer, raw_offset, getattr(result, "bottom_boundary", None))
    finally:
        writer.close()

    return CompressedChunkVoxelResult(
        chunk_x=int(result.chunk_x),
        chunk_y=int(getattr(result, "chunk_y", 0)),
        chunk_z=int(result.chunk_z),
        payload=payload_buffer.getvalue(),
        blocks=blocks_segment,
        materials=materials_segment,
        source=str(getattr(result, "source", "") or ""),
        top_boundary=top_segment,
        bottom_boundary=bottom_segment,
        is_empty=bool(getattr(result, "is_empty", False)),
    )


def decompress_chunk_voxel_result(
    result: ChunkVoxelResult | CompressedChunkVoxelResult,
    *,
    copy: bool = True,
) -> ChunkVoxelResult:
    if not isinstance(result, CompressedChunkVoxelResult):
        return result
    raw_payload = _zstd_decompressor().decompress(result.payload, max_output_size=int(result.raw_nbytes))
    return ChunkVoxelResult(
        chunk_x=int(result.chunk_x),
        chunk_y=int(result.chunk_y),
        chunk_z=int(result.chunk_z),
        blocks=_decompress_array(raw_payload, result.blocks, copy=copy),
        materials=_decompress_array(raw_payload, result.materials, copy=copy),
        source=str(result.source),
        top_boundary=None if result.top_boundary is None else _decompress_array(raw_payload, result.top_boundary, copy=copy),
        bottom_boundary=None if result.bottom_boundary is None else _decompress_array(raw_payload, result.bottom_boundary, copy=copy),
        is_empty=bool(result.is_empty),
    )


def chunk_voxel_result_raw_nbytes(result: ChunkVoxelResult | CompressedChunkVoxelResult) -> int:
    if isinstance(result, CompressedChunkVoxelResult):
        return int(result.raw_nbytes)
    total = int(result.blocks.nbytes) + int(result.materials.nbytes)
    top_boundary = getattr(result, "top_boundary", None)
    bottom_boundary = getattr(result, "bottom_boundary", None)
    if top_boundary is not None:
        total += int(top_boundary.nbytes)
    if bottom_boundary is not None:
        total += int(bottom_boundary.nbytes)
    return total


def chunk_voxel_result_stream_nbytes(result: ChunkVoxelResult | CompressedChunkVoxelResult) -> int:
    if isinstance(result, CompressedChunkVoxelResult):
        return int(result.blocks.raw_nbytes) + int(result.materials.raw_nbytes)
    return int(result.blocks.nbytes) + int(result.materials.nbytes)


def compressed_chunk_voxel_results_stats(values: Iterable[object], seen_ids: set[int] | None = None) -> dict[str, int]:
    entries = 0
    raw_bytes = 0
    compressed_bytes = 0
    for value in values:
        if not isinstance(value, CompressedChunkVoxelResult):
            continue
        entries += 1
        value_id = id(value)
        if seen_ids is not None:
            if value_id in seen_ids:
                continue
            seen_ids.add(value_id)
        raw_bytes += int(value.raw_nbytes)
        compressed_bytes += int(value.compressed_nbytes)
    return {
        "entries": entries,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
    }
