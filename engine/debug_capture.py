from __future__ import annotations

"""Small image-capture helpers used by renderer debug paths.

Keeping these pure helpers outside ``renderer.py`` makes the F7/RC capture path
unit-testable without importing WGPU or constructing a ``TerrainRenderer``.
"""

import os
import struct
import zlib
from typing import Any

import numpy as np


def safe_filename_component(value: str, *, fallback: str = "mode") -> str:
    """Return a filesystem-friendly slug for debug filenames."""

    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or fallback


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """Encode a PNG chunk with length and CRC."""

    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def write_rgba8_png(path: str, rgba: np.ndarray) -> None:
    """Write an RGBA8 numpy array as a minimal PNG file."""

    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError(f"expected RGBA8 image with 4 channels, got {channels}")
    scanlines = bytearray()
    for y in range(height):
        scanlines.append(0)
        scanlines.extend(rgba[y].tobytes())
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", int(width), int(height), 8, 6, 0, 0, 0),
    )
    payload += png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=4))
    payload += png_chunk(b"IEND", b"")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)


def readback_to_rgba8(mapped: Any, *, width: int, height: int, texture_format: str, padded_bpr: int) -> np.ndarray:
    """Convert a mapped WGPU readback buffer into an RGBA8 image."""

    if texture_format == "rgba16float":
        row_pixels = padded_bpr // 8
        values = np.frombuffer(mapped, dtype=np.float16, count=(padded_bpr * height) // 2)
        image_f16 = values.reshape((height, row_pixels, 4))[:, :width, :]
        image_f32 = np.nan_to_num(image_f16.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        rgb = np.clip(image_f32[:, :, 0:3], 0.0, 1.0)
        alpha = np.clip(image_f32[:, :, 3:4], 0.0, 1.0)
        out = np.empty((height, width, 4), dtype=np.uint8)
        out[:, :, 0:3] = (rgb * 255.0 + 0.5).astype(np.uint8)
        out[:, :, 3:4] = (alpha * 255.0 + 0.5).astype(np.uint8)
        return out
    if texture_format == "rgba8unorm":
        raw = np.frombuffer(mapped, dtype=np.uint8, count=padded_bpr * height)
        rows = raw.reshape((height, padded_bpr))[:, : width * 4]
        return rows.reshape((height, width, 4)).copy()
    raise ValueError(f"unsupported screenshot texture format: {texture_format!r}")
