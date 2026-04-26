import struct
import zlib

import numpy as np

from engine.debug_capture import readback_to_rgba8, safe_filename_component, write_rgba8_png


def test_safe_filename_component_collapses_symbols():
    assert safe_filename_component(" RC Debug: Sky Leak++ ") == "rc_debug_sky_leak"
    assert safe_filename_component("***", fallback="capture") == "capture"


def test_readback_to_rgba8_strips_padding():
    # Two rows, one real RGBA pixel per row, padded to 8 bytes per row.
    mapped = bytes([
        1, 2, 3, 4, 99, 99, 99, 99,
        5, 6, 7, 8, 88, 88, 88, 88,
    ])
    rgba = readback_to_rgba8(mapped, width=1, height=2, texture_format="rgba8unorm", padded_bpr=8)
    assert rgba.tolist() == [[[1, 2, 3, 4]], [[5, 6, 7, 8]]]


def test_write_rgba8_png_writes_valid_signature_and_ihdr(tmp_path):
    path = tmp_path / "debug.png"
    pixels = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8)
    write_rgba8_png(str(path), pixels)
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (2, 1)
    ihdr_crc = struct.unpack(">I", data[29:33])[0]
    assert ihdr_crc == (zlib.crc32(data[12:29]) & 0xFFFFFFFF)
