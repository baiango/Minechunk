from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes.util import find_library
from pathlib import Path


HUD_FONT_FALLBACK: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000"),
    "?": ("111", "001", "010", "000", "010"),
    ".": ("000", "000", "000", "000", "010"),
    ",": ("000", "000", "000", "010", "100"),
    ":": ("000", "010", "000", "010", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "/": ("001", "001", "010", "100", "100"),
    "(": ("010", "100", "100", "100", "010"),
    ")": ("010", "001", "001", "001", "010"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "101", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


def _find_hud_font_path() -> str | None:
    bundled_font = Path(__file__).resolve().with_name("res") / "Roboto-VariableFont_wdth,wght.ttf"
    if os.path.exists(bundled_font):
        return str(bundled_font)
    try:
        font_path = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n", "Roboto:style=Regular"],
            text=True,
        ).strip()
        if font_path and os.path.exists(font_path):
            return font_path
    except Exception:
        pass
    for candidate in (
        "/System/Library/Fonts/Supplemental/Roboto Regular.ttf",
        "/Library/Fonts/Roboto Regular.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _find_freetype_library_path() -> str | None:
    candidates: list[str] = []

    found = find_library("freetype")
    if found:
        candidates.append(found)

    if sys.platform == "darwin":
        candidates.extend(
            [
                "/opt/homebrew/opt/freetype/lib/libfreetype.dylib",
                "/usr/local/opt/freetype/lib/libfreetype.dylib",
                "/opt/homebrew/lib/libfreetype.dylib",
                "/usr/local/lib/libfreetype.dylib",
                "/usr/lib/libfreetype.dylib",
            ]
        )
    elif sys.platform.startswith("linux"):
        candidates.extend(
            [
                "/usr/lib/x86_64-linux-gnu/libfreetype.so.6",
                "/usr/lib64/libfreetype.so.6",
                "/usr/lib/libfreetype.so.6",
                "/lib/x86_64-linux-gnu/libfreetype.so.6",
                "/lib64/libfreetype.so.6",
                "/lib/libfreetype.so.6",
            ]
        )
    elif sys.platform.startswith("win"):
        candidates.extend(
            [
                "libfreetype-6.dll",
                "freetype.dll",
            ]
        )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    for candidate in candidates:
        if candidate:
            return candidate

    return None


def _build_hud_font_from_freetype() -> dict[str, tuple[str, ...]]:
    font_path = _find_hud_font_path()
    if not font_path:
        raise RuntimeError("No usable HUD font file found")

    library_path = _find_freetype_library_path()
    if not library_path or not os.path.exists(library_path):
        raise RuntimeError("FreeType library not available")

    freetype = ctypes.CDLL(library_path)
    c_void_p = ctypes.c_void_p

    class FT_Generic(ctypes.Structure):
        _fields_ = [("data", c_void_p), ("finalizer", c_void_p)]

    class FT_Vector(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class FT_BBox(ctypes.Structure):
        _fields_ = [
            ("xMin", ctypes.c_long),
            ("yMin", ctypes.c_long),
            ("xMax", ctypes.c_long),
            ("yMax", ctypes.c_long),
        ]

    class FT_Bitmap(ctypes.Structure):
        _fields_ = [
            ("rows", ctypes.c_uint),
            ("width", ctypes.c_uint),
            ("pitch", ctypes.c_int),
            ("buffer", ctypes.POINTER(ctypes.c_ubyte)),
            ("num_grays", ctypes.c_ushort),
            ("pixel_mode", ctypes.c_ubyte),
            ("palette_mode", ctypes.c_ubyte),
            ("palette", c_void_p),
        ]

    class FT_Glyph_Metrics(ctypes.Structure):
        _fields_ = [
            ("width", ctypes.c_long),
            ("height", ctypes.c_long),
            ("horiBearingX", ctypes.c_long),
            ("horiBearingY", ctypes.c_long),
            ("horiAdvance", ctypes.c_long),
            ("vertBearingX", ctypes.c_long),
            ("vertBearingY", ctypes.c_long),
            ("vertAdvance", ctypes.c_long),
        ]

    class FT_Outline(ctypes.Structure):
        _fields_ = [
            ("n_contours", ctypes.c_ushort),
            ("n_points", ctypes.c_ushort),
            ("points", c_void_p),
            ("tags", c_void_p),
            ("contours", c_void_p),
            ("flags", ctypes.c_int),
        ]

    class FT_SizeRec(ctypes.Structure):
        pass

    class FT_FaceRec(ctypes.Structure):
        pass

    class FT_GlyphSlotRec(ctypes.Structure):
        pass

    FT_Library = c_void_p
    FT_Size = ctypes.POINTER(FT_SizeRec)
    FT_CharMap = c_void_p
    FT_Driver = c_void_p
    FT_Memory = c_void_p
    FT_Stream = c_void_p
    FT_ListRec = c_void_p
    FT_Face_Internal = c_void_p
    FT_Slot_Internal = c_void_p
    FT_GlyphSlot = ctypes.POINTER(FT_GlyphSlotRec)
    FT_Face = ctypes.POINTER(FT_FaceRec)

    class FT_Size_Metrics(ctypes.Structure):
        _fields_ = [
            ("x_ppem", ctypes.c_ushort),
            ("y_ppem", ctypes.c_ushort),
            ("x_scale", ctypes.c_long),
            ("y_scale", ctypes.c_long),
            ("ascender", ctypes.c_long),
            ("descender", ctypes.c_long),
            ("height", ctypes.c_long),
            ("max_advance", ctypes.c_long),
        ]

    FT_SizeRec._fields_ = [
        ("face", FT_Face),
        ("generic", FT_Generic),
        ("metrics", FT_Size_Metrics),
        ("internal", c_void_p),
    ]

    FT_GlyphSlotRec._fields_ = [
        ("library", FT_Library),
        ("face", FT_Face),
        ("next", FT_GlyphSlot),
        ("glyph_index", ctypes.c_uint),
        ("generic", FT_Generic),
        ("metrics", FT_Glyph_Metrics),
        ("linearHoriAdvance", ctypes.c_long),
        ("linearVertAdvance", ctypes.c_long),
        ("advance", FT_Vector),
        ("format", ctypes.c_uint),
        ("bitmap", FT_Bitmap),
        ("bitmap_left", ctypes.c_int),
        ("bitmap_top", ctypes.c_int),
        ("outline", FT_Outline),
        ("num_subglyphs", ctypes.c_uint),
        ("subglyphs", c_void_p),
        ("control_data", c_void_p),
        ("control_len", ctypes.c_long),
        ("lsb_delta", ctypes.c_long),
        ("rsb_delta", ctypes.c_long),
        ("other", c_void_p),
        ("internal", FT_Slot_Internal),
    ]

    FT_FaceRec._fields_ = [
        ("num_faces", ctypes.c_long),
        ("face_index", ctypes.c_long),
        ("face_flags", ctypes.c_long),
        ("style_flags", ctypes.c_long),
        ("num_glyphs", ctypes.c_long),
        ("family_name", ctypes.c_char_p),
        ("style_name", ctypes.c_char_p),
        ("num_fixed_sizes", ctypes.c_int),
        ("available_sizes", c_void_p),
        ("num_charmaps", ctypes.c_int),
        ("charmaps", c_void_p),
        ("generic", FT_Generic),
        ("bbox", FT_BBox),
        ("units_per_EM", ctypes.c_ushort),
        ("ascender", ctypes.c_short),
        ("descender", ctypes.c_short),
        ("height", ctypes.c_short),
        ("max_advance_width", ctypes.c_short),
        ("max_advance_height", ctypes.c_short),
        ("underline_position", ctypes.c_short),
        ("underline_thickness", ctypes.c_short),
        ("glyph", FT_GlyphSlot),
        ("size", FT_Size),
        ("charmap", FT_CharMap),
        ("driver", FT_Driver),
        ("memory", FT_Memory),
        ("stream", FT_Stream),
        ("sizes_list", FT_ListRec),
        ("autohint", c_void_p),
        ("extensions", c_void_p),
        ("internal", FT_Face_Internal),
    ]

    face = FT_Face()
    lib_obj = FT_Library()

    FT_LOAD_RENDER = 0x4

    freetype.FT_Init_FreeType.argtypes = [ctypes.POINTER(FT_Library)]
    freetype.FT_Init_FreeType.restype = ctypes.c_int
    freetype.FT_New_Face.argtypes = [FT_Library, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(FT_Face)]
    freetype.FT_New_Face.restype = ctypes.c_int
    freetype.FT_Set_Pixel_Sizes.argtypes = [FT_Face, ctypes.c_uint, ctypes.c_uint]
    freetype.FT_Set_Pixel_Sizes.restype = ctypes.c_int
    freetype.FT_Load_Char.argtypes = [FT_Face, ctypes.c_ulong, ctypes.c_int]
    freetype.FT_Load_Char.restype = ctypes.c_int
    freetype.FT_Done_Face.argtypes = [FT_Face]
    freetype.FT_Done_FreeType.argtypes = [FT_Library]

    error = freetype.FT_Init_FreeType(ctypes.byref(lib_obj))
    if error != 0:
        raise RuntimeError(f"FT_Init_FreeType failed: {error}")
    try:
        error = freetype.FT_New_Face(lib_obj, font_path.encode("utf-8"), 0, ctypes.byref(face))
        if error != 0:
            raise RuntimeError(f"FT_New_Face failed: {error}")
        try:
            error = freetype.FT_Set_Pixel_Sizes(face, 0, 7)
            if error != 0:
                raise RuntimeError(f"FT_Set_Pixel_Sizes failed: {error}")
            size_metrics = face.contents.size.contents.metrics
            cell_width = max(1, int(round(size_metrics.max_advance / 64.0)) + 1)
            cell_height = max(1, int(round((size_metrics.ascender - size_metrics.descender) / 64.0)))
            baseline = int(round(size_metrics.ascender / 64.0))
            font: dict[str, tuple[str, ...]] = {}
            for code in range(32, 127):
                char = chr(code)
                error = freetype.FT_Load_Char(face, code, FT_LOAD_RENDER)
                if error != 0:
                    font[char] = HUD_FONT_FALLBACK.get(char, HUD_FONT_FALLBACK["?"])
                    continue
                slot = face.contents.glyph.contents
                bitmap = slot.bitmap
                rows = [["0"] * cell_width for _ in range(cell_height)]
                if bitmap.width > 0 and bitmap.rows > 0 and bool(bitmap.buffer):
                    buffer = ctypes.cast(bitmap.buffer, ctypes.POINTER(ctypes.c_ubyte))
                    row_offset = baseline - slot.bitmap_top
                    for y in range(bitmap.rows):
                        dest_y = row_offset + y
                        if not (0 <= dest_y < cell_height):
                            continue
                        for x in range(bitmap.width):
                            dest_x = slot.bitmap_left + x
                            if not (0 <= dest_x < cell_width):
                                continue
                            value = buffer[y * bitmap.pitch + x]
                            if value > 64:
                                rows[dest_y][dest_x] = "1"
                font[char] = tuple("".join(row) for row in rows)
            if "?" not in font:
                font["?"] = HUD_FONT_FALLBACK["?"]
            return font
        finally:
            freetype.FT_Done_Face(face)
    finally:
        freetype.FT_Done_FreeType(lib_obj)


def build_hud_font() -> dict[str, tuple[str, ...]]:
    try:
        return _build_hud_font_from_freetype()
    except Exception:
        return dict(HUD_FONT_FALLBACK)


HUD_FONT: dict[str, tuple[str, ...]] = {}
_HUD_FONT_INITIALIZED = False


def get_hud_font() -> dict[str, tuple[str, ...]]:
    global _HUD_FONT_INITIALIZED
    if not _HUD_FONT_INITIALIZED:
        HUD_FONT.clear()
        HUD_FONT.update(build_hud_font())
        _HUD_FONT_INITIALIZED = True
    return HUD_FONT


def hud_glyph_rows(char: str) -> tuple[str, ...]:
    font = get_hud_font()
    return font.get(char, font["?"])

