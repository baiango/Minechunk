from __future__ import annotations

import ctypes
import atexit
import os
import platform
from pathlib import Path
from typing import Final

import numpy as np

from . import terrain_profile as _py_terrain_profile
from . import voxel_fill as _py_voxel_fill


ABI_VERSION: Final = 2
_ENV_LIB_PATH: Final = "MINECHUNK_TERRAIN_ZIG_LIB"
_ENV_DISABLE: Final = "MINECHUNK_DISABLE_ZIG_TERRAIN"
_ENV_KERNEL: Final = "MINECHUNK_TERRAIN_KERNEL"
_KERNEL_AUTO: Final = "auto"
_KERNEL_NUMBA: Final = "numba"
_KERNEL_ZIG: Final = "zig"

_UINT8_PTR = ctypes.POINTER(ctypes.c_uint8)
_UINT32_PTR = ctypes.POINTER(ctypes.c_uint32)
_CONST_UINT32_PTR = ctypes.POINTER(ctypes.c_uint32)
_INT32_PTR = ctypes.POINTER(ctypes.c_int32)

_LIBRARY: ctypes.CDLL | None | bool = None
_LOAD_ERROR: str | None = None
_LOADED_PATH: Path | None = None
_LOAD_KEY: tuple[str, str | None, bool] | None = None


def is_zig_terrain_available() -> bool:
    return _load_library() is not None


def zig_terrain_library_path() -> str | None:
    if _load_library() is None:
        return None
    return str(_LOADED_PATH) if _LOADED_PATH is not None else None


def zig_terrain_load_error() -> str | None:
    _load_library()
    return _LOAD_ERROR


def terrain_kernel_label() -> str:
    return "Zig" if _load_library() is not None else "Numba"


def surface_profile_at(
    x: float,
    z: float,
    seed: int,
    height_limit: int,
) -> tuple[int, int]:
    lib = _load_library()
    if lib is None:
        height, material = _py_terrain_profile.surface_profile_at(float(x), float(z), int(seed), int(height_limit))
        return int(height), int(material)

    out_height = ctypes.c_uint32()
    out_material = ctypes.c_uint32()
    status = lib.minechunk_surface_profile_at(
        ctypes.c_double(float(x)),
        ctypes.c_double(float(z)),
        ctypes.c_int64(int(seed)),
        ctypes.c_int32(int(height_limit)),
        ctypes.byref(out_height),
        ctypes.byref(out_material),
    )
    _check_status(status, "minechunk_surface_profile_at")
    return int(out_height.value), int(out_material.value)


def terrain_block_material_at(
    world_x: int,
    world_y: int,
    world_z: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> int:
    lib = _load_library()
    if lib is None:
        return int(
            _py_terrain_profile.terrain_block_material_at(
                int(world_x),
                int(world_y),
                int(world_z),
                int(seed),
                int(world_height_limit),
                bool(carve_caves),
            )
        )

    return int(
        lib.minechunk_terrain_block_material_at(
            ctypes.c_int64(int(world_x)),
            ctypes.c_int64(int(world_y)),
            ctypes.c_int64(int(world_z)),
            ctypes.c_int64(int(seed)),
            ctypes.c_int32(int(world_height_limit)),
            ctypes.c_uint8(1 if carve_caves else 0),
        )
    )


def fill_chunk_surface_grids(
    heights: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    height_limit: int,
) -> None:
    lib = _load_library()
    if lib is None:
        _py_voxel_fill.fill_chunk_surface_grids(
            heights,
            materials,
            chunk_x,
            chunk_z,
            chunk_size,
            seed,
            height_limit,
        )
        return

    _require_uint32_array(heights, "heights")
    _require_uint32_array(materials, "materials")
    status = lib.minechunk_fill_chunk_surface_grids(
        _uint32_ptr(heights),
        _uint32_ptr(materials),
        ctypes.c_size_t(heights.size),
        ctypes.c_size_t(materials.size),
        ctypes.c_int32(int(chunk_x)),
        ctypes.c_int32(int(chunk_z)),
        ctypes.c_int32(int(chunk_size)),
        ctypes.c_int64(int(seed)),
        ctypes.c_int32(int(height_limit)),
    )
    _check_status(status, "minechunk_fill_chunk_surface_grids")


def fill_chunk_surface_grids_batch(
    heights: np.ndarray,
    materials: np.ndarray,
    chunk_xs: np.ndarray | list[int] | tuple[int, ...],
    chunk_zs: np.ndarray | list[int] | tuple[int, ...],
    chunk_size: int,
    seed: int,
    height_limit: int,
) -> None:
    chunk_xs_array = _as_int32_vector(chunk_xs, "chunk_xs")
    chunk_zs_array = _as_int32_vector(chunk_zs, "chunk_zs")
    if chunk_xs_array.shape != chunk_zs_array.shape:
        raise ValueError("chunk_xs and chunk_zs must have matching shapes")

    chunk_count = int(chunk_xs_array.size)
    sample_size = int(chunk_size) + 2
    plane_cells = sample_size * sample_size
    total_cells = chunk_count * plane_cells
    _require_uint32_array(heights, "heights")
    _require_uint32_array(materials, "materials")
    _require_min_cells(heights, total_cells, "heights")
    _require_min_cells(materials, total_cells, "materials")

    if chunk_count == 0:
        return

    lib = _load_library()
    if lib is None:
        flat_heights = heights.reshape(-1)
        flat_materials = materials.reshape(-1)
        for index, (chunk_x, chunk_z) in enumerate(zip(chunk_xs_array, chunk_zs_array, strict=True)):
            start = index * plane_cells
            end = start + plane_cells
            _py_voxel_fill.fill_chunk_surface_grids(
                flat_heights[start:end],
                flat_materials[start:end],
                int(chunk_x),
                int(chunk_z),
                chunk_size,
                seed,
                height_limit,
            )
        return

    status = lib.minechunk_fill_chunk_surface_grids_batch(
        _uint32_ptr(heights),
        _uint32_ptr(materials),
        _int32_ptr(chunk_xs_array),
        _int32_ptr(chunk_zs_array),
        ctypes.c_size_t(heights.size),
        ctypes.c_size_t(materials.size),
        ctypes.c_size_t(chunk_count),
        ctypes.c_int32(int(chunk_size)),
        ctypes.c_int64(int(seed)),
        ctypes.c_int32(int(height_limit)),
    )
    _check_status(status, "minechunk_fill_chunk_surface_grids_batch")


def fill_stacked_chunk_voxel_grid(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> None:
    lib = _load_library()
    if lib is None:
        _py_voxel_fill.fill_stacked_chunk_voxel_grid(
            blocks,
            materials,
            chunk_x,
            chunk_y,
            chunk_z,
            chunk_size,
            seed,
            world_height_limit,
            carve_caves,
        )
        return

    _require_uint8_array(blocks, "blocks", ndim=3)
    _require_uint32_array(materials, "materials", ndim=3)
    if blocks.shape != materials.shape:
        raise ValueError("blocks and materials must have matching shapes")
    status = lib.minechunk_fill_stacked_chunk_voxel_grid(
        _uint8_ptr(blocks),
        _uint32_ptr(materials),
        ctypes.c_size_t(blocks.size),
        ctypes.c_size_t(materials.size),
        ctypes.c_int32(int(blocks.shape[0])),
        ctypes.c_int32(int(chunk_x)),
        ctypes.c_int32(int(chunk_y)),
        ctypes.c_int32(int(chunk_z)),
        ctypes.c_int32(int(chunk_size)),
        ctypes.c_int64(int(seed)),
        ctypes.c_int32(int(world_height_limit)),
        ctypes.c_uint8(1 if carve_caves else 0),
    )
    _check_status(status, "minechunk_fill_stacked_chunk_voxel_grid")


def fill_stacked_chunk_vertical_neighbor_planes(
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> None:
    lib = _load_library()
    if lib is None:
        _py_voxel_fill.fill_stacked_chunk_vertical_neighbor_planes(
            top_plane,
            bottom_plane,
            chunk_x,
            chunk_y,
            chunk_z,
            chunk_size,
            seed,
            world_height_limit,
            carve_caves,
        )
        return

    sample_size = int(chunk_size) + 2
    surface_heights = np.empty(sample_size * sample_size, dtype=np.uint32)
    surface_materials = np.empty(sample_size * sample_size, dtype=np.uint32)
    blocks = np.empty((int(chunk_size), sample_size, sample_size), dtype=np.uint8)
    materials = np.empty((int(chunk_size), sample_size, sample_size), dtype=np.uint32)
    fill_chunk_surface_grids(surface_heights, surface_materials, chunk_x, chunk_z, chunk_size, seed, world_height_limit)
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        blocks,
        materials,
        top_plane,
        bottom_plane,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height_limit,
        carve_caves,
    )


def fill_stacked_chunk_voxel_grid_with_neighbor_planes(
    blocks: np.ndarray,
    materials: np.ndarray,
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> None:
    lib = _load_library()
    if lib is None:
        _py_voxel_fill.fill_stacked_chunk_voxel_grid_with_neighbor_planes(
            blocks,
            materials,
            top_plane,
            bottom_plane,
            chunk_x,
            chunk_y,
            chunk_z,
            chunk_size,
            seed,
            world_height_limit,
            carve_caves,
        )
        return

    sample_size = int(chunk_size) + 2
    surface_heights = np.empty(sample_size * sample_size, dtype=np.uint32)
    surface_materials = np.empty(sample_size * sample_size, dtype=np.uint32)
    fill_chunk_surface_grids(surface_heights, surface_materials, chunk_x, chunk_z, chunk_size, seed, world_height_limit)
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        blocks,
        materials,
        top_plane,
        bottom_plane,
        surface_heights,
        surface_materials,
        chunk_x,
        chunk_y,
        chunk_z,
        chunk_size,
        seed,
        world_height_limit,
        carve_caves,
    )


def fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
    blocks: np.ndarray,
    materials: np.ndarray,
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_x: int,
    chunk_y: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> None:
    lib = _load_library()
    if lib is None:
        _py_voxel_fill.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
            blocks,
            materials,
            top_plane,
            bottom_plane,
            surface_heights,
            surface_materials,
            chunk_x,
            chunk_y,
            chunk_z,
            chunk_size,
            seed,
            world_height_limit,
            carve_caves,
        )
        return

    _require_voxel_fill_arrays(blocks, materials, top_plane, bottom_plane, surface_heights, surface_materials)
    status = lib.minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        _uint8_ptr(blocks),
        _uint32_ptr(materials),
        _uint8_ptr(top_plane),
        _uint8_ptr(bottom_plane),
        _const_uint32_ptr(surface_heights),
        _const_uint32_ptr(surface_materials),
        ctypes.c_size_t(blocks.size),
        ctypes.c_size_t(materials.size),
        ctypes.c_size_t(top_plane.size),
        ctypes.c_size_t(surface_heights.size),
        ctypes.c_int32(int(blocks.shape[0])),
        ctypes.c_int32(int(chunk_x)),
        ctypes.c_int32(int(chunk_y)),
        ctypes.c_int32(int(chunk_z)),
        ctypes.c_int32(int(chunk_size)),
        ctypes.c_int64(int(seed)),
        ctypes.c_int32(int(world_height_limit)),
        ctypes.c_uint8(1 if carve_caves else 0),
    )
    _check_status(status, "minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface")


def fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch(
    blocks: np.ndarray,
    materials: np.ndarray,
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
    chunk_coords: np.ndarray | list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    chunk_size: int,
    seed: int,
    world_height_limit: int,
    carve_caves: bool = True,
) -> None:
    coords = _as_chunk_coords(chunk_coords)
    chunk_count = int(coords.shape[0])
    sample_size = int(chunk_size) + 2
    plane_cells = sample_size * sample_size
    total_plane_cells = chunk_count * plane_cells

    _require_uint8_array(blocks, "blocks", ndim=4)
    _require_uint32_array(materials, "materials", ndim=4)
    _require_uint8_array(top_plane, "top_plane", ndim=3)
    _require_uint8_array(bottom_plane, "bottom_plane", ndim=3)
    _require_uint32_array(surface_heights, "surface_heights")
    _require_uint32_array(surface_materials, "surface_materials")
    if blocks.shape != materials.shape:
        raise ValueError("blocks and materials must have matching shapes")
    if blocks.shape[0] != chunk_count:
        raise ValueError("blocks first dimension must match chunk count")
    if blocks.shape[2:] != (sample_size, sample_size):
        raise ValueError("blocks must have shape (chunk_count, local_height, sample_size, sample_size)")
    if top_plane.shape != (chunk_count, sample_size, sample_size):
        raise ValueError("top_plane must have shape (chunk_count, sample_size, sample_size)")
    if bottom_plane.shape != (chunk_count, sample_size, sample_size):
        raise ValueError("bottom_plane must have shape (chunk_count, sample_size, sample_size)")
    if surface_heights.shape != surface_materials.shape:
        raise ValueError("surface_heights and surface_materials must have matching shapes")
    _require_min_cells(surface_heights, total_plane_cells, "surface_heights")
    _require_min_cells(surface_materials, total_plane_cells, "surface_materials")

    if chunk_count == 0:
        return

    lib = _load_library()
    if lib is None:
        flat_surface_heights = surface_heights.reshape(-1)[:total_plane_cells].reshape(chunk_count, plane_cells)
        flat_surface_materials = surface_materials.reshape(-1)[:total_plane_cells].reshape(chunk_count, plane_cells)
        for index, (chunk_x, chunk_y, chunk_z) in enumerate(coords):
            _py_voxel_fill.fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
                blocks[index],
                materials[index],
                top_plane[index],
                bottom_plane[index],
                flat_surface_heights[index],
                flat_surface_materials[index],
                int(chunk_x),
                int(chunk_y),
                int(chunk_z),
                chunk_size,
                seed,
                world_height_limit,
                carve_caves,
            )
        return

    chunk_xs = np.ascontiguousarray(coords[:, 0])
    chunk_ys = np.ascontiguousarray(coords[:, 1])
    chunk_zs = np.ascontiguousarray(coords[:, 2])
    status = lib.minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch(
        _uint8_ptr(blocks),
        _uint32_ptr(materials),
        _uint8_ptr(top_plane),
        _uint8_ptr(bottom_plane),
        _const_uint32_ptr(surface_heights),
        _const_uint32_ptr(surface_materials),
        _int32_ptr(chunk_xs),
        _int32_ptr(chunk_ys),
        _int32_ptr(chunk_zs),
        ctypes.c_size_t(blocks.size),
        ctypes.c_size_t(materials.size),
        ctypes.c_size_t(top_plane.size),
        ctypes.c_size_t(surface_heights.size),
        ctypes.c_size_t(chunk_count),
        ctypes.c_int32(int(blocks.shape[1])),
        ctypes.c_int32(int(chunk_size)),
        ctypes.c_int64(int(seed)),
        ctypes.c_int32(int(world_height_limit)),
        ctypes.c_uint8(1 if carve_caves else 0),
    )
    _check_status(status, "minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch")


def fill_chunk_voxel_grid(
    blocks: np.ndarray,
    materials: np.ndarray,
    chunk_x: int,
    chunk_z: int,
    chunk_size: int,
    seed: int,
    height_limit: int,
    carve_caves: bool = True,
) -> None:
    lib = _load_library()
    if lib is None:
        _py_voxel_fill.fill_chunk_voxel_grid(
            blocks,
            materials,
            chunk_x,
            chunk_z,
            chunk_size,
            seed,
            height_limit,
            carve_caves,
        )
        return

    _require_uint8_array(blocks, "blocks", ndim=3)
    _require_uint32_array(materials, "materials", ndim=3)
    if blocks.shape != materials.shape:
        raise ValueError("blocks and materials must have matching shapes")
    status = lib.minechunk_fill_chunk_voxel_grid(
        _uint8_ptr(blocks),
        _uint32_ptr(materials),
        ctypes.c_size_t(blocks.size),
        ctypes.c_size_t(materials.size),
        ctypes.c_int32(int(blocks.shape[0])),
        ctypes.c_int32(int(chunk_x)),
        ctypes.c_int32(int(chunk_z)),
        ctypes.c_int32(int(chunk_size)),
        ctypes.c_int64(int(seed)),
        ctypes.c_int32(int(height_limit)),
        ctypes.c_uint8(1 if carve_caves else 0),
    )
    _check_status(status, "minechunk_fill_chunk_voxel_grid")


def expand_chunk_surface_to_voxel_grid(
    blocks: np.ndarray,
    materials: np.ndarray,
    height_grid: np.ndarray,
    material_grid: np.ndarray,
    chunk_size: int,
    height_limit: int,
) -> None:
    _py_voxel_fill.expand_chunk_surface_to_voxel_grid(
        blocks,
        materials,
        height_grid,
        material_grid,
        chunk_size,
        height_limit,
    )


def _load_library() -> ctypes.CDLL | None:
    global _LIBRARY, _LOAD_ERROR, _LOADED_PATH, _LOAD_KEY

    mode = _selected_kernel_mode()
    explicit_path = os.environ.get(_ENV_LIB_PATH)
    disabled = os.environ.get(_ENV_DISABLE) == "1"
    load_key = (mode, explicit_path, disabled)
    if load_key != _LOAD_KEY:
        _LIBRARY = None
        _LOAD_ERROR = None
        _LOADED_PATH = None
        _LOAD_KEY = load_key

    if _LIBRARY is False:
        return None
    if isinstance(_LIBRARY, ctypes.CDLL):
        return _LIBRARY
    if disabled or mode == _KERNEL_NUMBA:
        _LIBRARY = False
        return None

    errors: list[str] = []
    for path in _candidate_library_paths():
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            _configure_library(lib)
            abi_version = int(lib.minechunk_terrain_abi_version())
            if abi_version != ABI_VERSION:
                raise RuntimeError(f"ABI version {abi_version} does not match expected {ABI_VERSION}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            if explicit_path:
                raise RuntimeError(f"failed to load Zig terrain kernel from {path}: {exc}") from exc
            continue
        _LIBRARY = lib
        _LOADED_PATH = path
        _LOAD_ERROR = None
        atexit.register(_shutdown_library, lib)
        return lib

    _LIBRARY = False
    _LOAD_ERROR = "; ".join(errors) if errors else None
    if mode == _KERNEL_ZIG:
        detail = f": {_LOAD_ERROR}" if _LOAD_ERROR else ""
        raise RuntimeError(
            "Zig terrain kernel was requested, but no compatible shared library was found"
            f"{detail}. Run `python3 tools/build_zig_terrain.py`."
        )
    return None


def _configure_library(lib: ctypes.CDLL) -> None:
    lib.minechunk_terrain_abi_version.argtypes = []
    lib.minechunk_terrain_abi_version.restype = ctypes.c_uint32

    if hasattr(lib, "minechunk_shutdown_terrain_workers"):
        lib.minechunk_shutdown_terrain_workers.argtypes = []
        lib.minechunk_shutdown_terrain_workers.restype = None

    lib.minechunk_surface_profile_at.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.minechunk_surface_profile_at.restype = ctypes.c_int32

    lib.minechunk_terrain_block_material_at.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_uint8,
    ]
    lib.minechunk_terrain_block_material_at.restype = ctypes.c_uint32

    lib.minechunk_fill_chunk_surface_grids.argtypes = [
        _UINT32_PTR,
        _UINT32_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
    ]
    lib.minechunk_fill_chunk_surface_grids.restype = ctypes.c_int32

    lib.minechunk_fill_chunk_surface_grids_batch.argtypes = [
        _UINT32_PTR,
        _UINT32_PTR,
        _INT32_PTR,
        _INT32_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
    ]
    lib.minechunk_fill_chunk_surface_grids_batch.restype = ctypes.c_int32

    lib.minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface.argtypes = [
        _UINT8_PTR,
        _UINT32_PTR,
        _UINT8_PTR,
        _UINT8_PTR,
        _CONST_UINT32_PTR,
        _CONST_UINT32_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_uint8,
    ]
    lib.minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface.restype = ctypes.c_int32

    lib.minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch.argtypes = [
        _UINT8_PTR,
        _UINT32_PTR,
        _UINT8_PTR,
        _UINT8_PTR,
        _CONST_UINT32_PTR,
        _CONST_UINT32_PTR,
        _INT32_PTR,
        _INT32_PTR,
        _INT32_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_uint8,
    ]
    lib.minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch.restype = ctypes.c_int32

    lib.minechunk_fill_stacked_chunk_voxel_grid.argtypes = [
        _UINT8_PTR,
        _UINT32_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_uint8,
    ]
    lib.minechunk_fill_stacked_chunk_voxel_grid.restype = ctypes.c_int32

    lib.minechunk_fill_chunk_voxel_grid.argtypes = [
        _UINT8_PTR,
        _UINT32_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_uint8,
    ]
    lib.minechunk_fill_chunk_voxel_grid.restype = ctypes.c_int32


def _shutdown_library(lib: ctypes.CDLL) -> None:
    shutdown = getattr(lib, "minechunk_shutdown_terrain_workers", None)
    if shutdown is None:
        return
    try:
        shutdown()
    except Exception:
        pass


def _candidate_library_paths() -> list[Path]:
    explicit_path = os.environ.get(_ENV_LIB_PATH)
    if explicit_path:
        return [Path(explicit_path).expanduser()]

    filename = _library_filename()
    kernel_dir = Path(__file__).resolve().parent
    root = kernel_dir.parents[2]
    return [
        kernel_dir / "native" / filename,
        root / "zig-out" / "lib" / filename,
    ]


def _selected_kernel_mode() -> str:
    raw = os.environ.get(_ENV_KERNEL, _KERNEL_AUTO).strip().lower()
    if not raw or raw == "default":
        return _KERNEL_AUTO
    if raw in ("python", "py"):
        return _KERNEL_NUMBA
    if raw in ("native", "libzig"):
        return _KERNEL_ZIG
    if raw in (_KERNEL_AUTO, _KERNEL_NUMBA, _KERNEL_ZIG):
        return raw
    raise ValueError(f"{_ENV_KERNEL} must be one of: auto, numba, zig")


def _library_filename() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "libminechunk_terrain.dylib"
    if system == "windows":
        return "minechunk_terrain.dll"
    return "libminechunk_terrain.so"


def _check_status(status: int, function_name: str) -> None:
    if int(status) != 0:
        raise RuntimeError(f"{function_name} failed with status {int(status)}")


def _as_int32_vector(values: np.ndarray | list[int] | tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return np.ascontiguousarray(array)


def _as_chunk_coords(
    values: np.ndarray | list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    coords = np.asarray(values, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("chunk_coords must have shape (chunk_count, 3)")
    return np.ascontiguousarray(coords)


def _require_min_cells(array: np.ndarray, min_cells: int, name: str) -> None:
    if array.size < min_cells:
        raise ValueError(f"{name} must contain at least {min_cells} cells")


def _require_voxel_fill_arrays(
    blocks: np.ndarray,
    materials: np.ndarray,
    top_plane: np.ndarray,
    bottom_plane: np.ndarray,
    surface_heights: np.ndarray,
    surface_materials: np.ndarray,
) -> None:
    _require_uint8_array(blocks, "blocks", ndim=3)
    _require_uint32_array(materials, "materials", ndim=3)
    _require_uint8_array(top_plane, "top_plane", ndim=2)
    _require_uint8_array(bottom_plane, "bottom_plane", ndim=2)
    _require_uint32_array(surface_heights, "surface_heights")
    _require_uint32_array(surface_materials, "surface_materials")

    if blocks.shape != materials.shape:
        raise ValueError("blocks and materials must have matching shapes")
    if top_plane.shape != bottom_plane.shape:
        raise ValueError("top_plane and bottom_plane must have matching shapes")
    if surface_heights.shape != surface_materials.shape:
        raise ValueError("surface_heights and surface_materials must have matching shapes")


def _require_uint8_array(array: np.ndarray, name: str, ndim: int | None = None) -> None:
    _require_array(array, name, np.uint8, ndim)


def _require_uint32_array(array: np.ndarray, name: str, ndim: int | None = None) -> None:
    _require_array(array, name, np.uint32, ndim)


def _require_array(array: np.ndarray, name: str, dtype: np.dtype, ndim: int | None = None) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _uint8_ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return array.ctypes.data_as(_UINT8_PTR)


def _uint32_ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint32):
    return array.ctypes.data_as(_UINT32_PTR)


def _const_uint32_ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint32):
    return array.ctypes.data_as(_CONST_UINT32_PTR)


def _int32_ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
    return array.ctypes.data_as(_INT32_PTR)
