from __future__ import annotations

import ctypes
import os
import platform
import time
try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None

import numpy as np

from .. import render_contract as render_consts
from ..cache import mesh_allocator as mesh_cache
from ..cache import mesh_zstd
from ..cache import tile_zstd
from ..memory_pressure import memory_pressure_stats
from ..rendering import worldspace_rc
from ..terrain.compression import compressed_chunk_voxel_results_stats, chunk_voxel_result_raw_nbytes, is_compressed_chunk_voxel_result
from .hud_overlay import _ensure_hud_vertex_buffer, build_frame_breakdown_hud_vertices, build_profile_hud_vertices
from .profiling_stats import frame_breakdown_average, profile_average_fps, profile_frame_time_percentiles


def _renderer_module():
    """Return stable render constants without importing the renderer runtime."""
    return render_consts


def _resource_peak_rss_bytes() -> int:
    if resource is None:
        return 0
    try:
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0
    if peak_rss <= 0:
        return 0
    # macOS reports bytes, Linux reports KiB.
    if platform.system() == "Darwin":
        return peak_rss
    return peak_rss * 1024


def _darwin_current_rss_bytes() -> int:
    try:
        libc = ctypes.CDLL("libc.dylib")

        class TimeValue(ctypes.Structure):
            _fields_ = [
                ("seconds", ctypes.c_int32),
                ("microseconds", ctypes.c_int32),
            ]

        class MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", TimeValue),
                ("system_time", TimeValue),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        mach_task_self = libc.mach_task_self
        mach_task_self.restype = ctypes.c_uint32
        task_info = libc.task_info
        task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        task_info.restype = ctypes.c_int

        info = MachTaskBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        result = task_info(
            mach_task_self(),
            20,  # MACH_TASK_BASIC_INFO
            ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_uint32)),
            ctypes.byref(count),
        )
        if result != 0:
            return 0
        return int(info.resident_size)
    except Exception:
        return 0


def _darwin_task_vm_info_stats() -> dict[str, int]:
    try:
        libc = ctypes.CDLL("libc.dylib")

        class TaskVmInfoBase(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("region_count", ctypes.c_int32),
                ("page_size", ctypes.c_int32),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_peak", ctypes.c_uint64),
                ("device", ctypes.c_uint64),
                ("device_peak", ctypes.c_uint64),
                ("internal", ctypes.c_uint64),
                ("internal_peak", ctypes.c_uint64),
                ("external", ctypes.c_uint64),
                ("external_peak", ctypes.c_uint64),
                ("reusable", ctypes.c_uint64),
                ("reusable_peak", ctypes.c_uint64),
                ("purgeable_volatile_pmap", ctypes.c_uint64),
                ("purgeable_volatile_resident", ctypes.c_uint64),
                ("purgeable_volatile_virtual", ctypes.c_uint64),
                ("compressed", ctypes.c_uint64),
                ("compressed_peak", ctypes.c_uint64),
                ("compressed_lifetime", ctypes.c_uint64),
                ("phys_footprint", ctypes.c_uint64),
            ]

        class TaskVmInfo(TaskVmInfoBase):
            _fields_ = [
                ("min_address", ctypes.c_uint64),
                ("max_address", ctypes.c_uint64),
                ("ledger_phys_footprint_peak", ctypes.c_int64),
                ("ledger_purgeable_nonvolatile", ctypes.c_int64),
                ("ledger_purgeable_nonvolatile_compressed", ctypes.c_int64),
                ("ledger_purgeable_volatile", ctypes.c_int64),
                ("ledger_purgeable_volatile_compressed", ctypes.c_int64),
                ("ledger_tag_network_nonvolatile", ctypes.c_int64),
                ("ledger_tag_network_nonvolatile_compressed", ctypes.c_int64),
                ("ledger_tag_network_volatile", ctypes.c_int64),
                ("ledger_tag_network_volatile_compressed", ctypes.c_int64),
                ("ledger_tag_media_footprint", ctypes.c_int64),
                ("ledger_tag_media_footprint_compressed", ctypes.c_int64),
                ("ledger_tag_media_nofootprint", ctypes.c_int64),
                ("ledger_tag_media_nofootprint_compressed", ctypes.c_int64),
                ("ledger_tag_graphics_footprint", ctypes.c_int64),
                ("ledger_tag_graphics_footprint_compressed", ctypes.c_int64),
                ("ledger_tag_graphics_nofootprint", ctypes.c_int64),
                ("ledger_tag_graphics_nofootprint_compressed", ctypes.c_int64),
                ("ledger_tag_neural_footprint", ctypes.c_int64),
                ("ledger_tag_neural_footprint_compressed", ctypes.c_int64),
                ("ledger_tag_neural_nofootprint", ctypes.c_int64),
                ("ledger_tag_neural_nofootprint_compressed", ctypes.c_int64),
            ]

        mach_task_self = libc.mach_task_self
        mach_task_self.restype = ctypes.c_uint32
        task_info = libc.task_info
        task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        task_info.restype = ctypes.c_int

        def _load_info(info_type):
            info = info_type()
            count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
            result = task_info(
                mach_task_self(),
                22,  # TASK_VM_INFO
                ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_uint32)),
                ctypes.byref(count),
            )
            return info if result == 0 else None

        info = _load_info(TaskVmInfo)
        if info is None:
            info = _load_info(TaskVmInfoBase)
        if info is None:
            return {}
        return {
            "footprint_bytes": int(info.phys_footprint),
            "task_vm_rss_bytes": int(info.resident_size),
            "internal_bytes": int(info.internal),
            "external_bytes": int(info.external),
            "compressed_bytes": int(info.compressed),
            "iokit_bytes": int(info.device),
            "reusable_bytes": int(info.reusable),
            "graphics_footprint_bytes": max(0, int(getattr(info, "ledger_tag_graphics_footprint", 0))),
            "graphics_compressed_bytes": max(0, int(getattr(info, "ledger_tag_graphics_footprint_compressed", 0))),
            "media_footprint_bytes": max(0, int(getattr(info, "ledger_tag_media_footprint", 0))),
            "neural_footprint_bytes": max(0, int(getattr(info, "ledger_tag_neural_footprint", 0))),
            "footprint_peak_bytes": max(0, int(getattr(info, "ledger_phys_footprint_peak", 0))),
        }
    except Exception:
        return {}


def _linux_current_rss_bytes() -> int:
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            return 0
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 0


def process_memory_stats() -> dict[str, int | bool]:
    current_rss = 0
    footprint_bytes = 0
    footprint_supported = False
    darwin_vm_stats: dict[str, int] = {}
    system_name = platform.system()
    if system_name == "Darwin":
        current_rss = _darwin_current_rss_bytes()
        darwin_vm_stats = _darwin_task_vm_info_stats()
        footprint_bytes = int(darwin_vm_stats.get("footprint_bytes", 0))
        footprint_supported = footprint_bytes > 0
    elif system_name == "Linux":
        current_rss = _linux_current_rss_bytes()
    current_supported = current_rss > 0
    peak_rss = _resource_peak_rss_bytes()
    if current_rss <= 0:
        current_rss = peak_rss
    if footprint_bytes <= 0:
        footprint_bytes = current_rss
    return {
        "rss_bytes": int(current_rss),
        "footprint_bytes": int(footprint_bytes),
        "peak_rss_bytes": int(peak_rss),
        "task_vm_rss_bytes": int(darwin_vm_stats.get("task_vm_rss_bytes", 0)),
        "internal_bytes": int(darwin_vm_stats.get("internal_bytes", 0)),
        "external_bytes": int(darwin_vm_stats.get("external_bytes", 0)),
        "compressed_bytes": int(darwin_vm_stats.get("compressed_bytes", 0)),
        "iokit_bytes": int(darwin_vm_stats.get("iokit_bytes", 0)),
        "reusable_bytes": int(darwin_vm_stats.get("reusable_bytes", 0)),
        "graphics_footprint_bytes": int(darwin_vm_stats.get("graphics_footprint_bytes", 0)),
        "graphics_compressed_bytes": int(darwin_vm_stats.get("graphics_compressed_bytes", 0)),
        "media_footprint_bytes": int(darwin_vm_stats.get("media_footprint_bytes", 0)),
        "neural_footprint_bytes": int(darwin_vm_stats.get("neural_footprint_bytes", 0)),
        "footprint_peak_bytes": int(darwin_vm_stats.get("footprint_peak_bytes", 0)),
        "current_supported": bool(current_supported),
        "footprint_supported": bool(footprint_supported),
    }


def _format_bytes_short(byte_count: float | int) -> str:
    value = float(byte_count)
    if value >= 1024.0 * 1024.0:
        return f"{value / (1024.0 * 1024.0):4.1f}M"
    if value >= 1024.0:
        return f"{value / 1024.0:4.1f}K"
    return f"{int(value)}B"


def terrain_zstd_runtime_stats(renderer) -> dict[str, int | float | bool]:
    world = getattr(renderer, "world", None)
    terrain_backend_label = ""
    if world is not None:
        terrain_backend_label_fn = getattr(world, "terrain_backend_label", None)
        if callable(terrain_backend_label_fn):
            try:
                terrain_backend_label = str(terrain_backend_label_fn() or "").strip().lower()
            except Exception:
                terrain_backend_label = ""
    mesh_backend_label = str(getattr(renderer, "mesh_backend_label", "") or "").strip().lower()
    native_surface_bypass = bool(getattr(renderer, "use_gpu_meshing", False)) and (
        (terrain_backend_label == "wgpu" and mesh_backend_label == "wgpu")
        or (terrain_backend_label == "metal" and mesh_backend_label == "metal")
    )
    if world is not None and hasattr(world, "terrain_zstd_cache_stats"):
        cache_stats = world.terrain_zstd_cache_stats()
    else:
        cache_stats = {"enabled": False, "entries": 0, "raw_bytes": 0, "compressed_bytes": 0}
    seen_ids: set[int] = set()
    cache_values = getattr(getattr(world, "_terrain_zstd_cache", None), "values", None)
    if callable(cache_values):
        cache_byte_stats = compressed_chunk_voxel_results_stats(cache_values(), seen_ids=seen_ids)
        cache_raw_bytes = int(cache_byte_stats.get("raw_bytes", 0))
        cache_compressed_bytes = int(cache_byte_stats.get("compressed_bytes", 0))
    else:
        cache_raw_bytes = int(cache_stats.get("raw_bytes", 0))
        cache_compressed_bytes = int(cache_stats.get("compressed_bytes", 0))
    queue_stats = compressed_chunk_voxel_results_stats(
        getattr(renderer, "_pending_voxel_mesh_results", ()),
        seen_ids=seen_ids,
    )
    queue_raw_bytes = int(queue_stats.get("raw_bytes", 0))
    queue_compressed_bytes = int(queue_stats.get("compressed_bytes", 0))
    live_entries = (
        len(seen_ids)
        if callable(cache_values)
        else int(cache_stats.get("entries", 0)) + int(queue_stats.get("entries", 0))
    )
    stream_entries = float(frame_breakdown_average(renderer, "terrain_zstd_stream_entries"))
    stream_raw_bytes = float(frame_breakdown_average(renderer, "terrain_zstd_stream_raw_bytes"))
    stream_compressed_bytes = float(frame_breakdown_average(renderer, "terrain_zstd_stream_compressed_bytes"))
    total_entries = int(getattr(renderer, "_terrain_zstd_total_entries", 0))
    total_raw_bytes = int(getattr(renderer, "_terrain_zstd_total_raw_bytes", 0))
    total_compressed_bytes = int(getattr(renderer, "_terrain_zstd_total_compressed_bytes", 0))
    return {
        "enabled": bool(cache_stats.get("enabled", False)),
        "bypassed": bool(cache_stats.get("enabled", False)) and native_surface_bypass,
        "cache_entries": int(cache_stats.get("entries", 0)),
        "queue_entries": int(queue_stats.get("entries", 0)),
        "live_entries": live_entries,
        "cache_raw_bytes": cache_raw_bytes,
        "cache_compressed_bytes": cache_compressed_bytes,
        "queue_raw_bytes": queue_raw_bytes,
        "queue_compressed_bytes": queue_compressed_bytes,
        "raw_bytes": cache_raw_bytes + queue_raw_bytes,
        "compressed_bytes": cache_compressed_bytes + queue_compressed_bytes,
        "stream_entries": stream_entries,
        "stream_raw_bytes": stream_raw_bytes,
        "stream_compressed_bytes": stream_compressed_bytes,
        "total_entries": total_entries,
        "total_raw_bytes": total_raw_bytes,
        "total_compressed_bytes": total_compressed_bytes,
    }


def _numpy_array_nbytes(value, seen_ids: set[int]) -> int:
    if not isinstance(value, np.ndarray):
        return 0
    value_id = id(value)
    if value_id in seen_ids:
        return 0
    seen_ids.add(value_id)
    return int(value.nbytes)


def _chunk_result_array_bytes(result, seen_array_ids: set[int]) -> int:
    total = 0
    for name in ("blocks", "materials", "top_boundary", "bottom_boundary"):
        total += _numpy_array_nbytes(getattr(result, name, None), seen_array_ids)
    return total


def _pending_chunk_result_memory_stats(values, seen_array_ids: set[int]) -> dict[str, int]:
    entries = 0
    raw_entries = 0
    resident_bytes = 0
    logical_raw_bytes = 0
    for result in values or ():
        entries += 1
        if is_compressed_chunk_voxel_result(result):
            continue
        raw_entries += 1
        resident_bytes += _chunk_result_array_bytes(result, seen_array_ids)
        logical_raw_bytes += chunk_voxel_result_raw_nbytes(result)
    return {
        "entries": entries,
        "raw_entries": raw_entries,
        "resident_bytes": resident_bytes,
        "logical_raw_bytes": logical_raw_bytes,
    }


def _collision_cache_memory_stats(world, seen_array_ids: set[int]) -> dict[str, int]:
    cache = getattr(world, "_collision_block_chunk_cache", None)
    values = getattr(cache, "values", None)
    if not callable(values):
        return {"entries": 0, "bytes": 0, "limit": 0}
    total = 0
    entries = 0
    for blocks in values():
        entries += 1
        total += _numpy_array_nbytes(blocks, seen_array_ids)
    return {
        "entries": entries,
        "bytes": total,
        "limit": int(getattr(world, "_collision_block_chunk_cache_limit", 0)),
    }


def _direct_numpy_attr_memory_stats(owners, seen_array_ids: set[int]) -> dict[str, int]:
    total = 0
    entries = 0
    for owner in owners:
        try:
            values = vars(owner).values()
        except TypeError:
            continue
        for value in values:
            before = total
            total += _numpy_array_nbytes(value, seen_array_ids)
            if total != before:
                entries += 1
    return {"entries": entries, "bytes": total}


def _wgpu_mesher_scratch_gpu_bytes(renderer) -> int:
    capacity = int(getattr(renderer, "_voxel_mesh_scratch_capacity", 0) or 0)
    sample_size = int(getattr(renderer, "_voxel_mesh_scratch_sample_size", 0) or 0)
    height_limit = int(getattr(renderer, "_voxel_mesh_scratch_height_limit", 0) or 0)
    if capacity <= 0 or sample_size <= 0 or height_limit <= 0:
        return 0
    storage_height = height_limit + 2
    columns_per_chunk = max(1, (sample_size - 2) * (sample_size - 2))
    blocks_bytes = capacity * storage_height * sample_size * sample_size * 4
    coords_bytes = capacity * 4 * 4
    column_totals_bytes = capacity * columns_per_chunk * 4
    chunk_totals_bytes = capacity * 4
    return (
        blocks_bytes * 2
        + coords_bytes
        + column_totals_bytes
        + chunk_totals_bytes * 2
        + max(8, capacity * 8)
        + 32
        + int(_renderer_module().VERTEX_STRIDE)
    )


def _async_voxel_resource_gpu_bytes(resources) -> int:
    chunk_capacity = int(getattr(resources, "chunk_capacity", 0) or 0)
    sample_size = int(getattr(resources, "sample_size", 0) or 0)
    height_limit = int(getattr(resources, "height_limit", 0) or 0)
    column_capacity = int(getattr(resources, "column_capacity", 0) or 0)
    if chunk_capacity <= 0 or sample_size <= 0 or height_limit <= 0:
        return 0
    storage_height = height_limit + 2
    blocks_bytes = chunk_capacity * storage_height * sample_size * sample_size * 4
    coords_bytes = chunk_capacity * 4 * 4
    chunk_totals_bytes = chunk_capacity * 4
    return (
        blocks_bytes * 2
        + coords_bytes
        + column_capacity * 4
        + chunk_totals_bytes * 2
        + 32
        + 32
        + max(4, chunk_totals_bytes)
        + int(_renderer_module().VERTEX_STRIDE)
    )


def _pending_surface_gpu_bytes(renderer) -> int:
    total = 0
    for batch in getattr(renderer, "_pending_surface_gpu_batches", ()) or ():
        total += int(len(getattr(batch, "chunks", ()) or ())) * int(getattr(batch, "cell_count", 0) or 0) * 8
    return total


def _visibility_gpu_bytes(renderer) -> int:
    total = int(getattr(renderer, "_mesh_draw_indirect_capacity", 0) or 0) * 16
    visibility_capacity = int(getattr(renderer, "_mesh_visibility_record_capacity", 0) or 0)
    visibility_array = getattr(renderer, "_mesh_visibility_record_array", None)
    itemsize = int(getattr(getattr(visibility_array, "dtype", None), "itemsize", 0) or 0)
    total += visibility_capacity * itemsize
    return total


def _tile_render_batch_gpu_stats(renderer) -> dict[str, int]:
    owned_bytes = 0
    owned_entries = 0
    for batch in getattr(renderer, "_tile_render_batches", {}).values():
        if not getattr(batch, "owns_vertex_buffer", False) or getattr(batch, "vertex_buffer", None) is None:
            continue
        owned_entries += 1
        owned_bytes += int(
            getattr(batch, "owned_vertex_buffer_capacity_bytes", 0)
            or (int(getattr(batch, "vertex_count", 0)) * int(_renderer_module().VERTEX_STRIDE))
        )

    pool_bytes = 0
    pool_entries = 0
    for bucket_bytes, buffers in (getattr(renderer, "_merged_tile_buffer_pool", None) or {}).items():
        entry_count = len(buffers or ())
        pool_entries += entry_count
        pool_bytes += int(bucket_bytes) * entry_count

    reuse_bytes = 0
    reuse_entries = 0
    for entries in getattr(renderer, "_merged_tile_buffer_reuse_queue", None) or ():
        for _buffer, capacity_bytes in entries:
            reuse_entries += 1
            reuse_bytes += int(capacity_bytes)

    return {
        "owned_entries": int(owned_entries),
        "owned_bytes": int(owned_bytes),
        "pool_entries": int(pool_entries),
        "pool_bytes": int(pool_bytes),
        "reuse_entries": int(reuse_entries),
        "reuse_bytes": int(reuse_bytes),
        "bytes": int(owned_bytes + pool_bytes + reuse_bytes),
    }


def engine_memory_breakdown_stats(
    renderer,
    *,
    process_memory: dict[str, int | bool] | None = None,
    terrain_zstd: dict[str, int | float | bool] | None = None,
    slab_stats: tuple[int, int, int, int, int, int] | None = None,
) -> dict[str, int]:
    process_memory = process_memory_stats() if process_memory is None else process_memory
    terrain_zstd = terrain_zstd_runtime_stats(renderer) if terrain_zstd is None else terrain_zstd
    mesh_zstd_stats = mesh_zstd.mesh_zstd_runtime_stats(renderer)
    tile_zstd_stats = tile_zstd.tile_zstd_runtime_stats(renderer)
    mesh_compaction_stats = mesh_cache.mesh_output_compaction_stats(renderer)
    tile_gpu_stats = _tile_render_batch_gpu_stats(renderer)
    if slab_stats is None:
        slab_stats = mesh_cache.mesh_output_allocator_stats(renderer)
    slab_count, slab_total_bytes, slab_used_bytes, slab_free_bytes, slab_largest_free_bytes, slab_alloc_count = slab_stats

    seen_arrays: set[int] = set()
    world = getattr(renderer, "world", None)
    pending_voxel_stats = _pending_chunk_result_memory_stats(getattr(renderer, "_pending_voxel_mesh_results", ()), seen_arrays)
    pending_gpu_cpu_bytes = 0
    pending_gpu_raw_entries = 0
    pending_gpu_logical_bytes = 0
    for pending in getattr(renderer, "_pending_gpu_mesh_batches", ()) or ():
        chunk_results = getattr(pending, "chunk_results", None)
        if chunk_results:
            stats = _pending_chunk_result_memory_stats(chunk_results, seen_arrays)
            pending_gpu_cpu_bytes += int(stats["resident_bytes"])
            pending_gpu_raw_entries += int(stats["raw_entries"])
            pending_gpu_logical_bytes += int(stats["logical_raw_bytes"])
    collision_stats = _collision_cache_memory_stats(world, seen_arrays)
    backend = getattr(world, "_backend", None)
    scratch_owners = [renderer, world, backend]
    scratch_owners.extend(getattr(renderer, "_async_voxel_mesh_batch_pool", ()) or ())
    for pending in getattr(renderer, "_pending_gpu_mesh_batches", ()) or ():
        resources = getattr(pending, "resources", None)
        if resources is not None:
            scratch_owners.append(resources)
    scratch_stats = _direct_numpy_attr_memory_stats(scratch_owners, seen_arrays)

    terrain_payload_bytes = (
        int(terrain_zstd["cache_compressed_bytes"])
        + int(terrain_zstd["queue_compressed_bytes"])
        + int(pending_voxel_stats["resident_bytes"])
        + pending_gpu_cpu_bytes
    )
    mesh_zstd_cpu_bytes = int(mesh_zstd_stats["cache_compressed_bytes"])
    tile_zstd_cpu_bytes = int(tile_zstd_stats["cache_compressed_bytes"])
    tracked_cpu_bytes = terrain_payload_bytes + mesh_zstd_cpu_bytes + tile_zstd_cpu_bytes + int(collision_stats["bytes"]) + int(scratch_stats["bytes"])
    rss_bytes = int(process_memory["rss_bytes"])
    footprint_bytes = int(process_memory.get("footprint_bytes", rss_bytes))
    other_rss_bytes = max(0, rss_bytes - tracked_cpu_bytes)
    other_footprint_bytes = max(0, footprint_bytes - tracked_cpu_bytes)

    async_pool_gpu_bytes = sum(
        _async_voxel_resource_gpu_bytes(resources)
        for resources in getattr(renderer, "_async_voxel_mesh_batch_pool", ()) or ()
    )
    pending_async_gpu_bytes = sum(
        _async_voxel_resource_gpu_bytes(getattr(pending, "resources", None))
        for pending in getattr(renderer, "_pending_gpu_mesh_batches", ()) or ()
    )
    gpu_transient_bytes = (
        _wgpu_mesher_scratch_gpu_bytes(renderer)
        + async_pool_gpu_bytes
        + pending_async_gpu_bytes
        + _pending_surface_gpu_bytes(renderer)
        + _visibility_gpu_bytes(renderer)
        + int(mesh_zstd_stats["pending_raw_bytes"])
        + int(tile_zstd_stats["pending_raw_bytes"])
        + int(tile_zstd_stats["pending_source_bytes"])
        + int(mesh_compaction_stats["pending_retired_bytes"])
    )

    return {
        "rss_bytes": rss_bytes,
        "footprint_bytes": footprint_bytes,
        "tracked_cpu_bytes": tracked_cpu_bytes,
        "other_rss_bytes": other_rss_bytes,
        "other_footprint_bytes": other_footprint_bytes,
        "terrain_payload_cpu_bytes": terrain_payload_bytes,
        "terrain_payload_raw_logical_bytes": int(terrain_zstd["raw_bytes"]) + int(pending_voxel_stats["logical_raw_bytes"]) + pending_gpu_logical_bytes,
        "terrain_zstd_cpu_bytes": int(terrain_zstd["compressed_bytes"]),
        "terrain_raw_queue_cpu_bytes": int(pending_voxel_stats["resident_bytes"]) + pending_gpu_cpu_bytes,
        "terrain_raw_queue_entries": int(pending_voxel_stats["raw_entries"]) + pending_gpu_raw_entries,
        "mesh_zstd_cpu_bytes": mesh_zstd_cpu_bytes,
        "mesh_zstd_raw_bytes": int(mesh_zstd_stats["cache_raw_bytes"]),
        "tile_zstd_cpu_bytes": tile_zstd_cpu_bytes,
        "tile_zstd_raw_bytes": int(tile_zstd_stats["cache_raw_bytes"]),
        "collision_cpu_bytes": int(collision_stats["bytes"]),
        "collision_entries": int(collision_stats["entries"]),
        "collision_limit": int(collision_stats["limit"]),
        "scratch_numpy_cpu_bytes": int(scratch_stats["bytes"]),
        "scratch_numpy_entries": int(scratch_stats["entries"]),
        "mesh_slab_count": int(slab_count),
        "mesh_slab_total_bytes": int(slab_total_bytes),
        "mesh_slab_used_bytes": int(slab_used_bytes),
        "mesh_slab_free_bytes": int(slab_free_bytes),
        "mesh_slab_largest_free_bytes": int(slab_largest_free_bytes),
        "mesh_slab_alloc_count": int(slab_alloc_count),
        "tile_render_gpu_bytes": int(tile_gpu_stats["bytes"]),
        "tile_render_owned_gpu_bytes": int(tile_gpu_stats["owned_bytes"]),
        "tile_render_reuse_gpu_bytes": int(tile_gpu_stats["pool_bytes"]) + int(tile_gpu_stats["reuse_bytes"]),
        "tile_render_owned_entries": int(tile_gpu_stats["owned_entries"]),
        "tile_render_reuse_entries": int(tile_gpu_stats["pool_entries"]) + int(tile_gpu_stats["reuse_entries"]),
        "chunk_mesh_gpu_bytes": int(mesh_cache.chunk_cache_memory_bytes(renderer)),
        "gpu_transient_bytes": int(gpu_transient_bytes),
        "mesh_compaction_pending_retired_bytes": int(mesh_compaction_stats["pending_retired_bytes"]),
        "gpu_estimated_bytes": int(slab_total_bytes) + int(tile_gpu_stats["bytes"]) + int(gpu_transient_bytes),
    }


def refresh_profile_summary(renderer, now: float) -> None:
    renderer_module = _renderer_module()
    avg_cpu_ms = renderer.profile_window_cpu_ms / max(1, renderer.profile_window_frames)
    avg_fps = profile_average_fps(renderer)
    frame_p50_ms, frame_p95_ms, frame_p99_ms = profile_frame_time_percentiles(renderer)

    slab_stats = mesh_cache.mesh_output_allocator_stats(renderer)
    slab_count, slab_total_bytes, slab_used_bytes, slab_free_bytes, slab_largest_free_bytes, slab_alloc_count = slab_stats
    process_memory = process_memory_stats()
    terrain_zstd = terrain_zstd_runtime_stats(renderer)
    mesh_zstd_stats = mesh_zstd.mesh_zstd_runtime_stats(renderer)
    tile_zstd_stats = tile_zstd.tile_zstd_runtime_stats(renderer)
    mesh_compaction_stats = mesh_cache.mesh_output_compaction_stats(renderer)
    pressure_stats = memory_pressure_stats(renderer)
    memory = engine_memory_breakdown_stats(
        renderer,
        process_memory=process_memory,
        terrain_zstd=terrain_zstd,
        slab_stats=slab_stats,
    )
    terrain_zstd_stream_ratio = (
        float(terrain_zstd["stream_raw_bytes"]) / max(1.0, float(terrain_zstd["stream_compressed_bytes"]))
        if float(terrain_zstd["stream_compressed_bytes"]) > 0.0
        else 0.0
    )
    terrain_zstd_total_ratio = (
        float(terrain_zstd["total_raw_bytes"]) / max(1.0, float(terrain_zstd["total_compressed_bytes"]))
        if int(terrain_zstd["total_compressed_bytes"]) > 0
        else 0.0
    )
    terrain_zstd_status = "BYPASS" if terrain_zstd["bypassed"] else ("ON" if terrain_zstd["enabled"] else "OFF")
    mesh_zstd_ratio = (
        float(mesh_zstd_stats["cache_raw_bytes"]) / max(1.0, float(mesh_zstd_stats["cache_compressed_bytes"]))
        if int(mesh_zstd_stats["cache_compressed_bytes"]) > 0
        else 0.0
    )
    tile_zstd_ratio = (
        float(tile_zstd_stats["cache_raw_bytes"]) / max(1.0, float(tile_zstd_stats["cache_compressed_bytes"]))
        if int(tile_zstd_stats["cache_compressed_bytes"]) > 0
        else 0.0
    )
    lines = [
        f"AVG FPS {avg_fps:5.1f}  CPU {avg_cpu_ms:5.1f}MS",
        f"FRAME P50 {frame_p50_ms:5.1f}MS  P95 {frame_p95_ms:5.1f}MS  P99 {frame_p99_ms:5.1f}MS",
        f"RENDER API  {renderer.render_api_label}",
        f"RENDER BACKEND {renderer.render_backend_label}",
        f"ENGINE MODE {renderer.engine_mode_label}",
        f"PRESENT     FPS {renderer_module.SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if renderer_module.SWAPCHAIN_USE_VSYNC else 'OFF'}",
        f"PROCESS MEM FOOT {_format_bytes_short(int(process_memory['footprint_bytes']))}  RSS {_format_bytes_short(int(process_memory['rss_bytes']))}  PEAK {_format_bytes_short(int(process_memory['peak_rss_bytes']))}",
        f"MEM MAC INT {_format_bytes_short(int(process_memory['internal_bytes']))}  IO {_format_bytes_short(int(process_memory['iokit_bytes']))}  GFX {_format_bytes_short(int(process_memory['graphics_footprint_bytes']))}  REUSE {_format_bytes_short(int(process_memory['reusable_bytes']))}  COMP {_format_bytes_short(int(process_memory['compressed_bytes']))}  RELIEF {_format_bytes_short(int(pressure_stats['last_relief_bytes']))}",
        f"MEM CPU TRACK {_format_bytes_short(memory['tracked_cpu_bytes'])}  TERR {_format_bytes_short(memory['terrain_payload_cpu_bytes'])}  COLL {_format_bytes_short(memory['collision_cpu_bytes'])}  OTHER {_format_bytes_short(memory['other_footprint_bytes'])}",
        f"MEM GPU EST {_format_bytes_short(memory['gpu_estimated_bytes'])}  SLABS {_format_bytes_short(memory['mesh_slab_total_bytes'])}  TILE {_format_bytes_short(memory['tile_render_gpu_bytes'])}  TRANS {_format_bytes_short(memory['gpu_transient_bytes'])}",
        f"MESH ZSTD {'ON' if mesh_zstd_stats['enabled'] else 'OFF'}  CACHE {mesh_zstd_stats['cache_entries']}/{mesh_zstd_stats['cache_limit']} {_format_bytes_short(mesh_zstd_stats['cache_raw_bytes'])}->{_format_bytes_short(mesh_zstd_stats['cache_compressed_bytes'])} {mesh_zstd_ratio:4.1f}X  PENDING {mesh_zstd_stats['pending_entries']}/{_format_bytes_short(mesh_zstd_stats['pending_raw_bytes'])}",
        f"TILE ZSTD {'ON' if tile_zstd_stats['enabled'] else 'OFF'}  CACHE {tile_zstd_stats['cache_entries']}/{tile_zstd_stats['cache_limit']} {_format_bytes_short(tile_zstd_stats['cache_raw_bytes'])}->{_format_bytes_short(tile_zstd_stats['cache_compressed_bytes'])} {tile_zstd_ratio:4.1f}X  PENDING {tile_zstd_stats['pending_entries']}/{_format_bytes_short(tile_zstd_stats['pending_raw_bytes'])}",
        f"MESH COMPACT SLABS {mesh_compaction_stats['source_slabs']}  {_format_bytes_short(mesh_compaction_stats['retired_slab_bytes'])}->{_format_bytes_short(mesh_compaction_stats['new_slab_bytes'])}  RECLAIM {_format_bytes_short(mesh_compaction_stats['net_reclaimed_bytes'])}  TOTAL {_format_bytes_short(mesh_compaction_stats['before_slab_bytes'])}->{_format_bytes_short(mesh_compaction_stats['after_slab_bytes'])}  COPY {_format_bytes_short(mesh_compaction_stats['copied_bytes'])}  PEND {_format_bytes_short(mesh_compaction_stats['pending_retired_bytes'])}",
        f"TERRAIN ZSTD {terrain_zstd_status}  LIVE {terrain_zstd['live_entries']}  CACHE {terrain_zstd['cache_entries']}/{_format_bytes_short(float(terrain_zstd['cache_compressed_bytes']))}  QUEUE {terrain_zstd['queue_entries']}/{_format_bytes_short(float(terrain_zstd['queue_compressed_bytes']))}",
        f"ZSTD STREAM {float(terrain_zstd['stream_entries']):4.1f}/F  {_format_bytes_short(float(terrain_zstd['stream_raw_bytes']))}->{_format_bytes_short(float(terrain_zstd['stream_compressed_bytes']))}  {terrain_zstd_stream_ratio:4.1f}X",
        f"ZSTD TOTAL {int(terrain_zstd['total_entries'])}  {_format_bytes_short(int(terrain_zstd['total_raw_bytes']))}->{_format_bytes_short(int(terrain_zstd['total_compressed_bytes']))}  {terrain_zstd_total_ratio:4.1f}X",
        f"MESH SLABS {slab_count:2d}  USED {slab_used_bytes / (1024.0 * 1024.0):4.1f} MIB  FREE {slab_free_bytes / (1024.0 * 1024.0):4.1f} MIB",
        f"MESH BIGGEST GAP {slab_largest_free_bytes / (1024.0 * 1024.0):4.1f} MIB  ALLOCS {slab_alloc_count:3d}",
    ]
    if lines != renderer.profile_hud_lines or renderer.profile_hud_vertex_count <= 0:
        renderer.profile_hud_lines = lines
        renderer.profile_hud_vertex_bytes, renderer.profile_hud_vertex_count = build_profile_hud_vertices(renderer, lines)
        _ensure_hud_vertex_buffer(renderer, "profile_hud", renderer.profile_hud_vertex_bytes)
    renderer.profile_window_start = now
    renderer.profile_next_report = now + renderer_module.PROFILE_REPORT_INTERVAL
    renderer.profile_window_cpu_ms = 0.0
    renderer.profile_window_frames = 0
    renderer.profile_window_frame_times = []


def refresh_frame_breakdown_summary(renderer, now: float | None = None) -> None:
    if not renderer.profiling_enabled:
        return
    if now is None:
        now = time.perf_counter()
    next_refresh = float(getattr(renderer, "_frame_breakdown_next_refresh", 0.0))
    if renderer.frame_breakdown_vertex_count > 0 and now < next_refresh:
        return
    renderer._frame_breakdown_next_refresh = float(now) + 0.2
    renderer_module = _renderer_module()
    avg_world_update = frame_breakdown_average(renderer, "world_update")
    avg_visibility_lookup = frame_breakdown_average(renderer, "visibility_lookup")
    avg_chunk_stream = frame_breakdown_average(renderer, "chunk_stream")
    avg_camera_upload = frame_breakdown_average(renderer, "camera_upload")
    avg_swapchain_acquire = frame_breakdown_average(renderer, "swapchain_acquire")
    avg_render_encode = frame_breakdown_average(renderer, "render_encode")
    avg_command_finish = frame_breakdown_average(renderer, "command_finish")
    avg_queue_submit = frame_breakdown_average(renderer, "queue_submit")
    avg_wall_frame = frame_breakdown_average(renderer, "wall_frame")
    pending_chunk_requests = int(round(frame_breakdown_average(renderer, "pending_chunk_requests")))
    visible_vertices = int(round(frame_breakdown_average(renderer, "visible_vertices")))
    avg_issue_encode = (
        avg_world_update
        + avg_visibility_lookup
        + avg_chunk_stream
        + avg_camera_upload
        + avg_swapchain_acquire
        + avg_render_encode
        + avg_command_finish
        + avg_queue_submit
    )
    draw_calls = int(round(frame_breakdown_average(renderer, "draw_calls")))
    merged_chunks = int(round(frame_breakdown_average(renderer, "merged_chunks")))
    visible_chunk_targets = int(round(frame_breakdown_average(renderer, "visible_chunk_targets")))
    visible_chunks = int(round(frame_breakdown_average(renderer, "visible_chunks")))
    visible_but_not_ready = max(0, visible_chunk_targets - visible_chunks)
    process_memory = process_memory_stats()
    terrain_zstd = terrain_zstd_runtime_stats(renderer)
    mesh_zstd_stats = mesh_zstd.mesh_zstd_runtime_stats(renderer)
    tile_zstd_stats = tile_zstd.tile_zstd_runtime_stats(renderer)
    mesh_compaction_stats = mesh_cache.mesh_output_compaction_stats(renderer)
    pressure_stats = memory_pressure_stats(renderer)
    terrain_zstd_stream_ratio = (
        float(terrain_zstd["stream_raw_bytes"]) / max(1.0, float(terrain_zstd["stream_compressed_bytes"]))
        if float(terrain_zstd["stream_compressed_bytes"]) > 0.0
        else 0.0
    )
    terrain_zstd_total_ratio = (
        float(terrain_zstd["total_raw_bytes"]) / max(1.0, float(terrain_zstd["total_compressed_bytes"]))
        if int(terrain_zstd["total_compressed_bytes"]) > 0
        else 0.0
    )
    terrain_zstd_status = "BYPASS" if terrain_zstd["bypassed"] else ("ON" if terrain_zstd["enabled"] else "OFF")
    mesh_zstd_ratio = (
        float(mesh_zstd_stats["cache_raw_bytes"]) / max(1.0, float(mesh_zstd_stats["cache_compressed_bytes"]))
        if int(mesh_zstd_stats["cache_compressed_bytes"]) > 0
        else 0.0
    )
    tile_zstd_ratio = (
        float(tile_zstd_stats["cache_raw_bytes"]) / max(1.0, float(tile_zstd_stats["cache_compressed_bytes"]))
        if int(tile_zstd_stats["cache_compressed_bytes"]) > 0
        else 0.0
    )
    slab_stats = mesh_cache.mesh_output_allocator_stats(renderer)
    slab_count, slab_total_bytes, slab_used_bytes, slab_free_bytes, slab_largest_free_bytes, slab_alloc_count = slab_stats
    memory = engine_memory_breakdown_stats(
        renderer,
        process_memory=process_memory,
        terrain_zstd=terrain_zstd,
        slab_stats=slab_stats,
    )

    camera_x = float(renderer.camera.position[0])
    camera_y = float(renderer.camera.position[1])
    camera_z = float(renderer.camera.position[2])
    camera_block_x = camera_x / max(renderer.world.block_size, 1e-9)
    camera_block_y = camera_y / max(renderer.world.block_size, 1e-9)
    camera_block_z = camera_z / max(renderer.world.block_size, 1e-9)

    rc_debug_mode = int(getattr(renderer, "rc_debug_mode", 0))
    rc_debug_names = tuple(getattr(renderer, "rc_debug_mode_names", ("off",)))
    rc_debug_name = rc_debug_names[rc_debug_mode] if 0 <= rc_debug_mode < len(rc_debug_names) else "unknown"
    rc_frame = int(getattr(renderer, "_worldspace_rc_frame_index", 0))
    rc_burst = int(getattr(renderer, "_worldspace_rc_convergence_frames_remaining", 0))
    rc_kind = str(getattr(renderer, "_worldspace_rc_last_update_kind", "unknown"))

    def _fmt_cascade_list(values) -> str:
        vals = [int(v) for v in list(values or [])]
        return "-" if not vals else ",".join(f"C{v}" for v in vals)

    rc_scheduled = _fmt_cascade_list(getattr(renderer, "_worldspace_rc_last_scheduled_updates", []))
    rc_dirty = _fmt_cascade_list(getattr(renderer, "_worldspace_rc_last_dirty_indices", []))
    rc_reject = _fmt_cascade_list(getattr(renderer, "_worldspace_rc_last_history_reject_updates", []))
    rc_last_updates = list(getattr(renderer, "_worldspace_rc_last_update_frame", []))
    rc_ages: list[str] = []
    for idx in range(4):
        try:
            last_update = int(rc_last_updates[idx])
        except Exception:
            last_update = -1000000
        rc_ages.append("--" if last_update < -999000 else str(max(0, rc_frame - last_update)))
    rc_age_text = "/".join(rc_ages)
    rc_resolution = int(getattr(render_consts, "WORLDSPACE_RC_GRID_RESOLUTION", 0))
    rc_directions = int(getattr(render_consts, "WORLDSPACE_RC_DIRECTION_COUNT", 0))
    rc_filter_passes = int(getattr(render_consts, "WORLDSPACE_RC_SPATIAL_FILTER_PASSES", 0))
    rc_temporal_alpha = float(getattr(render_consts, "WORLDSPACE_RC_TEMPORAL_BLEND_ALPHA", 0.0))
    rc_interval_bands = list(getattr(renderer, "_worldspace_rc_last_interval_bands", []))
    if len(rc_interval_bands) < 4:
        rc_interval_bands = [worldspace_rc.interval_band(i) for i in range(4)]
    rc_interval_text = " ".join(
        f"C{idx}:{float(band[0]):.1f}-{float(band[1]):.1f}"
        for idx, band in enumerate(rc_interval_bands[:4])
    )
    rc_snapshot_path = str(getattr(renderer, "_worldspace_rc_last_snapshot_path", "") or "-")
    if len(rc_snapshot_path) > 54:
        rc_snapshot_path = "..." + rc_snapshot_path[-51:]

    lines = [
        f"FRAME BREAKDOWN @ DIMENSION {renderer.render_dimension_chunks}x{renderer.render_dimension_chunks} CHUNKS",
        f"MOVE SPEED: {renderer._current_move_speed / max(renderer.world.block_size, 1e-9):5.1f} B/S",
        f"CAM POS M: {camera_x:7.2f} {camera_y:7.2f} {camera_z:7.2f}",
        f"CAM POS B: {camera_block_x:7.1f} {camera_block_y:7.1f} {camera_block_z:7.1f}",
        f"RENDER BACKEND: {renderer.render_backend_label}",
        f"TERRAIN BACKEND: {renderer.world.terrain_backend_label()}",
        f"MESH BACKEND: {renderer.mesh_backend_label}",
        f"CHUNK DIMS: {renderer_module.CHUNK_SIZE}x{renderer_module.CHUNK_SIZE}x{renderer_module.CHUNK_SIZE}",
        f"BACKEND POLL SIZE: {renderer.terrain_batch_size}",
        f"MESH DRAIN SIZE: {renderer.mesh_batch_size}",
        f"RC: {'ON' if renderer.radiance_cascades_enabled else 'OFF'} DEBUG {rc_debug_mode}:{rc_debug_name}",
        f"RC FIELD: RES {rc_resolution} DIRS {rc_directions} FILTER {rc_filter_passes} TEMP {rc_temporal_alpha:.2f}",
        f"RC UPDATE: {rc_kind.upper()} SCHED {rc_scheduled} DIRTY {rc_dirty} REJECT {rc_reject} BURST {rc_burst}",
        f"RC AGE C0/C1/C2/C3: {rc_age_text} FRAMES",
        f"RC INTERVALS: {rc_interval_text}",
        f"RC SNAPSHOT F7: {rc_snapshot_path}",
        f"MESH SLABS: {slab_count}  USED {slab_used_bytes / (1024.0 * 1024.0):4.1f} MIB  FREE {slab_free_bytes / (1024.0 * 1024.0):4.1f} MIB",
        f"MESH BIGGEST GAP: {slab_largest_free_bytes / (1024.0 * 1024.0):4.1f} MIB  ALLOCS {slab_alloc_count}",
        f"CPU FRAME ISSUE: {avg_issue_encode:5.1f} MS",
        f"  WORLD UPDATE: {avg_world_update:5.1f} MS",
        f"  VISIBILITY LOOKUP: {avg_visibility_lookup:5.1f} MS",
        f"  CHUNK STREAM: {avg_chunk_stream:5.1f} MS",
        f"  CAMERA UPLOAD: {avg_camera_upload:5.1f} MS",
        f"  SWAPCHAIN ACQUIRE: {avg_swapchain_acquire:5.1f} MS",
        f"  RENDER ENCODE: {avg_render_encode:5.1f} MS",
        f"  COMMAND FINISH: {avg_command_finish:5.1f} MS",
        f"  QUEUE SUBMIT: {avg_queue_submit:5.1f} MS",
        f"WALL FRAME: {avg_wall_frame:5.1f} MS",
        f"PROCESS MEM: FOOT {_format_bytes_short(int(process_memory['footprint_bytes']))}  RSS {_format_bytes_short(int(process_memory['rss_bytes']))}  PEAK {_format_bytes_short(int(process_memory['peak_rss_bytes']))}",
        f"MEM MAC: INT {_format_bytes_short(int(process_memory['internal_bytes']))}  IO {_format_bytes_short(int(process_memory['iokit_bytes']))}  GFX {_format_bytes_short(int(process_memory['graphics_footprint_bytes']))}  REUSE {_format_bytes_short(int(process_memory['reusable_bytes']))}  COMP {_format_bytes_short(int(process_memory['compressed_bytes']))}  RELIEF {_format_bytes_short(int(pressure_stats['last_relief_bytes']))}",
        f"MEM CPU: TRACK {_format_bytes_short(memory['tracked_cpu_bytes'])}  TERR {_format_bytes_short(memory['terrain_payload_cpu_bytes'])}  COLL {memory['collision_entries']}/{_format_bytes_short(memory['collision_cpu_bytes'])}  SCR {_format_bytes_short(memory['scratch_numpy_cpu_bytes'])}",
        f"MEM CPU: OTHER FOOT {_format_bytes_short(memory['other_footprint_bytes'])}  OTHER RSS {_format_bytes_short(memory['other_rss_bytes'])}  RAWQ {memory['terrain_raw_queue_entries']}/{_format_bytes_short(memory['terrain_raw_queue_cpu_bytes'])}",
        f"MEM GPU: EST {_format_bytes_short(memory['gpu_estimated_bytes'])}  SLABS {_format_bytes_short(memory['mesh_slab_total_bytes'])}  TILE {_format_bytes_short(memory['tile_render_gpu_bytes'])}  TRANS {_format_bytes_short(memory['gpu_transient_bytes'])}",
        f"MESH ZSTD: {'ON' if mesh_zstd_stats['enabled'] else 'OFF'}  CACHE {mesh_zstd_stats['cache_entries']}/{mesh_zstd_stats['cache_limit']} RAW {_format_bytes_short(mesh_zstd_stats['cache_raw_bytes'])} COMP {_format_bytes_short(mesh_zstd_stats['cache_compressed_bytes'])} RATIO {mesh_zstd_ratio:4.1f}X  PENDING {mesh_zstd_stats['pending_entries']}/{_format_bytes_short(mesh_zstd_stats['pending_raw_bytes'])}",
        f"TILE ZSTD: {'ON' if tile_zstd_stats['enabled'] else 'OFF'}  CACHE {tile_zstd_stats['cache_entries']}/{tile_zstd_stats['cache_limit']} RAW {_format_bytes_short(tile_zstd_stats['cache_raw_bytes'])} COMP {_format_bytes_short(tile_zstd_stats['cache_compressed_bytes'])} RATIO {tile_zstd_ratio:4.1f}X  PENDING {tile_zstd_stats['pending_entries']}/{_format_bytes_short(tile_zstd_stats['pending_raw_bytes'])}",
        f"MESH COMPACT: SLABS {mesh_compaction_stats['source_slabs']}  {_format_bytes_short(mesh_compaction_stats['retired_slab_bytes'])}->{_format_bytes_short(mesh_compaction_stats['new_slab_bytes'])}  RECLAIM {_format_bytes_short(mesh_compaction_stats['net_reclaimed_bytes'])}  TOTAL {_format_bytes_short(mesh_compaction_stats['before_slab_bytes'])}->{_format_bytes_short(mesh_compaction_stats['after_slab_bytes'])}  COPY {_format_bytes_short(mesh_compaction_stats['copied_bytes'])}  PEND {_format_bytes_short(mesh_compaction_stats['pending_retired_bytes'])}",
        f"TERRAIN ZSTD: {terrain_zstd_status}  LIVE {terrain_zstd['live_entries']}  CACHE {terrain_zstd['cache_entries']}/{_format_bytes_short(float(terrain_zstd['cache_compressed_bytes']))}  QUEUE {terrain_zstd['queue_entries']}/{_format_bytes_short(float(terrain_zstd['queue_compressed_bytes']))}",
        f"ZSTD STREAM: {float(terrain_zstd['stream_entries']):4.1f}/F  RAW {_format_bytes_short(float(terrain_zstd['stream_raw_bytes']))}  COMP {_format_bytes_short(float(terrain_zstd['stream_compressed_bytes']))}  RATIO {terrain_zstd_stream_ratio:4.1f}X",
        f"ZSTD TOTAL: {int(terrain_zstd['total_entries'])}  RAW {_format_bytes_short(int(terrain_zstd['total_raw_bytes']))}  COMP {_format_bytes_short(int(terrain_zstd['total_compressed_bytes']))}  RATIO {terrain_zstd_total_ratio:4.1f}X",
        f"TOTAL DRAW VERTICES: {visible_vertices:,}",
        f"VISIBLE BUT NOT READY: {visible_but_not_ready}",
        f"PENDING CHUNK REQUESTS: {pending_chunk_requests}",
        f"DRAW CALLS: {draw_calls}",
        f"VISIBLE MERGED CHUNKS (VISIBLE ONLY): {merged_chunks}",
    ]
    if lines != renderer.frame_breakdown_lines or renderer.frame_breakdown_vertex_count <= 0:
        renderer.frame_breakdown_lines = lines
        renderer.frame_breakdown_vertex_bytes, renderer.frame_breakdown_vertex_count = build_frame_breakdown_hud_vertices(renderer, lines)
        _ensure_hud_vertex_buffer(renderer, "frame_breakdown", renderer.frame_breakdown_vertex_bytes)
