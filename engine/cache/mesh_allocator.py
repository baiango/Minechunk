from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_left
import math
import struct
import time

import numpy as np
import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkDrawBatch, ChunkMesh, ChunkRenderBatch, MeshBufferAllocation, MeshOutputSlab
from ..meshing import gpu_mesher as wgpu_mesher
from ..visibility import coord_manager


MERGED_TILE_AGE_REFRESH_INTERVAL_SECONDS = 0.25

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


def _renderer_module():
    """Return the small render contract used by cache/allocator helpers.

    Importing ``engine.renderer`` from here creates a reverse dependency on the
    god-object runtime module.  The allocator only needs stable constants, so keep
    it pointed at ``render_contract`` instead.
    """
    return render_consts


def chunk_cache_memory_bytes(renderer) -> int:
    renderer_module = _renderer_module()
    vertex_stride = int(renderer_module.VERTEX_STRIDE)

    allocation_bytes: dict[int, int] = {}
    fallback_buffer_bytes: dict[int, int] = {}

    for mesh in renderer.chunk_cache.values():
        if mesh.allocation_id is not None:
            allocation = renderer._mesh_allocations.get(mesh.allocation_id)
            if allocation is not None:
                allocation_bytes[allocation.allocation_id] = int(allocation.size_bytes)
                continue
        key = id(mesh.vertex_buffer)
        fallback_buffer_bytes[key] = max(
            fallback_buffer_bytes.get(key, 0),
            mesh.vertex_offset + mesh.vertex_count * vertex_stride,
        )

    return sum(allocation_bytes.values()) + sum(fallback_buffer_bytes.values())


def mesh_output_allocator_stats(renderer) -> tuple[int, int, int, int, int, int]:
    slab_count = len(renderer._mesh_output_slabs)
    total_bytes = 0
    free_bytes = 0
    largest_free_bytes = 0
    for slab in renderer._mesh_output_slabs.values():
        total_bytes += int(slab.size_bytes)
        tail_free_bytes = max(0, int(slab.size_bytes) - int(slab.append_offset))
        free_bytes += tail_free_bytes
        if tail_free_bytes > largest_free_bytes:
            largest_free_bytes = tail_free_bytes
        for _, size in slab.free_ranges:
            size = int(size)
            free_bytes += size
            if size > largest_free_bytes:
                largest_free_bytes = size
    used_bytes = max(0, total_bytes - free_bytes)
    allocation_count = len(renderer._mesh_allocations)
    return slab_count, total_bytes, used_bytes, free_bytes, largest_free_bytes, allocation_count


def mesh_output_slab_size_for_request(renderer, request_bytes: int) -> int:
    size_bytes = max(renderer._mesh_output_min_slab_bytes, int(request_bytes))
    slab_bytes = renderer._mesh_output_min_slab_bytes
    while slab_bytes < size_bytes:
        slab_bytes *= 2
    return slab_bytes


def slab_tail_free_bytes(slab: MeshOutputSlab) -> int:
    return max(0, int(slab.size_bytes) - int(slab.append_offset))


def _ensure_slab_free_range_indexes(slab: MeshOutputSlab) -> None:
    free_by_offset = getattr(slab, "_free_ranges_by_offset", None)
    free_by_size = getattr(slab, "_free_ranges_by_size", None)
    if free_by_offset is not None and free_by_size is not None:
        return
    free_ranges = coalesce_mesh_output_free_ranges(list(getattr(slab, "free_ranges", [])))
    slab._free_ranges_by_offset = list(free_ranges)
    slab._free_ranges_by_size = sorted((int(size), int(offset)) for offset, size in free_ranges)
    slab.free_ranges = list(free_ranges)


def _sync_slab_free_ranges(slab: MeshOutputSlab) -> None:
    _ensure_slab_free_range_indexes(slab)
    slab.free_ranges = list(slab._free_ranges_by_offset)


def _remove_slab_size_entry(slab: MeshOutputSlab, offset: int, size: int) -> None:
    entries = slab._free_ranges_by_size
    index = bisect_left(entries, (int(size), int(offset)))
    while index < len(entries):
        entry_size, entry_offset = entries[index]
        if entry_size != int(size):
            break
        if entry_offset == int(offset):
            entries.pop(index)
            return
        index += 1


def _add_slab_size_entry(slab: MeshOutputSlab, offset: int, size: int) -> None:
    entries = slab._free_ranges_by_size
    entry = (int(size), int(offset))
    entries.insert(bisect_left(entries, entry), entry)


def _pop_slab_free_range_at(slab: MeshOutputSlab, index: int) -> tuple[int, int]:
    _ensure_slab_free_range_indexes(slab)
    offset, size = slab._free_ranges_by_offset.pop(index)
    _remove_slab_size_entry(slab, offset, size)
    slab.free_ranges = list(slab._free_ranges_by_offset)
    return int(offset), int(size)


def _insert_slab_free_range(slab: MeshOutputSlab, offset: int, size: int) -> None:
    _ensure_slab_free_range_indexes(slab)
    offset = int(offset)
    size = int(size)
    if size <= 0:
        return

    free_by_offset = slab._free_ranges_by_offset
    index = bisect_left(free_by_offset, (offset, 0))
    merge_offset = offset
    merge_end = offset + size

    if index > 0:
        prev_offset, prev_size = free_by_offset[index - 1]
        prev_end = int(prev_offset) + int(prev_size)
        if prev_end >= offset:
            _pop_slab_free_range_at(slab, index - 1)
            index -= 1
            merge_offset = min(int(prev_offset), merge_offset)
            merge_end = max(prev_end, merge_end)

    while index < len(free_by_offset):
        next_offset, next_size = free_by_offset[index]
        next_offset = int(next_offset)
        next_end = next_offset + int(next_size)
        if next_offset > merge_end:
            break
        _pop_slab_free_range_at(slab, index)
        merge_offset = min(merge_offset, next_offset)
        merge_end = max(merge_end, next_end)

    merged = (merge_offset, merge_end - merge_offset)
    free_by_offset.insert(index, merged)
    _add_slab_size_entry(slab, merged[0], merged[1])
    slab.free_ranges = list(free_by_offset)


def slab_largest_free_range_bytes(slab: MeshOutputSlab) -> int:
    _ensure_slab_free_range_indexes(slab)
    largest = slab_tail_free_bytes(slab)
    if slab._free_ranges_by_size:
        largest = max(largest, int(slab._free_ranges_by_size[-1][0]))
    return largest


def slab_total_free_bytes(slab: MeshOutputSlab) -> int:
    _ensure_slab_free_range_indexes(slab)
    total = slab_tail_free_bytes(slab)
    for _, size in slab._free_ranges_by_offset:
        total += int(size)
    return total


def mesh_output_request_size_class(renderer, request_bytes: int) -> int:
    needed_bytes = max(renderer._mesh_output_binding_alignment, int(request_bytes))
    size_class = renderer._mesh_output_binding_alignment
    while size_class < needed_bytes:
        size_class <<= 1
    return size_class


def slab_accepts_size_class(slab: MeshOutputSlab, size_class_bytes: int) -> bool:
    slab_class = max(0, int(getattr(slab, "size_class_bytes", 0)))
    if slab_class <= 0:
        return True
    return slab_class == int(size_class_bytes)


def _mesh_output_slabs_for_size_class(renderer, size_class_bytes: int) -> OrderedDict[int, MeshOutputSlab]:
    slabs_by_class = getattr(renderer, "_mesh_output_slabs_by_size_class", None)
    if slabs_by_class is None:
        slabs_by_class = {}
        setattr(renderer, "_mesh_output_slabs_by_size_class", slabs_by_class)
    class_key = int(size_class_bytes)
    class_slabs = slabs_by_class.get(class_key)
    if class_slabs is None:
        class_slabs = OrderedDict()
        slabs_by_class[class_key] = class_slabs
    elif class_slabs:
        live_slabs = renderer._mesh_output_slabs
        stale_ids = [slab_id for slab_id in class_slabs.keys() if slab_id not in live_slabs]
        for slab_id in stale_ids:
            class_slabs.pop(slab_id, None)
        if not class_slabs:
            slabs_by_class.pop(class_key, None)
            class_slabs = OrderedDict()
            slabs_by_class[class_key] = class_slabs
    return class_slabs


def _index_mesh_output_slab(renderer, slab: MeshOutputSlab) -> None:
    _mesh_output_slabs_for_size_class(renderer, int(getattr(slab, "size_class_bytes", 0)))[slab.slab_id] = slab


def _unindex_mesh_output_slab(renderer, slab: MeshOutputSlab) -> None:
    slabs_by_class = getattr(renderer, "_mesh_output_slabs_by_size_class", None)
    if not slabs_by_class:
        return
    class_key = int(getattr(slab, "size_class_bytes", 0))
    class_slabs = slabs_by_class.get(class_key)
    if class_slabs is None:
        return
    class_slabs.pop(slab.slab_id, None)
    if not class_slabs:
        slabs_by_class.pop(class_key, None)


def _touch_mesh_output_slab(renderer, slab: MeshOutputSlab) -> None:
    if slab.slab_id not in renderer._mesh_output_slabs:
        _unindex_mesh_output_slab(renderer, slab)
        return
    renderer._mesh_output_slabs.move_to_end(slab.slab_id)
    class_slabs = _mesh_output_slabs_for_size_class(renderer, int(getattr(slab, "size_class_bytes", 0)))
    if slab.slab_id in class_slabs:
        class_slabs.move_to_end(slab.slab_id)
    else:
        class_slabs[slab.slab_id] = slab


@profile
def create_mesh_output_slab(renderer, size_bytes: int, size_class_bytes: int) -> MeshOutputSlab:
    if getattr(renderer, "_device_lost", False):
        raise RuntimeError("Render device is unavailable after device loss.")
    slab = MeshOutputSlab(
        slab_id=renderer._next_mesh_output_slab_id,
        buffer=renderer.device.create_buffer(
            size=max(1, int(size_bytes)),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        ),
        size_bytes=int(size_bytes),
        free_ranges=[],
        append_offset=0,
        size_class_bytes=int(size_class_bytes),
    )
    renderer._next_mesh_output_slab_id += 1
    renderer._mesh_output_slabs[slab.slab_id] = slab
    _index_mesh_output_slab(renderer, slab)
    renderer._mesh_output_append_slab_id = slab.slab_id
    return slab


@profile
def register_mesh_output_allocation(
    renderer,
    slab: MeshOutputSlab,
    offset_bytes: int,
    size_bytes: int,
) -> MeshBufferAllocation:
    allocation = MeshBufferAllocation(
        allocation_id=renderer._next_mesh_allocation_id,
        buffer=slab.buffer,
        offset_bytes=int(offset_bytes),
        size_bytes=int(size_bytes),
        slab_id=slab.slab_id,
        refcount=0,
    )
    renderer._next_mesh_allocation_id += 1
    renderer._mesh_allocations[allocation.allocation_id] = allocation
    _touch_mesh_output_slab(renderer, slab)
    renderer._mesh_output_append_slab_id = slab.slab_id
    return allocation


@profile
def allocate_from_mesh_output_slab_bump(
    renderer,
    slab: MeshOutputSlab,
    request_bytes: int,
) -> MeshBufferAllocation | None:
    _ensure_slab_free_range_indexes(slab)
    needed_bytes = max(1, int(request_bytes))
    current_offset = int(slab.append_offset)
    aligned_offset = render_consts.align_up(current_offset, renderer._mesh_output_binding_alignment)
    alloc_end = aligned_offset + needed_bytes
    if alloc_end > int(slab.size_bytes):
        return None
    padding = aligned_offset - current_offset
    if padding > 0:
        _insert_slab_free_range(slab, current_offset, padding)
    slab.append_offset = alloc_end
    return register_mesh_output_allocation(renderer, slab, aligned_offset, needed_bytes)


@profile
def find_mesh_output_slab_free_range_choice(
    renderer,
    slab: MeshOutputSlab,
    request_bytes: int,
) -> tuple[int, int, int, int, int] | None:
    _ensure_slab_free_range_indexes(slab)
    needed_bytes = max(1, int(request_bytes))
    alignment = renderer._mesh_output_binding_alignment
    size_entries = slab._free_ranges_by_size
    if not size_entries:
        return None

    start = bisect_left(size_entries, (needed_bytes, -1))
    best_choice: tuple[int, int, int, int, int] | None = None
    max_checks = min(len(size_entries), start + 8)
    for size_index in range(start, max_checks):
        range_size, range_offset = size_entries[size_index]
        aligned_offset = render_consts.align_up(int(range_offset), alignment)
        padding = aligned_offset - int(range_offset)
        usable_size = int(range_size) - padding
        if usable_size < needed_bytes:
            continue
        waste = usable_size - needed_bytes
        choice = (waste, int(range_size), int(size_index), int(range_offset), int(aligned_offset))
        if best_choice is None or choice < best_choice:
            best_choice = choice
            if waste == 0:
                break
    if best_choice is not None:
        return best_choice

    for size_index in range(max_checks, len(size_entries)):
        range_size, range_offset = size_entries[size_index]
        aligned_offset = render_consts.align_up(int(range_offset), alignment)
        padding = aligned_offset - int(range_offset)
        usable_size = int(range_size) - padding
        if usable_size < needed_bytes:
            continue
        waste = usable_size - needed_bytes
        return (waste, int(range_size), int(size_index), int(range_offset), int(aligned_offset))
    return None


@profile
def allocate_from_mesh_output_slab_free_ranges(
    renderer,
    slab: MeshOutputSlab,
    request_bytes: int,
) -> MeshBufferAllocation | None:
    _ensure_slab_free_range_indexes(slab)
    needed_bytes = max(1, int(request_bytes))
    best_choice = find_mesh_output_slab_free_range_choice(renderer, slab, needed_bytes)
    if best_choice is None:
        return None

    _, range_size, size_index, range_offset, aligned_offset = best_choice
    alloc_end = aligned_offset + needed_bytes
    try:
        slab._free_ranges_by_size.pop(size_index)
    except IndexError:
        _remove_slab_size_entry(slab, range_offset, range_size)
    offset_entries = slab._free_ranges_by_offset
    offset_index = bisect_left(offset_entries, (int(range_offset), 0))
    while offset_index < len(offset_entries):
        current_offset, current_size = offset_entries[offset_index]
        if int(current_offset) == int(range_offset) and int(current_size) == int(range_size):
            offset_entries.pop(offset_index)
            break
        offset_index += 1
    new_ranges: list[tuple[int, int]] = []
    padding = aligned_offset - range_offset
    if padding > 0:
        new_ranges.append((range_offset, padding))
    tail_size = (range_offset + range_size) - alloc_end
    if tail_size > 0:
        new_ranges.append((alloc_end, tail_size))
    for new_offset, new_size in new_ranges:
        _insert_slab_free_range(slab, new_offset, new_size)
    _sync_slab_free_ranges(slab)
    return register_mesh_output_allocation(renderer, slab, aligned_offset, needed_bytes)


@profile
def allocate_from_mesh_output_slab(
    renderer,
    slab: MeshOutputSlab,
    request_bytes: int,
) -> MeshBufferAllocation | None:
    allocation = allocate_from_mesh_output_slab_bump(renderer, slab, request_bytes)
    if allocation is not None:
        return allocation
    return allocate_from_mesh_output_slab_free_ranges(renderer, slab, request_bytes)


@profile
def allocate_mesh_output_range(renderer, request_bytes: int) -> MeshBufferAllocation:
    needed_bytes = max(1, int(request_bytes))
    needed_bytes = render_consts.align_up(needed_bytes, renderer._mesh_output_binding_alignment)
    size_class_bytes = mesh_output_request_size_class(renderer, needed_bytes)

    class_slabs = _mesh_output_slabs_for_size_class(renderer, size_class_bytes)
    if class_slabs:
        append_slab_id = renderer._mesh_output_append_slab_id
        if append_slab_id is not None:
            append_slab = class_slabs.get(int(append_slab_id))
            if append_slab is not None:
                allocation = allocate_from_mesh_output_slab_bump(renderer, append_slab, needed_bytes)
                if allocation is not None:
                    return allocation

        recent_candidates: list[MeshOutputSlab] = []
        for slab_id, slab in reversed(class_slabs.items()):
            if append_slab_id is not None and int(slab_id) == int(append_slab_id):
                continue
            recent_candidates.append(slab)
            if len(recent_candidates) >= 4:
                break
        for slab in recent_candidates:
            allocation = allocate_from_mesh_output_slab_bump(renderer, slab, needed_bytes)
            if allocation is not None:
                return allocation

        best_free_choice: tuple[int, int, int, MeshOutputSlab] | None = None
        for slab_id, slab in class_slabs.items():
            choice = find_mesh_output_slab_free_range_choice(renderer, slab, needed_bytes)
            if choice is None:
                continue
            waste, range_size, _, _, _ = choice
            ranked = (waste, range_size, int(slab_id))
            if best_free_choice is None or ranked < best_free_choice[:3]:
                best_free_choice = (waste, range_size, int(slab_id), slab)
        if best_free_choice is not None:
            allocation = allocate_from_mesh_output_slab_free_ranges(renderer, best_free_choice[3], needed_bytes)
            if allocation is not None:
                return allocation

        best_append_choice: tuple[int, int, int, MeshOutputSlab] | None = None
        for slab_id, slab in class_slabs.items():
            tail_free_bytes = slab_tail_free_bytes(slab)
            if tail_free_bytes < needed_bytes:
                continue
            ranked = (tail_free_bytes - needed_bytes, abs(int(slab.size_bytes) - needed_bytes), int(slab_id), slab)
            if best_append_choice is None or ranked[:3] < best_append_choice[:3]:
                best_append_choice = ranked
        if best_append_choice is not None:
            allocation = allocate_from_mesh_output_slab_bump(renderer, best_append_choice[3], needed_bytes)
            if allocation is not None:
                return allocation

    slab = create_mesh_output_slab(
        renderer,
        mesh_output_slab_size_for_request(renderer, needed_bytes),
        size_class_bytes,
    )
    allocation = allocate_from_mesh_output_slab_bump(renderer, slab, needed_bytes)
    if allocation is None:
        raise RuntimeError("Failed to suballocate mesh output slab.")
    return allocation


@profile
def coalesce_mesh_output_free_ranges(free_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not free_ranges:
        return []
    merged: list[list[int]] = []
    for offset, size in sorted(free_ranges):
        if size <= 0:
            continue
        if not merged:
            merged.append([offset, size])
            continue
        last = merged[-1]
        last_end = last[0] + last[1]
        if offset <= last_end:
            last[1] = max(last_end, offset + size) - last[0]
        else:
            merged.append([offset, size])
    return [(offset, size) for offset, size in merged]


@profile
def trim_mesh_output_slab_tail(renderer, slab: MeshOutputSlab) -> None:
    _ensure_slab_free_range_indexes(slab)
    tail_offset = int(slab.append_offset)
    if tail_offset <= 0 or not slab._free_ranges_by_offset:
        return
    while slab._free_ranges_by_offset:
        range_offset, range_size = slab._free_ranges_by_offset[-1]
        range_end = int(range_offset) + int(range_size)
        if range_end != tail_offset:
            break
        tail_offset = int(range_offset)
        _pop_slab_free_range_at(slab, len(slab._free_ranges_by_offset) - 1)
    slab.append_offset = tail_offset
    _sync_slab_free_ranges(slab)


@profile
def refresh_mesh_output_append_slab(renderer) -> None:
    renderer._mesh_output_append_slab_id = None
    best_choice: tuple[int, int, int] | None = None
    slab_items = list(renderer._mesh_output_slabs.items())
    total_slabs = len(slab_items)
    for recency, (slab_id, slab) in enumerate(slab_items):
        tail_free_bytes = slab_tail_free_bytes(slab)
        if tail_free_bytes <= 0:
            continue
        choice = (-tail_free_bytes, -(recency - total_slabs), slab_id)
        if best_choice is None or choice < best_choice:
            best_choice = choice
    if best_choice is not None:
        renderer._mesh_output_append_slab_id = best_choice[2]


def retire_mesh_output_slab_if_empty(renderer, slab: MeshOutputSlab) -> None:
    if int(slab.append_offset) != 0 or slab.free_ranges:
        return
    _unindex_mesh_output_slab(renderer, slab)
    renderer._mesh_output_slabs.pop(slab.slab_id, None)
    if renderer._mesh_output_append_slab_id == slab.slab_id:
        refresh_mesh_output_append_slab(renderer)
    slab.buffer.destroy()


@profile
def free_mesh_output_range(renderer, slab_id: int, offset_bytes: int, size_bytes: int) -> None:
    slab = renderer._mesh_output_slabs.get(int(slab_id))
    if slab is None:
        return
    _insert_slab_free_range(slab, int(offset_bytes), int(size_bytes))
    trim_mesh_output_slab_tail(renderer, slab)
    retire_mesh_output_slab_if_empty(renderer, slab)
    if slab.slab_id not in renderer._mesh_output_slabs:
        return
    _touch_mesh_output_slab(renderer, slab)
    refresh_mesh_output_append_slab(renderer)


def schedule_mesh_output_range_free(renderer, slab_id: int, offset_bytes: int, size_bytes: int) -> None:
    delay_frames = max(1, int(render_consts.MESH_OUTPUT_FREE_DELAY_FRAMES))
    renderer._deferred_mesh_output_frees.append(
        (delay_frames, int(slab_id), int(offset_bytes), int(size_bytes))
    )


@profile
def process_deferred_mesh_output_frees(renderer) -> None:
    if not renderer._deferred_mesh_output_frees:
        return
    next_queue = []
    while renderer._deferred_mesh_output_frees:
        frames_left, slab_id, offset_bytes, size_bytes = renderer._deferred_mesh_output_frees.popleft()
        frames_left -= 1
        if frames_left <= 0:
            free_mesh_output_range(renderer, slab_id, offset_bytes, size_bytes)
        else:
            next_queue.append((frames_left, slab_id, offset_bytes, size_bytes))
    renderer._deferred_mesh_output_frees.extend(next_queue)


@profile
def retain_chunk_mesh_storage(renderer, mesh: ChunkMesh) -> None:
    if mesh.allocation_id is None:
        renderer._retain_mesh_buffer(mesh.vertex_buffer)
        return
    allocation = renderer._mesh_allocations.get(mesh.allocation_id)
    if allocation is None:
        raise RuntimeError(f"Unknown mesh allocation id: {mesh.allocation_id}")
    allocation.refcount += 1


@profile
def release_chunk_mesh_storage(renderer, mesh: ChunkMesh) -> None:
    if mesh.allocation_id is None:
        shared_empty_buffer = getattr(renderer, "_shared_empty_chunk_vertex_buffer", None)
        if shared_empty_buffer is not None and mesh.vertex_buffer is shared_empty_buffer:
            return
        renderer._release_mesh_buffer(mesh.vertex_buffer)
        return
    allocation = renderer._mesh_allocations.get(mesh.allocation_id)
    if allocation is None:
        return
    allocation.refcount -= 1
    if allocation.refcount > 0:
        return
    renderer._mesh_allocations.pop(mesh.allocation_id, None)
    if allocation.slab_id is None:
        allocation.buffer.destroy()
        return
    schedule_mesh_output_range_free(
        renderer,
        allocation.slab_id,
        allocation.offset_bytes,
        allocation.size_bytes,
    )


@profile
def clear_tile_render_batches(renderer) -> None:
    for batch in renderer._tile_render_batches.values():
        if getattr(batch, "owns_vertex_buffer", False) and batch.vertex_buffer is not None:
            batch.vertex_buffer.destroy()
    _destroy_merged_tile_buffer_reuse_state(renderer)
    renderer._tile_render_batches.clear()
    renderer._tile_dirty_keys.clear()
    renderer._visible_tile_dirty_keys.clear()
    renderer._visible_tile_key_set.clear()
    renderer._visible_active_tile_key_set.clear()
    renderer._tile_versions.clear()
    renderer._tile_mutation_version += 1
    renderer._visible_tile_mutation_version += 1
    renderer._cached_tile_draw_batches.clear()
    renderer._cached_visible_render_batches.clear()


@profile
def mark_tile_dirty(renderer, chunk_x: int, chunk_z: int, chunk_y: int = 0) -> None:
    key = tile_key(renderer, int(chunk_x), int(chunk_z), int(chunk_y))
    renderer._tile_dirty_keys.add(key)
    renderer._tile_versions[key] = int(renderer._tile_versions.get(key, 0)) + 1
    renderer._tile_mutation_version += 1
    if key in renderer._visible_tile_key_set:
        renderer._visible_tile_dirty_keys.add(key)
        renderer._visible_tile_mutation_version += 1
        renderer._cached_tile_draw_batches.clear()
        renderer._cached_visible_render_batches.clear()




def _sync_visible_active_tile_keys(renderer) -> None:
    visible_tile_keys = getattr(renderer, "_visible_tile_keys", ())
    active_key_set = getattr(renderer, "_visible_active_tile_key_set", set())
    renderer._visible_active_tile_keys = [tile_key_value for tile_key_value in visible_tile_keys if tile_key_value in active_key_set]


def _refresh_visible_tile_active_meshes_for_key(renderer, tile_key_value, slots) -> None:
    active_meshes = [mesh for mesh in slots if mesh is not None and mesh.vertex_count > 0]
    if active_meshes:
        renderer._visible_tile_active_meshes[tile_key_value] = active_meshes
        renderer._visible_active_tile_key_set.add(tile_key_value)
    else:
        renderer._visible_tile_active_meshes.pop(tile_key_value, None)
        renderer._visible_active_tile_key_set.discard(tile_key_value)
    _sync_visible_active_tile_keys(renderer)


def _remove_visible_active_mesh_for_key(renderer, tile_key_value, mesh) -> None:
    active_meshes = renderer._visible_tile_active_meshes.get(tile_key_value)
    if not active_meshes:
        return
    try:
        active_meshes.remove(mesh)
    except ValueError:
        return
    if not active_meshes:
        renderer._visible_tile_active_meshes.pop(tile_key_value, None)
        renderer._visible_active_tile_key_set.discard(tile_key_value)
    _sync_visible_active_tile_keys(renderer)


def _append_visible_active_mesh_for_key(renderer, tile_key_value, mesh) -> None:
    if mesh is None or int(getattr(mesh, "vertex_count", 0)) <= 0:
        return
    active_meshes = renderer._visible_tile_active_meshes.get(tile_key_value)
    if active_meshes is None:
        renderer._visible_tile_active_meshes[tile_key_value] = [mesh]
        renderer._visible_active_tile_key_set.add(tile_key_value)
    else:
        active_meshes.append(mesh)
    _sync_visible_active_tile_keys(renderer)


@profile
def store_chunk_meshes(renderer, meshes: list[ChunkMesh]) -> None:
    if not meshes:
        return

    chunk_cache = renderer.chunk_cache
    cache_get = chunk_cache.get
    cache_move_to_end = chunk_cache.move_to_end
    release = lambda mesh: release_chunk_mesh_storage(renderer, mesh)
    retain = lambda mesh: retain_chunk_mesh_storage(renderer, mesh)
    visible_chunk_coord_set = renderer._visible_chunk_coord_set
    visible_tile_mesh_slots = getattr(renderer, "_visible_tile_mesh_slots", {})
    visible_origin = getattr(renderer, "_visible_chunk_origin", None)
    visible_rel_coord_to_tile_slot = getattr(renderer, "_visible_rel_coord_to_tile_slot", {})
    visible_tile_base = getattr(renderer, "_visible_tile_base", (0, 0, 0))

    def _visible_tile_slot_meta(coord):
        if visible_origin is None:
            return None
        rel_coord = (
            int(coord[0]) - int(visible_origin[0]),
            int(coord[1]) - int(visible_origin[1]),
            int(coord[2]) - int(visible_origin[2]),
        )
        rel_slot = visible_rel_coord_to_tile_slot.get(rel_coord)
        if rel_slot is None:
            return None
        rel_tile_key, slot_index = rel_slot
        tile_key_value = (
            int(visible_tile_base[0]) + int(rel_tile_key[0]),
            int(visible_tile_base[1]) + int(rel_tile_key[1]),
            int(visible_tile_base[2]) + int(rel_tile_key[2]),
        )
        return tile_key_value, int(slot_index)

    mesh_key_set: set[tuple[int, int, int]] = set()

    for mesh in meshes:
        key = (mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z)
        mesh_key_set.add(key)
        mark_tile_dirty(renderer, mesh.chunk_x, mesh.chunk_z, getattr(mesh, "chunk_y", 0))

        existing = cache_get(key)
        if existing is mesh:
            cache_move_to_end(key)
            continue

        if existing is not None:
            mark_tile_dirty(renderer, existing.chunk_x, existing.chunk_z, getattr(existing, "chunk_y", 0))
            release(existing)

        chunk_cache[key] = mesh
        if existing is not None:
            cache_move_to_end(key)
        retain(mesh)

        slot_meta = _visible_tile_slot_meta(key)
        if slot_meta is not None:
            tile_key_value, slot_index = slot_meta
            if existing is not None and int(getattr(existing, "vertex_count", 0)) > 0:
                _remove_visible_active_mesh_for_key(renderer, tile_key_value, existing)
            if mesh.vertex_count > 0:
                _append_visible_active_mesh_for_key(renderer, tile_key_value, mesh)
            slots = visible_tile_mesh_slots.get(tile_key_value)
            if slots is not None and 0 <= slot_index < len(slots):
                slots[slot_index] = mesh if mesh.vertex_count > 0 else None
                _refresh_visible_tile_active_meshes_for_key(renderer, tile_key_value, slots)

    pending_chunk_coords = renderer._pending_chunk_coords
    pending_chunk_coords.difference_update(mesh_key_set)

    visible_keys = mesh_key_set & visible_chunk_coord_set
    if visible_keys:
        renderer._visible_displayed_coords.update(visible_keys)
        renderer._visible_missing_coords.difference_update(visible_keys)
        renderer._visible_display_state_dirty = True

    max_cached_chunks = renderer.max_cached_chunks
    if len(chunk_cache) <= max_cached_chunks:
        return

    visible_displayed_coords = renderer._visible_displayed_coords
    visible_missing_coords = renderer._visible_missing_coords
    pop_oldest = chunk_cache.popitem
    queue_dirty = False

    while len(chunk_cache) > max_cached_chunks:
        old_key, old_mesh = pop_oldest(last=False)
        mark_tile_dirty(renderer, old_mesh.chunk_x, old_mesh.chunk_z, getattr(old_mesh, "chunk_y", 0))
        slot_meta = _visible_tile_slot_meta(old_key)
        if slot_meta is not None:
            tile_key_value, slot_index = slot_meta
            if int(getattr(old_mesh, "vertex_count", 0)) > 0:
                _remove_visible_active_mesh_for_key(renderer, tile_key_value, old_mesh)
            slots = visible_tile_mesh_slots.get(tile_key_value)
            if slots is not None and 0 <= slot_index < len(slots):
                slots[slot_index] = None
                _refresh_visible_tile_active_meshes_for_key(renderer, tile_key_value, slots)
        release(old_mesh)
        if old_key in visible_chunk_coord_set:
            visible_displayed_coords.discard(old_key)
            if old_key not in pending_chunk_coords:
                visible_missing_coords.add(old_key)
                queue_dirty = True

    if queue_dirty:
        renderer._chunk_request_queue_dirty = True
        renderer._visible_display_state_dirty = True


@profile
def store_chunk_mesh(renderer, mesh: ChunkMesh) -> None:
    store_chunk_meshes(renderer, [mesh])


def chunk_mesh_age(renderer, mesh: ChunkMesh) -> float:
    return max(0.0, time.perf_counter() - mesh.created_at)


def merge_chunk_bounds(renderer, tile_meshes: list[ChunkMesh]) -> tuple[float, float, float, float]:
    if not tile_meshes:
        raise ValueError("min() arg is an empty sequence")

    bounds = tile_meshes[0].bounds
    bx = bounds[0]
    by = bounds[1]
    bz = bounds[2]
    br = bounds[3]

    min_x = bx - br
    max_x = bx + br
    min_y = by - br
    max_y = by + br
    min_z = bz - br
    max_z = bz + br

    for index in range(1, len(tile_meshes)):
        bounds = tile_meshes[index].bounds
        bx = bounds[0]
        by = bounds[1]
        bz = bounds[2]
        br = bounds[3]

        left = bx - br
        right = bx + br
        bottom = by - br
        top = by + br
        near = bz - br
        far = bz + br

        if left < min_x:
            min_x = left
        if right > max_x:
            max_x = right
        if bottom < min_y:
            min_y = bottom
        if top > max_y:
            max_y = top
        if near < min_z:
            min_z = near
        if far > max_z:
            max_z = far

    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    center_z = (min_z + max_z) * 0.5

    dx = max_x - center_x
    dy = max_y - center_y
    dz = max_z - center_z
    radius = math.sqrt(dx * dx + dy * dy + dz * dz)

    return center_x, center_y, center_z, radius


_MERGED_TILE_BUFFER_POOL_BUCKET_MIN_BYTES = 64 * 1024
_MERGED_TILE_BUFFER_POOL_PER_BUCKET_LIMIT = 8


def _merged_tile_buffer_bucket_bytes(required_bytes: int) -> int:
    needed_bytes = max(int(_renderer_module().VERTEX_STRIDE), int(required_bytes))
    return max(_MERGED_TILE_BUFFER_POOL_BUCKET_MIN_BYTES, 1 << (needed_bytes - 1).bit_length())


def _get_merged_tile_buffer_pool(renderer) -> dict[int, list[wgpu.GPUBuffer]]:
    pool = getattr(renderer, "_merged_tile_buffer_pool", None)
    if pool is None:
        pool = {}
        setattr(renderer, "_merged_tile_buffer_pool", pool)
    return pool


def _queue_merged_tile_buffer_for_reuse(renderer, buffer, capacity_bytes: int) -> None:
    if buffer is None:
        return
    reuse_capacity_bytes = int(capacity_bytes)
    if reuse_capacity_bytes <= 0:
        try:
            buffer.destroy()
        except Exception:
            pass
        return
    reuse_queue = getattr(renderer, "_merged_tile_buffer_reuse_queue", None)
    if reuse_queue is None:
        reuse_queue = []
        setattr(renderer, "_merged_tile_buffer_reuse_queue", reuse_queue)
    reuse_queue.append([(buffer, reuse_capacity_bytes)])


def _flush_merged_tile_buffer_reuse_queue(renderer) -> None:
    reuse_queue = getattr(renderer, "_merged_tile_buffer_reuse_queue", None)
    if not reuse_queue:
        return
    pool = _get_merged_tile_buffer_pool(renderer)
    while len(reuse_queue) > 3:
        expired_entries = reuse_queue.pop(0)
        for buffer, capacity_bytes in expired_entries:
            bucket_bytes = _merged_tile_buffer_bucket_bytes(capacity_bytes)
            bucket = pool.get(bucket_bytes)
            if bucket is None:
                bucket = []
                pool[bucket_bytes] = bucket
            if len(bucket) < _MERGED_TILE_BUFFER_POOL_PER_BUCKET_LIMIT:
                bucket.append(buffer)
            else:
                try:
                    buffer.destroy()
                except Exception:
                    pass


def _destroy_merged_tile_buffer_reuse_state(renderer) -> None:
    reuse_queue = getattr(renderer, "_merged_tile_buffer_reuse_queue", None) or []
    for entries in reuse_queue:
        for buffer, _capacity_bytes in entries:
            try:
                buffer.destroy()
            except Exception:
                pass
    setattr(renderer, "_merged_tile_buffer_reuse_queue", [])

    pool = getattr(renderer, "_merged_tile_buffer_pool", None) or {}
    for buffers in pool.values():
        for buffer in buffers:
            try:
                buffer.destroy()
            except Exception:
                pass
    setattr(renderer, "_merged_tile_buffer_pool", {})


def _acquire_merged_tile_buffer(renderer, required_bytes: int) -> tuple[wgpu.GPUBuffer, int]:
    bucket_bytes = _merged_tile_buffer_bucket_bytes(required_bytes)
    pool = _get_merged_tile_buffer_pool(renderer)
    bucket = pool.get(bucket_bytes)
    if bucket:
        return bucket.pop(), bucket_bytes
    return (
        renderer.device.create_buffer(
            size=max(1, bucket_bytes),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        ),
        bucket_bytes,
    )


@profile
def merge_tile_meshes(
    renderer,
    tile_meshes: list[ChunkMesh],
    encoder,
    existing_buffer=None,
    existing_capacity_bytes: int = 0,
) -> tuple[wgpu.GPUBuffer, int]:
    renderer_module = _renderer_module()
    vertex_stride = int(renderer_module.VERTEX_STRIDE)
    merged_tile_max_chunks = int(renderer_module.MERGED_TILE_MAX_CHUNKS)

    total_vertices = sum(mesh.vertex_count for mesh in tile_meshes)
    total_vertex_bytes = total_vertices * vertex_stride
    merged_buffer = existing_buffer
    merged_buffer_capacity_bytes = int(existing_capacity_bytes)
    if merged_buffer is None or merged_buffer_capacity_bytes < total_vertex_bytes:
        merged_buffer, merged_buffer_capacity_bytes = _acquire_merged_tile_buffer(renderer, total_vertex_bytes)
    if (
        renderer.tile_merge_pipeline is None
        or renderer.tile_merge_bind_group_layout is None
        or len(tile_meshes) > merged_tile_max_chunks
    ):
        dest_offset = 0
        for mesh in tile_meshes:
            copy_size = mesh.vertex_count * vertex_stride
            encoder.copy_buffer_to_buffer(
                mesh.vertex_buffer,
                mesh.vertex_offset,
                merged_buffer,
                dest_offset,
                copy_size,
            )
            dest_offset += copy_size
        return merged_buffer, merged_buffer_capacity_bytes

    metadata_array = np.zeros((merged_tile_max_chunks, 4), dtype=np.uint32)
    dst_first_vertex = 0
    for index, mesh in enumerate(tile_meshes):
        metadata_array[index, 0] = np.uint32(mesh.vertex_count)
        metadata_array[index, 1] = np.uint32(dst_first_vertex)
        dst_first_vertex += int(mesh.vertex_count)

    metadata_buffer = renderer.device.create_buffer_with_data(
        data=metadata_array.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    params_buffer = renderer.device.create_buffer_with_data(
        data=struct.pack("<4I", len(tile_meshes), total_vertices, 0, 0),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    entries = []
    for index in range(merged_tile_max_chunks):
        if index < len(tile_meshes):
            mesh = tile_meshes[index]
            entries.append(
                {
                    "binding": index,
                    "resource": {
                        "buffer": mesh.vertex_buffer,
                        "offset": mesh.vertex_offset,
                        "size": max(1, mesh.vertex_count * vertex_stride),
                    },
                }
            )
        else:
            entries.append({"binding": index, "resource": {"buffer": renderer._tile_merge_dummy_buffer}})
    entries.extend(
        [
            {"binding": merged_tile_max_chunks, "resource": {"buffer": metadata_buffer, "offset": 0, "size": merged_tile_max_chunks * 16}},
            {"binding": merged_tile_max_chunks + 1, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
            {"binding": merged_tile_max_chunks + 2, "resource": {"buffer": merged_buffer, "offset": 0, "size": max(1, total_vertex_bytes)}},
        ]
    )
    bind_group = renderer.device.create_bind_group(
        layout=renderer.tile_merge_bind_group_layout,
        entries=entries,
    )
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(renderer.tile_merge_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(max(1, (total_vertices + 63) // 64), 1, 1)
    compute_pass.end()
    wgpu_mesher.schedule_gpu_buffer_cleanup(renderer, [metadata_buffer, params_buffer], frames=3)
    return merged_buffer, merged_buffer_capacity_bytes


def visible_tile_mesh_groups(renderer) -> dict[tuple[int, int, int], list[ChunkMesh]]:
    if not renderer._visible_chunk_coords:
        coord_manager.refresh_visible_chunk_coords(renderer)
    tile_active_meshes = getattr(renderer, "_visible_tile_active_meshes", None) or {}
    return {
        tile_key_value: list(meshes)
        for tile_key_value, meshes in tile_active_meshes.items()
        if meshes
    }


def visible_tile_chunk_groups(renderer) -> tuple[dict[tuple[int, int, int], list[tuple[int, int, int, ChunkMesh]]], int]:
    if not renderer._visible_chunk_coords:
        coord_manager.refresh_visible_chunk_coords(renderer)
    tile_active_meshes = getattr(renderer, "_visible_tile_active_meshes", None) or {}
    tile_groups: dict[tuple[int, int, int], list[tuple[int, int, int, ChunkMesh]]] = {}
    visible_count = 0
    for tile_key_value, meshes in tile_active_meshes.items():
        if not meshes:
            continue
        chunks = [
            (mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z, mesh)
            for mesh in meshes
            if mesh.vertex_count > 0
        ]
        if chunks:
            tile_groups[tile_key_value] = chunks
            visible_count += len(chunks)
    return tile_groups, visible_count


def _append_direct_render_batch(
    render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]],
    vertex_buffer: wgpu.GPUBuffer,
    binding_offset: int,
    vertex_count: int,
    first_vertex: int,
) -> None:
    vertex_count = int(vertex_count)
    if vertex_count <= 0:
        return
    binding_offset = int(binding_offset)
    first_vertex = int(first_vertex)
    if render_batches:
        last_vertex_buffer, last_binding_offset, last_vertex_count, last_first_vertex = render_batches[-1]
        if (
            last_vertex_buffer is vertex_buffer
            and int(last_binding_offset) == binding_offset
            and int(last_first_vertex) + int(last_vertex_count) == first_vertex
        ):
            render_batches[-1] = (
                last_vertex_buffer,
                binding_offset,
                int(last_vertex_count) + vertex_count,
                int(last_first_vertex),
            )
            return
    render_batches.append((vertex_buffer, binding_offset, vertex_count, first_vertex))



def _extend_direct_render_batches(
    render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]],
    batches: tuple[tuple[wgpu.GPUBuffer, int, int, int], ...] | list[tuple[wgpu.GPUBuffer, int, int, int]],
) -> None:
    for vertex_buffer, binding_offset, vertex_count, first_vertex in batches:
        _append_direct_render_batch(render_batches, vertex_buffer, binding_offset, vertex_count, first_vertex)


def _create_direct_render_batch_group_entry(
    vertex_buffer: wgpu.GPUBuffer,
    binding_offset: int,
    vertex_count: int,
    first_vertex: int,
) -> list[object]:
    batch = (vertex_buffer, binding_offset, vertex_count, first_vertex)
    return [vertex_buffer, binding_offset, [batch], int(first_vertex), False]


def _append_direct_render_batch_grouped(
    render_batch_groups: OrderedDict[tuple[int, int], list[object]],
    vertex_buffer: wgpu.GPUBuffer,
    binding_offset: int,
    vertex_count: int,
    first_vertex: int,
) -> None:
    vertex_count = int(vertex_count)
    if vertex_count <= 0:
        return
    binding_offset = int(binding_offset)
    first_vertex = int(first_vertex)
    key = (id(vertex_buffer), binding_offset)
    entry = render_batch_groups.get(key)
    if entry is None:
        render_batch_groups[key] = _create_direct_render_batch_group_entry(vertex_buffer, binding_offset, vertex_count, first_vertex)
        return
    batches = entry[2]
    assert isinstance(batches, list)
    if batches:
        last_vertex_buffer, last_binding_offset, last_vertex_count, last_first_vertex = batches[-1]
        if (
            last_vertex_buffer is vertex_buffer
            and int(last_binding_offset) == binding_offset
            and int(last_first_vertex) + int(last_vertex_count) == first_vertex
        ):
            batches[-1] = (vertex_buffer, binding_offset, int(last_vertex_count) + vertex_count, int(last_first_vertex))
            entry[3] = int(last_first_vertex)
            return
    if first_vertex < int(entry[3]):
        entry[4] = True
    batches.append((vertex_buffer, binding_offset, vertex_count, first_vertex))
    entry[3] = first_vertex


def _group_render_batches(
    batches: tuple[tuple[wgpu.GPUBuffer, int, int, int], ...] | list[tuple[wgpu.GPUBuffer, int, int, int]],
) -> tuple[tuple[tuple[int, int], wgpu.GPUBuffer, int, tuple[tuple[wgpu.GPUBuffer, int, int, int], ...]], ...]:
    if not batches:
        return ()
    grouped: OrderedDict[tuple[int, int], list[tuple[wgpu.GPUBuffer, int, int, int]]] = OrderedDict()
    for vertex_buffer, binding_offset, vertex_count, first_vertex in batches:
        vertex_count = int(vertex_count)
        if vertex_count <= 0:
            continue
        binding_offset = int(binding_offset)
        first_vertex = int(first_vertex)
        key = (id(vertex_buffer), binding_offset)
        group = grouped.get(key)
        if group is None:
            grouped[key] = [(vertex_buffer, binding_offset, vertex_count, first_vertex)]
            continue
        last_vertex_buffer, last_binding_offset, last_vertex_count, last_first_vertex = group[-1]
        if (
            last_vertex_buffer is vertex_buffer
            and int(last_binding_offset) == binding_offset
            and int(last_first_vertex) + int(last_vertex_count) == first_vertex
        ):
            group[-1] = (vertex_buffer, binding_offset, int(last_vertex_count) + vertex_count, int(last_first_vertex))
            continue
        group.append((vertex_buffer, binding_offset, vertex_count, first_vertex))
    grouped_batches = []
    for key, group in grouped.items():
        if not group:
            continue
        grouped_batches.append((key, group[0][0], int(group[0][1]), tuple(group)))
    return tuple(grouped_batches)


def _extend_direct_render_batches_grouped(
    render_batch_groups: OrderedDict[tuple[int, int], list[object]],
    batches: tuple[tuple[wgpu.GPUBuffer, int, int, int], ...] | list[tuple[wgpu.GPUBuffer, int, int, int]],
) -> None:
    if not batches:
        return
    append_grouped = _append_direct_render_batch_grouped
    if len(batches) == 1:
        vertex_buffer, binding_offset, vertex_count, first_vertex = batches[0]
        append_grouped(render_batch_groups, vertex_buffer, binding_offset, vertex_count, first_vertex)
        return
    for vertex_buffer, binding_offset, vertex_count, first_vertex in batches:
        append_grouped(render_batch_groups, vertex_buffer, binding_offset, vertex_count, first_vertex)


def _extend_grouped_render_batch_groups(
    render_batch_groups: OrderedDict[tuple[int, int], list[object]],
    grouped_batches: tuple[tuple[tuple[int, int], wgpu.GPUBuffer, int, tuple[tuple[wgpu.GPUBuffer, int, int, int], ...]], ...],
) -> None:
    if not grouped_batches:
        return
    for key, vertex_buffer, binding_offset, batches_to_add in grouped_batches:
        if not batches_to_add:
            continue
        entry = render_batch_groups.get(key)
        if entry is None:
            render_batch_groups[key] = [vertex_buffer, int(binding_offset), list(batches_to_add), int(batches_to_add[-1][3]), False]
            continue
        batches = entry[2]
        assert isinstance(batches, list)
        first_new = batches_to_add[0]
        last_existing = batches[-1] if batches else None
        if last_existing is not None:
            last_vertex_buffer, last_binding_offset, last_vertex_count, last_first_vertex = last_existing
            if (
                last_vertex_buffer is first_new[0]
                and int(last_binding_offset) == int(first_new[1])
                and int(last_first_vertex) + int(last_vertex_count) == int(first_new[3])
            ):
                batches[-1] = (
                    first_new[0],
                    int(first_new[1]),
                    int(last_vertex_count) + int(first_new[2]),
                    int(last_first_vertex),
                )
                if len(batches_to_add) > 1:
                    if int(batches_to_add[1][3]) < int(last_first_vertex):
                        entry[4] = True
                    batches.extend(batches_to_add[1:])
                    entry[3] = int(batches_to_add[-1][3])
                else:
                    entry[3] = int(last_first_vertex)
                continue
        if int(first_new[3]) < int(entry[3]):
            entry[4] = True
        batches.extend(batches_to_add)
        entry[3] = int(batches_to_add[-1][3])


def _finalize_direct_render_batch_groups(
    render_batch_groups: OrderedDict[tuple[int, int], list[object]],
) -> list[tuple[wgpu.GPUBuffer, int, int, int]]:
    if not render_batch_groups:
        return []
    if len(render_batch_groups) == 1:
        only_entry = next(iter(render_batch_groups.values()))
        only_batches = only_entry[2]
        assert isinstance(only_batches, list)
        if len(only_batches) <= 1:
            return list(only_batches)
        if not bool(only_entry[4]):
            return list(only_batches)

    normalized: list[tuple[wgpu.GPUBuffer, int, int, int]] = []
    normalized_extend = normalized.extend
    normalized_append = normalized.append
    for entry in render_batch_groups.values():
        batches = entry[2]
        assert isinstance(batches, list)
        batch_count = len(batches)
        if batch_count <= 0:
            continue
        if batch_count == 1:
            normalized_append(batches[0])
            continue
        if not bool(entry[4]):
            normalized_extend(batches)
            continue

        batches.sort(key=lambda item: item[3])
        last_vertex_buffer, last_binding_offset, last_vertex_count, last_first_vertex = batches[0]
        last_binding_offset = int(last_binding_offset)
        last_vertex_count = int(last_vertex_count)
        last_first_vertex = int(last_first_vertex)
        for vertex_buffer, binding_offset, vertex_count, first_vertex in batches[1:]:
            binding_offset = int(binding_offset)
            vertex_count = int(vertex_count)
            first_vertex = int(first_vertex)
            if (
                last_vertex_buffer is vertex_buffer
                and last_binding_offset == binding_offset
                and last_first_vertex + last_vertex_count == first_vertex
            ):
                last_vertex_count += vertex_count
                continue
            normalized_append((last_vertex_buffer, last_binding_offset, last_vertex_count, last_first_vertex))
            last_vertex_buffer = vertex_buffer
            last_binding_offset = binding_offset
            last_vertex_count = vertex_count
            last_first_vertex = first_vertex
        normalized_append((last_vertex_buffer, last_binding_offset, last_vertex_count, last_first_vertex))
    return normalized


def _normalize_direct_render_batches(
    render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]],
) -> list[tuple[wgpu.GPUBuffer, int, int, int]]:
    if len(render_batches) <= 1:
        return render_batches

    grouped: OrderedDict[tuple[int, int], list[tuple[wgpu.GPUBuffer, int, int, int]]] = OrderedDict()
    for vertex_buffer, binding_offset, vertex_count, first_vertex in render_batches:
        key = (id(vertex_buffer), int(binding_offset))
        group = grouped.get(key)
        if group is None:
            group = []
            grouped[key] = group
        group.append((vertex_buffer, int(binding_offset), int(vertex_count), int(first_vertex)))

    normalized: list[tuple[wgpu.GPUBuffer, int, int, int]] = []
    for batches in grouped.values():
        if len(batches) > 1:
            batches.sort(key=lambda item: item[3])
        for vertex_buffer, binding_offset, vertex_count, first_vertex in batches:
            _append_direct_render_batch(normalized, vertex_buffer, binding_offset, vertex_count, first_vertex)
    return normalized



def _draw_batches_to_render_batches(draw_batches: tuple[ChunkDrawBatch, ...] | list[ChunkDrawBatch]) -> tuple[tuple[wgpu.GPUBuffer, int, int, int], ...]:
    render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]] = []
    for batch in draw_batches:
        _append_direct_render_batch(
            render_batches,
            batch.vertex_buffer,
            int(batch.binding_offset),
            int(batch.vertex_count),
            int(batch.first_vertex),
        )
    return tuple(render_batches)


def _cached_tile_batch_stats(batch: ChunkRenderBatch) -> tuple[int, int, int, float]:
    return (
        int(getattr(batch, "merged_chunk_count", 0)),
        int(getattr(batch, "visible_chunk_count", 0)),
        int(getattr(batch, "visible_vertex_count", 0)),
        float(getattr(batch, "next_refresh_at", 0.0)),
    )


def _build_visible_tile_iterable(
    visible_active_tile_keys: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    active_key_set: set[tuple[int, int, int]],
    tile_render_batches: dict[tuple[int, int, int], ChunkRenderBatch],
    visible_tile_dirty_keys: set[tuple[int, int, int]],
) -> list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...]:
    if not visible_tile_dirty_keys:
        return visible_active_tile_keys
    tile_iterable = list(visible_active_tile_keys)
    seen = set(tile_iterable)
    seen_add = seen.add
    tile_iterable_append = tile_iterable.append
    for tile_key_value in tile_render_batches.keys():
        if tile_key_value not in seen:
            seen_add(tile_key_value)
            tile_iterable_append(tile_key_value)
    for tile_key_value in visible_tile_dirty_keys:
        if tile_key_value not in seen:
            seen_add(tile_key_value)
            tile_iterable_append(tile_key_value)
    return tile_iterable


def _store_cached_tile_render_batch(
    renderer,
    tile_key_value: tuple[int, int, int],
    existing: ChunkRenderBatch | None,
    *,
    signature: tuple[tuple[int, int, int], ...],
    vertex_count: int,
    vertex_buffer,
    bounds: tuple[float, float, float, float],
    chunk_count: int,
    complete_tile: bool,
    all_mature: bool,
    visible_mask: int,
    source_version: int,
    cached_draw_batches: tuple[ChunkDrawBatch, ...],
    cached_render_batches: tuple[tuple[wgpu.GPUBuffer, int, int, int], ...],
    cached_grouped_render_batches: tuple[tuple[tuple[int, int], wgpu.GPUBuffer, int, tuple[tuple[wgpu.GPUBuffer, int, int, int], ...]], ...],
    next_refresh_at: float,
    visible_chunk_count: int,
    merged_chunk_count: int,
    visible_vertex_count: int,
    owns_vertex_buffer: bool,
    owned_vertex_buffer_capacity_bytes: int = 0,
) -> ChunkRenderBatch:
    old_buffer = existing.vertex_buffer if (existing is not None and getattr(existing, "owns_vertex_buffer", False)) else None
    old_buffer_capacity_bytes = 0
    if old_buffer is not None and existing is not None:
        old_buffer_capacity_bytes = int(
            getattr(existing, "owned_vertex_buffer_capacity_bytes", 0)
            or (int(getattr(existing, "vertex_count", 0)) * int(_renderer_module().VERTEX_STRIDE))
        )
    batch = ChunkRenderBatch(
        signature=signature,
        vertex_count=vertex_count,
        vertex_buffer=vertex_buffer,
        bounds=bounds,
        chunk_count=chunk_count,
        complete_tile=complete_tile,
        all_mature=all_mature,
        visible_mask=visible_mask,
        source_version=source_version,
        cached_draw_batches=cached_draw_batches,
        cached_render_batches=cached_render_batches,
        cached_grouped_render_batches=cached_grouped_render_batches,
        next_refresh_at=float(next_refresh_at),
        visible_chunk_count=int(visible_chunk_count),
        merged_chunk_count=int(merged_chunk_count),
        visible_vertex_count=int(visible_vertex_count),
        owns_vertex_buffer=bool(owns_vertex_buffer),
        owned_vertex_buffer_capacity_bytes=int(owned_vertex_buffer_capacity_bytes),
    )
    renderer._tile_render_batches[tile_key_value] = batch
    renderer._tile_dirty_keys.discard(tile_key_value)
    renderer._visible_tile_dirty_keys.discard(tile_key_value)
    if old_buffer is not None and old_buffer is not vertex_buffer:
        _queue_merged_tile_buffer_for_reuse(renderer, old_buffer, old_buffer_capacity_bytes)
    return batch


@profile
def build_tile_draw_batches(
    renderer,
    meshes: list[ChunkMesh] | None,
    encoder,
    *,
    age_gate: bool,
    direct_render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]] | None = None,
    direct_render_batch_groups: OrderedDict[tuple[int, int], list[tuple[wgpu.GPUBuffer, int, int, int]]] | None = None,
) -> tuple[list[ChunkDrawBatch], int, int, int, float]:
    renderer_module = _renderer_module()
    merged_tile_min_age_seconds = float(renderer_module.MERGED_TILE_MIN_AGE_SECONDS)

    cache_key: tuple[int, int, int] | None = None
    now = 0.0
    current_tile_keys: set[tuple[int, int, int]]

    if meshes is None:
        visible_tile_dirty_keys = renderer._visible_tile_dirty_keys
        cache_key = (int(renderer._visible_layout_version), int(renderer._visible_tile_mutation_version), 1 if age_gate else 0)
        cached = renderer._cached_tile_draw_batches.get(cache_key)
        if cached is not None and not visible_tile_dirty_keys:
            cached_until, cached_batches, cached_merged, cached_visible, cached_vertices = cached
            if cached_until <= 0.0:
                return cached_batches, cached_merged, cached_visible, cached_vertices, 0.0
            now = time.perf_counter()
            if now < cached_until:
                return cached_batches, cached_merged, cached_visible, cached_vertices, cached_until
        if not renderer._visible_chunk_coords:
            coord_manager.refresh_visible_chunk_coords(renderer)
            visible_tile_dirty_keys = renderer._visible_tile_dirty_keys
        visible_tile_active_meshes = getattr(renderer, "_visible_tile_active_meshes", None) or {}
        current_tile_keys = getattr(renderer, "_visible_tile_key_set", None) or set(getattr(renderer, "_visible_tile_keys", None) or ())
    else:
        tile_groups: dict[tuple[int, int, int], list[ChunkMesh]] = {}
        for mesh in meshes:
            if mesh.vertex_count <= 0:
                continue
            tile_key_value = tile_key(renderer, mesh.chunk_x, mesh.chunk_z, getattr(mesh, "chunk_y", 0))
            group = tile_groups.get(tile_key_value)
            if group is None:
                group = []
                tile_groups[tile_key_value] = group
            group.append(mesh)
        current_tile_keys = set(tile_groups.keys())
        visible_tile_dirty_keys = set()

    tile_render_batches = renderer._tile_render_batches
    if meshes is None:
        visible_layout_version = int(getattr(renderer, "_visible_layout_version", 0))
        last_cleanup_layout_version = int(getattr(renderer, "_tile_render_batch_cleanup_layout_version", -1))
        if last_cleanup_layout_version != visible_layout_version:
            stale_keys = [tile_key_value for tile_key_value in tile_render_batches if tile_key_value not in current_tile_keys]
            for tile_key_value in stale_keys:
                batch = tile_render_batches.pop(tile_key_value)
                if getattr(batch, "owns_vertex_buffer", False) and batch.vertex_buffer is not None:
                    _queue_merged_tile_buffer_for_reuse(
                        renderer,
                        batch.vertex_buffer,
                        int(
                            getattr(batch, "owned_vertex_buffer_capacity_bytes", 0)
                            or (int(getattr(batch, "vertex_count", 0)) * int(renderer_module.VERTEX_STRIDE))
                        ),
                    )
            renderer._tile_render_batch_cleanup_layout_version = visible_layout_version
    else:
        stale_keys = [tile_key_value for tile_key_value in tile_render_batches if tile_key_value not in current_tile_keys]
        for tile_key_value in stale_keys:
            batch = tile_render_batches.pop(tile_key_value)
            if getattr(batch, "owns_vertex_buffer", False) and batch.vertex_buffer is not None:
                renderer._transient_render_buffers.append([batch.vertex_buffer])

    draw_batches: list[ChunkDrawBatch] = []
    merged_chunk_count = 0
    visible_chunk_count = 0
    next_refresh_at = 0.0
    merged_tile_max_chunks = int(renderer_module.MERGED_TILE_MAX_CHUNKS)
    visible_masks = getattr(renderer, "_visible_tile_masks", {})
    if age_gate and now <= 0.0:
        now = time.perf_counter()

    visible_vertex_count = 0

    if meshes is None:
        visible_active_tile_keys = getattr(renderer, "_visible_active_tile_keys", None) or ()
        active_key_set = getattr(renderer, "_visible_active_tile_key_set", None) or set(visible_active_tile_keys)
        tile_iterable = _build_visible_tile_iterable(
            visible_active_tile_keys,
            active_key_set,
            tile_render_batches,
            visible_tile_dirty_keys,
        )
        tile_render_batches_get = tile_render_batches.get
        tile_versions_get = renderer._tile_versions.get
        visible_masks_get = visible_masks.get
        visible_tile_active_meshes_get = visible_tile_active_meshes.get
        visible_tile_dirty_contains = visible_tile_dirty_keys.__contains__
    else:
        tile_iterable = sorted(tile_groups)
        tile_render_batches_get = tile_render_batches.get
        tile_versions_get = renderer._tile_versions.get
        visible_masks_get = visible_masks.get
        visible_tile_active_meshes_get = None
        visible_tile_dirty_contains = visible_tile_dirty_keys.__contains__

    draw_batches_extend = draw_batches.extend
    draw_batches_append = draw_batches.append
    grouped_extend = _extend_direct_render_batches_grouped
    direct_extend = _extend_direct_render_batches

    for tile_key_value in tile_iterable:
        existing = tile_render_batches_get(tile_key_value)
        tile_version = int(tile_versions_get(tile_key_value, 0))
        tile_is_dirty = visible_tile_dirty_contains(tile_key_value) or (existing is not None and existing.source_version != tile_version)
        current_visible_mask = int(visible_masks_get(tile_key_value, 0))

        if meshes is None and existing is not None and not tile_is_dirty and existing.visible_mask == current_visible_mask:
            tile_merged, tile_visible, tile_vertices, tile_refresh_at = _cached_tile_batch_stats(existing)
            if tile_visible > 0 and (tile_refresh_at <= 0.0 or now < tile_refresh_at):
                cached_draw_batches = getattr(existing, "cached_draw_batches", ())
                if cached_draw_batches:
                    cached_render_batches = getattr(existing, "cached_render_batches", ())
                    if direct_render_batch_groups is not None:
                        cached_grouped_render_batches = getattr(existing, "cached_grouped_render_batches", ())
                        if cached_grouped_render_batches:
                            _extend_grouped_render_batch_groups(direct_render_batch_groups, cached_grouped_render_batches)
                        else:
                            grouped_extend(direct_render_batch_groups, cached_render_batches)
                    elif direct_render_batches is not None:
                        direct_extend(direct_render_batches, cached_render_batches)
                    else:
                        draw_batches_extend(cached_draw_batches)
                merged_chunk_count += tile_merged
                visible_chunk_count += tile_visible
                visible_vertex_count += tile_vertices
                if tile_refresh_at > 0.0 and (next_refresh_at <= 0.0 or tile_refresh_at < next_refresh_at):
                    next_refresh_at = tile_refresh_at
                continue

        if meshes is None:
            tile_meshes = visible_tile_active_meshes_get(tile_key_value)
            if not tile_meshes:
                continue
        else:
            tile_meshes = tile_groups[tile_key_value]
            if current_visible_mask == 0:
                current_visible_mask = int(tile_visible_mask(renderer, tile_key_value, tile_meshes))

        tile_mesh_count = len(tile_meshes)
        if (
            existing is not None
            and not tile_is_dirty
            and existing.chunk_count == tile_mesh_count
            and existing.visible_mask == current_visible_mask
        ):
            tile_merged, tile_visible, tile_vertices, tile_refresh_at = _cached_tile_batch_stats(existing)
            if tile_visible > 0 and (tile_refresh_at <= 0.0 or now < tile_refresh_at):
                cached_draw_batches = getattr(existing, "cached_draw_batches", ())
                if cached_draw_batches:
                    if direct_render_batch_groups is not None:
                        cached_grouped_render_batches = getattr(existing, "cached_grouped_render_batches", ())
                        if cached_grouped_render_batches:
                            _extend_grouped_render_batch_groups(direct_render_batch_groups, cached_grouped_render_batches)
                        else:
                            _extend_direct_render_batches_grouped(direct_render_batch_groups, getattr(existing, "cached_render_batches", ()))
                    elif direct_render_batches is not None:
                        _extend_direct_render_batches(direct_render_batches, getattr(existing, "cached_render_batches", ()))
                    else:
                        draw_batches.extend(cached_draw_batches)
                merged_chunk_count += tile_merged
                visible_chunk_count += tile_visible
                visible_vertex_count += tile_vertices
                if tile_refresh_at > 0.0 and (next_refresh_at <= 0.0 or tile_refresh_at < next_refresh_at):
                    next_refresh_at = tile_refresh_at
                continue

        mature_meshes: list[ChunkMesh] = []
        immature_meshes: list[ChunkMesh] = []
        tile_next_refresh = 0.0
        for mesh in tile_meshes:
            if age_gate and now - float(mesh.created_at) < merged_tile_min_age_seconds:
                immature_meshes.append(mesh)
                mesh_refresh = float(mesh.created_at) + merged_tile_min_age_seconds
                if tile_next_refresh <= 0.0 or mesh_refresh < tile_next_refresh:
                    tile_next_refresh = mesh_refresh
            else:
                mature_meshes.append(mesh)
        if meshes is not None:
            mature_meshes.sort(key=lambda mesh: (mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z))
            immature_meshes.sort(key=lambda mesh: (mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z))
        if tile_next_refresh > 0.0 and (next_refresh_at <= 0.0 or tile_next_refresh < next_refresh_at):
            next_refresh_at = tile_next_refresh

        if tile_mesh_count == 1:
            mesh = tile_meshes[0]
            single_draw_batch = ChunkDrawBatch(
                vertex_buffer=mesh.vertex_buffer,
                binding_offset=mesh.binding_offset,
                vertex_count=mesh.vertex_count,
                first_vertex=mesh.first_vertex,
                bounds=mesh.bounds,
            )
            single_cached_render_batches = ((mesh.vertex_buffer, int(mesh.binding_offset), int(mesh.vertex_count), int(mesh.first_vertex)),)
            batch = _store_cached_tile_render_batch(
                renderer,
                tile_key_value,
                existing,
                signature=((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z),),
                vertex_count=int(mesh.vertex_count),
                vertex_buffer=mesh.vertex_buffer,
                bounds=mesh.bounds,
                chunk_count=1,
                complete_tile=False,
                all_mature=not age_gate or now - float(mesh.created_at) >= merged_tile_min_age_seconds,
                visible_mask=current_visible_mask,
                source_version=tile_version,
                cached_draw_batches=(single_draw_batch,),
                cached_render_batches=single_cached_render_batches,
                cached_grouped_render_batches=_group_render_batches(single_cached_render_batches),
                next_refresh_at=0.0 if (not age_gate or now - float(mesh.created_at) >= merged_tile_min_age_seconds) else float(mesh.created_at) + merged_tile_min_age_seconds,
                visible_chunk_count=1,
                merged_chunk_count=0,
                visible_vertex_count=int(mesh.vertex_count),
                owns_vertex_buffer=False,
            )
            if direct_render_batch_groups is not None:
                _extend_grouped_render_batch_groups(direct_render_batch_groups, batch.cached_grouped_render_batches)
            elif direct_render_batches is not None:
                direct_extend(direct_render_batches, batch.cached_render_batches)
            else:
                draw_batches_append(single_draw_batch)
            visible_chunk_count += 1
            visible_vertex_count += int(mesh.vertex_count)
            mesh_refresh_at = 0.0 if (not age_gate or now - float(mesh.created_at) >= merged_tile_min_age_seconds) else float(mesh.created_at) + merged_tile_min_age_seconds
            if mesh_refresh_at > 0.0 and (next_refresh_at <= 0.0 or mesh_refresh_at < next_refresh_at):
                next_refresh_at = mesh_refresh_at
            continue

        if len(mature_meshes) < 2:
            tile_draw_batches: list[ChunkDrawBatch] = []
            tile_visible_chunk_count = 0
            tile_visible_vertex_count = 0
            for mesh in immature_meshes:
                tile_draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                tile_visible_chunk_count += 1
                tile_visible_vertex_count += int(mesh.vertex_count)
            for mesh in mature_meshes:
                tile_draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                tile_visible_chunk_count += 1
                tile_visible_vertex_count += int(mesh.vertex_count)
            tile_cached_draw_batches = tuple(tile_draw_batches)
            tile_cached_render_batches = _draw_batches_to_render_batches(tile_draw_batches)
            batch = _store_cached_tile_render_batch(
                renderer,
                tile_key_value,
                existing,
                signature=tuple((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z) for mesh in tile_meshes),
                vertex_count=tile_visible_vertex_count,
                vertex_buffer=tile_draw_batches[0].vertex_buffer if tile_draw_batches else None,
                bounds=merge_chunk_bounds(renderer, tile_meshes),
                chunk_count=tile_visible_chunk_count,
                complete_tile=False,
                all_mature=(len(immature_meshes) == 0),
                visible_mask=current_visible_mask,
                source_version=tile_version,
                cached_draw_batches=tile_cached_draw_batches,
                cached_render_batches=tile_cached_render_batches,
                cached_grouped_render_batches=_group_render_batches(tile_cached_render_batches),
                next_refresh_at=tile_next_refresh,
                visible_chunk_count=tile_visible_chunk_count,
                merged_chunk_count=0,
                visible_vertex_count=tile_visible_vertex_count,
                owns_vertex_buffer=False,
            )
            if direct_render_batch_groups is not None:
                _extend_grouped_render_batch_groups(direct_render_batch_groups, batch.cached_grouped_render_batches)
            elif direct_render_batches is not None:
                direct_extend(direct_render_batches, batch.cached_render_batches)
            else:
                draw_batches_extend(tile_draw_batches)
            visible_chunk_count += tile_visible_chunk_count
            visible_vertex_count += tile_visible_vertex_count
            if tile_next_refresh > 0.0 and (next_refresh_at <= 0.0 or tile_next_refresh < next_refresh_at):
                next_refresh_at = tile_next_refresh
            continue

        signature = tuple((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z) for mesh in mature_meshes)
        batch_vertex_count = sum(mesh.vertex_count for mesh in mature_meshes)
        batch_vertex_bytes = int(batch_vertex_count) * int(renderer_module.VERTEX_STRIDE)
        batch_bounds = merge_chunk_bounds(renderer, mature_meshes)
        rebuild_merged = (
            existing is None
            or tile_is_dirty
            or existing.signature != signature
            or existing.vertex_count != batch_vertex_count
            or existing.chunk_count != len(mature_meshes)
            or existing.vertex_buffer is None
            or not getattr(existing, "owns_vertex_buffer", False)
        )
        merged_buffer = existing.vertex_buffer if not rebuild_merged else None
        merged_buffer_capacity_bytes = int(batch_vertex_bytes)
        if not rebuild_merged and existing is not None:
            merged_buffer_capacity_bytes = int(
                getattr(existing, "owned_vertex_buffer_capacity_bytes", batch_vertex_bytes)
                or batch_vertex_bytes
            )
        if rebuild_merged:
            reuse_merged_buffer = None
            reuse_capacity_bytes = 0
            if (
                existing is not None
                and getattr(existing, "owns_vertex_buffer", False)
                and existing.vertex_buffer is not None
            ):
                existing_capacity_bytes = int(
                    getattr(existing, "owned_vertex_buffer_capacity_bytes", 0)
                    or (int(existing.vertex_count) * int(renderer_module.VERTEX_STRIDE))
                )
                if existing_capacity_bytes >= batch_vertex_bytes:
                    reuse_merged_buffer = existing.vertex_buffer
                    reuse_capacity_bytes = existing_capacity_bytes
            merged_buffer, merged_buffer_capacity_bytes = merge_tile_meshes(
                renderer,
                mature_meshes,
                encoder,
                reuse_merged_buffer,
                reuse_capacity_bytes,
            )

        tile_draw_batches: list[ChunkDrawBatch] = [
            ChunkDrawBatch(
                vertex_buffer=merged_buffer,
                binding_offset=0,
                vertex_count=batch_vertex_count,
                first_vertex=0,
                bounds=batch_bounds,
                chunk_count=len(mature_meshes),
            )
        ]
        tile_visible_vertex_count = int(batch_vertex_count)
        for mesh in immature_meshes:
            tile_draw_batches.append(
                ChunkDrawBatch(
                    vertex_buffer=mesh.vertex_buffer,
                    binding_offset=mesh.binding_offset,
                    vertex_count=mesh.vertex_count,
                    first_vertex=mesh.first_vertex,
                    bounds=mesh.bounds,
                )
            )
            tile_visible_vertex_count += int(mesh.vertex_count)

        tile_cached_draw_batches = tuple(tile_draw_batches)
        tile_cached_render_batches = _draw_batches_to_render_batches(tile_draw_batches)
        batch = _store_cached_tile_render_batch(
            renderer,
            tile_key_value,
            existing,
            signature=signature,
            vertex_count=batch_vertex_count,
            vertex_buffer=merged_buffer,
            bounds=batch_bounds,
            chunk_count=len(mature_meshes),
            complete_tile=(tile_mesh_count == merged_tile_max_chunks and len(mature_meshes) == tile_mesh_count),
            all_mature=(len(mature_meshes) == tile_mesh_count),
            visible_mask=current_visible_mask,
            source_version=tile_version,
            cached_draw_batches=tile_cached_draw_batches,
            cached_render_batches=tile_cached_render_batches,
            cached_grouped_render_batches=_group_render_batches(tile_cached_render_batches),
            next_refresh_at=tile_next_refresh,
            visible_chunk_count=tile_mesh_count,
            merged_chunk_count=len(mature_meshes),
            visible_vertex_count=tile_visible_vertex_count,
            owns_vertex_buffer=True,
            owned_vertex_buffer_capacity_bytes=merged_buffer_capacity_bytes,
        )
        if direct_render_batch_groups is not None:
            _extend_grouped_render_batch_groups(direct_render_batch_groups, batch.cached_grouped_render_batches)
        elif direct_render_batches is not None:
            direct_extend(direct_render_batches, batch.cached_render_batches)
        else:
            draw_batches_extend(batch.cached_draw_batches)
        merged_chunk_count += batch.merged_chunk_count
        visible_chunk_count += batch.visible_chunk_count
        visible_vertex_count += batch.visible_vertex_count
        if tile_next_refresh > 0.0 and (next_refresh_at <= 0.0 or tile_next_refresh < next_refresh_at):
            next_refresh_at = tile_next_refresh

    _flush_merged_tile_buffer_reuse_queue(renderer)
    while len(renderer._transient_render_buffers) > 3:
        old_buffers = renderer._transient_render_buffers.pop(0)
        for buffer in old_buffers:
            buffer.destroy()

    if cache_key is not None and not renderer._visible_tile_dirty_keys:
        cache_until = float(next_refresh_at)
        if cache_until > 0.0:
            if now <= 0.0:
                now = time.perf_counter()
            cache_until = max(cache_until, now + MERGED_TILE_AGE_REFRESH_INTERVAL_SECONDS)
        renderer._cached_tile_draw_batches[cache_key] = (
            cache_until,
            draw_batches,
            merged_chunk_count,
            visible_chunk_count,
            visible_vertex_count,
        )
    return draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count, next_refresh_at


@profile
def ensure_mesh_draw_indirect_scratch(renderer, command_capacity: int) -> None:
    renderer_module = _renderer_module()
    indirect_draw_command_stride = int(renderer_module.INDIRECT_DRAW_COMMAND_STRIDE)

    needed = max(1, int(command_capacity))
    target_capacity = max(256, 1 << (needed - 1).bit_length())
    if (
        renderer._mesh_draw_indirect_capacity >= target_capacity
        and renderer._mesh_draw_indirect_buffer is not None
    ):
        return
    old_buffer = renderer._mesh_draw_indirect_buffer
    renderer._mesh_draw_indirect_capacity = target_capacity
    renderer._mesh_draw_indirect_buffer = renderer.device.create_buffer(
        size=max(indirect_draw_command_stride, target_capacity * indirect_draw_command_stride),
        usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    renderer._mesh_draw_indirect_array = np.empty((target_capacity, 4), dtype=np.uint32)
    if old_buffer is not None:
        wgpu_mesher.schedule_gpu_buffer_cleanup(renderer, [old_buffer], frames=6)


@profile
def ensure_mesh_visibility_scratch(renderer, record_capacity: int) -> None:
    renderer_module = _renderer_module()
    mesh_visibility_record_dtype = renderer_module.MESH_VISIBILITY_RECORD_DTYPE

    needed = max(1, int(record_capacity))
    target_capacity = max(256, 1 << (needed - 1).bit_length())
    if (
        renderer._mesh_visibility_record_capacity >= target_capacity
        and renderer._mesh_visibility_record_buffer is not None
        and renderer._mesh_draw_indirect_capacity >= target_capacity
        and renderer._mesh_draw_indirect_buffer is not None
    ):
        return
    old_buffer = renderer._mesh_visibility_record_buffer
    renderer._mesh_visibility_record_capacity = target_capacity
    renderer._mesh_visibility_record_buffer = renderer.device.create_buffer(
        size=max(mesh_visibility_record_dtype.itemsize, target_capacity * mesh_visibility_record_dtype.itemsize),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    renderer._mesh_visibility_record_array = np.empty(target_capacity, dtype=mesh_visibility_record_dtype)
    ensure_mesh_draw_indirect_scratch(renderer, target_capacity)
    if old_buffer is not None:
        wgpu_mesher.schedule_gpu_buffer_cleanup(renderer, [old_buffer], frames=6)


@profile
def build_gpu_visibility_records(
    renderer,
    encoder,
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], int, int, int, int]:
    draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count, _ = build_tile_draw_batches(
        renderer,
        None,
        encoder,
        age_gate=True,
    )

    groups: OrderedDict[tuple[int, int], tuple[wgpu.GPUBuffer, int, list[ChunkDrawBatch]]] = OrderedDict()
    for batch in draw_batches:
        key = (id(batch.vertex_buffer), batch.binding_offset)
        if key not in groups:
            groups[key] = (batch.vertex_buffer, batch.binding_offset, [])
        groups[key][2].append(batch)

    command_count = len(draw_batches)
    ensure_mesh_visibility_scratch(renderer, command_count)
    metadata_buffer = renderer._mesh_visibility_record_buffer
    metadata_array = renderer._mesh_visibility_record_array
    params_buffer = renderer._mesh_visibility_params_buffer
    assert metadata_buffer is not None
    assert params_buffer is not None

    render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]] = []
    command_index = 0
    for vertex_buffer, binding_offset, batches in groups.values():
        batch_start = command_index
        for batch in batches:
            metadata_array[command_index]["bounds"] = batch.bounds
            metadata_array[command_index]["draw"] = (
                batch.vertex_count,
                1,
                batch.first_vertex,
                0,
            )
            command_index += 1
        render_batches.append((vertex_buffer, binding_offset, batch_start, command_index - batch_start))

    if command_count > 0:
        renderer.device.queue.write_buffer(metadata_buffer, 0, memoryview(metadata_array[:command_count]))
        renderer.device.queue.write_buffer(params_buffer, 0, struct.pack("<4I", int(command_count), 0, 0, 0))

    return render_batches, command_count, merged_chunk_count, visible_chunk_count, visible_vertex_count


@profile
def visible_render_batches_indirect(
    renderer,
    encoder,
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], float, int, int, int, int]:
    encode_start = time.perf_counter()
    draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count, _ = build_tile_draw_batches(
        renderer,
        None,
        encoder,
        age_gate=True,
    )

    groups: OrderedDict[tuple[int, int], tuple[wgpu.GPUBuffer, int, list[ChunkDrawBatch]]] = OrderedDict()
    for batch in draw_batches:
        key = (id(batch.vertex_buffer), batch.binding_offset)
        if key not in groups:
            groups[key] = (batch.vertex_buffer, batch.binding_offset, [])
        groups[key][2].append(batch)

    command_count = len(draw_batches)
    ensure_mesh_draw_indirect_scratch(renderer, command_count)
    indirect_buffer = renderer._mesh_draw_indirect_buffer
    indirect_array = renderer._mesh_draw_indirect_array
    assert indirect_buffer is not None

    render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]] = []
    command_index = 0
    for vertex_buffer, binding_offset, batches in groups.values():
        batch_start = command_index
        for batch in batches:
            indirect_array[command_index, 0] = np.uint32(batch.vertex_count)
            indirect_array[command_index, 1] = np.uint32(1)
            indirect_array[command_index, 2] = np.uint32(batch.first_vertex)
            indirect_array[command_index, 3] = np.uint32(0)
            command_index += 1
        render_batches.append((vertex_buffer, binding_offset, batch_start, command_index - batch_start))

    if command_count > 0:
        renderer.device.queue.write_buffer(indirect_buffer, 0, memoryview(indirect_array[:command_count]))

    render_encode_ms = (time.perf_counter() - encode_start) * 1000.0
    return render_batches, render_encode_ms, command_count, merged_chunk_count, visible_chunk_count, visible_vertex_count


def visible_chunks(renderer) -> list[tuple[int, int, int, ChunkMesh]]:
    visible: list[tuple[int, int, int, ChunkMesh]] = []
    if not renderer._visible_chunk_coords:
        coord_manager.refresh_visible_chunk_coords(renderer)
    for chunk_x, chunk_y, chunk_z in renderer._visible_chunk_coords:
        key = (chunk_x, chunk_y, chunk_z)
        mesh = renderer.chunk_cache.get(key)
        if mesh is None:
            continue
        visible.append((chunk_x, chunk_y, chunk_z, mesh))
    return visible


def tile_key(renderer, chunk_x: int, chunk_z: int, chunk_y: int = 0) -> tuple[int, int, int]:
    renderer_module = _renderer_module()
    merged_tile_size_chunks = int(renderer_module.MERGED_TILE_SIZE_CHUNKS)
    return chunk_x // merged_tile_size_chunks, int(chunk_y), chunk_z // merged_tile_size_chunks


def tile_visible_mask(renderer, tile_key_value: tuple[int, int, int], tile_meshes: list[ChunkMesh]) -> int:
    renderer_module = _renderer_module()
    merged_tile_size_chunks = int(renderer_module.MERGED_TILE_SIZE_CHUNKS)
    tile_origin_x = int(tile_key_value[0]) * merged_tile_size_chunks
    tile_origin_z = int(tile_key_value[2]) * merged_tile_size_chunks
    mask = 0
    for mesh in tile_meshes:
        local_x = int(mesh.chunk_x) - tile_origin_x
        local_z = int(mesh.chunk_z) - tile_origin_z
        if 0 <= local_x < merged_tile_size_chunks and 0 <= local_z < merged_tile_size_chunks:
            mask |= 1 << (local_z * merged_tile_size_chunks + local_x)
    return mask


@profile
def visible_render_batches(
    renderer,
    encoder,
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], float, int, int, int, int]:
    cache_key = (int(renderer._visible_layout_version), int(renderer._visible_tile_mutation_version), 1)
    now = time.perf_counter()
    cached_entry = renderer._cached_visible_render_batches.get(cache_key)
    if cached_entry is not None and not renderer._visible_tile_dirty_keys:
        cached_until, cached_batches, cached_draw_calls, cached_merged, cached_visible, cached_vertices = cached_entry
        if cached_until <= 0.0 or now < cached_until:
            return cached_batches, 0.0, cached_draw_calls, cached_merged, cached_visible, cached_vertices

    encode_start = time.perf_counter()
    render_batch_groups: OrderedDict[tuple[int, int], list[tuple[wgpu.GPUBuffer, int, int, int]]] = OrderedDict()
    _draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count, next_refresh_at = build_tile_draw_batches(
        renderer,
        None,
        encoder,
        age_gate=True,
        direct_render_batch_groups=render_batch_groups,
    )
    render_batches = _finalize_direct_render_batch_groups(render_batch_groups)
    render_encode_ms = (time.perf_counter() - encode_start) * 1000.0
    if not renderer._visible_tile_dirty_keys:
        cache_until = float(next_refresh_at)
        if cache_until > 0.0:
            cache_until = max(cache_until, now + MERGED_TILE_AGE_REFRESH_INTERVAL_SECONDS)
        renderer._cached_visible_render_batches[cache_key] = (
            cache_until,
            render_batches,
            len(render_batches),
            merged_chunk_count,
            visible_chunk_count,
            visible_vertex_count,
        )
    return render_batches, render_encode_ms, len(render_batches), merged_chunk_count, visible_chunk_count, visible_vertex_count
