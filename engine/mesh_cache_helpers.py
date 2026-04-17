from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_left
import math
import struct
import time

import numpy as np
import wgpu

from . import render_constants as render_consts
from .meshing_types import ChunkDrawBatch, ChunkMesh, ChunkRenderBatch, MeshBufferAllocation, MeshOutputSlab
from . import wgpu_chunk_mesher as wgpu_mesher


def _renderer_module():
    from . import renderer as renderer_module

    return renderer_module


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
    renderer._mesh_output_append_slab_id = slab.slab_id
    return slab


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
    renderer._mesh_output_slabs.move_to_end(slab.slab_id)
    renderer._mesh_output_append_slab_id = slab.slab_id
    return allocation


def allocate_from_mesh_output_slab_bump(
    renderer,
    slab: MeshOutputSlab,
    request_bytes: int,
) -> MeshBufferAllocation | None:
    _ensure_slab_free_range_indexes(slab)
    needed_bytes = max(1, int(request_bytes))
    current_offset = int(slab.append_offset)
    aligned_offset = renderer._align_up(current_offset, renderer._mesh_output_binding_alignment)
    alloc_end = aligned_offset + needed_bytes
    if alloc_end > int(slab.size_bytes):
        return None
    padding = aligned_offset - current_offset
    if padding > 0:
        _insert_slab_free_range(slab, current_offset, padding)
    slab.append_offset = alloc_end
    return register_mesh_output_allocation(renderer, slab, aligned_offset, needed_bytes)


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
        aligned_offset = renderer._align_up(int(range_offset), alignment)
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
        aligned_offset = renderer._align_up(int(range_offset), alignment)
        padding = aligned_offset - int(range_offset)
        usable_size = int(range_size) - padding
        if usable_size < needed_bytes:
            continue
        waste = usable_size - needed_bytes
        return (waste, int(range_size), int(size_index), int(range_offset), int(aligned_offset))
    return None


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


def allocate_from_mesh_output_slab(
    renderer,
    slab: MeshOutputSlab,
    request_bytes: int,
) -> MeshBufferAllocation | None:
    allocation = allocate_from_mesh_output_slab_bump(renderer, slab, request_bytes)
    if allocation is not None:
        return allocation
    return allocate_from_mesh_output_slab_free_ranges(renderer, slab, request_bytes)


def allocate_mesh_output_range(renderer, request_bytes: int) -> MeshBufferAllocation:
    needed_bytes = max(1, int(request_bytes))
    needed_bytes = renderer._align_up(needed_bytes, renderer._mesh_output_binding_alignment)
    size_class_bytes = mesh_output_request_size_class(renderer, needed_bytes)

    slab_items = list(renderer._mesh_output_slabs.items())
    if slab_items:
        for _, slab in slab_items:
            trim_mesh_output_slab_tail(renderer, slab)

        class_slab_items = [
            (slab_id, slab)
            for slab_id, slab in slab_items
            if slab_accepts_size_class(slab, size_class_bytes)
        ]

        best_free_choice: tuple[int, int, int, MeshOutputSlab] | None = None
        for slab_id, slab in class_slab_items:
            choice = find_mesh_output_slab_free_range_choice(renderer, slab, needed_bytes)
            if choice is None:
                continue
            waste, range_size, _, _, _ = choice
            ranked = (waste, range_size, slab_id)
            if best_free_choice is None or ranked < best_free_choice[:3]:
                best_free_choice = (waste, range_size, slab_id, slab)
        if best_free_choice is not None:
            allocation = allocate_from_mesh_output_slab_free_ranges(renderer, best_free_choice[3], needed_bytes)
            if allocation is not None:
                return allocation

        append_candidates: list[tuple[int, int, int, MeshOutputSlab]] = []
        for slab_id, slab in class_slab_items:
            tail_free_bytes = slab_tail_free_bytes(slab)
            if tail_free_bytes < needed_bytes:
                continue
            append_candidates.append((tail_free_bytes - needed_bytes, abs(int(slab.size_bytes) - needed_bytes), slab_id, slab))
        if append_candidates:
            _, _, _, slab = min(append_candidates)
            allocation = allocate_from_mesh_output_slab_bump(renderer, slab, needed_bytes)
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
    renderer._mesh_output_slabs.pop(slab.slab_id, None)
    if renderer._mesh_output_append_slab_id == slab.slab_id:
        refresh_mesh_output_append_slab(renderer)
    slab.buffer.destroy()


def free_mesh_output_range(renderer, slab_id: int, offset_bytes: int, size_bytes: int) -> None:
    slab = renderer._mesh_output_slabs.get(int(slab_id))
    if slab is None:
        return
    _insert_slab_free_range(slab, int(offset_bytes), int(size_bytes))
    trim_mesh_output_slab_tail(renderer, slab)
    retire_mesh_output_slab_if_empty(renderer, slab)
    if slab.slab_id not in renderer._mesh_output_slabs:
        return
    renderer._mesh_output_slabs.move_to_end(slab.slab_id)
    refresh_mesh_output_append_slab(renderer)


def schedule_mesh_output_range_free(renderer, slab_id: int, offset_bytes: int, size_bytes: int) -> None:
    delay_frames = max(1, int(render_consts.MESH_OUTPUT_FREE_DELAY_FRAMES))
    renderer._deferred_mesh_output_frees.append(
        (delay_frames, int(slab_id), int(offset_bytes), int(size_bytes))
    )


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


def retain_chunk_mesh_storage(renderer, mesh: ChunkMesh) -> None:
    if mesh.allocation_id is None:
        renderer._retain_mesh_buffer(mesh.vertex_buffer)
        return
    allocation = renderer._mesh_allocations.get(mesh.allocation_id)
    if allocation is None:
        raise RuntimeError(f"Unknown mesh allocation id: {mesh.allocation_id}")
    allocation.refcount += 1


def release_chunk_mesh_storage(renderer, mesh: ChunkMesh) -> None:
    if mesh.allocation_id is None:
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


def clear_tile_render_batches(renderer) -> None:
    for batch in renderer._tile_render_batches.values():
        batch.vertex_buffer.destroy()
    renderer._tile_render_batches.clear()
    renderer._tile_dirty_keys.clear()
    renderer._tile_versions.clear()
    renderer._tile_mutation_version += 1
    renderer._cached_tile_draw_batches.clear()


def mark_tile_dirty(renderer, chunk_x: int, chunk_z: int, chunk_y: int = 0) -> None:
    key = tile_key(renderer, int(chunk_x), int(chunk_z), int(chunk_y))
    renderer._tile_dirty_keys.add(key)
    renderer._tile_versions[key] = int(renderer._tile_versions.get(key, 0)) + 1
    renderer._tile_mutation_version += 1
    renderer._cached_tile_draw_batches.clear()


def store_chunk_meshes(renderer, meshes: list[ChunkMesh]) -> None:
    if not meshes:
        return

    chunk_cache = renderer.chunk_cache
    cache_get = chunk_cache.get
    cache_move_to_end = chunk_cache.move_to_end
    release = lambda mesh: release_chunk_mesh_storage(renderer, mesh)
    retain = lambda mesh: retain_chunk_mesh_storage(renderer, mesh)
    visible_chunk_coord_set = renderer._visible_chunk_coord_set

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

    pending_chunk_coords = renderer._pending_chunk_coords
    pending_chunk_coords.difference_update(mesh_key_set)

    visible_keys = mesh_key_set & visible_chunk_coord_set
    if visible_keys:
        renderer._visible_displayed_coords.update(visible_keys)
        renderer._visible_missing_coords.difference_update(visible_keys)

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
        release(old_mesh)
        if old_key in visible_chunk_coord_set:
            visible_displayed_coords.discard(old_key)
            if old_key not in pending_chunk_coords:
                visible_missing_coords.add(old_key)
                queue_dirty = True

    if queue_dirty:
        renderer._chunk_request_queue_dirty = True


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


def merge_tile_meshes(renderer, tile_meshes: list[ChunkMesh], encoder) -> wgpu.GPUBuffer:
    renderer_module = _renderer_module()
    vertex_stride = int(renderer_module.VERTEX_STRIDE)
    merged_tile_max_chunks = int(renderer_module.MERGED_TILE_MAX_CHUNKS)

    total_vertices = sum(mesh.vertex_count for mesh in tile_meshes)
    total_vertex_bytes = total_vertices * vertex_stride
    merged_buffer = renderer.device.create_buffer(
        size=max(1, total_vertex_bytes),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
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
        return merged_buffer

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
    return merged_buffer


def visible_tile_mesh_groups(renderer) -> dict[tuple[int, int, int], list[ChunkMesh]]:
    if not renderer._visible_chunk_coords:
        renderer._refresh_visible_chunk_coords()
    tile_keys = getattr(renderer, "_visible_tile_keys", None) or []
    tile_coords = getattr(renderer, "_visible_tile_coords", None) or {}
    chunk_cache = renderer.chunk_cache
    tile_groups: dict[tuple[int, int, int], list[ChunkMesh]] = {}
    for tile_key_value in tile_keys:
        coords = tile_coords.get(tile_key_value)
        if not coords:
            continue
        meshes: list[ChunkMesh] = []
        for coord in coords:
            mesh = chunk_cache.get(coord)
            if mesh is None or mesh.vertex_count <= 0:
                continue
            chunk_cache.move_to_end(coord)
            meshes.append(mesh)
        if meshes:
            tile_groups[tile_key_value] = meshes
    return tile_groups


def visible_tile_chunk_groups(renderer) -> tuple[dict[tuple[int, int, int], list[tuple[int, int, int, ChunkMesh]]], int]:
    if not renderer._visible_chunk_coords:
        renderer._refresh_visible_chunk_coords()
    tile_keys = getattr(renderer, "_visible_tile_keys", None) or []
    tile_coords = getattr(renderer, "_visible_tile_coords", None) or {}
    chunk_cache = renderer.chunk_cache
    tile_groups: dict[tuple[int, int, int], list[tuple[int, int, int, ChunkMesh]]] = {}
    visible_count = 0
    for tile_key_value in tile_keys:
        coords = tile_coords.get(tile_key_value)
        if not coords:
            continue
        chunks: list[tuple[int, int, int, ChunkMesh]] = []
        for chunk_x, chunk_y, chunk_z in coords:
            coord = (chunk_x, chunk_y, chunk_z)
            mesh = chunk_cache.get(coord)
            if mesh is None or mesh.vertex_count <= 0:
                continue
            chunk_cache.move_to_end(coord)
            chunks.append((chunk_x, chunk_y, chunk_z, mesh))
        if chunks:
            tile_groups[tile_key_value] = chunks
            visible_count += len(chunks)
    return tile_groups, visible_count


def build_tile_draw_batches(
    renderer,
    meshes: list[ChunkMesh] | None,
    encoder,
    *,
    age_gate: bool,
) -> tuple[list[ChunkDrawBatch], int, int, int]:
    renderer_module = _renderer_module()
    merged_tile_min_age_seconds = float(renderer_module.MERGED_TILE_MIN_AGE_SECONDS)

    cache_key: tuple[int, int, int] | None = None
    if meshes is None:
        cache_key = (int(renderer._visible_layout_version), int(renderer._tile_mutation_version), 1 if age_gate else 0)
        cached = renderer._cached_tile_draw_batches.get(cache_key)
        if cached is not None and not renderer._tile_dirty_keys:
            cached_batches, cached_merged, cached_visible, cached_vertices = cached
            return list(cached_batches), cached_merged, cached_visible, cached_vertices
        tile_groups = visible_tile_mesh_groups(renderer)
    else:
        tile_groups: dict[tuple[int, int], list[ChunkMesh]] = {}
        for mesh in meshes:
            if mesh.vertex_count <= 0:
                continue
            tile_groups.setdefault(tile_key(renderer, mesh.chunk_x, mesh.chunk_z, getattr(mesh, "chunk_y", 0)), []).append(mesh)

    current_tile_keys = set(tile_groups.keys())
    stale_keys = [tile_key_value for tile_key_value in renderer._tile_render_batches if tile_key_value not in current_tile_keys]
    for tile_key_value in stale_keys:
        batch = renderer._tile_render_batches.pop(tile_key_value)
        renderer._transient_render_buffers.append([batch.vertex_buffer])

    draw_batches: list[ChunkDrawBatch] = []
    merged_chunk_count = 0
    visible_chunk_count = 0

    merged_tile_max_chunks = int(renderer_module.MERGED_TILE_MAX_CHUNKS)
    for tile_key_value in sorted(tile_groups):
        tile_meshes = tile_groups[tile_key_value]
        existing = renderer._tile_render_batches.get(tile_key_value)
        tile_version = int(renderer._tile_versions.get(tile_key_value, 0))
        tile_is_dirty = tile_key_value in renderer._tile_dirty_keys or (existing is not None and existing.source_version != tile_version)
        current_visible_mask = int(getattr(renderer, "_visible_tile_masks", {}).get(tile_key_value, tile_visible_mask(renderer, tile_key_value, tile_meshes)))
        if (
            existing is not None
            and not tile_is_dirty
            and existing.all_mature
            and existing.chunk_count == len(tile_meshes)
            and existing.visible_mask == current_visible_mask
        ):
            draw_batches.append(
                ChunkDrawBatch(
                    vertex_buffer=existing.vertex_buffer,
                    binding_offset=0,
                    vertex_count=existing.vertex_count,
                    first_vertex=0,
                    bounds=existing.bounds,
                    chunk_count=existing.chunk_count,
                )
            )
            merged_chunk_count += existing.chunk_count
            visible_chunk_count += existing.chunk_count
            continue

        mature_meshes = sorted(
            (mesh for mesh in tile_meshes if chunk_mesh_age(renderer, mesh) >= merged_tile_min_age_seconds),
            key=lambda mesh: (mesh.chunk_x, mesh.chunk_z),
        )
        immature_meshes = sorted(
            (mesh for mesh in tile_meshes if chunk_mesh_age(renderer, mesh) < merged_tile_min_age_seconds),
            key=lambda mesh: (mesh.chunk_x, mesh.chunk_z),
        )

        if len(tile_meshes) == 1:
            mesh = tile_meshes[0]
            draw_batches.append(
                ChunkDrawBatch(
                    vertex_buffer=mesh.vertex_buffer,
                    binding_offset=mesh.binding_offset,
                    vertex_count=mesh.vertex_count,
                    first_vertex=mesh.first_vertex,
                    bounds=mesh.bounds,
                )
            )
            renderer._tile_dirty_keys.discard(tile_key_value)
            visible_chunk_count += 1
            continue

        if len(mature_meshes) < 2:
            if existing is not None:
                renderer._tile_render_batches.pop(tile_key_value, None)
                renderer._transient_render_buffers.append([existing.vertex_buffer])
            for mesh in immature_meshes:
                draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                visible_chunk_count += 1
            for mesh in mature_meshes:
                draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                visible_chunk_count += 1
            renderer._tile_dirty_keys.discard(tile_key_value)
            continue

        signature = tuple((mesh.chunk_x, getattr(mesh, "chunk_y", 0), mesh.chunk_z) for mesh in mature_meshes)
        batch_vertex_count = sum(mesh.vertex_count for mesh in mature_meshes)
        batch_bounds = merge_chunk_bounds(renderer, mature_meshes)
        if (
            existing is None
            or tile_is_dirty
            or existing.signature != signature
            or existing.vertex_count != batch_vertex_count
            or existing.chunk_count != len(mature_meshes)
        ):
            old_buffer = existing.vertex_buffer if existing is not None else None
            merged_buffer = merge_tile_meshes(renderer, mature_meshes, encoder)
            renderer._tile_render_batches[tile_key_value] = ChunkRenderBatch(
                signature=signature,
                vertex_count=batch_vertex_count,
                vertex_buffer=merged_buffer,
                bounds=batch_bounds,
                chunk_count=len(mature_meshes),
                complete_tile=(len(tile_meshes) == merged_tile_max_chunks and len(mature_meshes) == len(tile_meshes)),
                all_mature=(len(mature_meshes) == len(tile_meshes)),
                visible_mask=current_visible_mask,
                source_version=tile_version,
            )
            renderer._tile_dirty_keys.discard(tile_key_value)
            if old_buffer is not None:
                renderer._transient_render_buffers.append([old_buffer])

        batch = renderer._tile_render_batches[tile_key_value]
        draw_batches.append(
            ChunkDrawBatch(
                vertex_buffer=batch.vertex_buffer,
                binding_offset=0,
                vertex_count=batch.vertex_count,
                first_vertex=0,
                bounds=batch.bounds,
                chunk_count=batch.chunk_count,
            )
        )
        merged_chunk_count += len(mature_meshes)
        visible_chunk_count += len(mature_meshes)
        for mesh in immature_meshes:
            draw_batches.append(
                ChunkDrawBatch(
                    vertex_buffer=mesh.vertex_buffer,
                    binding_offset=mesh.binding_offset,
                    vertex_count=mesh.vertex_count,
                    first_vertex=mesh.first_vertex,
                    bounds=mesh.bounds,
                )
            )
            visible_chunk_count += 1

    while len(renderer._transient_render_buffers) > 3:
        old_buffers = renderer._transient_render_buffers.pop(0)
        for buffer in old_buffers:
            buffer.destroy()

    visible_vertex_count = sum(batch.vertex_count for batch in draw_batches)
    if (
        cache_key is not None
        and not renderer._tile_dirty_keys
        and merged_chunk_count == visible_chunk_count
    ):
        renderer._cached_tile_draw_batches[cache_key] = (
            list(draw_batches),
            merged_chunk_count,
            visible_chunk_count,
            visible_vertex_count,
        )
    return draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count


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


def build_gpu_visibility_records(
    renderer,
    encoder,
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], int, int, int, int]:
    draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count = build_tile_draw_batches(
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


def visible_render_batches_indirect(
    renderer,
    encoder,
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], float, int, int, int, int]:
    encode_start = time.perf_counter()
    draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count = build_tile_draw_batches(
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
        renderer._refresh_visible_chunk_coords()
    for chunk_x, chunk_y, chunk_z in renderer._visible_chunk_coords:
        key = (chunk_x, chunk_y, chunk_z)
        mesh = renderer.chunk_cache.get(key)
        if mesh is None:
            continue
        renderer.chunk_cache.move_to_end(key)
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


def visible_render_batches(
    renderer,
    encoder,
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int]], float, int, int, int, int]:
    encode_start = time.perf_counter()
    visible = visible_chunks(renderer)
    render_batches: list[tuple[wgpu.GPUBuffer, int, int]] = []
    visible_count = 0
    visible_vertex_count = 0
    for _, _, _, mesh in visible:
        if mesh.vertex_count <= 0:
            continue
        render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))
        visible_count += 1
        visible_vertex_count += int(mesh.vertex_count)
    render_encode_ms = (time.perf_counter() - encode_start) * 1000.0
    return render_batches, render_encode_ms, len(render_batches), 0, visible_count, visible_vertex_count
