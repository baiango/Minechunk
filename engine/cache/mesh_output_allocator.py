from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_left

import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkMesh, MeshBufferAllocation, MeshOutputSlab


try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func

def chunk_cache_memory_bytes(renderer) -> int:
    vertex_stride = int(render_consts.VERTEX_STRIDE)

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

