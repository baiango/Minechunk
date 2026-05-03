from __future__ import annotations

from collections import OrderedDict, deque
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


def _mesh_compaction_empty_stats(renderer, enabled: bool) -> dict[str, int | bool]:
    if hasattr(renderer, "_mesh_output_slabs"):
        slab_count, slab_total_bytes, _, _, _, _ = mesh_output_allocator_stats(renderer)
    else:
        slab_count = 0
        slab_total_bytes = 0
    return {
        "enabled": bool(enabled),
        "source_slabs": 0,
        "retired_slab_bytes": 0,
        "new_slab_bytes": 0,
        "net_reclaimed_bytes": 0,
        "copied_bytes": 0,
        "moved_allocations": 0,
        "moved_meshes": 0,
        "pending_retired_bytes": 0,
        "before_slab_bytes": int(slab_total_bytes),
        "after_slab_bytes": int(slab_total_bytes),
        "slab_count": int(slab_count),
    }


def _ensure_mesh_compaction_state(renderer) -> None:
    if not hasattr(renderer, "_mesh_compaction_retired_cleanup_bytes") or renderer._mesh_compaction_retired_cleanup_bytes is None:
        renderer._mesh_compaction_retired_cleanup_bytes = deque()
    if not hasattr(renderer, "_mesh_compaction_last_stats") or renderer._mesh_compaction_last_stats is None:
        renderer._mesh_compaction_last_stats = {}


def _process_mesh_compaction_retired_bytes(renderer) -> int:
    _ensure_mesh_compaction_state(renderer)
    queue = renderer._mesh_compaction_retired_cleanup_bytes
    if not queue:
        return 0
    next_queue = deque()
    pending_bytes = 0
    while queue:
        frames_left, byte_count = queue.popleft()
        frames_left = int(frames_left) - 1
        byte_count = int(byte_count)
        if frames_left > 0:
            next_queue.append((frames_left, byte_count))
            pending_bytes += byte_count
    renderer._mesh_compaction_retired_cleanup_bytes = next_queue
    return int(pending_bytes)


def _mesh_compaction_pending_retired_bytes(renderer) -> int:
    _ensure_mesh_compaction_state(renderer)
    return sum(int(byte_count) for _, byte_count in renderer._mesh_compaction_retired_cleanup_bytes)


def mesh_output_compaction_stats(renderer) -> dict[str, int | bool]:
    _ensure_mesh_compaction_state(renderer)
    stats = dict(getattr(renderer, "_mesh_compaction_last_stats", {}) or {})
    if not stats:
        enabled = bool(getattr(render_consts, "MESH_ZSTD_COMPACTION_ENABLED", True)) and bool(getattr(renderer, "mesh_zstd_enabled", False))
        stats = _mesh_compaction_empty_stats(renderer, enabled)
    stats["pending_retired_bytes"] = _mesh_compaction_pending_retired_bytes(renderer)
    return stats


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


def _mesh_zstd_readback_blocked_ids(renderer) -> tuple[set[int], set[int]]:
    pending_buffers: set[int] = set()
    pending_allocations: set[int] = set()
    for pending in getattr(renderer, "_pending_mesh_zstd_readbacks", ()) or ():
        vertex_buffer_id = int(getattr(pending, "vertex_buffer_id", 0) or 0)
        if vertex_buffer_id:
            pending_buffers.add(vertex_buffer_id)
        allocation_id = getattr(pending, "allocation_id", None)
        if allocation_id is not None:
            pending_allocations.add(int(allocation_id))
    return pending_buffers, pending_allocations


def _chunk_meshes_by_allocation(renderer) -> dict[int, list[ChunkMesh]]:
    meshes_by_allocation: dict[int, list[ChunkMesh]] = {}
    for mesh in getattr(renderer, "chunk_cache", {}).values():
        allocation_id = getattr(mesh, "allocation_id", None)
        if allocation_id is None:
            continue
        allocation_id = int(allocation_id)
        if allocation_id not in renderer._mesh_allocations:
            continue
        meshes_by_allocation.setdefault(allocation_id, []).append(mesh)
    return meshes_by_allocation


def _mesh_allocation_meshes_are_movable(allocation: MeshBufferAllocation, meshes: list[ChunkMesh]) -> bool:
    if not meshes:
        return False
    if int(getattr(allocation, "refcount", 0)) != len(meshes):
        return False
    allocation_start = int(allocation.offset_bytes)
    allocation_end = allocation_start + int(allocation.size_bytes)
    vertex_stride = int(render_consts.VERTEX_STRIDE)
    for mesh in meshes:
        if mesh.vertex_buffer is not allocation.buffer:
            return False
        mesh_start = int(getattr(mesh, "vertex_offset", 0))
        mesh_end = mesh_start + max(0, int(getattr(mesh, "vertex_count", 0))) * vertex_stride
        if mesh_start < allocation_start or mesh_end > allocation_end:
            return False
    return True


def _movable_mesh_allocations_by_slab(renderer) -> tuple[dict[int, list[tuple[MeshBufferAllocation, list[ChunkMesh]]]], set[int]]:
    meshes_by_allocation = _chunk_meshes_by_allocation(renderer)
    movable_by_slab: dict[int, list[tuple[MeshBufferAllocation, list[ChunkMesh]]]] = {}
    blocked_slab_ids: set[int] = set()
    slabs = getattr(renderer, "_mesh_output_slabs", {})
    for allocation_id, allocation in list(getattr(renderer, "_mesh_allocations", {}).items()):
        slab_id = getattr(allocation, "slab_id", None)
        if slab_id is None or int(slab_id) not in slabs:
            continue
        slab_id = int(slab_id)
        meshes = meshes_by_allocation.get(int(allocation_id), [])
        if not _mesh_allocation_meshes_are_movable(allocation, meshes):
            blocked_slab_ids.add(slab_id)
            continue
        movable_by_slab.setdefault(slab_id, []).append((allocation, meshes))
    for slab_id in blocked_slab_ids:
        movable_by_slab.pop(slab_id, None)
    return movable_by_slab, blocked_slab_ids


def _simulate_mesh_compaction_pack(renderer, allocation_sizes: list[int]) -> int:
    alignment = max(1, int(getattr(renderer, "_mesh_output_binding_alignment", 1)))
    simulated_slabs_by_class: dict[int, list[list[int]]] = {}
    total_slab_bytes = 0
    for allocation_size in sorted((int(size) for size in allocation_sizes), reverse=True):
        needed_bytes = render_consts.align_up(max(1, allocation_size), alignment)
        size_class_bytes = mesh_output_request_size_class(renderer, needed_bytes)
        class_slabs = simulated_slabs_by_class.setdefault(size_class_bytes, [])
        placed = False
        for slab in class_slabs:
            slab_size, append_offset = slab
            aligned_offset = render_consts.align_up(append_offset, alignment)
            alloc_end = aligned_offset + needed_bytes
            if alloc_end <= slab_size:
                slab[1] = alloc_end
                placed = True
                break
        if placed:
            continue
        slab_size = mesh_output_slab_size_for_request(renderer, needed_bytes)
        class_slabs.append([int(slab_size), int(needed_bytes)])
        total_slab_bytes += int(slab_size)
    return int(total_slab_bytes)


def _allocate_mesh_compaction_range(
    renderer,
    new_slabs_by_class: dict[int, list[MeshOutputSlab]],
    request_bytes: int,
) -> tuple[MeshOutputSlab, int, int]:
    alignment = max(1, int(getattr(renderer, "_mesh_output_binding_alignment", 1)))
    needed_bytes = render_consts.align_up(max(1, int(request_bytes)), alignment)
    size_class_bytes = mesh_output_request_size_class(renderer, needed_bytes)
    class_slabs = new_slabs_by_class.setdefault(size_class_bytes, [])
    for slab in class_slabs:
        _ensure_slab_free_range_indexes(slab)
        current_offset = int(slab.append_offset)
        aligned_offset = render_consts.align_up(current_offset, alignment)
        alloc_end = aligned_offset + needed_bytes
        if alloc_end > int(slab.size_bytes):
            continue
        padding = aligned_offset - current_offset
        if padding > 0:
            _insert_slab_free_range(slab, current_offset, padding)
        slab.append_offset = alloc_end
        _touch_mesh_output_slab(renderer, slab)
        return slab, int(aligned_offset), int(needed_bytes)

    slab = create_mesh_output_slab(
        renderer,
        mesh_output_slab_size_for_request(renderer, needed_bytes),
        size_class_bytes,
    )
    class_slabs.append(slab)
    current_offset = int(slab.append_offset)
    aligned_offset = render_consts.align_up(current_offset, alignment)
    alloc_end = aligned_offset + needed_bytes
    if alloc_end > int(slab.size_bytes):
        raise RuntimeError("Failed to allocate compacted mesh output range.")
    slab.append_offset = alloc_end
    return slab, int(aligned_offset), int(needed_bytes)


def _destroy_new_compaction_slabs(renderer, new_slabs_by_class: dict[int, list[MeshOutputSlab]]) -> None:
    for class_slabs in new_slabs_by_class.values():
        for slab in class_slabs:
            _unindex_mesh_output_slab(renderer, slab)
            renderer._mesh_output_slabs.pop(int(slab.slab_id), None)
            try:
                slab.buffer.destroy()
            except Exception:
                pass
    refresh_mesh_output_append_slab(renderer)


def _invalidate_moved_mesh_batches(renderer, affected_meshes: list[ChunkMesh]) -> None:
    if not affected_meshes:
        return
    from .tile_mesh_cache import mark_tile_dirty

    renderer._cached_tile_draw_batches.clear()
    renderer._cached_visible_render_batches.clear()
    seen_coords: set[tuple[int, int, int]] = set()
    for mesh in affected_meshes:
        coord = (int(mesh.chunk_x), int(getattr(mesh, "chunk_y", 0)), int(mesh.chunk_z))
        if coord in seen_coords:
            continue
        seen_coords.add(coord)
        mark_tile_dirty(renderer, coord[0], coord[2], coord[1])


@profile
def compact_mesh_output_slabs(
    renderer,
    *,
    enabled: bool | None = None,
    max_copy_bytes: int | None = None,
    max_source_slabs: int | None = None,
    min_reclaim_bytes: int | None = None,
) -> dict[str, int | bool]:
    _ensure_mesh_compaction_state(renderer)
    pending_retired_bytes = _process_mesh_compaction_retired_bytes(renderer)
    compaction_enabled = (
        bool(getattr(render_consts, "MESH_ZSTD_COMPACTION_ENABLED", True))
        and bool(getattr(renderer, "mesh_zstd_enabled", False))
        if enabled is None
        else bool(enabled)
    )
    stats = _mesh_compaction_empty_stats(renderer, compaction_enabled)
    stats["pending_retired_bytes"] = int(pending_retired_bytes)
    if not compaction_enabled:
        renderer._mesh_compaction_last_stats = stats
        return stats

    max_copy_bytes = int(
        getattr(render_consts, "MESH_ZSTD_COMPACTION_MAX_COPY_BYTES_PER_FRAME", 1024 * 1024 * 1024)
        if max_copy_bytes is None
        else max_copy_bytes
    )
    max_source_slabs = int(
        getattr(render_consts, "MESH_ZSTD_COMPACTION_MAX_SOURCE_SLABS_PER_FRAME", 32)
        if max_source_slabs is None
        else max_source_slabs
    )
    min_reclaim_bytes = int(
        getattr(render_consts, "MESH_ZSTD_COMPACTION_MIN_RECLAIM_BYTES", 32 * 1024 * 1024)
        if min_reclaim_bytes is None
        else min_reclaim_bytes
    )
    if max_copy_bytes <= 0 or max_source_slabs <= 0:
        renderer._mesh_compaction_last_stats = stats
        return stats

    pending_buffer_ids, pending_allocation_ids = _mesh_zstd_readback_blocked_ids(renderer)
    movable_by_slab, blocked_slab_ids = _movable_mesh_allocations_by_slab(renderer)
    candidates: list[tuple[int, int, int, MeshOutputSlab, list[tuple[MeshBufferAllocation, list[ChunkMesh]]]]] = []
    for slab_id, allocations in movable_by_slab.items():
        if slab_id in blocked_slab_ids:
            continue
        slab = renderer._mesh_output_slabs.get(int(slab_id))
        if slab is None:
            continue
        if id(slab.buffer) in pending_buffer_ids:
            continue
        if any(int(allocation.allocation_id) in pending_allocation_ids for allocation, _ in allocations):
            continue
        # Retiring the source slab is only valid if every live allocation in it
        # was found in chunk_cache and can be moved.
        live_allocations_in_slab = [
            allocation
            for allocation in renderer._mesh_allocations.values()
            if getattr(allocation, "slab_id", None) is not None and int(allocation.slab_id) == int(slab_id)
        ]
        if len(live_allocations_in_slab) != len(allocations):
            continue
        free_bytes = slab_total_free_bytes(slab)
        if free_bytes <= 0:
            continue
        live_bytes = sum(int(allocation.size_bytes) for allocation, _ in allocations)
        if live_bytes <= 0:
            continue
        candidates.append((int(free_bytes), int(live_bytes), int(slab_id), slab, allocations))

    if not candidates:
        renderer._mesh_compaction_last_stats = stats
        return stats

    selected: list[tuple[int, int, int, MeshOutputSlab, list[tuple[MeshBufferAllocation, list[ChunkMesh]]]]] = []
    selected_live_bytes = 0
    for candidate in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if len(selected) >= max_source_slabs:
            break
        candidate_live_bytes = int(candidate[1])
        if selected_live_bytes + candidate_live_bytes > max_copy_bytes:
            continue
        selected.append(candidate)
        selected_live_bytes += candidate_live_bytes

    if not selected:
        renderer._mesh_compaction_last_stats = stats
        return stats

    source_slabs = [candidate[3] for candidate in selected]
    source_slab_ids = {int(slab.slab_id) for slab in source_slabs}
    source_total_bytes = sum(int(slab.size_bytes) for slab in source_slabs)
    allocation_infos: list[tuple[MeshBufferAllocation, list[ChunkMesh]]] = []
    seen_allocation_ids: set[int] = set()
    for _, _, _, _, allocations in selected:
        for allocation, meshes in allocations:
            allocation_id = int(allocation.allocation_id)
            if allocation_id in seen_allocation_ids:
                continue
            seen_allocation_ids.add(allocation_id)
            allocation_infos.append((allocation, meshes))
    allocation_infos.sort(key=lambda item: int(item[0].size_bytes), reverse=True)
    new_total_bytes = _simulate_mesh_compaction_pack(
        renderer,
        [int(allocation.size_bytes) for allocation, _ in allocation_infos],
    )
    net_reclaimed_bytes = int(source_total_bytes) - int(new_total_bytes)
    if net_reclaimed_bytes < max(0, min_reclaim_bytes):
        renderer._mesh_compaction_last_stats = stats
        return stats

    new_slabs_by_class: dict[int, list[MeshOutputSlab]] = {}
    moves: list[tuple[MeshBufferAllocation, object, int, int | None, MeshOutputSlab, int, int, list[ChunkMesh], list[int]]] = []
    encoder = None
    try:
        encoder = renderer.device.create_command_encoder()
        for allocation, meshes in allocation_infos:
            old_buffer = allocation.buffer
            old_offset = int(allocation.offset_bytes)
            old_slab_id = allocation.slab_id
            new_slab, new_offset, new_size = _allocate_mesh_compaction_range(
                renderer,
                new_slabs_by_class,
                int(allocation.size_bytes),
            )
            encoder.copy_buffer_to_buffer(old_buffer, old_offset, new_slab.buffer, new_offset, int(allocation.size_bytes))
            relative_offsets = [int(mesh.vertex_offset) - old_offset for mesh in meshes]
            moves.append((allocation, old_buffer, old_offset, old_slab_id, new_slab, new_offset, new_size, meshes, relative_offsets))
        if moves:
            renderer.device.queue.submit([encoder.finish()])
    except Exception:
        _destroy_new_compaction_slabs(renderer, new_slabs_by_class)
        renderer._mesh_compaction_last_stats = stats
        return stats

    affected_meshes: list[ChunkMesh] = []
    vertex_stride = int(render_consts.VERTEX_STRIDE)
    for allocation, _old_buffer, _old_offset, _old_slab_id, new_slab, new_offset, _new_size, meshes, relative_offsets in moves:
        allocation.buffer = new_slab.buffer
        allocation.offset_bytes = int(new_offset)
        allocation.slab_id = int(new_slab.slab_id)
        for mesh, relative_offset in zip(meshes, relative_offsets):
            mesh.vertex_buffer = new_slab.buffer
            mesh.vertex_offset = int(new_offset) + int(relative_offset)
            mesh.binding_offset = int(mesh.vertex_offset % vertex_stride)
            mesh.first_vertex = int((int(mesh.vertex_offset) - int(mesh.binding_offset)) // vertex_stride)
            affected_meshes.append(mesh)

    _invalidate_moved_mesh_batches(renderer, affected_meshes)

    old_buffers = []
    for slab in source_slabs:
        _unindex_mesh_output_slab(renderer, slab)
        renderer._mesh_output_slabs.pop(int(slab.slab_id), None)
        old_buffers.append(slab.buffer)
    if getattr(renderer, "_mesh_output_append_slab_id", None) in source_slab_ids:
        refresh_mesh_output_append_slab(renderer)

    if old_buffers:
        from ..meshing import gpu_mesher as wgpu_mesher

        delay_frames = max(1, int(getattr(render_consts, "MESH_OUTPUT_FREE_DELAY_FRAMES", 16)))
        wgpu_mesher.schedule_gpu_buffer_cleanup(renderer, old_buffers, frames=delay_frames)
        renderer._mesh_compaction_retired_cleanup_bytes.append((delay_frames, int(source_total_bytes)))
        pending_retired_bytes = _mesh_compaction_pending_retired_bytes(renderer)

    after_slab_count, after_slab_total_bytes, _, _, _, _ = mesh_output_allocator_stats(renderer)
    stats.update(
        {
            "source_slabs": int(len(source_slabs)),
            "retired_slab_bytes": int(source_total_bytes),
            "new_slab_bytes": int(new_total_bytes),
            "net_reclaimed_bytes": max(0, int(net_reclaimed_bytes)),
            "copied_bytes": int(sum(int(allocation.size_bytes) for allocation, _, _, _, _, _, _, _, _ in moves)),
            "moved_allocations": int(len(moves)),
            "moved_meshes": int(len(affected_meshes)),
            "pending_retired_bytes": int(pending_retired_bytes),
            "after_slab_bytes": int(after_slab_total_bytes),
            "slab_count": int(after_slab_count),
        }
    )
    renderer._mesh_compaction_last_stats = stats
    return stats


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
        shared_empty_buffer = getattr(renderer, "_shared_empty_chunk_vertex_buffer", None)
        if shared_empty_buffer is not None and mesh.vertex_buffer is shared_empty_buffer:
            return
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
