from __future__ import annotations

from collections import OrderedDict
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


def create_mesh_output_slab(renderer, size_bytes: int) -> MeshOutputSlab:
    slab = MeshOutputSlab(
        slab_id=renderer._next_mesh_output_slab_id,
        buffer=renderer.device.create_buffer(
            size=max(1, int(size_bytes)),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        ),
        size_bytes=int(size_bytes),
        free_ranges=[],
        append_offset=0,
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
    needed_bytes = max(1, int(request_bytes))
    aligned_offset = renderer._align_up(int(slab.append_offset), renderer._mesh_output_binding_alignment)
    alloc_end = aligned_offset + needed_bytes
    if alloc_end > int(slab.size_bytes):
        return None
    slab.append_offset = alloc_end
    return register_mesh_output_allocation(renderer, slab, aligned_offset, needed_bytes)


def allocate_from_mesh_output_slab_free_ranges(
    renderer,
    slab: MeshOutputSlab,
    request_bytes: int,
) -> MeshBufferAllocation | None:
    needed_bytes = max(1, int(request_bytes))
    alignment = renderer._mesh_output_binding_alignment
    for index, (range_offset, range_size) in enumerate(slab.free_ranges):
        aligned_offset = renderer._align_up(range_offset, alignment)
        padding = aligned_offset - range_offset
        usable_size = range_size - padding
        if usable_size < needed_bytes:
            continue

        alloc_end = aligned_offset + needed_bytes
        new_ranges: list[tuple[int, int]] = []
        if padding > 0:
            new_ranges.append((range_offset, padding))
        tail_size = (range_offset + range_size) - alloc_end
        if tail_size > 0:
            new_ranges.append((alloc_end, tail_size))
        slab.free_ranges[index:index + 1] = new_ranges
        return register_mesh_output_allocation(renderer, slab, aligned_offset, needed_bytes)
    return None


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

    append_slab = None
    if renderer._mesh_output_append_slab_id is not None:
        append_slab = renderer._mesh_output_slabs.get(renderer._mesh_output_append_slab_id)
    if append_slab is not None:
        allocation = allocate_from_mesh_output_slab_bump(renderer, append_slab, needed_bytes)
        if allocation is not None:
            return allocation
        allocation = allocate_from_mesh_output_slab_free_ranges(renderer, append_slab, needed_bytes)
        if allocation is not None:
            return allocation

    renderer_module = _renderer_module()
    scan_limit = max(0, int(renderer_module.MESH_OUTPUT_FREERANGE_SCAN_LIMIT))
    if renderer._mesh_output_slabs:
        scanned = 0
        for slab_id, slab in reversed(list(renderer._mesh_output_slabs.items())):
            if append_slab is not None and slab_id == append_slab.slab_id:
                continue
            allocation = allocate_from_mesh_output_slab_free_ranges(renderer, slab, needed_bytes)
            if allocation is not None:
                return allocation
            scanned += 1
            if scan_limit > 0 and scanned >= scan_limit:
                break

    slab = create_mesh_output_slab(renderer, mesh_output_slab_size_for_request(renderer, needed_bytes))
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
    tail_offset = int(slab.append_offset)
    if tail_offset <= 0 or not slab.free_ranges:
        return
    while slab.free_ranges:
        range_offset, range_size = slab.free_ranges[-1]
        range_end = int(range_offset) + int(range_size)
        if range_end != tail_offset:
            break
        tail_offset = int(range_offset)
        slab.free_ranges.pop()
    slab.append_offset = tail_offset


def refresh_mesh_output_append_slab(renderer) -> None:
    renderer._mesh_output_append_slab_id = None
    for slab_id, slab in reversed(list(renderer._mesh_output_slabs.items())):
        if int(slab.append_offset) < int(slab.size_bytes):
            renderer._mesh_output_append_slab_id = slab_id
            return


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
    slab.free_ranges.append((int(offset_bytes), int(size_bytes)))
    slab.free_ranges = coalesce_mesh_output_free_ranges(slab.free_ranges)
    trim_mesh_output_slab_tail(renderer, slab)
    retire_mesh_output_slab_if_empty(renderer, slab)
    if slab.slab_id not in renderer._mesh_output_slabs:
        return
    if renderer._mesh_output_append_slab_id is None or int(slab.append_offset) < int(slab.size_bytes):
        renderer._mesh_output_append_slab_id = slab.slab_id


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


def store_chunk_meshes(renderer, meshes: list[ChunkMesh]) -> None:
    if not meshes:
        return

    chunk_cache = renderer.chunk_cache
    cache_get = chunk_cache.get
    cache_move_to_end = chunk_cache.move_to_end
    release = lambda mesh: release_chunk_mesh_storage(renderer, mesh)
    retain = lambda mesh: retain_chunk_mesh_storage(renderer, mesh)
    visible_chunk_coord_set = renderer._visible_chunk_coord_set

    mesh_key_set: set[tuple[int, int]] = set()

    for mesh in meshes:
        key = (mesh.chunk_x, mesh.chunk_z)
        mesh_key_set.add(key)

        existing = cache_get(key)
        if existing is mesh:
            cache_move_to_end(key)
            continue

        if existing is not None:
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


def build_tile_draw_batches(
    renderer,
    meshes: list[ChunkMesh],
    encoder,
    *,
    age_gate: bool,
) -> tuple[list[ChunkDrawBatch], int, int, int]:
    renderer_module = _renderer_module()
    merged_tile_min_age_seconds = float(renderer_module.MERGED_TILE_MIN_AGE_SECONDS)

    tile_groups: dict[tuple[int, int], list[ChunkMesh]] = {}
    for mesh in meshes:
        if mesh.vertex_count <= 0:
            continue
        tile_groups.setdefault(tile_key(renderer, mesh.chunk_x, mesh.chunk_z), []).append(mesh)

    current_tile_keys = set(tile_groups.keys())
    stale_keys = [tile_key_value for tile_key_value in renderer._tile_render_batches if tile_key_value not in current_tile_keys]
    for tile_key_value in stale_keys:
        batch = renderer._tile_render_batches.pop(tile_key_value)
        renderer._transient_render_buffers.append([batch.vertex_buffer])

    draw_batches: list[ChunkDrawBatch] = []
    merged_chunk_count = 0
    visible_chunk_count = 0

    for tile_key_value in sorted(tile_groups):
        tile_meshes = tile_groups[tile_key_value]
        mature_meshes = [mesh for mesh in tile_meshes if chunk_mesh_age(renderer, mesh) >= merged_tile_min_age_seconds]
        immature_meshes = [mesh for mesh in tile_meshes if chunk_mesh_age(renderer, mesh) < merged_tile_min_age_seconds]

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
            visible_chunk_count += 1
            continue

        existing = renderer._tile_render_batches.get(tile_key_value)
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
            continue

        signature = tuple((mesh.chunk_x, mesh.chunk_z) for mesh in mature_meshes)
        batch_vertex_count = sum(mesh.vertex_count for mesh in mature_meshes)
        if (
            existing is None
            or existing.signature != signature
            or existing.vertex_count != batch_vertex_count
        ):
            old_buffer = existing.vertex_buffer if existing is not None else None
            merged_buffer = merge_tile_meshes(renderer, mature_meshes, encoder)
            renderer._tile_render_batches[tile_key_value] = ChunkRenderBatch(
                signature=signature,
                vertex_count=batch_vertex_count,
                vertex_buffer=merged_buffer,
            )
            if old_buffer is not None:
                renderer._transient_render_buffers.append([old_buffer])

        batch = renderer._tile_render_batches[tile_key_value]
        draw_batches.append(
            ChunkDrawBatch(
                vertex_buffer=batch.vertex_buffer,
                binding_offset=0,
                vertex_count=batch.vertex_count,
                first_vertex=0,
                bounds=merge_chunk_bounds(renderer, mature_meshes),
                chunk_count=len(mature_meshes),
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
    visible_meshes = [mesh for _, _, mesh in visible_chunks(renderer) if mesh.vertex_count > 0]
    draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count = build_tile_draw_batches(
        renderer,
        visible_meshes,
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
    visible_meshes = [mesh for _, _, mesh in visible_chunks(renderer) if mesh.vertex_count > 0]
    draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count = build_tile_draw_batches(
        renderer,
        visible_meshes,
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


def visible_chunks(renderer) -> list[tuple[int, int, ChunkMesh]]:
    visible: list[tuple[int, int, ChunkMesh]] = []
    if not renderer._visible_chunk_coords:
        renderer._refresh_visible_chunk_coords()
    for chunk_x, chunk_z in renderer._visible_chunk_coords:
        mesh = renderer.chunk_cache.get((chunk_x, chunk_z))
        if mesh is None:
            continue
        renderer.chunk_cache.move_to_end((chunk_x, chunk_z))
        visible.append((chunk_x, chunk_z, mesh))
    return visible


def tile_key(renderer, chunk_x: int, chunk_z: int) -> tuple[int, int]:
    renderer_module = _renderer_module()
    merged_tile_size_chunks = int(renderer_module.MERGED_TILE_SIZE_CHUNKS)
    return chunk_x // merged_tile_size_chunks, chunk_z // merged_tile_size_chunks


def visible_render_batches(
    renderer,
    encoder,
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int]], float, int, int, int, int]:
    encode_start = time.perf_counter()
    visible = visible_chunks(renderer)

    tile_groups: dict[tuple[int, int], list[tuple[int, int, ChunkMesh]]] = {}
    for chunk_x, chunk_z, mesh in visible:
        tile_groups.setdefault(tile_key(renderer, chunk_x, chunk_z), []).append((chunk_x, chunk_z, mesh))

    render_batches: list[tuple[wgpu.GPUBuffer, int, int]] = []
    merged_chunk_count = 0

    current_tile_keys = set(tile_groups.keys())
    stale_keys = [tile_key_value for tile_key_value in renderer._tile_render_batches if tile_key_value not in current_tile_keys]
    for tile_key_value in stale_keys:
        batch = renderer._tile_render_batches.pop(tile_key_value)
        renderer._transient_render_buffers.append([batch.vertex_buffer])

    renderer_module = _renderer_module()
    merged_tile_min_age_seconds = float(renderer_module.MERGED_TILE_MIN_AGE_SECONDS)

    for tile_key_value in sorted(tile_groups):
        tile_chunks = tile_groups[tile_key_value]
        mature_chunks = [(chunk_x, chunk_z, mesh) for chunk_x, chunk_z, mesh in tile_chunks if chunk_mesh_age(renderer, mesh) >= merged_tile_min_age_seconds]
        immature_chunks = [(chunk_x, chunk_z, mesh) for chunk_x, chunk_z, mesh in tile_chunks if chunk_mesh_age(renderer, mesh) < merged_tile_min_age_seconds]

        if len(tile_chunks) == 1:
            _, _, mesh = tile_chunks[0]
            render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))
            continue

        existing = renderer._tile_render_batches.get(tile_key_value)
        if len(mature_chunks) < 2:
            if existing is not None:
                renderer._tile_render_batches.pop(tile_key_value, None)
                renderer._transient_render_buffers.append([existing.vertex_buffer])
            for _, _, mesh in immature_chunks:
                render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))
            for _, _, mesh in mature_chunks:
                render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))
            continue

        merged_chunk_count += len(mature_chunks)
        batch_vertex_count = sum(mesh.vertex_count for _, _, mesh in mature_chunks)
        if (
            existing is None
            or existing.signature != tuple((chunk_x, chunk_z) for chunk_x, chunk_z, _ in mature_chunks)
            or existing.vertex_count != batch_vertex_count
        ):
            old_buffer = existing.vertex_buffer if existing is not None else None
            merged_buffer = merge_tile_meshes(renderer, [mesh for _, _, mesh in mature_chunks], encoder)
            renderer._tile_render_batches[tile_key_value] = ChunkRenderBatch(
                signature=tuple((chunk_x, chunk_z) for chunk_x, chunk_z, _ in mature_chunks),
                vertex_count=batch_vertex_count,
                vertex_buffer=merged_buffer,
            )
            if old_buffer is not None:
                renderer._transient_render_buffers.append([old_buffer])

        batch = renderer._tile_render_batches[tile_key_value]
        render_batches.append((batch.vertex_buffer, batch.vertex_count, 0))
        for _, _, mesh in immature_chunks:
            render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))

    while len(renderer._transient_render_buffers) > 3:
        old_buffers = renderer._transient_render_buffers.pop(0)
        for buffer in old_buffers:
            buffer.destroy()

    render_encode_ms = (time.perf_counter() - encode_start) * 1000.0
    visible_vertex_count = sum(vertex_count for _, vertex_count, _ in render_batches)
    return render_batches, render_encode_ms, len(render_batches), merged_chunk_count, len(visible), visible_vertex_count
