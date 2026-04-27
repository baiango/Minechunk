from __future__ import annotations

import math
import struct
import time

import numpy as np
import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkMesh
from ..meshing import gpu_mesher as wgpu_mesher
from ..visibility import coord_manager
from .mesh_output_allocator import release_chunk_mesh_storage, retain_chunk_mesh_storage
from .tile_cache_constants import MERGED_TILE_AGE_REFRESH_INTERVAL_SECONDS


def _renderer_module():
    return render_consts


try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func

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

