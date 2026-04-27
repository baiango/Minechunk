from __future__ import annotations

from collections import OrderedDict
import struct
import time

import numpy as np
import wgpu

from .. import render_contract as render_consts
from ..meshing_types import ChunkDrawBatch, ChunkMesh
from ..meshing import gpu_mesher as wgpu_mesher
from ..visibility import coord_manager
from .tile_draw_batches import (
    MERGED_TILE_AGE_REFRESH_INTERVAL_SECONDS,
    _finalize_direct_render_batch_groups,
    build_tile_draw_batches,
)


def _renderer_module():
    return render_consts


try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func

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
