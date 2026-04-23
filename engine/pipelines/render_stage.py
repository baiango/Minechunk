from __future__ import annotations

"""Render-stage entry points."""

from collections import OrderedDict
import struct
import time
from typing import Iterable

import numpy as np
import wgpu

from ..cache.mesh_allocator import (
    ensure_mesh_draw_indirect_scratch,
    ensure_mesh_visibility_scratch,
)
from ..meshing_types import ChunkDrawBatch, ChunkMesh
from ..rendering.tile_batcher import build_tile_draw_batches
from ..rendering.direct_render import build_gpu_visibility_records, visible_render_batches, visible_render_batches_indirect
from ..rendering.merge_pipeline import merge_chunk_bounds, merge_tile_meshes


def _mesh_draw_batches(meshes: Iterable[ChunkMesh]) -> tuple[list[ChunkDrawBatch], int, int]:
    draw_batches: list[ChunkDrawBatch] = []
    visible_chunk_count = 0
    visible_vertex_count = 0
    for mesh in meshes:
        if mesh is None:
            continue
        vertex_buffer = getattr(mesh, "vertex_buffer", None)
        vertex_count = int(getattr(mesh, "vertex_count", 0))
        if vertex_buffer is None or vertex_count <= 0:
            continue
        draw_batches.append(
            ChunkDrawBatch(
                vertex_buffer=vertex_buffer,
                binding_offset=int(getattr(mesh, "binding_offset", 0)),
                vertex_count=vertex_count,
                first_vertex=int(getattr(mesh, "first_vertex", 0)),
                bounds=tuple(getattr(mesh, "bounds", (0.0, 0.0, 0.0, 0.0))),
                chunk_count=1,
            )
        )
        visible_chunk_count += 1
        visible_vertex_count += vertex_count
    return draw_batches, visible_chunk_count, visible_vertex_count


def _group_draw_batches(
    draw_batches: list[ChunkDrawBatch],
) -> OrderedDict[tuple[int, int], tuple[wgpu.GPUBuffer, int, list[ChunkDrawBatch]]]:
    groups: OrderedDict[tuple[int, int], tuple[wgpu.GPUBuffer, int, list[ChunkDrawBatch]]] = OrderedDict()
    for batch in draw_batches:
        key = (id(batch.vertex_buffer), int(batch.binding_offset))
        if key not in groups:
            groups[key] = (batch.vertex_buffer, int(batch.binding_offset), [])
        groups[key][2].append(batch)
    return groups


def visible_render_batches_for_meshes(
    meshes: Iterable[ChunkMesh],
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], float, int, int, int, int]:
    draw_batches, visible_chunk_count, visible_vertex_count = _mesh_draw_batches(meshes)
    render_batches = [
        (
            batch.vertex_buffer,
            int(batch.binding_offset),
            int(batch.vertex_count),
            int(batch.first_vertex),
        )
        for batch in draw_batches
    ]
    return render_batches, 0.0, len(render_batches), 0, visible_chunk_count, visible_vertex_count


def visible_render_batches_indirect_for_meshes(
    renderer,
    meshes: Iterable[ChunkMesh],
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], float, int, int, int, int]:
    encode_start = time.perf_counter()
    draw_batches, visible_chunk_count, visible_vertex_count = _mesh_draw_batches(meshes)
    groups = _group_draw_batches(draw_batches)

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
    return render_batches, render_encode_ms, command_count, 0, visible_chunk_count, visible_vertex_count


def build_gpu_visibility_records_for_meshes(
    renderer,
    meshes: Iterable[ChunkMesh],
) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], int, int, int, int]:
    draw_batches, visible_chunk_count, visible_vertex_count = _mesh_draw_batches(meshes)
    groups = _group_draw_batches(draw_batches)

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
                int(batch.vertex_count),
                1,
                int(batch.first_vertex),
                0,
            )
            command_index += 1
        render_batches.append((vertex_buffer, binding_offset, batch_start, command_index - batch_start))

    if command_count > 0:
        renderer.device.queue.write_buffer(metadata_buffer, 0, memoryview(metadata_array[:command_count]))
        renderer.device.queue.write_buffer(params_buffer, 0, struct.pack("<4I", int(command_count), 0, 0, 0))

    return render_batches, command_count, 0, visible_chunk_count, visible_vertex_count
