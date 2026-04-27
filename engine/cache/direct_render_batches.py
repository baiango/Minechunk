from __future__ import annotations

from collections import OrderedDict

import wgpu

from ..meshing_types import ChunkDrawBatch


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
