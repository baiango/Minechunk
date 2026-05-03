from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from engine import render_contract as render_consts
from engine.meshing.cpu_mesher import cpu_make_chunk_mesh_batch_from_terrain_results
from engine.terrain.kernels.materials import GRASS
from engine.terrain.kernels.voxel_mesher import count_chunk_surface_vertices_from_heightmap_clipped
from engine.terrain.types import ChunkVoxelResult


class FakeBuffer:
    def __init__(self, size: int) -> None:
        self.data = bytearray(int(size))


class FakeQueue:
    def __init__(self) -> None:
        self.write_calls: list[tuple[FakeBuffer, int, int]] = []

    def write_buffer(self, buffer, offset: int, data) -> None:
        payload = bytes(data)
        end = int(offset) + len(payload)
        if end > len(buffer.data):
            buffer.data.extend(b"\0" * (end - len(buffer.data)))
        buffer.data[int(offset) : end] = payload
        self.write_calls.append((buffer, int(offset), len(payload)))


class FakeDevice:
    def __init__(self) -> None:
        self.queue = FakeQueue()

    def create_buffer(self, *, size: int, usage) -> FakeBuffer:
        return FakeBuffer(size)


def test_cpu_surface_batch_emits_directly_into_one_upload(monkeypatch) -> None:
    chunk_size = int(render_consts.CHUNK_SIZE)
    sample_size = chunk_size + 2
    heights = np.zeros(sample_size * sample_size, dtype=np.uint32)
    materials = np.full(sample_size * sample_size, GRASS, dtype=np.uint32)
    for local_z in range(1, chunk_size + 1):
        for local_x in range(1, chunk_size + 1):
            heights[local_z * sample_size + local_x] = 4

    batch_buffer = FakeBuffer(0)
    allocation = SimpleNamespace(buffer=batch_buffer, offset_bytes=0, allocation_id=77)
    requested_sizes: list[int] = []

    def fake_allocate(_renderer, size_bytes: int):
        requested_sizes.append(int(size_bytes))
        return allocation

    monkeypatch.setattr("engine.meshing.cpu_mesher.mesh_cache.allocate_mesh_output_range", fake_allocate)

    renderer = SimpleNamespace(device=FakeDevice(), _shared_empty_chunk_vertex_buffer=None)
    blocks = np.zeros((chunk_size, sample_size, sample_size), dtype=np.uint8)
    empty_result = ChunkVoxelResult(
        chunk_x=1,
        chunk_y=0,
        chunk_z=0,
        blocks=blocks,
        materials=np.zeros_like(blocks, dtype=np.uint32),
        is_fully_occluded=True,
    )
    surface_result = ChunkVoxelResult(
        chunk_x=0,
        chunk_y=0,
        chunk_z=0,
        blocks=blocks,
        materials=np.zeros_like(blocks, dtype=np.uint32),
        surface_heights=heights,
        surface_materials=materials,
        use_surface_mesher=True,
    )

    meshes = cpu_make_chunk_mesh_batch_from_terrain_results(renderer, [surface_result, empty_result])
    expected_vertices = count_chunk_surface_vertices_from_heightmap_clipped(heights, materials, chunk_size, chunk_size, 0)

    assert [mesh.vertex_count for mesh in meshes] == [expected_vertices, 0]
    assert requested_sizes == [expected_vertices * render_consts.VERTEX_STRIDE]
    assert renderer.device.queue.write_calls == [(batch_buffer, 0, expected_vertices * render_consts.VERTEX_STRIDE)]
    assert meshes[0].vertex_buffer is batch_buffer
    assert meshes[0].allocation_id == 77
    assert meshes[1].vertex_buffer is renderer._shared_empty_chunk_vertex_buffer
    assert meshes[1].allocation_id is None
