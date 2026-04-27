from __future__ import annotations

"""Metal buffer and single/grid surface sampling helpers."""

import struct
from typing import Optional

import numpy as np

from .metal_terrain_common import Metal, profile


class MetalTerrainBufferOps:
    def _create_pipeline(self, library, function_name: str):
        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Missing Metal kernel function '{function_name}'.")
        pipeline, err = self.device.newComputePipelineStateWithFunction_error_(function, None)
        if err is not None or pipeline is None:
            raise RuntimeError(f"Failed to create Metal compute pipeline for '{function_name}': {err}")
        return pipeline

    def _make_buffer(self, size: int):
        return self.device.newBufferWithLength_options_(max(4, int(size)), self._resource_options)

    @staticmethod
    def _buffer_memoryview(buffer, size: Optional[int] = None) -> memoryview:
        length = int(buffer.length()) if size is None else int(size)
        return memoryview(buffer.contents().as_buffer(length))

    def _write_buffer_bytes(self, buffer, data: bytes | bytearray | memoryview) -> None:
        payload = memoryview(data).cast("B")
        self._buffer_memoryview(buffer, len(payload))[: len(payload)] = payload

    def _write_buffer_array(self, buffer, array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array)
        payload = memoryview(contiguous).cast("B")
        self._buffer_memoryview(buffer, contiguous.nbytes)[: contiguous.nbytes] = payload

    @staticmethod
    def _buffer_uint32_view(buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(int(count) * 4), dtype=np.uint32, count=int(count))

    def _dispatch(self, pipeline, buffers: list[tuple[object, int]], grid_size: tuple[int, int, int], group_size: tuple[int, int, int]):
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for buffer, index in buffers:
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*map(int, grid_size)),
            Metal.MTLSizeMake(*map(int, group_size)),
        )
        encoder.endEncoding()
        command_buffer.commit()
        return command_buffer

    def _write_params(
        self,
        *,
        sample_origin_x: float,
        sample_origin_z: float,
        height_limit: int,
        chunk_x: int,
        chunk_z: int,
        sample_size: int,
        seed: int,
    ) -> None:
        payload = struct.pack(
            "<4f4i4I",
            float(sample_origin_x), float(sample_origin_z), 0.0, float(height_limit),
            int(chunk_x), int(chunk_z), int(sample_size), int(self.chunk_size),
            int(seed) & 0xFFFFFFFF, int(self.chunk_size) & 0xFFFFFFFF, 0, 0,
        )
        self._write_buffer_bytes(self._params_buffer, payload)

    @profile
    def surface_profile_at(self, x: int, z: int) -> tuple[int, int]:
        self._write_params(
            sample_origin_x=float(x),
            sample_origin_z=float(z),
            height_limit=self.height_limit,
            chunk_x=0,
            chunk_z=0,
            sample_size=1,
            seed=self.seed,
        )
        cb = self._dispatch(
            self._single_pipeline,
            [(self._single_heights_buffer, 0), (self._single_materials_buffer, 1), (self._params_buffer, 2)],
            (1, 1, 1),
            (1, 1, 1),
        )
        cb.waitUntilCompleted()
        return (
            int(self._buffer_uint32_view(self._single_heights_buffer, 1)[0]),
            int(self._buffer_uint32_view(self._single_materials_buffer, 1)[0]),
        )

    @profile
    def fill_chunk_surface_grids(self, heights: np.ndarray, materials: np.ndarray, chunk_x: int, chunk_z: int) -> None:
        self._write_params(
            sample_origin_x=0.0,
            sample_origin_z=0.0,
            height_limit=self.height_limit,
            chunk_x=int(chunk_x),
            chunk_z=int(chunk_z),
            sample_size=self.sample_size,
            seed=self.seed,
        )
        cb = self._dispatch(
            self._grid_pipeline,
            [(self._grid_heights_buffer, 0), (self._grid_materials_buffer, 1), (self._params_buffer, 2)],
            (self.sample_size, self.sample_size, 1),
            (8, 8, 1),
        )
        cb.waitUntilCompleted()
        heights[:] = self._buffer_uint32_view(self._grid_heights_buffer, self.cell_count)
        materials[:] = self._buffer_uint32_view(self._grid_materials_buffer, self.cell_count)

    @profile
    def chunk_surface_grids(self, chunk_x: int, chunk_z: int) -> tuple[np.ndarray, np.ndarray]:
        heights = np.empty(self.cell_count, dtype=np.uint32)
        materials = np.empty(self.cell_count, dtype=np.uint32)
        self.fill_chunk_surface_grids(heights, materials, chunk_x, chunk_z)
        return heights, materials
