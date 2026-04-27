from __future__ import annotations

from .. import render_contract as render_consts

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


def _renderer_module():
    """Return stable render constants without importing the renderer runtime."""
    return render_consts


def _chunk_half() -> float:
    return float(_renderer_module().CHUNK_WORLD_SIZE) * 0.5


def _storage_height(local_height: int) -> int:
    return int(local_height) + 2


def _storage_binding_size(size_bytes: int) -> int:
    # WGPU storage buffer binding sizes must be at least 4 bytes and
    # must be 4-byte aligned.
    size_bytes = max(4, int(size_bytes))
    return (size_bytes + 3) & ~3


def _emit_vertex_binding_size(size_bytes: int) -> int:
    # The emit shader's output binding is an array of packed ChunkVertex
    # records. Even when a high-altitude batch emits zero vertices, WGPU
    # validates the declared storage binding against the minimum element
    # footprint, so a dummy binding must cover at least one full vertex.
    vertex_stride = int(_renderer_module().VERTEX_STRIDE)
    return _storage_binding_size(max(vertex_stride, int(size_bytes)))


def _mesh_output_request_bytes(renderer, size_bytes: int) -> int:
    # Keep output allocations friendly to dynamic storage-buffer offset
    # alignment even when a whole high-altitude batch emits zero vertices.
    alignment = max(4, int(getattr(renderer, "_mesh_output_binding_alignment", 256)))
    return max(alignment, _emit_vertex_binding_size(size_bytes))


def _normalize_chunk_coord(coord) -> tuple[int, int, int]:
    if len(coord) >= 3:
        return int(coord[0]), int(coord[1]), int(coord[2])
    if len(coord) == 2:
        return int(coord[0]), 0, int(coord[1])
    raise ValueError(f"Invalid chunk coordinate: {coord!r}")


def _normalize_chunk_coords(coords) -> list[tuple[int, int, int]]:
    return [_normalize_chunk_coord(coord) for coord in coords]
