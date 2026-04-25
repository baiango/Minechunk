"""Meshing-stage entry points.

The terrain backend and meshing backend are intentionally decoupled.

Fast path:
    terrain GPU surface buffers -> same API GPU mesher

Portable path:
    any terrain backend -> ChunkVoxelResult arrays -> selected mesher

This prevents invalid cross-API binding such as feeding a Metal MTLBuffer into
wgpu, or a wgpu.GPUBuffer into Metal.  The selected mesher should be able to
consume any terrain backend; only the zero-copy surface-buffer handoff is
backend-specific.
"""

from __future__ import annotations

from ..meshing.cpu_mesher import (
    build_chunk_vertex_array,
    cpu_make_chunk_mesh_batch_from_terrain_results,
    cpu_make_chunk_mesh_from_terrain_result,
)
from ..meshing import gpu_mesher, metal_mesher


def selected_mesher_kind(renderer) -> str:
    if not getattr(renderer, "use_gpu_meshing", False):
        return "cpu"
    return "metal" if getattr(renderer, "_using_metal_meshing", False) else "wgpu"


def selected_mesher_label(renderer) -> str:
    kind = selected_mesher_kind(renderer)
    if kind == "metal":
        return "Metal"
    if kind == "wgpu":
        return "Wgpu"
    return "CPU"


def _select_mesher(renderer):
    return metal_mesher if selected_mesher_kind(renderer) == "metal" else gpu_mesher


def terrain_surface_device_kind(renderer) -> str:
    label = ""
    try:
        label = renderer.world.terrain_backend_label()
    except Exception:
        return ""
    lowered = str(label).strip().lower()
    if lowered == "metal":
        return "metal"
    if lowered == "wgpu":
        return "wgpu"
    return "cpu"


def surface_gpu_batch_device_kind(surface_batch) -> str:
    kind = str(getattr(surface_batch, "device_kind", "") or "").strip().lower()
    if kind:
        return kind
    source = str(getattr(surface_batch, "source", "") or "").strip().lower()
    if "metal" in source:
        return "metal"
    if "wgpu" in source or source in {"gpu", "gpu_leased"}:
        return "wgpu"
    return ""


def can_selected_mesher_consume_surface_gpu_batch(renderer, surface_batch) -> bool:
    mesher_kind = selected_mesher_kind(renderer)
    if mesher_kind not in {"metal", "wgpu"}:
        return False
    return surface_gpu_batch_device_kind(surface_batch) == mesher_kind


def can_use_native_surface_gpu_handoff(renderer) -> bool:
    """Return True only when terrain buffers are native to the selected mesher.

    CPU terrain has no native GPU surface buffers.  Cross-API GPU pairs fall
    back to poll_ready_chunk_voxel_batches(), where the terrain backend returns
    neutral CPU-side ChunkVoxelResult data for any mesher.
    """

    mesher_kind = selected_mesher_kind(renderer)
    if mesher_kind not in {"metal", "wgpu"}:
        return False
    return terrain_surface_device_kind(renderer) == mesher_kind


def make_chunk_mesh_batch_from_terrain(renderer, terrain_results):
    if getattr(renderer, "use_gpu_meshing", False):
        return _select_mesher(renderer).make_chunk_mesh_batch_from_terrain_results(renderer, terrain_results)
    return cpu_make_chunk_mesh_batch_from_terrain_results(renderer, terrain_results)


def make_chunk_mesh_from_terrain_result(renderer, terrain_result):
    if getattr(renderer, "use_gpu_meshing", False):
        meshes = _select_mesher(renderer).make_chunk_mesh_batch_from_terrain_results(renderer, [terrain_result])
        return meshes[0] if meshes else None
    return cpu_make_chunk_mesh_from_terrain_result(renderer, terrain_result)


def enqueue_surface_gpu_batches_for_meshing(renderer, surface_batches):
    mesher = _select_mesher(renderer)
    compatible = []
    incompatible = []
    for surface_batch in surface_batches:
        if can_selected_mesher_consume_surface_gpu_batch(renderer, surface_batch):
            compatible.append(surface_batch)
        else:
            incompatible.append(surface_batch)

    if incompatible:
        # This should only happen if an older backend failed to mark its device
        # kind or the pipeline changed backend while work was in flight.  Do not
        # bind foreign buffers.  Release them and let the caller request/poll via
        # the neutral voxel path on the next frame.
        release = getattr(mesher, "release_surface_gpu_batch_immediately", None)
        fallback_release = getattr(gpu_mesher, "release_surface_gpu_batch_immediately", None)
        for surface_batch in incompatible:
            for coord in getattr(surface_batch, "chunks", []) or []:
                try:
                    key = (int(coord[0]), int(coord[1]), int(coord[2]))
                except Exception:
                    continue
                try:
                    renderer._pending_chunk_coords.discard(key)
                    if key in getattr(renderer, "_visible_chunk_coord_set", set()) and key not in getattr(renderer, "chunk_cache", {}):
                        renderer._visible_missing_coords.add(key)
                        renderer._chunk_request_queue_dirty = True
                except Exception:
                    pass
            released = False
            if callable(release):
                try:
                    release(surface_batch) if mesher is metal_mesher else release(renderer, surface_batch)
                    released = True
                except TypeError:
                    try:
                        release(surface_batch)
                        released = True
                    except Exception:
                        pass
                except Exception:
                    pass
            if not released and callable(fallback_release):
                try:
                    fallback_release(renderer, surface_batch)
                except Exception:
                    pass

    if not compatible:
        return []

    if hasattr(mesher, "enqueue_surface_gpu_batches_for_meshing"):
        return mesher.enqueue_surface_gpu_batches_for_meshing(renderer, compatible)
    all_meshes = []
    for surface_batch in compatible:
        all_meshes.extend(make_chunk_mesh_batch_from_surface_gpu_batch(renderer, surface_batch))
    return all_meshes


def drain_pending_surface_gpu_batches_to_meshing(renderer):
    mesher = _select_mesher(renderer)
    if hasattr(mesher, "drain_pending_surface_gpu_batches_to_meshing"):
        return mesher.drain_pending_surface_gpu_batches_to_meshing(renderer)
    return 0


def make_chunk_mesh_batch_from_surface_gpu_batch(renderer, surface_batch):
    if not can_selected_mesher_consume_surface_gpu_batch(renderer, surface_batch):
        raise RuntimeError(
            "Selected mesher cannot consume this terrain GPU surface batch: "
            f"mesher={selected_mesher_kind(renderer)!r}, "
            f"surface_device_kind={surface_gpu_batch_device_kind(surface_batch)!r}. "
            "Use the neutral ChunkVoxelResult path for cross-backend terrain/meshing."
        )
    mesher = _select_mesher(renderer)
    if hasattr(mesher, "make_chunk_mesh_batch_from_surface_gpu_batch"):
        return mesher.make_chunk_mesh_batch_from_surface_gpu_batch(renderer, surface_batch)
    raise RuntimeError("Selected mesher does not support surface-GPU terrain batches.")
