"""Meshing-stage entry points."""

from ..meshing.cpu_mesher import (
    build_chunk_vertex_array,
    cpu_make_chunk_mesh_batch_from_terrain_results,
    cpu_make_chunk_mesh_from_terrain_result,
)
from ..meshing import gpu_mesher, metal_mesher


def _select_mesher(renderer):
    return metal_mesher if getattr(renderer, "_using_metal_meshing", False) else gpu_mesher


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
    if hasattr(mesher, "enqueue_surface_gpu_batches_for_meshing"):
        return mesher.enqueue_surface_gpu_batches_for_meshing(renderer, surface_batches)
    all_meshes = []
    for surface_batch in surface_batches:
        all_meshes.extend(make_chunk_mesh_batch_from_surface_gpu_batch(renderer, surface_batch))
    return all_meshes


def drain_pending_surface_gpu_batches_to_meshing(renderer):
    mesher = _select_mesher(renderer)
    if hasattr(mesher, "drain_pending_surface_gpu_batches_to_meshing"):
        return mesher.drain_pending_surface_gpu_batches_to_meshing(renderer)
    return 0


def make_chunk_mesh_batch_from_surface_gpu_batch(renderer, surface_batch):
    mesher = _select_mesher(renderer)
    if hasattr(mesher, "make_chunk_mesh_batch_from_surface_gpu_batch"):
        return mesher.make_chunk_mesh_batch_from_surface_gpu_batch(renderer, surface_batch)
    raise RuntimeError("Selected mesher does not support surface-GPU terrain batches.")

