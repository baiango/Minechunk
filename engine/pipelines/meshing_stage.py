"""Meshing-stage entry points."""

from ..meshing.cpu_mesher import build_chunk_vertex_array, cpu_make_chunk_mesh_batch_from_voxels, cpu_make_chunk_mesh_from_voxels
from ..meshing import gpu_mesher, metal_mesher


def make_chunk_mesh_batch_from_voxels(renderer, chunk_results):
    if getattr(renderer, "use_gpu_meshing", False):
        mesher = metal_mesher if getattr(renderer, "_using_metal_meshing", False) else gpu_mesher
        return mesher.make_chunk_mesh_batch_from_voxels(renderer, chunk_results)
    return cpu_make_chunk_mesh_batch_from_voxels(renderer, chunk_results)
