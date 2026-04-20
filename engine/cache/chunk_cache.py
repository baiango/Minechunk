"""Chunk cache helpers for the split layout."""

from ..cache import mesh_allocator as mesh_cache


def get_chunk_mesh(renderer, coord):
    return renderer.chunk_cache.get(coord)


def store_chunk_mesh(renderer, mesh):
    return mesh_cache.store_chunk_mesh(renderer, mesh)


def store_chunk_meshes(renderer, meshes):
    return mesh_cache.store_chunk_meshes(renderer, meshes)


def release_chunk_mesh_storage(renderer, mesh):
    return mesh_cache.release_chunk_mesh_storage(renderer, mesh)


def clear_chunk_cache(renderer):
    for mesh in list(renderer.chunk_cache.values()):
        mesh_cache.release_chunk_mesh_storage(renderer, mesh)
    renderer.chunk_cache.clear()
