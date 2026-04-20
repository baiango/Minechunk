"""Terrain-stage entry points."""


def request_chunk_voxel_batch(world, chunks):
    return world.request_chunk_voxel_batch(chunks)


def poll_ready_chunk_voxel_batches(world):
    return world.poll_ready_chunk_voxel_batches()


def request_chunk_surface_batch(world, chunks):
    return world.request_chunk_surface_batch(chunks)


def poll_ready_chunk_surface_gpu_batches(world):
    return world.poll_ready_chunk_surface_gpu_batches()
