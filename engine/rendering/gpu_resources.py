from __future__ import annotations

from .gpu_resource_buffers import create_gpu_buffers
from .gpu_resource_layouts import create_gpu_bind_group_layouts
from .gpu_resource_pipelines import create_gpu_pipelines


def initialize_gpu_resources(renderer) -> None:
    """Create long-lived GPU buffers, bind group layouts, and pipelines.

    This keeps TerrainRenderer focused on runtime orchestration. The function
    intentionally mutates renderer attributes to preserve the existing renderer
    contract used by the cache, mesher, and render stages.
    """
    create_gpu_buffers(renderer)
    create_gpu_bind_group_layouts(renderer)
    create_gpu_pipelines(renderer)
