from __future__ import annotations

from engine.renderer import TerrainRenderer
def main() -> None:
    renderer = TerrainRenderer(
        fixed_view_dimensions=(16, 16, 16),
        freeze_view_origin=True,
        freeze_camera=True,
        exit_when_view_ready=True,
        terrain_batch_size=128,
        mesh_batch_size=32,
    )
    renderer.run()


if __name__ == "__main__":
    main()
