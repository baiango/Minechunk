from __future__ import annotations

from engine.renderer import TerrainRenderer


def main() -> None:
    renderer = TerrainRenderer(
        chunk_radius=8,
        vertical_chunk_radius=8,
        exit_when_view_ready=True,
    )
    renderer.run()


if __name__ == "__main__":
    main()
