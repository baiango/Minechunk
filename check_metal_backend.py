from __future__ import annotations

from engine.terrain.backends.metal_terrain_backend import MetalTerrainBackend


def main() -> None:
    backend = MetalTerrainBackend(None, seed=1337, chunk_size=64, height_limit=512, chunks_per_poll=1)
    print(f"OK: Metal terrain backend initialized on {backend.device.name() if hasattr(backend.device, 'name') else backend.device!r}")
    backend.destroy()


if __name__ == "__main__":
    main()
