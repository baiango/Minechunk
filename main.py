from __future__ import annotations

import argparse


def main() -> None:
    try:
        from engine.renderer import TerrainRenderer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This build targets wgpu-py. Install the runtime dependencies with "
            "`pip install -r requirements.txt` first."
        ) from exc

    parser = argparse.ArgumentParser(description="Run Minechunk.")
    parser.parse_args()

    TerrainRenderer().run()


if __name__ == "__main__":
    main()
