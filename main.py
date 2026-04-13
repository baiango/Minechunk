from __future__ import annotations

import argparse


def main() -> None:
    try:
        from renderer import TerrainRenderer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This build targets wgpu-py. Install the runtime dependencies with "
            "`pip install -r requirements.txt` first."
        ) from exc

    parser = argparse.ArgumentParser(description="Run Minechunk.")
    parser.add_argument(
        "--chunk-prep-preset",
        choices=("conservative", "balanced", "aggressive"),
        default="balanced",
        help="Tune how aggressively chunks are requested and processed.",
    )
    args = parser.parse_args()

    TerrainRenderer(chunk_prep_preset=args.chunk_prep_preset).run()


if __name__ == "__main__":
    main()
