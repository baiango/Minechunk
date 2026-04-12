from __future__ import annotations


def main() -> None:
    try:
        from renderer import TerrainRenderer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This build targets wgpu-py. Install the runtime dependencies with "
            "`pip install -r requirements.txt` first."
        ) from exc

    TerrainRenderer().run()


if __name__ == "__main__":
    main()
