from __future__ import annotations

import argparse

from engine.renderer import TerrainRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Minechunk.")
    parser.parse_args()

    TerrainRenderer().run()


if __name__ == "__main__":
    main()
