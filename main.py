from __future__ import annotations

import argparse
import os


def _configure_engine_mode(value: str | None) -> None:
    if value is None:
        return
    from engine import renderer_config as cfg

    mode = value.strip().lower()
    mapping = {
        "cpu": cfg.ENGINE_MODE_CPU,
        "wgpu": cfg.ENGINE_MODE_WGPU,
        "metal": cfg.ENGINE_MODE_METAL,
    }
    cfg.engine_mode = mapping[mode]
    cfg.chunk_prep_request_budget_cap = 8 if cfg.engine_mode != cfg.ENGINE_MODE_CPU else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Minechunk.")
    parser.add_argument(
        "--engine",
        choices=("cpu", "wgpu", "metal"),
        default=None,
        help="Override the configured backend. Default is the value in engine/renderer_config.py.",
    )
    parser.add_argument(
        "--allow-metal-fallback",
        action="store_true",
        help="Allow Metal mode to fall back to WGPU/CPU instead of failing loudly.",
    )
    args = parser.parse_args()

    if args.allow_metal_fallback:
        os.environ["MINECHUNK_ALLOW_METAL_FALLBACK"] = "1"
    _configure_engine_mode(args.engine)

    from engine.renderer import TerrainRenderer

    TerrainRenderer().run()


if __name__ == "__main__":
    main()
