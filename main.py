from __future__ import annotations

import argparse
import os


def _parse_view_dimensions(value: str) -> tuple[int, int, int]:
    raw = value.strip().lower().replace("×", "x").replace(",", "x")
    parts = [part.strip() for part in raw.split("x") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("view dimensions must look like 16x16x16")
    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("view dimensions must contain integers") from exc
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("view dimensions must be greater than 0")
    return dims  # type: ignore[return-value]


def _configure_runtime_knobs(
    *,
    engine: str | None,
    chunk_request_budget_cap: int | None,
    rc_enabled: bool | None,
    terrain_zstd_enabled: bool | None,
    mesh_zstd_enabled: bool | None,
    tile_merging_enabled: bool | None,
    allow_metal_fallback: bool,
) -> None:
    from engine import render_contract
    from engine import renderer_config as cfg

    if allow_metal_fallback:
        os.environ["MINECHUNK_ALLOW_METAL_FALLBACK"] = "1"
    else:
        os.environ.pop("MINECHUNK_ALLOW_METAL_FALLBACK", None)

    if engine is not None:
        mode = engine.strip().lower()
        mapping = {
            "cpu": cfg.ENGINE_MODE_CPU,
            "wgpu": cfg.ENGINE_MODE_WGPU,
            "metal": cfg.ENGINE_MODE_METAL,
        }
        cfg.engine_mode = mapping[mode]
        if chunk_request_budget_cap is None:
            default_budget_for_engine = getattr(
                cfg,
                "default_chunk_prep_request_budget_cap_for_engine",
                lambda engine_mode: 2 if engine_mode == cfg.ENGINE_MODE_CPU else 8,
            )
            cfg.chunk_prep_request_budget_cap = int(default_budget_for_engine(cfg.engine_mode))

    if chunk_request_budget_cap is not None:
        cfg.chunk_prep_request_budget_cap = max(1, int(chunk_request_budget_cap))

    if rc_enabled is not None:
        cfg.RADIANCE_CASCADES_ENABLED = bool(rc_enabled)
        render_contract.RADIANCE_CASCADES_ENABLED = bool(rc_enabled)

    if terrain_zstd_enabled is not None:
        cfg.TERRAIN_ZSTD_ENABLED = bool(terrain_zstd_enabled)
        render_contract.TERRAIN_ZSTD_ENABLED = bool(terrain_zstd_enabled)
        render_contract.terrain_zstd_enabled = bool(terrain_zstd_enabled)

    if mesh_zstd_enabled is not None:
        cfg.MESH_ZSTD_ENABLED = bool(mesh_zstd_enabled)
        render_contract.MESH_ZSTD_ENABLED = bool(mesh_zstd_enabled)
        render_contract.mesh_zstd_enabled = bool(mesh_zstd_enabled)

    if tile_merging_enabled is not None:
        cfg.TILE_MERGING_ENABLED = bool(tile_merging_enabled)
        render_contract.TILE_MERGING_ENABLED = bool(tile_merging_enabled)
        render_contract.tile_merging_enabled = bool(tile_merging_enabled)

    render_contract.engine_mode = cfg.engine_mode
    render_contract.chunk_prep_request_budget_cap = int(cfg.chunk_prep_request_budget_cap)


def _coalesce_bool(value: bool | None, default: bool) -> bool:
    return default if value is None else bool(value)


def _build_renderer_from_args(args: argparse.Namespace):
    from engine.benchmark_runtime import RendererLaunchConfig, make_renderer

    mode = str(args.benchmark_mode).strip().lower().replace("-", "_")
    fixed_view_dimensions = args.fixed_view
    if fixed_view_dimensions is None and mode in ("fixed", "fly_forward"):
        fixed_view_dimensions = (16, 16, 16)

    config = RendererLaunchConfig(
        mode=mode,
        seed=int(args.seed),
        fixed_view_dimensions=fixed_view_dimensions,
        terrain_batch_size=args.terrain_batch_size,
        mesh_batch_size=args.mesh_batch_size,
        terrain_zstd_enabled=args.terrain_zstd,
        mesh_zstd_enabled=args.mesh_zstd,
        tile_merging_enabled=args.tile_merge,
        freeze_view_origin=_coalesce_bool(args.freeze_view_origin, mode == "fixed"),
        freeze_camera=_coalesce_bool(args.freeze_camera, mode == "fixed"),
        exit_when_view_ready=_coalesce_bool(args.exit_when_view_ready, False),
        fly_speed_mps=float(args.fly_speed_mps),
        target_rendered_chunks=int(args.target_rendered_chunks),
        status_log_interval_s=float(args.status_log_interval_s),
    )
    return make_renderer(config)


def _build_arg_parser() -> argparse.ArgumentParser:
    try:
        from engine.benchmark_runtime import DEFAULT_BENCHMARK_FLY_SPEED_MPS
    except Exception:
        DEFAULT_BENCHMARK_FLY_SPEED_MPS = 20.0

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
    parser.add_argument(
        "--rc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Radiance Cascades for this run. Use --no-rc for raster/no-RC profiling.",
    )
    parser.add_argument(
        "--terrain-zstd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable zstd level-1 compression for CPU-side terrain chunk payloads.",
    )
    parser.add_argument(
        "--mesh-zstd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable experimental zstd readback compression for offscreen mesh buffers.",
    )
    parser.add_argument(
        "--tile-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable merged visible tile GPU buffers. Default is off to reduce footprint.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="World seed.")
    parser.add_argument(
        "--benchmark-mode",
        choices=("interactive", "fixed", "fly_forward"),
        default="interactive",
        help="Run mode. interactive is the normal game window; fixed/fly_forward are CLI benchmark modes.",
    )
    parser.add_argument(
        "--fixed-view",
        "--view",
        dest="fixed_view",
        type=_parse_view_dimensions,
        default=None,
        help="Fixed visible chunk dimensions, for example 16x16x16. Used by fixed and fly_forward modes.",
    )
    parser.add_argument("--terrain-batch-size", type=int, default=None, help="Terrain backend poll/batch size.")
    parser.add_argument("--mesh-batch-size", type=int, default=None, help="Mesh drain/finalize batch size.")
    parser.add_argument("--chunk-request-budget-cap", type=int, default=None, help="Chunk prep request budget cap.")
    parser.add_argument("--freeze-view-origin", action=argparse.BooleanOptionalAction, default=None, help="Freeze the visible chunk origin.")
    parser.add_argument("--freeze-camera", action=argparse.BooleanOptionalAction, default=None, help="Freeze camera updates.")
    parser.add_argument("--exit-when-view-ready", action=argparse.BooleanOptionalAction, default=None, help="Close after the fixed visible view is ready.")
    parser.add_argument("--fly-speed-mps", type=float, default=DEFAULT_BENCHMARK_FLY_SPEED_MPS, help="Fly-forward benchmark speed in meters per second.")
    parser.add_argument("--target-rendered-chunks", type=int, default=4096, help="Unique rendered chunk target for fly-forward mode.")
    parser.add_argument("--status-log-interval-s", type=float, default=1.0, help="Fly-forward status log interval in seconds.")
    parser.add_argument("--start-profiling-hud", action="store_true", help="Enable the profiling HUD immediately after renderer creation.")
    return parser


def _summarize_launch(args: argparse.Namespace) -> str:
    rc_text = "default" if args.rc is None else ("on" if args.rc else "off")
    terrain_zstd_text = "default" if args.terrain_zstd is None else ("on" if args.terrain_zstd else "off")
    mesh_zstd_text = "default" if args.mesh_zstd is None else ("on" if args.mesh_zstd else "off")
    tile_merge_text = "default" if args.tile_merge is None else ("on" if args.tile_merge else "off")
    engine_text = args.engine or "configured"
    view_text = "default" if args.fixed_view is None else "×".join(str(value) for value in args.fixed_view)
    return (
        f"mode={args.benchmark_mode}, engine={engine_text}, rc={rc_text}, "
        f"terrain_zstd={terrain_zstd_text}, mesh_zstd={mesh_zstd_text}, tile_merge={tile_merge_text}, "
        f"view={view_text}, terrain_batch={args.terrain_batch_size or 'default'}, "
        f"mesh_batch={args.mesh_batch_size or 'default'}, chunk_budget={args.chunk_request_budget_cap or 'default'}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_runtime_knobs(
        engine=args.engine,
        chunk_request_budget_cap=args.chunk_request_budget_cap,
        rc_enabled=args.rc,
        terrain_zstd_enabled=args.terrain_zstd,
        mesh_zstd_enabled=args.mesh_zstd,
        tile_merging_enabled=args.tile_merge,
        allow_metal_fallback=bool(args.allow_metal_fallback),
    )

    print(f"Info: Minechunk launch config {_summarize_launch(args)}")
    renderer = _build_renderer_from_args(args)

    if args.start_profiling_hud:
        from engine import profiling_runtime
        profiling_runtime.enable(renderer)

    renderer.run()


if __name__ == "__main__":
    main()
