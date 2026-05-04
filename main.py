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
    renderer_backend: str | None,
    terrain_backend: str | None,
    meshing_backend: str | None,
    terrain_kernel: str | None,
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

    if renderer_backend is not None and renderer_backend.strip().lower() != "wgpu":
        raise ValueError("renderer backend must be wgpu")

    for label, backend in (("terrain backend", terrain_backend), ("meshing backend", meshing_backend)):
        if backend is not None and backend.strip().lower() not in ("cpu", "wgpu", "metal"):
            raise ValueError(f"{label} must be one of: cpu, wgpu, metal")

    if terrain_kernel is not None:
        kernel = terrain_kernel.strip().lower()
        if kernel not in ("auto", "numba", "zig"):
            raise ValueError("terrain kernel must be one of: auto, numba, zig")
        os.environ["MINECHUNK_TERRAIN_KERNEL"] = kernel
        if kernel == "numba":
            os.environ["MINECHUNK_DISABLE_ZIG_TERRAIN"] = "1"
        else:
            os.environ.pop("MINECHUNK_DISABLE_ZIG_TERRAIN", None)

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


def _warm_numba_cache() -> None:
    import numpy as np

    from engine.render_constants import VERTEX_COMPONENTS
    from engine.world_constants import BLOCK_SIZE, CHUNK_SIZE, WORLD_HEIGHT_BLOCKS
    from engine.terrain.kernels import (
        build_chunk_surface_run_table_from_heightmap_clipped,
        build_chunk_surface_vertex_array_from_heightmap_clipped,
        build_chunk_vertex_array_from_voxels_with_boundaries,
        count_chunk_surface_vertices_from_heightmap_clipped,
        emit_chunk_surface_run_table_vertices,
        emit_chunk_surface_vertices_from_heightmap_clipped,
        fill_chunk_surface_grids,
        fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface,
        surface_profile_at,
    )

    seed = 1337
    chunk_size = int(CHUNK_SIZE)
    sample_size = chunk_size + 2
    world_height = int(WORLD_HEIGHT_BLOCKS)
    local_height = chunk_size

    heights = np.empty(sample_size * sample_size, dtype=np.uint32)
    surface_materials = np.empty(sample_size * sample_size, dtype=np.uint32)
    blocks = np.zeros((local_height, sample_size, sample_size), dtype=np.uint8)
    materials = np.zeros((local_height, sample_size, sample_size), dtype=np.uint32)
    top_boundary = np.zeros((sample_size, sample_size), dtype=np.uint8)
    bottom_boundary = np.zeros((sample_size, sample_size), dtype=np.uint8)

    surface_profile_at(0.0, 0.0, seed, world_height)
    fill_chunk_surface_grids(heights, surface_materials, 0, 0, chunk_size, seed, world_height)
    fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
        blocks,
        materials,
        top_boundary,
        bottom_boundary,
        heights,
        surface_materials,
        0,
        0,
        0,
        chunk_size,
        seed,
        world_height,
        True,
    )
    build_chunk_vertex_array_from_voxels_with_boundaries(
        blocks,
        materials,
        0,
        0,
        chunk_size,
        local_height,
        top_boundary,
        bottom_boundary,
        float(BLOCK_SIZE),
        0,
    )
    build_chunk_surface_vertex_array_from_heightmap_clipped(
        heights,
        surface_materials,
        0,
        0,
        chunk_size,
        local_height,
        float(BLOCK_SIZE),
        0,
    )
    surface_vertex_count = count_chunk_surface_vertices_from_heightmap_clipped(
        heights,
        surface_materials,
        chunk_size,
        local_height,
        0,
    )
    surface_vertices = np.empty((surface_vertex_count, int(VERTEX_COMPONENTS)), dtype=np.float32)
    if surface_vertex_count > 0:
        emit_chunk_surface_vertices_from_heightmap_clipped(
            surface_vertices,
            0,
            heights,
            surface_materials,
            0,
            0,
            chunk_size,
            local_height,
            float(BLOCK_SIZE),
            0,
        )
    surface_run_table, surface_run_count, surface_run_vertex_count = build_chunk_surface_run_table_from_heightmap_clipped(
        heights,
        surface_materials,
        chunk_size,
        local_height,
        0,
    )
    surface_run_vertices = np.empty((surface_run_vertex_count, int(VERTEX_COMPONENTS)), dtype=np.float32)
    if surface_run_vertex_count > 0:
        emit_chunk_surface_run_table_vertices(
            surface_run_vertices,
            0,
            surface_run_table,
            surface_run_count,
            0,
            0,
            chunk_size,
            float(BLOCK_SIZE),
        )


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
        terrain_caves_enabled=args.terrain_caves,
        mesh_zstd_enabled=args.mesh_zstd,
        tile_merging_enabled=args.tile_merge,
        postprocess_enabled=args.postprocess,
        renderer_backend=args.renderer_backend,
        terrain_backend=args.terrain_backend,
        meshing_backend=args.meshing_backend,
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
        "--renderer-backend",
        choices=("wgpu",),
        default=None,
        help="Renderer/presentation backend. WGPU is currently the supported renderer.",
    )
    parser.add_argument(
        "--terrain-backend",
        choices=("cpu", "wgpu", "metal"),
        default=None,
        help="Terrain generation backend. Omit for the configured runtime default.",
    )
    parser.add_argument(
        "--meshing-backend",
        choices=("cpu", "wgpu", "metal"),
        default=None,
        help="Voxel meshing backend. Omit for the configured runtime default.",
    )
    parser.add_argument(
        "--allow-metal-fallback",
        action="store_true",
        help="Allow Metal mode to fall back to WGPU/CPU instead of failing loudly.",
    )
    parser.add_argument(
        "--terrain-kernel",
        choices=("auto", "numba", "zig"),
        default=None,
        help="CPU terrain kernel selection. auto uses Zig when the shared library is present, otherwise Numba.",
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
        default=False,
        help="Enable or disable zstd level-1 compression for CPU-side terrain chunk payloads. Default is off.",
    )
    parser.add_argument(
        "--terrain-caves",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable cave carving in terrain voxel fills. Use --no-terrain-caves to isolate surface/meshing cost.",
    )
    parser.add_argument(
        "--mesh-zstd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable experimental zstd readback compression for offscreen mesh buffers. Default is off.",
    )
    parser.add_argument(
        "--tile-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable merged visible tile GPU buffers. Default is off to reduce footprint.",
    )
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the postprocess/G-buffer/final-blit path. Use --no-postprocess for direct no-RC profiling.",
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
    parser.add_argument(
        "--cache-numba-only",
        action="store_true",
        help="Compile/cache the CPU Numba terrain and meshing kernels, then exit without creating a renderer.",
    )
    return parser


def _summarize_launch(args: argparse.Namespace) -> str:
    rc_text = "default" if args.rc is None else ("on" if args.rc else "off")
    terrain_zstd_text = "default" if args.terrain_zstd is None else ("on" if args.terrain_zstd else "off")
    terrain_caves_text = "default" if args.terrain_caves is None else ("on" if args.terrain_caves else "off")
    mesh_zstd_text = "default" if args.mesh_zstd is None else ("on" if args.mesh_zstd else "off")
    tile_merge_text = "default" if args.tile_merge is None else ("on" if args.tile_merge else "off")
    postprocess_text = "default" if args.postprocess is None else ("on" if args.postprocess else "off")
    renderer_backend_text = args.renderer_backend or "configured"
    terrain_backend_text = args.terrain_backend or "configured"
    meshing_backend_text = args.meshing_backend or "configured"
    terrain_kernel_text = args.terrain_kernel or "auto"
    view_text = "default" if args.fixed_view is None else "×".join(str(value) for value in args.fixed_view)
    return (
        f"mode={args.benchmark_mode}, renderer={renderer_backend_text}, terrain_backend={terrain_backend_text}, "
        f"meshing_backend={meshing_backend_text}, terrain_kernel={terrain_kernel_text}, rc={rc_text}, "
        f"terrain_zstd={terrain_zstd_text}, terrain_caves={terrain_caves_text}, "
        f"mesh_zstd={mesh_zstd_text}, tile_merge={tile_merge_text}, "
        f"postprocess={postprocess_text}, view={view_text}, terrain_batch={args.terrain_batch_size or 'default'}, "
        f"mesh_batch={args.mesh_batch_size or 'default'}, chunk_budget={args.chunk_request_budget_cap or 'default'}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_runtime_knobs(
        renderer_backend=args.renderer_backend,
        terrain_backend=args.terrain_backend,
        meshing_backend=args.meshing_backend,
        terrain_kernel=args.terrain_kernel,
        chunk_request_budget_cap=args.chunk_request_budget_cap,
        rc_enabled=args.rc,
        terrain_zstd_enabled=args.terrain_zstd,
        mesh_zstd_enabled=args.mesh_zstd,
        tile_merging_enabled=args.tile_merge,
        allow_metal_fallback=bool(args.allow_metal_fallback),
    )

    if args.cache_numba_only:
        print("Info: warming Numba CPU kernel cache")
        _warm_numba_cache()
        print("Info: Numba CPU kernel cache warmed")
        return

    print(f"Info: Minechunk launch config {_summarize_launch(args)}")
    renderer = _build_renderer_from_args(args)

    if args.start_profiling_hud:
        from engine import profiling_runtime
        profiling_runtime.enable(renderer)

    renderer.run()


if __name__ == "__main__":
    main()
