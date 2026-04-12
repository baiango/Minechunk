from __future__ import annotations

import argparse
import math
import resource
import struct
import sys
import time
from dataclasses import dataclass
from typing import Callable

import wgpu

from terrain_kernels import build_chunk_vertex_array
from voxel_world import CHUNK_SIZE, VoxelWorld


def _bench_surface_grids(world: VoxelWorld, coords: list[tuple[int, int]], rounds: int) -> tuple[float, int]:
    start = time.perf_counter()
    total_cells = 0
    for _ in range(rounds):
        for chunk_x, chunk_z in coords:
            heights, materials = world.chunk_surface_grids(chunk_x, chunk_z)
            total_cells += int(heights.size + materials.size)
    elapsed = time.perf_counter() - start
    return elapsed, total_cells


def _bench_mesh_build(
    world: VoxelWorld, coords: list[tuple[int, int]], rounds: int
) -> tuple[float, int, int, list[float]]:
    start = time.perf_counter()
    total_vertices = 0
    total_chunks = 0
    latencies: list[float] = []
    for _ in range(rounds):
        for chunk_x, chunk_z in coords:
            chunk_start = time.perf_counter()
            height_grid, material_grid = world.chunk_surface_grids(chunk_x, chunk_z)
            vertex_array, vertex_count = build_chunk_vertex_array(
                height_grid,
                material_grid,
                chunk_x,
                chunk_z,
                CHUNK_SIZE,
            )
            total_vertices += int(vertex_count)
            total_chunks += 1
            # Keep the array alive long enough for the compiler to not optimize away the call.
            if vertex_array.shape[0] == 0:
                raise RuntimeError("Unexpected empty vertex array.")
            latencies.append(time.perf_counter() - chunk_start)
    elapsed = time.perf_counter() - start
    return elapsed, total_vertices, total_chunks, latencies


def _format_rate(count: int, elapsed: float, label: str) -> str:
    rate = count / elapsed if elapsed > 0.0 else math.inf
    ms_per = (elapsed / count) * 1000.0 if count > 0 else math.inf
    return f"{label}: {rate:,.1f}/s ({ms_per:.3f} ms each)"


def _format_bytes(num_bytes: int) -> str:
    mib = num_bytes / (1024.0 * 1024.0)
    return f"{num_bytes:,} bytes ({mib:.2f} MiB)"


def pack_camera_uniform(
    position: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    forward: tuple[float, float, float],
    focal: float,
    aspect: float,
    near: float,
    far: float,
) -> bytes:
    import struct

    return struct.pack(
        "<20f",
        position[0],
        position[1],
        position[2],
        0.0,
        right[0],
        right[1],
        right[2],
        0.0,
        up[0],
        up[1],
        up[2],
        0.0,
        forward[0],
        forward[1],
        forward[2],
        0.0,
        focal,
        aspect,
        near,
        far,
    )


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index] * 1000.0


def _expected_visible_chunk_count(radius: int) -> int:
    side = radius * 2 + 1
    return side * side


def _infinite_cache_limit(max_radius: int) -> int:
    # Big enough to hold the full visible set for the largest tested radius.
    side = max(1, max_radius * 2 + 1)
    return side * side


def _make_render_coords(radius: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for dz in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            coords.append((dx, dz))
    return coords


@dataclass(frozen=True)
class BenchmarkScene:
    visible_coords: list[tuple[int, int]]
    culled_chunks: list[tuple[int, int, object]]
    no_cull_chunks: list[tuple[int, int, object]]
    full_batches: list[tuple[object, int]]
    no_cull_batches: list[tuple[object, int]]
    dummy_batches: list[tuple[object, int]]
    merged_batches: list[tuple[object, int]]
    visible_count: int
    candidate_count: int
    visible_vertices: int
    visible_memory_bytes: int
    primed_memory_bytes: int
    chunk_allocations: int
    upload_bytes: int
    prime_elapsed: float


def _align_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _timestamp_period_ns(renderer) -> float:
    for source in (
        getattr(renderer, "timestamp_period_ns", None),
        getattr(renderer, "timestamp_period", None),
        getattr(getattr(renderer, "adapter", None), "timestamp_period_ns", None),
        getattr(getattr(renderer, "adapter", None), "timestamp_period", None),
    ):
        if isinstance(source, (int, float)) and source > 0.0:
            return float(source)

    for source in (
        getattr(getattr(renderer, "adapter", None), "limits", None),
        getattr(getattr(renderer, "adapter", None), "info", None),
        getattr(getattr(renderer, "device", None), "adapter_info", None),
    ):
        if source is None:
            continue
        getter = source.get if hasattr(source, "get") else None
        for key in ("timestamp_period_ns", "timestamp_period", "timestampPeriod"):
            if getter is not None:
                value = getter(key, None)
            else:
                value = getattr(source, key, None)
            if isinstance(value, (int, float)) and value > 0.0:
                return float(value)

    return 1.0


class BenchmarkTimestampTimer:
    def __init__(self, renderer, frame_count: int) -> None:
        self.enabled = bool(getattr(renderer, "timestamp_query_supported", False))
        self.renderer = renderer
        self.frame_count = max(0, frame_count)
        self.query_set = None
        self.resolve_buffer = None
        self.timestamp_period_ns = _timestamp_period_ns(renderer)
        self._next_frame = 0
        if not self.enabled or self.frame_count <= 0:
            self.enabled = False
            return

        query_count = self.frame_count * 2
        device = renderer.device
        create_query_set = getattr(device, "create_query_set", None)
        if not callable(create_query_set):
            self.enabled = False
            return

        try:
            self.query_set = create_query_set(type="timestamp", count=query_count)
        except TypeError:
            try:
                self.query_set = create_query_set(query_type="timestamp", count=query_count)
            except Exception:
                self.enabled = False
                self.query_set = None
                return
        except Exception:
            self.enabled = False
            self.query_set = None
            return

        resolve_size = _align_up(query_count * 8, 256)
        create_buffer = getattr(device, "create_buffer", None)
        if callable(create_buffer):
            try:
                self.resolve_buffer = create_buffer(
                    size=resolve_size,
                    usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
                )
            except Exception:
                self.enabled = False
                self.query_set = None
                self.resolve_buffer = None
                return
        else:
            self.enabled = False
            self.query_set = None
            self.resolve_buffer = None

    def next_timestamp_writes(self) -> dict[str, object] | None:
        if not self.enabled or self.query_set is None:
            return None
        frame_index = self._next_frame
        if frame_index >= self.frame_count:
            return None
        self._next_frame += 1
        query_index = frame_index * 2
        return {
            "query_set": self.query_set,
            "beginning_of_pass_write_index": query_index,
            "end_of_pass_write_index": query_index + 1,
        }

    def read_frame_times_ms(self) -> list[float]:
        if not self.enabled or self.query_set is None or self.resolve_buffer is None:
            return []
        if self._next_frame <= 0:
            return []

        encoder = self.renderer.device.create_command_encoder()
        resolve_count = self._next_frame * 2
        try:
            encoder.resolve_query_set(self.query_set, 0, resolve_count, self.resolve_buffer, 0)
        except TypeError:
            try:
                encoder.resolve_query_set(self.query_set, 0, resolve_count, self.resolve_buffer, 0)
            except Exception:
                return []
        except Exception:
            return []

        self.renderer.device.queue.submit([encoder.finish()])
        queue = self.renderer.device.queue
        wait_sync = getattr(queue, "on_submitted_work_done_sync", None)
        if callable(wait_sync):
            wait_sync()
        else:
            wait_legacy = getattr(queue, "on_submitted_work_done", None)
            if callable(wait_legacy):
                try:
                    result = wait_legacy()
                    if hasattr(result, "__await__"):
                        import asyncio

                        asyncio.run(result)
                except TypeError:
                    pass

        raw = self.renderer.device.queue.read_buffer(self.resolve_buffer, 0, resolve_count * 8)
        values = struct.unpack(f"<{resolve_count}Q", bytes(raw))
        period_ns = self.timestamp_period_ns
        frame_times_ms: list[float] = []
        for frame_index in range(self._next_frame):
            start = values[frame_index * 2]
            end = values[frame_index * 2 + 1]
            delta_ns = max(0, int(end) - int(start)) * period_ns
            frame_times_ms.append(delta_ns / 1_000_000.0)
        return frame_times_ms


FrameScene = BenchmarkScene


def _prime_frame_scene(renderer, radius: int) -> FrameScene:
    _clear_renderer_chunk_cache(renderer)
    renderer.chunk_radius = radius
    renderer._chunk_prep_tokens = 0.0
    renderer._refresh_visible_chunk_coords()
    visible_coords = list(renderer._visible_chunk_coords)
    chunk_allocations, upload_bytes, prime_elapsed = _prime_renderer_chunks(renderer, visible_coords)
    renderer._refresh_visible_chunk_coords()
    culled_chunks = renderer._visible_chunks()
    no_cull_chunks: list[tuple[int, int, object]] = [
        (chunk_x, chunk_z, renderer.chunk_cache[(chunk_x, chunk_z)])
        for chunk_x, chunk_z in visible_coords
        if (chunk_x, chunk_z) in renderer.chunk_cache
    ]
    if len(no_cull_chunks) != len(culled_chunks):
        raise RuntimeError(
            "No-cull candidate set mismatch: "
            f"expected {len(culled_chunks)} candidates, got {len(no_cull_chunks)}."
        )
    full_batches, visible_vertices = _build_batches_from_chunks(culled_chunks)
    no_cull_batches, _ = _build_batches_from_chunks(no_cull_chunks)

    dummy_buffer = renderer.device.create_buffer_with_data(
        data=_make_vertex_bytes(36),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    dummy_batches = [(dummy_buffer, 36) for _ in culled_chunks]

    merged_buffer = renderer.device.create_buffer_with_data(
        data=_make_vertex_bytes(visible_vertices),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    merged_batches = [(merged_buffer, visible_vertices)]

    visible_memory_bytes = visible_vertices * 48
    primed_memory_bytes = _chunk_mesh_memory_bytes(renderer, list(renderer.chunk_cache.keys()))

    return BenchmarkScene(
        visible_coords=visible_coords,
        culled_chunks=culled_chunks,
        no_cull_chunks=no_cull_chunks,
        full_batches=full_batches,
        no_cull_batches=no_cull_batches,
        dummy_batches=dummy_batches,
        merged_batches=merged_batches,
        visible_count=len(culled_chunks),
        candidate_count=len(no_cull_chunks),
        visible_vertices=visible_vertices,
        visible_memory_bytes=visible_memory_bytes,
        primed_memory_bytes=primed_memory_bytes,
        chunk_allocations=chunk_allocations,
        upload_bytes=upload_bytes,
        prime_elapsed=prime_elapsed,
    )


def _build_batches_from_chunks(
    chunks: list[tuple[int, int, object]],
) -> tuple[list[tuple[object, int]], int]:
    full_batches: list[tuple[object, int]] = []
    total_vertices = 0
    for _, _, mesh in chunks:
        vertex_count = int(mesh.vertex_count)
        full_batches.append((mesh.vertex_buffer, vertex_count))
        total_vertices += vertex_count
    return full_batches, total_vertices


def _clear_renderer_chunk_cache(renderer) -> None:
    for mesh in renderer.chunk_cache.values():
        mesh.vertex_buffer.destroy()
    renderer.chunk_cache.clear()


def _prime_renderer_chunks(renderer, coords: list[tuple[int, int]]) -> tuple[int, int, float]:
    start = time.perf_counter()
    allocations = 0
    upload_bytes = 0
    for chunk_x, chunk_z in coords:
        key = (chunk_x, chunk_z)
        was_cached = key in renderer.chunk_cache
        renderer._ensure_chunk_mesh(chunk_x, chunk_z)
        if not was_cached:
            allocations += 1
            mesh = renderer.chunk_cache[key]
            upload_bytes += mesh.vertex_count * 48
    renderer._refresh_visible_chunk_coords()
    return allocations, upload_bytes, time.perf_counter() - start


def _chunk_mesh_memory_bytes(renderer, coords: list[tuple[int, int]]) -> int:
    total = 0
    for chunk_x, chunk_z in coords:
        mesh = renderer.chunk_cache.get((chunk_x, chunk_z))
        if mesh is None:
            continue
        total += mesh.vertex_count * 48
    return total


def _make_vertex_bytes(vertex_count: int) -> bytes:
    return bytes(vertex_count * 48)


def _trivial_fragment_shader() -> str:
    return """
struct CameraUniform {
    position: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    proj: vec4f,
}

struct VertexInput {
    @location(0) position: vec4f,
    @location(1) normal: vec4f,
    @location(2) color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) color: vec3f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = input.position;
    out.normal = input.normal.xyz;
    out.color = input.color.xyz;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return vec4f(input.color, 1.0);
}
"""


def _create_trivial_pipeline(renderer):
    return renderer.device.create_render_pipeline(
        layout=renderer.device.create_pipeline_layout(bind_group_layouts=[renderer.render_bind_group_layout]),
        vertex={
            "module": renderer.device.create_shader_module(code=_trivial_fragment_shader()),
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 48,
                    "step_mode": "vertex",
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": "float32x4"},
                        {"shader_location": 1, "offset": 16, "format": "float32x4"},
                        {"shader_location": 2, "offset": 32, "format": "float32x4"},
                    ],
                }
            ],
        },
        fragment={
            "module": renderer.device.create_shader_module(code=_trivial_fragment_shader()),
            "entry_point": "fs_main",
            "targets": [{"format": renderer.color_format}],
        },
        primitive={"topology": "triangle-list", "cull_mode": "none"},
        depth_stencil={
            "format": "depth24plus",
            "depth_write_enabled": True,
            "depth_compare": "less",
        },
    )


def _submit_custom_render(
    renderer,
    pipeline,
    draw_batches: list[tuple[object, int]],
    *,
    upload_camera: bool = True,
    timestamp_writes: dict[str, object] | None = None,
) -> tuple[object, object]:
    renderer._ensure_depth_buffer()
    if upload_camera:
        _upload_camera_uniform(renderer)
    encoder = renderer.device.create_command_encoder()
    current_texture = renderer.context.get_current_texture()
    color_view = current_texture.create_view()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": color_view,
                "resolve_target": None,
                "clear_value": (0.60, 0.80, 0.98, 1.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
        depth_stencil_attachment={
            "view": renderer.depth_view,
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
        },
        **({"timestamp_writes": timestamp_writes} if timestamp_writes is not None else {}),
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, renderer.camera_bind_group)
    for buffer, vertex_count in draw_batches:
        render_pass.set_vertex_buffer(0, buffer)
        render_pass.draw(int(vertex_count), 1, 0, 0)
    render_pass.end()
    return encoder, color_view


def _upload_camera_uniform(renderer) -> None:
    right, up, forward = renderer._camera_basis()
    renderer.device.queue.write_buffer(
        renderer.camera_buffer,
        0,
        pack_camera_uniform(
            tuple(renderer.camera.position),
            right,
            up,
            forward,
            1.0 / math.tan(math.radians(90.0) * 0.5),
            max(
                1.0,
                renderer.canvas.get_physical_size()[0] / max(1.0, float(renderer.canvas.get_physical_size()[1])),
            ),
            0.1,
            1024.0,
        ),
    )


def _measure_mode_frames(renderer, frames: int, warmup: int, frame_fn) -> tuple[float, list[float]]:
    _drain_gpu(renderer)
    for _ in range(max(0, warmup)):
        frame_fn(False)
    _drain_gpu(renderer)
    frame_times: list[float] = []
    start = time.perf_counter()
    for _ in range(max(1, frames)):
        frame_start = time.perf_counter()
        frame_fn(True)
        frame_times.append(time.perf_counter() - frame_start)
    elapsed = time.perf_counter() - start
    _drain_gpu(renderer)
    return elapsed, frame_times


def _drain_gpu(renderer) -> None:
    queue = renderer.device.queue
    wait_sync = getattr(queue, "on_submitted_work_done_sync", None)
    if callable(wait_sync):
        wait_sync()
        return
    wait_legacy = getattr(queue, "on_submitted_work_done", None)
    if callable(wait_legacy):
        try:
            result = wait_legacy()
            if hasattr(result, "__await__"):
                import asyncio

                asyncio.run(result)
            return
        except TypeError:
            pass


def _run_frame_mode(
    renderer,
    scene: FrameScene,
    mode: str,
    trivial_pipeline,
    dt: float,
    *,
    timestamp_writes: dict[str, object] | None = None,
) -> None:
    renderer._update_camera(dt)

    if mode == "full normal":
        renderer._prepare_chunks(dt)
        _ = renderer._visible_chunks()
        encoder, color_view = _submit_custom_render(
            renderer,
            renderer.render_pipeline,
            scene.full_batches,
            upload_camera=True,
            timestamp_writes=timestamp_writes,
        )
    else:
        if mode != "no generation":
            renderer._prepare_chunks(dt)
        else:
            renderer._refresh_visible_chunk_coords()

        if mode != "no culling":
            _ = renderer._visible_chunks()

        if mode == "no culling":
            draw_batches = scene.no_cull_batches
        elif mode == "dummy mesh (synthetic)":
            draw_batches = scene.dummy_batches
        elif mode == "merged draws (synthetic)":
            draw_batches = scene.merged_batches
        else:
            draw_batches = scene.full_batches

        if mode == "no draw":
            draw_batches = []

        pipeline = trivial_pipeline if mode == "reduced shader" else renderer.render_pipeline
        upload_camera = mode != "no upload"
        encoder, color_view = _submit_custom_render(
            renderer,
            pipeline,
            draw_batches,
            upload_camera=upload_camera,
            timestamp_writes=timestamp_writes,
        )

    renderer.device.queue.submit([encoder.finish()])
    _ = color_view


def _make_frame_fn(
    renderer,
    scene: FrameScene,
    mode: str,
    trivial_pipeline,
    timestamp_timer: BenchmarkTimestampTimer | None = None,
) -> Callable[[bool], None]:
    def frame(measured: bool) -> None:
        dt = 0.0
        timestamp_writes = timestamp_timer.next_timestamp_writes() if measured and timestamp_timer is not None else None
        _run_frame_mode(
            renderer,
            scene,
            mode,
            trivial_pipeline,
            dt,
            timestamp_writes=timestamp_writes,
        )

    return frame


def _measure_frame_breakdown(renderer, frames: int, warmup: int, scene: FrameScene) -> dict[str, float]:
    phase_samples: dict[str, list[float]] = {
        "cpu frame issue / encode": [],
        "cpu world update": [],
        "cpu chunk prep": [],
        "cpu visibility lookup": [],
        "cpu camera upload": [],
        "cpu render encode": [],
        "cpu queue submit": [],
        "wall frame": [],
    }

    def one_frame() -> None:
        wall_start = time.perf_counter()
        cpu_start = wall_start

        dt = 0.0
        phase_start = time.perf_counter()
        renderer._update_camera(dt)
        phase_samples["cpu world update"].append(time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        renderer._prepare_chunks(dt)
        phase_samples["cpu chunk prep"].append(time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        _ = renderer._visible_chunks()
        phase_samples["cpu visibility lookup"].append(time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        _upload_camera_uniform(renderer)
        phase_samples["cpu camera upload"].append(time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        encoder, color_view = _submit_custom_render(
            renderer,
            renderer.render_pipeline,
            scene.full_batches,
            upload_camera=False,
        )
        phase_samples["cpu render encode"].append(time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        renderer.device.queue.submit([encoder.finish()])
        _ = color_view
        phase_samples["cpu queue submit"].append(time.perf_counter() - phase_start)

        phase_samples["cpu frame issue / encode"].append(time.perf_counter() - cpu_start)
        phase_samples["wall frame"].append(time.perf_counter() - wall_start)

    _drain_gpu(renderer)
    for _ in range(max(0, warmup)):
        one_frame()

    _drain_gpu(renderer)
    for _ in range(max(1, frames)):
        one_frame()

    _drain_gpu(renderer)

    return {
        phase: _percentile_ms(samples, 0.50)
        for phase, samples in phase_samples.items()
    }


def _bench_isolation_modes(
    renderer, scene: FrameScene, frames: int, warmup: int
) -> tuple[dict[str, float], dict[str, tuple[float, float, float, float]]]:
    trivial_pipeline = _create_trivial_pipeline(renderer)

    mode_order = [
        "full normal",
        "no generation",
        "no upload",
        "no draw",
        "no culling",
        "dummy mesh (synthetic)",
        "merged draws (synthetic)",
        "reduced shader",
    ]
    wall_times: dict[str, float] = {}
    frame_stats: dict[str, tuple[float, float, float, float]] = {}

    for mode in mode_order:
        frame_fn = _make_frame_fn(renderer, scene, mode, trivial_pipeline)
        elapsed, frame_times = _measure_mode_frames(renderer, frames, warmup, frame_fn)
        wall_times[mode] = (elapsed / max(1, frames)) * 1000.0
        frame_stats[mode] = (
            _percentile_ms(frame_times, 0.50),
            _percentile_ms(frame_times, 0.95),
            _percentile_ms(frame_times, 0.99),
            wall_times[mode],
        )

    return wall_times, frame_stats


def _measure_render_stats(
    renderer, scene: FrameScene, rounds: int, warmup: int
) -> tuple[float, float, float, float, float | None, float | None, float | None]:
    gpu_timer = BenchmarkTimestampTimer(renderer, max(1, rounds))
    frame_fn = _make_frame_fn(renderer, scene, "full normal", None, gpu_timer if gpu_timer.enabled else None)
    elapsed, frame_times = _measure_mode_frames(renderer, rounds, warmup, frame_fn)
    measured_rounds = max(1, rounds)
    fps = measured_rounds / elapsed if elapsed > 0.0 else math.inf
    p50_ms = _percentile_ms(frame_times, 0.50)
    p95_ms = _percentile_ms(frame_times, 0.95)
    p99_ms = _percentile_ms(frame_times, 0.99)
    if gpu_timer.enabled:
        gpu_frame_times = gpu_timer.read_frame_times_ms()
        if not gpu_frame_times:
            return fps, p50_ms, p95_ms, p99_ms, None, None, None
        gpu_p50_ms = _percentile_ms([value / 1000.0 for value in gpu_frame_times], 0.50)
        gpu_p95_ms = _percentile_ms([value / 1000.0 for value in gpu_frame_times], 0.95)
        gpu_p99_ms = _percentile_ms([value / 1000.0 for value in gpu_frame_times], 0.99)
        return fps, p50_ms, p95_ms, p99_ms, gpu_p50_ms, gpu_p95_ms, gpu_p99_ms
    return fps, p50_ms, p95_ms, p99_ms, None, None, None


def _find_render_capacity(
    renderer,
    target_fps: float,
    max_radius: int,
    rounds: int,
    warmup: int,
    *,
    show_progress: bool = True,
) -> tuple[int, float, dict[int, tuple[float, float, float, float, float | None, float | None, float | None]]]:
    measured: dict[int, tuple[float, float, float, float, float | None, float | None, float | None]] = {}
    best_radius = 0
    best_rate = 0.0
    fallback_radius = 0
    fallback_rate = 0.0

    # Search for the largest radius that still meets the target rate.
    # This assumes performance generally worsens as radius grows, which is
    # the behavior we want to characterize in this benchmark.
    lo = 0
    hi = max(0, max_radius)
    step = 0
    while lo <= hi:
        step += 1
        radius = (lo + hi) // 2
        if show_progress:
            print(
                f"radius search step {step}: testing radius {radius} "
                f"(search range {lo}-{hi})",
                flush=True,
            )
        scene = _prime_frame_scene(renderer, radius)
        stats = _measure_render_stats(renderer, scene, rounds, warmup)
        measured[radius] = stats
        if stats[0] > fallback_rate:
            fallback_radius = radius
            fallback_rate = stats[0]
        if stats[0] >= target_fps:
            best_radius = radius
            best_rate = stats[0]
            lo = radius + 1
            if show_progress:
                print(
                    f"radius search step {step}: radius {radius} passed "
                    f"({stats[0]:.1f}/s >= {target_fps:.1f}/s)",
                    flush=True,
                )
        else:
            hi = radius - 1
            if show_progress:
                print(
                    f"radius search step {step}: radius {radius} failed "
                    f"({stats[0]:.1f}/s < {target_fps:.1f}/s)",
                    flush=True,
                )

    if measured:
        if best_rate > 0.0:
            return best_radius, best_rate, measured

        return fallback_radius, fallback_rate, measured

    return 0, 0.0, measured


def _parse_batch_sizes(spec: str) -> list[int]:
    sizes: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError(f"Terrain batch sizes must be positive integers, got {size}.")
        sizes.append(size)
    if not sizes:
        raise ValueError("At least one terrain batch size is required.")
    return sizes


def _binary_batch_sizes(max_size: int) -> list[int]:
    limit = max(1, int(max_size))
    sizes: list[int] = []
    size = 1
    while size < limit:
        sizes.append(size)
        size *= 2
    if not sizes or sizes[-1] != limit:
        sizes.append(limit)
    return sizes


def _drain_terrain_voxel_batches(world: VoxelWorld, coords: list[tuple[int, int]]) -> tuple[list[float], int, int]:
    world.request_chunk_voxel_batch(coords)

    poll_times: list[float] = []
    drained_chunks = 0
    poll_count = 0
    max_polls = max(8, len(coords) * 8)

    while drained_chunks < len(coords):
        poll_start = time.perf_counter()
        ready = world.poll_ready_chunk_voxel_batches()
        poll_times.append(time.perf_counter() - poll_start)
        poll_count += 1
        drained_chunks += len(ready)
        if poll_count > max_polls:
            raise RuntimeError(
                "Terrain batch benchmark did not drain within the expected poll budget; "
                "the backend may be stalled."
            )

    return poll_times, drained_chunks, poll_count


def _measure_terrain_batch_size(
    world: VoxelWorld,
    coords: list[tuple[int, int]],
    rounds: int,
    warmup: int,
    *,
    show_progress: bool = True,
    batch_size: int | None = None,
) -> tuple[float, list[float], int, int]:
    if show_progress:
        label = f" batch_size={batch_size}" if batch_size is not None else ""
        print(
            f"terrain batch benchmark{label}: warmup={max(0, warmup)} rounds={max(1, rounds)}",
            flush=True,
        )
    for _ in range(max(0, warmup)):
        _drain_terrain_voxel_batches(world, coords)

    poll_times: list[float] = []
    drained_chunks = 0
    poll_count = 0
    start = time.perf_counter()
    total_rounds = max(1, rounds)
    for round_index in range(total_rounds):
        if show_progress:
            label = f" batch_size={batch_size}" if batch_size is not None else ""
            print(
                f"terrain batch benchmark{label}: round {round_index + 1}/{total_rounds}",
                flush=True,
            )
        round_poll_times, round_drained_chunks, round_poll_count = _drain_terrain_voxel_batches(world, coords)
        poll_times.extend(round_poll_times)
        drained_chunks += round_drained_chunks
        poll_count += round_poll_count
    elapsed = time.perf_counter() - start
    return elapsed, poll_times, drained_chunks, poll_count


def _bench_terrain_batch_sizes(
    renderer,
    seed: int,
    coords: list[tuple[int, int]],
    batch_sizes: list[int],
    rounds: int,
    warmup: int,
    *,
    show_progress: bool = True,
) -> list[tuple[int, str, float, list[float], int, int]]:
    results: list[tuple[int, str, float, list[float], int, int]] = []
    for batch_size in batch_sizes:
        if show_progress:
            print(
                f"terrain batch benchmark: testing batch_size={batch_size}",
                flush=True,
            )
        world = VoxelWorld(
            seed,
            gpu_device=renderer.device,
            prefer_gpu_terrain=renderer.use_gpu_terrain,
            terrain_batch_size=batch_size,
        )
        elapsed, poll_times, drained_chunks, poll_count = _measure_terrain_batch_size(
            world,
            coords,
            rounds,
            warmup,
            show_progress=show_progress,
            batch_size=batch_size,
        )
        results.append(
            (
                batch_size,
                world.terrain_backend_label(),
                elapsed,
                poll_times,
                drained_chunks,
                poll_count,
            )
        )
        if show_progress:
            chunks_per_s = drained_chunks / elapsed if elapsed > 0.0 else math.inf
            print(
                f"terrain batch benchmark: batch_size={batch_size} done "
                f"({chunks_per_s:.1f} chunks/s)",
                flush=True,
            )
    return results


def _print_terrain_batch_benchmark(
    coords: list[tuple[int, int]],
    rounds: int,
    warmup: int,
    results: list[tuple[int, str, float, list[float], int, int]],
) -> None:
    print()
    print("Terrain batch size benchmark")
    print(f"sampled chunks per round: {len(coords)}")
    print(f"measurement rounds: {rounds}")
    print(f"warmup rounds: {warmup}")
    print("batch size | backend | chunks/s | poll p50/p95/p99 ms | avg poll ms | polls")
    for batch_size, backend_label, elapsed, poll_times, drained_chunks, poll_count in results:
        chunks_per_s = drained_chunks / elapsed if elapsed > 0.0 else math.inf
        poll_p50 = _percentile_ms(poll_times, 0.50)
        poll_p95 = _percentile_ms(poll_times, 0.95)
        poll_p99 = _percentile_ms(poll_times, 0.99)
        avg_poll_ms = (sum(poll_times) / len(poll_times)) * 1000.0 if poll_times else 0.0
        print(
            f"{batch_size:9d} | {backend_label:<7} | {chunks_per_s:8.1f} | "
            f"{poll_p50:5.2f}/{poll_p95:5.2f}/{poll_p99:5.2f} | {avg_poll_ms:9.2f} | {poll_count:5d}"
        )


def _print_validation_report(report) -> None:
    print()
    print("Terrain validation")
    print(f"backend: {report.backend_label}")
    print(f"chunk: ({report.chunk_x}, {report.chunk_z})")
    print(f"total cells: {report.total_cells}")
    print(f"height mismatches: {report.height_mismatches}")
    print(f"material mismatches: {report.material_mismatches}")
    if report.first_height_mismatch is not None:
        index, cpu_height, active_height, active_material = report.first_height_mismatch
        print(
            "first height mismatch: "
            f"cell {index} cpu={cpu_height} active={active_height} active_material={active_material}"
        )
    if report.first_material_mismatch is not None:
        index, cpu_material, active_material, active_height = report.first_material_mismatch
        print(
            "first material mismatch: "
            f"cell {index} cpu={cpu_material} active={active_material} active_height={active_height}"
        )
    print(f"match: {'yes' if report.matches else 'no'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark chunk generation speed.")
    parser.add_argument("--seed", type=int, default=1337, help="World seed to benchmark.")
    parser.add_argument("--chunks", type=int, default=2048, help="Number of chunks to sample per round.")
    parser.add_argument("--rounds", type=int, default=6, help="Number of measurement rounds.")
    parser.add_argument("--warmup", type=int, default=64, help="Warmup chunks before timing.")
    parser.add_argument("--target-render-fps", type=float, default=10.0, help="Target render rate to hold.")
    parser.add_argument("--render-capacity-max-radius", type=int, default=32, help="Maximum chunk radius to search.")
    parser.add_argument("--render-rounds", type=int, default=120, help="Number of GPU render submissions to time.")
    parser.add_argument("--render-warmup", type=int, default=8, help="Warmup render submissions before timing.")
    parser.add_argument("--isolation-radius", type=int, default=32, help="Radius to use for frame breakdown and isolation passes.")
    parser.add_argument("--isolation-frames", type=int, default=60, help="Timed frames per isolation mode.")
    parser.add_argument("--isolation-warmup", type=int, default=8, help="Warmup frames per isolation mode.")
    parser.add_argument("--terrain-batch-size", type=int, default=4096, help="How many chunks to batch per terrain backend poll.")
    parser.add_argument(
        "--terrain-batch-sizes",
        type=str,
        default="",
        help="Comma-separated terrain batch sizes to benchmark. Defaults to the configured terrain batch size only.",
    )
    parser.add_argument(
        "--terrain-batch-rounds",
        type=int,
        default=1,
        help="Number of measured rounds for the terrain batch benchmark.",
    )
    parser.add_argument(
        "--terrain-batch-warmup",
        type=int,
        default=2,
        help="Warmup rounds for the terrain batch benchmark.",
    )
    parser.add_argument(
        "--terrain-batch-chunks",
        type=int,
        default=128,
        help="Number of chunks to sample per terrain batch benchmark round.",
    )
    parser.add_argument(
        "--terrain-only",
        action="store_true",
        help="Stop after the terrain batch benchmark and skip the render-capacity sweep.",
    )
    parser.add_argument("--mesh-batch-size", type=int, default=16, help="How many voxel chunks to batch per mesh build pass.")
    parser.add_argument(
        "--skip-terrain-validation",
        action="store_true",
        help="Skip comparing the active terrain backend against the CPU reference before the benchmark runs.",
    )
    parser.add_argument("--validate-chunk-x", type=int, default=0, help="Chunk X coordinate to validate.")
    parser.add_argument("--validate-chunk-z", type=int, default=0, help="Chunk Z coordinate to validate.")
    args = parser.parse_args()

    world = VoxelWorld(args.seed)

    # Use a compact square around the origin so the benchmark is reproducible.
    side = max(1, math.ceil(math.sqrt(args.chunks)))
    coords: list[tuple[int, int]] = []
    for index in range(args.chunks):
        coords.append((index % side, index // side))

    warmup_coords = coords[: min(len(coords), max(1, args.warmup))]

    # Warm up Numba and any cache paths before measuring.
    for chunk_x, chunk_z in warmup_coords:
        height_grid, material_grid = world.chunk_surface_grids(chunk_x, chunk_z)
        build_chunk_vertex_array(height_grid, material_grid, chunk_x, chunk_z, CHUNK_SIZE)

    surface_elapsed, total_cells = _bench_surface_grids(world, coords, args.rounds)
    mesh_elapsed, total_vertices, total_chunks, cpu_latencies = _bench_mesh_build(world, coords, args.rounds)
    cpu_p50_ms = _percentile_ms(cpu_latencies, 0.50)
    cpu_p95_ms = _percentile_ms(cpu_latencies, 0.95)
    cpu_p99_ms = _percentile_ms(cpu_latencies, 0.99)

    total_chunk_samples = len(coords) * args.rounds

    print("Chunk generation benchmark")
    print(f"seed: {args.seed}")
    print(f"chunk size: {CHUNK_SIZE}")
    print(f"sampled chunks per round: {len(coords)}")
    print(f"measurement rounds: {args.rounds}")
    print()
    print(_format_rate(total_chunk_samples, surface_elapsed, "surface grid chunks"))
    print(_format_rate(total_chunk_samples, mesh_elapsed, "full chunk meshes"))
    print(
        "p50/p95/p99 chunk build latency: "
        f"{cpu_p50_ms:.3f} / {cpu_p95_ms:.3f} / {cpu_p99_ms:.3f} ms"
    )
    print(f"surface grid cells processed: {total_cells:,}")
    print(f"vertices emitted: {total_vertices:,}")
    print(f"average vertices per chunk: {total_vertices / max(1, total_chunks):.1f}")

    try:
        from renderer import TerrainRenderer
    except Exception as exc:
        print()
        print(f"GPU render benchmark skipped: {exc.__class__.__name__}: {exc}")
        return

    renderer = TerrainRenderer(
        args.seed,
        terrain_batch_size=args.terrain_batch_size,
        mesh_batch_size=args.mesh_batch_size,
    )
    if not args.skip_terrain_validation:
        report = renderer.world.validate_chunk_surface_grids(args.validate_chunk_x, args.validate_chunk_z)
        _print_validation_report(report)

    terrain_chunks = max(1, int(args.terrain_batch_chunks))
    terrain_side = max(1, math.ceil(math.sqrt(terrain_chunks)))
    terrain_coords: list[tuple[int, int]] = []
    for index in range(terrain_chunks):
        terrain_coords.append((index % terrain_side, index // terrain_side))

    if args.terrain_batch_sizes.strip():
        batch_sizes = _parse_batch_sizes(args.terrain_batch_sizes)
    else:
        batch_sizes = _binary_batch_sizes(args.terrain_batch_size)
    terrain_batch_results = _bench_terrain_batch_sizes(
        renderer,
        args.seed,
        terrain_coords,
        batch_sizes,
        max(1, args.terrain_batch_rounds),
        max(0, args.terrain_batch_warmup),
    )

    _print_terrain_batch_benchmark(
        terrain_coords,
        max(1, args.terrain_batch_rounds),
        max(0, args.terrain_batch_warmup),
        terrain_batch_results,
    )

    if args.terrain_only:
        return

    original_cache_limit = renderer.max_cached_chunks
    renderer.max_cached_chunks = _infinite_cache_limit(args.render_capacity_max_radius)
    search_radius_limit = max(0, args.render_capacity_max_radius)
    renderer.chunk_radius = search_radius_limit
    renderer.camera.position[:] = [0.0, 200.0, 0.0]
    renderer.camera.yaw = math.pi
    renderer.camera.pitch = -1.20
    renderer.camera.clamp_pitch()
    renderer._refresh_visible_chunk_coords()

    render_coords = _make_render_coords(search_radius_limit)
    full_prime_allocations, full_prime_upload_bytes, full_prime_elapsed = _prime_renderer_chunks(renderer, render_coords)
    best_radius, best_rate, measured_radii = _find_render_capacity(
        renderer,
        max(0.0, float(args.target_render_fps)),
        search_radius_limit,
        max(1, args.render_rounds),
        max(0, args.render_warmup),
        show_progress=True,
    )

    selected_scene = _prime_frame_scene(renderer, best_radius)
    expected_visible_chunks = _expected_visible_chunk_count(best_radius)
    if selected_scene.visible_count != expected_visible_chunks:
        raise RuntimeError(
            f"Cache-safe benchmark mismatch: expected {expected_visible_chunks} visible chunks, got {selected_scene.visible_count}."
        )
    chunk_span = best_radius * 2 + 1
    block_radius = best_radius * CHUNK_SIZE
    target_reached = best_rate >= max(0.0, float(args.target_render_fps))
    visible_memory_bytes = selected_scene.visible_memory_bytes
    primed_memory_bytes = selected_scene.primed_memory_bytes
    visible_vertices = selected_scene.visible_vertices
    full_prime_upload_mb_s = (full_prime_upload_bytes / max(1e-9, full_prime_elapsed)) / (1024.0 * 1024.0)
    selected_prime_upload_mb_s = (selected_scene.upload_bytes / max(1e-9, selected_scene.prime_elapsed)) / (1024.0 * 1024.0)
    peak_rss_bytes = _peak_rss_bytes()
    bytes_per_chunk = visible_memory_bytes / max(1, selected_scene.visible_count)
    full_prime_upload_bytes_per_allocation = full_prime_upload_bytes / max(1, full_prime_allocations)
    selected_prime_upload_bytes_per_allocation = selected_scene.upload_bytes / max(1, selected_scene.chunk_allocations)
    best_stats = measured_radii.get(best_radius)
    gpu_stats_available = bool(best_stats and best_stats[4] is not None)

    print()
    print("GPU render capacity")
    print(f"target render rate: {args.target_render_fps:.1f}/s")
    print(f"benchmark cache limit: infinite")
    print(f"max tested radius: {search_radius_limit}")
    print(f"render radius at target: {best_radius}")
    print(f"render distance in chunks: {chunk_span} x {chunk_span}")
    print(f"render distance in blocks: +/-{block_radius}")
    print(f"expected visible chunks per frame: {expected_visible_chunks}")
    print(f"visible chunks per frame: {selected_scene.visible_count}")
    print(f"candidate chunks per frame: {selected_scene.candidate_count}")
    print(f"visible chunk memory: {_format_bytes(visible_memory_bytes)}")
    print(f"primed cache memory: {_format_bytes(primed_memory_bytes)}")
    print(f"vertices/frame: {visible_vertices:,}")
    print(f"draw calls/frame: {selected_scene.visible_count}")
    print(f"full-search-radius priming upload MB/s: {full_prime_upload_mb_s:.1f}")
    print(f"selected-radius priming upload MB/s: {selected_prime_upload_mb_s:.1f}")
    print(f"radius priming upload MB/s: {selected_prime_upload_mb_s:.1f}")
    print(f"bytes per chunk: {_format_bytes(int(bytes_per_chunk))}")
    print(f"peak RAM: {_format_bytes(peak_rss_bytes)}")
    print(f"full-search-radius allocations / upload bytes: {full_prime_allocations} / {_format_bytes(full_prime_upload_bytes)}")
    print(f"selected-radius allocations / upload bytes: {selected_scene.chunk_allocations} / {_format_bytes(selected_scene.upload_bytes)}")
    print(f"full-search-radius upload bytes per allocation: {_format_bytes(int(full_prime_upload_bytes_per_allocation))}")
    print(f"selected-radius upload bytes per allocation: {_format_bytes(int(selected_prime_upload_bytes_per_allocation))}")
    print(f"measured render rate: {best_rate:.1f}/s")
    if best_stats and best_stats[4] is not None:
        print(
            "gpu frame time: "
            f"{best_stats[4]:.3f} / {best_stats[5]:.3f} / {best_stats[6]:.3f} ms "
            "(timestamp query)"
        )
    else:
        print("gpu frame time: unavailable (timestamp query unsupported)")
    if not target_reached:
        print(f"note: {args.target_render_fps:.1f}/s is not reachable within the tested radius range")
    print()
    print("radius vs fps/frame time")
    if gpu_stats_available:
        print("radius      fps     p50 ms   p95 ms   p99 ms   gpu p50   gpu p95   gpu p99")
    else:
        print("radius      fps     p50 ms   p95 ms   p99 ms")
    for radius in sorted(measured_radii):
        fps, p50_ms, p95_ms, p99_ms, gpu_p50_ms, gpu_p95_ms, gpu_p99_ms = measured_radii[radius]
        if gpu_stats_available and gpu_p50_ms is not None:
            print(
                f"{radius:>6}  {fps:>8.1f}  {p50_ms:>8.3f}  {p95_ms:>7.3f}  {p99_ms:>7.3f}  "
                f"{gpu_p50_ms:>7.3f}  {gpu_p95_ms:>7.3f}  {gpu_p99_ms:>7.3f}"
            )
        else:
            print(f"{radius:>6}  {fps:>8.1f}  {p50_ms:>8.3f}  {p95_ms:>7.3f}  {p99_ms:>7.3f}")

    _drain_gpu(renderer)
    isolation_radius = max(0, min(args.isolation_radius, search_radius_limit))
    isolation_scene = _prime_frame_scene(renderer, isolation_radius)
    frame_breakdown = _measure_frame_breakdown(
        renderer,
        max(1, args.isolation_frames),
        max(0, args.isolation_warmup),
        isolation_scene,
    )
    _, isolation_stats = _bench_isolation_modes(
        renderer,
        isolation_scene,
        max(1, args.isolation_frames),
        max(0, args.isolation_warmup),
    )

    print()
    print(f"frame breakdown @ radius {isolation_radius}")
    print(f"cpu frame issue / encode: {frame_breakdown['cpu frame issue / encode']:.1f} ms")
    print(f"  cpu world update: {frame_breakdown['cpu world update']:.1f} ms")
    print(f"  cpu visibility lookup: {frame_breakdown['cpu visibility lookup']:.1f} ms")
    print(f"  cpu chunk prep: {frame_breakdown['cpu chunk prep']:.1f} ms")
    print(f"  cpu camera upload: {frame_breakdown['cpu camera upload']:.1f} ms")
    print(f"  cpu render encode: {frame_breakdown['cpu render encode']:.1f} ms")
    print(f"  cpu queue submit: {frame_breakdown['cpu queue submit']:.1f} ms")
    print(f"wall frame: {frame_breakdown['wall frame']:.1f} ms")

    print()
    print(f"isolation passes @ radius {isolation_radius} (diagnostic isolates)")
    print(f"{'mode':<18} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'wall ms':>8}")
    for mode in [
        "full normal",
        "no generation",
        "no upload",
        "no draw",
        "no culling",
        "dummy mesh (synthetic)",
        "merged draws (synthetic)",
        "reduced shader",
    ]:
        p50_ms, p95_ms, p99_ms, wall_ms = isolation_stats[mode]
        print(f"{mode:<18} {p50_ms:8.1f} {p95_ms:8.1f} {p99_ms:8.1f} {wall_ms:8.1f}")

    renderer.max_cached_chunks = original_cache_limit


if __name__ == "__main__":
    main()
