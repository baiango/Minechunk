from __future__ import annotations

import math
import time


def profile_begin_frame(renderer) -> float | None:
    if not renderer.profiling_enabled:
        return None
    return time.perf_counter()


def profile_end_frame(renderer, started_at: float | None, frame_dt: float) -> None:
    if started_at is None:
        return
    ended_at = time.perf_counter()
    renderer.profile_window_cpu_ms += (ended_at - started_at) * 1000.0
    renderer.profile_window_frames += 1
    renderer.profile_window_frame_times.append(frame_dt)
    if ended_at >= renderer.profile_next_report:
        from .profiling_summary import refresh_profile_summary

        refresh_profile_summary(renderer, ended_at)


def profile_frame_time_percentiles(renderer) -> tuple[float, float, float, float]:
    if not renderer.profile_window_frame_times:
        return 0.0, 0.0, 0.0, 0.0
    ordered = sorted(renderer.profile_window_frame_times)
    count = len(ordered)

    def pick(percentile: float) -> float:
        index = max(0, min(count - 1, math.ceil(percentile * count) - 1))
        return ordered[index] * 1000.0

    return pick(0.50), pick(0.95), pick(0.99), pick(0.999)


def profile_average_fps(renderer) -> float:
    if not renderer.profile_window_frame_times:
        return 0.0
    avg_frame_time = sum(renderer.profile_window_frame_times) / len(renderer.profile_window_frame_times)
    return 1.0 / max(1e-6, avg_frame_time)


def record_frame_breakdown_sample(renderer, name: str, value: float) -> None:
    samples = renderer.frame_breakdown_samples.get(name)
    if samples is None:
        return
    sample_sums = getattr(renderer, "frame_breakdown_sample_sums", None)
    if sample_sums is None:
        sample_sums = {key: float(sum(existing_samples)) for key, existing_samples in renderer.frame_breakdown_samples.items()}
        renderer.frame_breakdown_sample_sums = sample_sums
    if samples.maxlen is not None and len(samples) >= samples.maxlen:
        sample_sums[name] -= float(samples[0])
    value_f = float(value)
    samples.append(value_f)
    sample_sums[name] += value_f


def frame_breakdown_average(renderer, name: str) -> float:
    samples = renderer.frame_breakdown_samples.get(name)
    if not samples:
        return 0.0
    sample_sums = getattr(renderer, "frame_breakdown_sample_sums", None)
    if sample_sums is not None:
        return float(sample_sums.get(name, 0.0)) / max(1, len(samples))
    return sum(samples) / len(samples)
