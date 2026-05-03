from __future__ import annotations

import ctypes
import gc
import platform
import time


_libc = None
_pressure_relief_fn = None


def _darwin_pressure_relief_fn():
    global _libc, _pressure_relief_fn
    if platform.system() != "Darwin":
        return None
    if _pressure_relief_fn is not None:
        return _pressure_relief_fn
    try:
        _libc = ctypes.CDLL("libc.dylib")
        fn = _libc.malloc_zone_pressure_relief
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_size_t
    except Exception:
        _pressure_relief_fn = False
        return None
    _pressure_relief_fn = fn
    return fn


def malloc_pressure_relief() -> int:
    fn = _darwin_pressure_relief_fn()
    if not fn:
        return 0
    try:
        return int(fn(None, 0))
    except Exception:
        return 0


def service_process_memory_pressure_relief(renderer, *, interval_seconds: float = 1.0, force: bool = False) -> int:
    now = time.perf_counter()
    next_at = float(getattr(renderer, "_memory_pressure_next_relief_at", 0.0))
    if not force and now < next_at:
        return 0
    renderer._memory_pressure_next_relief_at = now + max(0.1, float(interval_seconds))
    # Large NumPy arrays are refcounted, but cyclic Python objects can still
    # keep wrappers alive. Run the cheapest generation first, then ask malloc to
    # return unused pages to macOS.
    try:
        gc.collect(0)
    except Exception:
        pass
    relieved_bytes = malloc_pressure_relief()
    renderer._memory_pressure_last_relief_bytes = int(relieved_bytes)
    renderer._memory_pressure_last_relief_at = float(now)
    renderer._memory_pressure_relief_calls = int(getattr(renderer, "_memory_pressure_relief_calls", 0)) + 1
    return int(relieved_bytes)


def memory_pressure_stats(renderer) -> dict[str, int]:
    return {
        "last_relief_bytes": int(getattr(renderer, "_memory_pressure_last_relief_bytes", 0)),
        "relief_calls": int(getattr(renderer, "_memory_pressure_relief_calls", 0)),
    }
