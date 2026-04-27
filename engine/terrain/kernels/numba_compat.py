from __future__ import annotations

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - fallback for environments without numba
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range
