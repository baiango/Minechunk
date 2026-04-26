from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_SHADER_DIR = Path(__file__).with_name("shaders")


@lru_cache(maxsize=None)
def load_shader_text(filename: str) -> str:
    """Load a checked-in shader asset beside the engine package.

    Keeping WGSL/MSL in real shader files makes diffs, editor highlighting, and
    copy/paste into shader tools much less painful than editing giant Python
    triple-quoted strings.
    """
    path = _SHADER_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - packaging/config error
        raise FileNotFoundError(f"Missing shader asset: {path}") from exc
