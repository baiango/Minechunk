from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine" / "terrain" / "kernels" / "native" / "terrain_kernel.zig"
OUTPUT_DIR = SOURCE.parent


def _library_name() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "libminechunk_terrain.dylib"
    if system == "windows":
        return "minechunk_terrain.dll"
    return "libminechunk_terrain.so"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the optional Zig terrain kernel shared library.")
    parser.add_argument("--debug", action="store_true", help="build with Zig Debug optimization instead of ReleaseFast")
    parser.add_argument("--zig", default="zig", help="path to the zig executable")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / _library_name(), help="output shared library path")
    args = parser.parse_args()

    zig = shutil.which(args.zig) if args.zig == "zig" else args.zig
    if zig is None:
        raise SystemExit("zig was not found on PATH; install Zig or pass --zig /path/to/zig")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    optimize = "Debug" if args.debug else "ReleaseFast"
    command = [
        str(zig),
        "build-lib",
        "-dynamic",
        "-O",
        optimize,
        f"-femit-bin={output}",
        str(SOURCE),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    print(output)


if __name__ == "__main__":
    main()
