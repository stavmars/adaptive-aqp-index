#!/usr/bin/env python3
"""Thin wrapper around the compiled ``generate_dataset`` tool.

Locates the built binary and forwards all arguments to it, so the seeded
synthetic generator is reachable from the data-prep scripts directory without
duplicating its logic. Determinism and the Parquet encoding live entirely in
the C++ tool; this wrapper only resolves the executable path.

Usage:
    scripts/generate_dataset.py --output <parquet> --seed <N> --rows <N> \\
        --column <name>:<dist>:<p0>:<p1> [--column ...] [--overwrite]

Run with --help to see the full tool usage. Set A3I_BUILD_DIR to point at a
non-default build directory.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_tool() -> Path:
    """Resolve the generate_dataset executable.

    Search order: A3I_BUILD_DIR/tools, ./build/tools, then PATH.
    """
    candidates = []
    env_build = os.environ.get("A3I_BUILD_DIR")
    if env_build:
        candidates.append(Path(env_build) / "tools" / "generate_dataset")
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "build" / "tools" / "generate_dataset")

    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand

    on_path = shutil.which("generate_dataset")
    if on_path:
        return Path(on_path)

    searched = "\n  ".join(str(c) for c in candidates)
    sys.exit(
        "generate_dataset binary not found. Build the project first "
        "(cmake --build build), or set A3I_BUILD_DIR.\nSearched:\n  " + searched
    )


def main() -> int:
    tool = find_tool()
    os.execv(str(tool), [str(tool), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
