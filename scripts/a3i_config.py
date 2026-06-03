"""Shared directory resolution for the experiment scripts.

Every script resolves its roots the same way and in the same order, so a dataset
prepared by one tool is found by another regardless of the current directory:

    explicit CLI flag  >  process environment  >  repo-root .env file  >  repo default

Real, machine-specific paths live in a git-ignored ``.env`` at the repo root
(copy ``.env.example``); they never appear in committed code. A fresh shell needs
no setup: the scripts read ``.env`` themselves, so nothing has to be exported or
sourced, and cron/IDE invocations behave identically to an interactive shell.

Importing this module loads ``.env`` into the process environment, so the
existing ``os.environ.get(...)`` lookups elsewhere (e.g. the build-dir search via
``A3I_BUILD_DIR``) pick up ``.env`` values too.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Populate os.environ from ``REPO_ROOT/.env`` for keys not already set.

    A real ``export`` therefore wins over ``.env`` (standard dotenv precedence).
    Minimal ``KEY=VALUE`` parser, stdlib only: blank lines and ``#`` comments are
    skipped, surrounding quotes on the value are stripped, and a leading
    ``export`` is tolerated. A missing or unreadable ``.env`` is silently ignored.
    """
    try:
        text = (REPO_ROOT / ".env").read_text()
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def resolve(flag: str | None, env_key: str, default_relpath: str) -> Path:
    """``flag`` > ``$env_key`` > ``REPO_ROOT/default_relpath``.

    A relative value (from any source) is taken relative to the repo root, so the
    result is independent of the current working directory; ``~`` is expanded.
    """
    value = flag or os.environ.get(env_key) or default_relpath
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


# Named roots: their env keys and repo-relative defaults in one place.
def prepared_root(flag: str | None = None) -> Path:
    return resolve(flag, "A3I_PREPARED_ROOT", "data/prepared")


def parquet_dir(flag: str | None = None) -> Path:
    return resolve(flag, "A3I_PARQUET_DIR", "data/raw")


def results_root(flag: str | None = None) -> Path:
    return resolve(flag, "A3I_RESULTS_ROOT", "experiments/results")


def workloads_dir(flag: str | None = None) -> Path:
    return resolve(flag, "A3I_WORKLOADS_DIR", "experiments/workloads")
