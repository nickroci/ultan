"""Logging setup with size-based rotation.

Pattern lifted from mann1x's ``run_daemon.py`` (prior-art/claude-hooks):
on startup, if the log file is over the rotate threshold, rename it to
``daemon.log.1`` (clobbering any existing ``.1``) and start fresh. We
don't use ``RotatingFileHandler`` because we want rotation *at startup*
and across restarts, not just within a single process — simpler to do
ourselves.

Logging is a debugging aid, never a hard dependency: every failure path
here is best-effort and silent so a logging hiccup can never block the
daemon from coming up.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 5 MiB — matches mann1x's choice. Enough for a multi-week steady state,
# small enough that the rotated file fits in one editor scroll.
LOG_ROTATE_MAX_BYTES = 5 * 1024 * 1024


def rotate_if_large(path: Path, *, max_bytes: int = LOG_ROTATE_MAX_BYTES) -> None:
    """Rename ``path`` -> ``path.name + ".1"`` if it exceeds max_bytes.

    Best-effort. Never raises.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= max_bytes:
        return
    backup = path.parent / (path.name + ".1")
    try:
        if backup.exists():
            backup.unlink()
    except OSError:
        pass
    try:
        path.rename(backup)
    except OSError:
        pass


def configure(
    log_file: Path,
    *,
    level: int = logging.INFO,
    foreground: bool = False,
    max_bytes: int = LOG_ROTATE_MAX_BYTES,
) -> logging.Logger:
    """Configure root logger to write to ``log_file`` (and stderr in
    foreground mode). Rotates the file at startup if too large.

    Returns the daemon's named logger.
    """
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    rotate_if_large(log_file, max_bytes=max_bytes)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Wipe any handlers a test or import might have installed — we want
    # a known clean state.
    for h in list(root.handlers):
        root.removeHandler(h)

    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        # If we can't open the log file, fall back to stderr so the user
        # at least sees something.
        foreground = True

    if foreground:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    return logging.getLogger("agent_mem_daemon")
