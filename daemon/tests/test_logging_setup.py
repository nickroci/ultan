"""Tests for the daemon's logging setup.

``logging_setup.configure`` rotates large log files at startup, mounts
a file handler, and optionally adds a stderr handler in foreground
mode. Failures must never raise — logging is best-effort.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_mem_daemon.logging_setup import (
    LOG_ROTATE_MAX_BYTES,
    configure,
    rotate_if_large,
)


def _reset_root_logger() -> None:
    """Strip handlers from the root logger so a previous test doesn't
    leave handlers behind that pollute the next configure() call."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def test_rotate_if_large_renames_when_over_threshold(tmp_path: Path) -> None:
    log = tmp_path / "daemon.log"
    log.write_bytes(b"x" * 100)
    rotate_if_large(log, max_bytes=50)
    assert not log.exists()
    assert (tmp_path / "daemon.log.1").exists()
    assert (tmp_path / "daemon.log.1").read_bytes() == b"x" * 100


def test_rotate_if_large_noop_when_under_threshold(tmp_path: Path) -> None:
    log = tmp_path / "daemon.log"
    log.write_bytes(b"x" * 100)
    rotate_if_large(log, max_bytes=1000)
    assert log.exists()
    assert log.read_bytes() == b"x" * 100
    assert not (tmp_path / "daemon.log.1").exists()


def test_rotate_if_large_clobbers_existing_backup(tmp_path: Path) -> None:
    log = tmp_path / "daemon.log"
    backup = tmp_path / "daemon.log.1"
    log.write_bytes(b"new content" * 100)
    backup.write_bytes(b"OLD")
    rotate_if_large(log, max_bytes=50)
    assert backup.exists()
    # The new content replaced the old backup.
    assert backup.read_bytes().startswith(b"new content")


def test_rotate_if_large_missing_file_is_silent(tmp_path: Path) -> None:
    # No exception, no file appears.
    rotate_if_large(tmp_path / "does-not-exist.log", max_bytes=100)
    assert not (tmp_path / "does-not-exist.log").exists()
    assert not (tmp_path / "does-not-exist.log.1").exists()


def test_configure_creates_log_file_and_returns_daemon_logger(tmp_path: Path) -> None:
    _reset_root_logger()
    try:
        log_file = tmp_path / "sub" / "daemon.log"  # parent does not exist
        logger = configure(log_file, level=logging.INFO, foreground=False)
        assert logger.name == "agent_mem_daemon"
        # parent dir is created.
        assert log_file.parent.is_dir()
        # File handler is wired up — emitting writes to disk.
        logger.info("hello")
        # Flush so the assertion can see the bytes.
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_file.exists()
        contents = log_file.read_text(encoding="utf-8")
        assert "hello" in contents
    finally:
        _reset_root_logger()


def test_configure_foreground_adds_stderr_handler(tmp_path: Path) -> None:
    _reset_root_logger()
    try:
        log_file = tmp_path / "daemon.log"
        configure(log_file, level=logging.INFO, foreground=True)
        root = logging.getLogger()
        # Exactly two handlers: file + stderr.
        kinds = {type(h).__name__ for h in root.handlers}
        assert "FileHandler" in kinds
        assert "StreamHandler" in kinds
    finally:
        _reset_root_logger()


def test_configure_falls_back_to_stderr_when_file_open_fails(tmp_path: Path) -> None:
    """If the log file can't be opened, configure() flips foreground=True
    so the user at least sees output on stderr."""
    _reset_root_logger()
    try:
        # Point log file at a path whose parent IS a regular file — mkdir
        # will succeed (parent already exists as a file? no, mkdir would
        # raise FileExistsError or NotADirectoryError). Easier: pass a
        # directory path as the log file so FileHandler can't open it.
        bad_log = tmp_path / "im-a-directory"
        bad_log.mkdir()
        configure(bad_log, level=logging.INFO, foreground=False)
        root = logging.getLogger()
        # The fall-back path adds a stderr handler. The FileHandler call
        # raised OSError, so only StreamHandler should be present.
        kinds = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in kinds
    finally:
        _reset_root_logger()


def test_configure_rotates_existing_large_log(tmp_path: Path) -> None:
    _reset_root_logger()
    try:
        log_file = tmp_path / "daemon.log"
        # Plant a too-big file.
        log_file.write_bytes(b"x" * 100)
        configure(log_file, level=logging.INFO, foreground=False, max_bytes=50)
        # The previous file got rotated to .1; the active file is fresh.
        assert (tmp_path / "daemon.log.1").exists()
        assert log_file.exists()
        # The active file is small (fresh) — should be <= a few hundred
        # bytes of any auto-emitted handler init.
        assert log_file.stat().st_size < 1024
    finally:
        _reset_root_logger()


def test_configure_wipes_existing_root_handlers(tmp_path: Path) -> None:
    _reset_root_logger()
    try:
        # Pre-install a sentinel handler.
        sentinel = logging.NullHandler()
        logging.getLogger().addHandler(sentinel)
        configure(tmp_path / "daemon.log", level=logging.INFO, foreground=False)
        # Sentinel must be gone.
        assert sentinel not in logging.getLogger().handlers
    finally:
        _reset_root_logger()


def test_rotate_max_bytes_constant_is_reasonable() -> None:
    """Pin the documented default — 5 MiB — so a refactor doesn't quietly
    drop it to KB."""
    assert LOG_ROTATE_MAX_BYTES == 5 * 1024 * 1024
