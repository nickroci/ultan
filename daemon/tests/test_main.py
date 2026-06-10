"""Tests for the daemon CLI entry point (``__main__.py``).

These tests exercise the entry point as close to end-to-end as possible
while keeping the LLM stubs and the long-running threads under control:

  - the argparse builder is tested by parsing real argv lists,
  - the PID-file helpers are tested against tmp files,
  - the run loop is driven by raising SIGTERM right after start so we
    end up exercising the lifecycle (configure logging → ensure_home →
    acquire pid → prewarm indexes → spin threads → graceful drain).

We never spawn a subprocess — every test stays in-process so coverage
counts the lines.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from agent_mem_daemon import __main__ as daemon_main

# ── PID-file helpers ─────────────────────────────────────────────────────


def test_read_pid_missing_file_returns_none(tmp_path: Path) -> None:
    assert daemon_main._read_pid(tmp_path / "missing.pid") is None


def test_read_pid_empty_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "empty.pid"
    p.write_text("", encoding="utf-8")
    assert daemon_main._read_pid(p) is None


def test_read_pid_garbage_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "garbage.pid"
    p.write_text("not a number", encoding="utf-8")
    assert daemon_main._read_pid(p) is None


def test_read_pid_with_trailing_whitespace(tmp_path: Path) -> None:
    p = tmp_path / "ok.pid"
    p.write_text("4321\n", encoding="utf-8")
    assert daemon_main._read_pid(p) == 4321


def test_pid_alive_for_current_process() -> None:
    assert daemon_main._pid_alive(os.getpid()) is True


def test_pid_alive_for_nonexistent_pid() -> None:
    # PID 1 is init and very unlikely to be absent; pick a deliberately
    # impossible-to-exist PID (huge) and assert it reports dead.
    assert daemon_main._pid_alive(2**30) is False


def test_pid_alive_with_invalid_pid() -> None:
    assert daemon_main._pid_alive(0) is False
    assert daemon_main._pid_alive(-1) is False


def test_acquire_pid_file_writes_pid(tmp_path: Path) -> None:
    pid_file = tmp_path / "sub" / "daemon.pid"
    daemon_main.acquire_pid_file(pid_file)
    assert pid_file.exists()
    assert int(pid_file.read_text(encoding="utf-8").strip()) == os.getpid()


def test_acquire_pid_file_replaces_stale_pid(tmp_path: Path) -> None:
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(f"{2**30}\n", encoding="utf-8")  # stale (never alive)
    daemon_main.acquire_pid_file(pid_file)
    assert int(pid_file.read_text(encoding="utf-8").strip()) == os.getpid()


def test_acquire_pid_file_refuses_when_live_owner(tmp_path: Path) -> None:
    pid_file = tmp_path / "daemon.pid"
    # Plant our own PID — we're alive by definition, so this must fail.
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        daemon_main.acquire_pid_file(pid_file)
    assert exc.value.code == 2


def test_acquire_pid_file_handles_write_failure(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "daemon.pid"

    def _fail(self, *args, **kwargs):
        raise OSError("simulated write error")

    monkeypatch.setattr(Path, "write_text", _fail)
    with pytest.raises(SystemExit) as exc:
        daemon_main.acquire_pid_file(pid_file)
    assert exc.value.code == 1


def test_release_pid_file_removes_own_pid(tmp_path: Path) -> None:
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    daemon_main.release_pid_file(pid_file)
    assert not pid_file.exists()


def test_release_pid_file_leaves_other_pid_alone(tmp_path: Path) -> None:
    pid_file = tmp_path / "daemon.pid"
    other = os.getpid() + 1  # almost certainly not us
    pid_file.write_text(f"{other}\n", encoding="utf-8")
    daemon_main.release_pid_file(pid_file)
    assert pid_file.exists()


def test_release_pid_file_silent_on_missing(tmp_path: Path) -> None:
    # Must not raise.
    daemon_main.release_pid_file(tmp_path / "never-existed.pid")


# ── Argparse ─────────────────────────────────────────────────────────────


def test_parser_defaults() -> None:
    parser = daemon_main._build_parser()
    args = parser.parse_args([])
    assert args.foreground is True
    assert args.verbose is False
    assert args.events_file is None
    assert args.log_file is None
    assert args.pid_file is None


def test_parser_accepts_all_knobs(tmp_path: Path) -> None:
    parser = daemon_main._build_parser()
    args = parser.parse_args(
        [
            "--events-file",
            str(tmp_path / "events.jsonl"),
            "--log-file",
            str(tmp_path / "daemon.log"),
            "--pid-file",
            str(tmp_path / "daemon.pid"),
            "--poll-interval",
            "0.5",
            "--max-turns",
            "20",
            "--inactivity-seconds",
            "120",
            "--librarian-concurrency",
            "4",
            "--librarian-debounce-secs",
            "12.5",
            "--session-end-debounce-secs",
            "3.0",
            "--scholar-every-k",
            "7",
            "--scholar-every-m-secs",
            "45.0",
            "--scholar-max-batch",
            "15",
            "--queue-ceiling",
            "250",
            "--sweep-interval-secs",
            "120.0",
            "--verbose",
        ]
    )
    assert args.events_file == tmp_path / "events.jsonl"
    assert args.log_file == tmp_path / "daemon.log"
    assert args.pid_file == tmp_path / "daemon.pid"
    assert args.poll_interval == 0.5
    assert args.max_turns == 20
    assert args.inactivity_seconds == 120
    assert args.librarian_concurrency == 4
    assert args.librarian_debounce_secs == 12.5
    assert args.session_end_debounce_secs == 3.0
    assert args.scholar_every_k == 7
    assert args.scholar_every_m_secs == 45.0
    assert args.scholar_max_batch == 15
    assert args.queue_ceiling == 250
    assert args.sweep_interval_secs == 120.0
    assert args.verbose is True


# ── _prewarm_indexes ────────────────────────────────────────────────────


def test_prewarm_indexes_skips_when_dir_missing(tmp_path: Path, caplog) -> None:
    missing = tmp_path / "no-such-knowledge"
    caplog.set_level(logging.INFO)
    daemon_main._prewarm_indexes(missing, logging.getLogger("test"))
    assert any("missing" in r.message for r in caplog.records)


def test_prewarm_indexes_logs_success_when_dir_exists(tmp_path: Path, caplog) -> None:
    """Just ensure no exception escapes when the dir exists; bm25 may or
    may not be importable depending on whether the path dep is built."""
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "x.md").write_text("# x\n", encoding="utf-8")
    caplog.set_level(logging.INFO)
    daemon_main._prewarm_indexes(k, logging.getLogger("test"))
    # No assertion needed — pass if no exception.


def test_prewarm_indexes_swallows_bm25_failure(tmp_path: Path, monkeypatch, caplog) -> None:
    """If bm25 raises during load, we log and continue (lazy rebuild on
    first use). The function must not propagate the exception."""
    k = tmp_path / "knowledge"
    k.mkdir()

    # Inject a bm25 module shim that raises on load_or_build.
    import sys
    import types

    fake = types.ModuleType("bm25")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated bm25 boom")

    fake.load_or_build = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bm25", fake)

    caplog.set_level(logging.INFO)
    daemon_main._prewarm_indexes(k, logging.getLogger("test"))
    # An exception trace was logged but no exception propagated.
    assert any("bm25 prewarm failed" in r.message for r in caplog.records)


# ── Signal handlers ─────────────────────────────────────────────────────


def test_install_signal_handlers_sets_event_on_sigterm(monkeypatch) -> None:
    """We can't actually deliver a signal in a unit test without
    affecting the test runner. Instead we capture the handler that
    ``signal.signal`` was called with and invoke it directly — proves the
    handler flips the event."""
    captured = {}
    real_signal = signal.signal

    def _capture(sig, handler):
        captured[sig] = handler
        # Don't actually install — keep pytest's own handlers intact.
        return real_signal(sig, signal.SIG_DFL) if False else None  # noqa: SIM108

    monkeypatch.setattr(signal, "signal", _capture)

    ev = threading.Event()
    daemon_main._install_signal_handlers(ev)
    assert signal.SIGTERM in captured
    assert signal.SIGINT in captured
    # Invoke the captured handler — must flip the event.
    handler = captured[signal.SIGTERM]
    assert handler is not None
    handler(signal.SIGTERM, None)
    assert ev.is_set()


def test_install_signal_handlers_handles_unknown_signum(monkeypatch) -> None:
    """The handler formats signum via ``signal.Signals(n).name`` and
    falls back to ``str(signum)`` for unknown numbers. Just driving the
    fall-back path."""
    captured = {}
    monkeypatch.setattr(signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    ev = threading.Event()
    daemon_main._install_signal_handlers(ev)
    handler = captured[signal.SIGTERM]
    # Pass a deliberately-unmapped signum.
    handler(999, None)
    assert ev.is_set()


# ── Full run() lifecycle ────────────────────────────────────────────────


def test_main_run_starts_and_shuts_down(monkeypatch, tmp_path: Path) -> None:
    """Drive ``main()`` end-to-end with a tiny argv. We stop the daemon
    by setting the internal stop_event from a background thread so we
    don't have to deliver a real signal during the test run."""

    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))

    # Stub _install_signal_handlers to capture the stop event so we can
    # flip it ourselves (and avoid hijacking pytest's signal handlers).
    captured_event = {}
    real_install = daemon_main._install_signal_handlers

    def _capture(ev):
        captured_event["ev"] = ev
        # Don't actually install signal handlers; we'll flip the event.
        _ = real_install  # keep ref alive for coverage of import path

    monkeypatch.setattr(daemon_main, "_install_signal_handlers", _capture)

    # Set up the kill thread *before* run() so the daemon stops promptly.
    def _killer():
        # Wait for the event to be installed.
        deadline = time.monotonic() + 5.0
        while "ev" not in captured_event and time.monotonic() < deadline:
            time.sleep(0.01)
        # Give the daemon a chance to enter its main loop.
        time.sleep(0.10)
        ev = captured_event.get("ev")
        if ev is not None:
            ev.set()

    t = threading.Thread(target=_killer, daemon=True)
    t.start()

    # main() returns the exit code from run() (0 on clean shutdown).
    rc = daemon_main.main(
        [
            "--events-file",
            str(tmp_path / "events.jsonl"),
            "--log-file",
            str(tmp_path / "daemon.log"),
            "--pid-file",
            str(tmp_path / "daemon.pid"),
            "--poll-interval",
            "0.05",
            "--librarian-debounce-secs",
            "0.05",
            "--session-end-debounce-secs",
            "0.05",
            "--sweep-interval-secs",
            "60.0",
            "--scholar-every-m-secs",
            "60.0",
        ]
    )
    t.join(timeout=2.0)
    assert rc == 0
    # PID file was created during startup and removed at shutdown.
    assert not (tmp_path / "daemon.pid").exists()
    # The log file got written.
    log_file = tmp_path / "daemon.log"
    assert log_file.exists()
    # Reset the root logger so other tests aren't affected.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


# ── daemon.state lifecycle flag ───────────────────────────────────────


def test_write_daemon_state_publishes_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    daemon_main.write_daemon_state("warming")
    raw = json.loads((tmp_path / "daemon.state").read_text(encoding="utf-8"))
    assert raw["phase"] == "warming"
    assert raw["pid"] == os.getpid()
    assert raw["since"]


def test_clear_daemon_state_removes_own_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    daemon_main.write_daemon_state("ready")
    daemon_main.clear_daemon_state()
    assert not (tmp_path / "daemon.state").exists()


def test_clear_daemon_state_leaves_other_daemons_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected duplicate exiting must not erase the live daemon's flag."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    (tmp_path / "daemon.state").write_text(
        json.dumps({"phase": "ready", "pid": os.getpid() + 99999, "since": "x"}),
        encoding="utf-8",
    )
    daemon_main.clear_daemon_state()
    assert (tmp_path / "daemon.state").exists()
