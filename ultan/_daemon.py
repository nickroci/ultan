"""Daemon launcher for the `ultan` wrapper.

The plugin provisioner (scripts/ensure-ultan.sh) installs `ultan[retrieval]`,
which puts the heavy retrieval stack (agent-mem-daemon + torch) into the SAME
venv as the `ultan` entry point — so the daemon is the `agent-mem-daemon`
console script sitting right next to this interpreter. We run THAT directly
and never re-provision from git/uvx: reusing the installed venv is the whole
point. (A bare `ultan` install — e.g. the uvx-launched MCP server — has no
daemon binary; it degrades to the lexical fallback and never spawns.)

If the daemon binary is missing the install is broken; we fail (or, on the hook
hot path, degrade to the lexical fallback) rather than silently rebuilding.
Provisioning the venv is `scripts/ensure-ultan.sh`'s job on SessionStart.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

# The daemon console script ships in the same venv as this `ultan` install, so
# it lives next to the running interpreter (sys.executable).
_DAEMON_ENTRYPOINT = "agent-mem-daemon"
_SPAWN_BACKOFF_S = 30.0


def _home() -> Path:
    return Path(os.environ.get("AGENT_MEM_HOME") or (Path.home() / ".agent-mem"))


def _socket_path() -> Path:
    return _home() / "priming.sock"


def _daemon_bin() -> Path:
    """The agent-mem-daemon console script installed alongside this `ultan`
    (same venv → sibling of the running interpreter)."""
    return Path(sys.executable).with_name(_DAEMON_ENTRYPOINT)


def _socket_answering() -> bool:
    """True if a daemon is already listening on the priming socket."""
    p = _socket_path()
    if not p.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            s.connect(str(p))
    except OSError:
        return False
    return True


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _read_daemon_state() -> dict[str, Any] | None:
    """The daemon's lifecycle flag ``{phase, pid, since, version}`` (written
    between pid-acquire and exit), or ``None`` when absent/unreadable."""
    try:
        raw: Any = json.loads((_home() / "daemon.state").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cast("dict[str, Any]", raw) if isinstance(raw, dict) else None


def _installed_daemon_version() -> str | None:
    """The agent-mem-daemon version installed in THIS venv right now, read live
    from on-disk metadata — so it reflects an ``uv tool install`` update even
    while an older daemon process keeps serving the previous code. ``None`` when
    the daemon dist isn't installed (a thin/uvx env) or metadata is unreadable.

    Cold path only (session-start / doctor): reading dist metadata is stdlib but
    does I/O, so it must never land on the per-turn hook hot path."""
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415 — cold path

    try:
        return version(_DAEMON_ENTRYPOINT)  # dist name == console-script name here
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — a broken install must not crash the caller
        return None


def status() -> str:
    """Coarse daemon state for consumers that must phrase fallbacks honestly:
    ``"ready"`` (socket answering), ``"warming"`` (daemon alive or freshly
    spawned but not serving yet), or ``"down"``. One socket probe plus cheap
    file stats — fine on the hook hot path."""
    if _socket_answering():
        return "ready"
    home = _home()
    # The daemon's own lifecycle flag (written between pid-acquire and exit).
    state = _read_daemon_state()
    if state is not None and _pid_alive(state.get("pid")):
        return "warming"
    # A spawn was attempted moments ago (pre-flag window, or older daemon).
    try:
        if (time.time() - (home / ".daemon-spawn-attempt").stat().st_mtime) < 300.0:
            return "warming"
    except OSError:
        pass
    return "down"


def _stop_daemon(pid: int, *, timeout_s: float = 3.0) -> None:
    """Stop a running daemon: SIGTERM, wait briefly for a clean exit, SIGKILL as
    a last resort. The daemon installs a SIGTERM handler that flips its stop
    event and shuts down gracefully (releasing the socket), so the common case is
    a clean stop well under ``timeout_s``."""
    import signal  # noqa: PLC0415 — cold path (session-start restart only)

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return  # already gone
    deadline = time.monotonic() + timeout_s
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def restart_if_stale() -> bool:
    """If a daemon is running OLDER code than what's now installed in the venv,
    stop it so a fresh daemon starts on current code. Returns True when a stale
    daemon was stopped — the caller's :func:`ensure_running` then spawns the new
    one (the stopped daemon no longer answers the socket).

    SESSION-START PATH ONLY — never the per-turn hook hot path: it reads package
    metadata (cold-cheap, hot-needless). The running version is what the daemon
    stamped into ``daemon.state`` at ITS startup; the installed version is read
    live, so an ``uv tool install`` update is detected even though the old
    process keeps serving the previous code.

    Anchored on the INSTALLED version: if we can't read it (a thin/uvx env with
    no daemon dist, or unreadable metadata) we never restart — there's nothing to
    compare against. Given a readable installed version, a daemon whose recorded
    version differs is stale; so is one with NO recorded version, because that is
    a legacy daemon from before this stamp existed (a current daemon always
    stamps a version when the metadata it reads is itself readable). Restarting
    that legacy daemon is the whole point — it's exactly the process left on old
    code that this feature exists to replace."""
    state = _read_daemon_state()
    if not state:
        return False
    pid = state.get("pid")
    if not _pid_alive(pid):
        return False
    installed = _installed_daemon_version()
    if not installed:
        return False  # can't tell what's installed — never restart blindly
    if state.get("version") == installed:
        return False  # up to date
    _stop_daemon(cast("int", pid))
    # Clear the spawn backoff so ensure_running() respawns immediately rather
    # than throttling this intentional restart as if it were a crash loop.
    try:
        (_home() / ".daemon-spawn-attempt").unlink()
    except OSError:
        pass
    return True


def run_foreground(extra_args: list[str]) -> int:
    """`ultan daemon` — run the installed agent-mem-daemon in the foreground."""
    daemon = _daemon_bin()
    if not daemon.exists():
        print(
            f"ultan: daemon binary not found at {daemon} — the install is "
            "incomplete. Reinstall the `ultan` tool (uv tool install ...).",
            file=sys.stderr,
        )
        return 1
    return subprocess.call([str(daemon), *extra_args])


def ensure_running() -> bool:
    """Lazy start. If the daemon isn't answering, detach-spawn the installed
    agent-mem-daemon and return immediately (never blocks on the ~25s warmup —
    the caller falls back to its lexical scan for this turn). A backoff stamp
    stops a crash-looping daemon from being re-spawned on every hook. If the
    daemon binary isn't installed we return False without spawning — provisioning
    the venv is ensure-ultan.sh's job, not the hook hot path's. This is what the
    hooks call."""
    if _socket_answering():
        return True
    daemon = _daemon_bin()
    if not daemon.exists():
        return False  # install incomplete — degrade to the in-hook lexical scan
    home = _home()
    stamp_path = home / ".daemon-spawn-attempt"
    try:
        home.mkdir(parents=True, exist_ok=True)
        # Existence check BEFORE the O_CREAT open: a freshly created stamp has
        # mtime=now, so the backoff test below would wrongly suppress the
        # first-ever spawn without this. (The check-then-open gap is benign —
        # a peer that wins it either still holds the flock, failing ours, or
        # already spawned a daemon whose PID file rejects our duplicate.)
        existed = stamp_path.exists()
        stamp_fd = os.open(str(stamp_path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return False
    try:
        try:
            # Non-blocking exclusive lock: if another hook holds it, that hook
            # is mid-spawn right this instant — don't pile a second daemon on.
            fcntl.flock(stamp_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        st = os.fstat(stamp_fd)
        if existed and (time.time() - st.st_mtime) < _SPAWN_BACKOFF_S:
            return False  # spawned recently — don't thrash
        os.pwrite(stamp_fd, b"x", 0)  # mark the attempt; the write refreshes mtime
        _spawn_detached(daemon, home)
    except OSError:
        return False
    finally:
        try:
            os.close(stamp_fd)  # releases the flock
        except OSError:
            pass
    return True


def _spawn_detached(daemon: Path, home: Path) -> None:
    """Detach-spawn the daemon with output teed to ``daemon-spawn.log``.

    NOT DEVNULL: everything that fails before the daemon configures its own
    logging — broken venv, import error, and acquire_pid_file's stale-PID
    "Refusing to start" on stderr — would otherwise vanish, turning a dead
    daemon into silent permanent lexical-fallback mode.
    """
    log_path = home / "daemon-spawn.log"
    try:
        if log_path.stat().st_size > 1_000_000:
            log_path.unlink()  # crude size cap; spawn attempts are rare
    except OSError:
        pass
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        stamp_line = f"--- spawn attempt {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        os.write(log_fd, stamp_line.encode("utf-8"))
        subprocess.Popen(
            [str(daemon), "-v"],
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        try:
            os.close(log_fd)
        except OSError:
            pass
