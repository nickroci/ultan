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


def status() -> str:
    """Coarse daemon state for consumers that must phrase fallbacks honestly:
    ``"ready"`` (socket answering), ``"warming"`` (daemon alive or freshly
    spawned but not serving yet), or ``"down"``. One socket probe plus cheap
    file stats — fine on the hook hot path."""
    if _socket_answering():
        return "ready"
    home = _home()
    # The daemon's own lifecycle flag (written between pid-acquire and exit).
    try:
        raw: Any = json.loads((home / "daemon.state").read_text(encoding="utf-8"))
        if isinstance(raw, dict) and _pid_alive(cast("dict[str, Any]", raw).get("pid")):
            return "warming"
    except (OSError, ValueError):
        pass
    # A spawn was attempted moments ago (pre-flag window, or older daemon).
    try:
        if (time.time() - (home / ".daemon-spawn-attempt").stat().st_mtime) < 300.0:
            return "warming"
    except OSError:
        pass
    return "down"


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
