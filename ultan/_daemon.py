"""Daemon launcher for the `ultan` wrapper.

`uv tool install ultan` installs the heavy retrieval stack (agent-mem-daemon +
torch) into the SAME venv as the `ultan` entry point — agent-mem-daemon is a
base dependency — so the daemon is the `agent-mem-daemon` console script sitting
right next to this interpreter. We run THAT directly and never re-provision from
git/uvx: reusing the installed venv is the whole point.

If the daemon binary is missing the install is broken; we fail (or, on the hook
hot path, degrade to the lexical fallback) rather than silently rebuilding.
Provisioning the venv is `scripts/ensure-ultan.sh`'s job on SessionStart.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

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
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.2)
        s.connect(str(p))
        s.close()
    except OSError:
        return False
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
    try:
        home.mkdir(parents=True, exist_ok=True)
        stamp = home / ".daemon-spawn-attempt"
        if stamp.exists() and (time.time() - stamp.stat().st_mtime) < _SPAWN_BACKOFF_S:
            return False  # spawned recently — don't thrash
        stamp.touch()
        subprocess.Popen(
            [str(daemon), "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    return True
