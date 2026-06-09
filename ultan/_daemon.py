"""Daemon launcher for the thin `ultan` wrapper.

The thin `ultan` env has no torch, so it doesn't run the daemon in-process — it
runs it through `uvx`, which provisions (and caches) an env that has the heavy
`retrieval` stack and runs the daemon's entry point there. The whole "launcher"
is this file: uvx does the environment management, so there's nothing to
hand-build.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

# What uvx resolves the heavy daemon from. Defaults to this repo's `retrieval`
# extra (pulls agent-mem-daemon + torch); override in dev with a local path,
# e.g. ULTAN_DAEMON_SPEC="/path/to/ultan[retrieval]".
_DEFAULT_SPEC = "ultan[retrieval] @ git+https://github.com/nickroci/ultan@experiment/uv-tool-install"
_DAEMON_ENTRYPOINT = "agent-mem-daemon"
_SPAWN_BACKOFF_S = 30.0


def _home() -> Path:
    return Path(os.environ.get("AGENT_MEM_HOME") or (Path.home() / ".agent-mem"))


def _socket_path() -> Path:
    return _home() / "priming.sock"


def _spec() -> str:
    return os.environ.get("ULTAN_DAEMON_SPEC") or _DEFAULT_SPEC


def _uvx_command(extra_args: list[str]) -> list[str]:
    return ["uvx", "--from", _spec(), _DAEMON_ENTRYPOINT, *extra_args]


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
    """`ultan daemon` — provision via uvx and run the daemon in the foreground."""
    return subprocess.call(_uvx_command(extra_args))


def ensure_running() -> bool:
    """Lazy start. If the daemon isn't answering, detach-spawn it via uvx and
    return immediately (never blocks on the ~25s warmup — the caller falls back
    to its lexical scan for this turn). A backoff stamp stops a crash-looping
    daemon from being re-spawned on every hook. This is what the hooks call."""
    if _socket_answering():
        return True
    home = _home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        stamp = home / ".daemon-spawn-attempt"
        if stamp.exists() and (time.time() - stamp.stat().st_mtime) < _SPAWN_BACKOFF_S:
            return False  # spawned recently — don't thrash
        stamp.touch()
        subprocess.Popen(
            _uvx_command(["-v"]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    return True
