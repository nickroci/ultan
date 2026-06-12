"""`ultan doctor` — one-stop health report for the installed runtime.

Answers "is Ultan working, and if not, where is it stuck?": runtime deps,
daemon process/socket, priming round-trip, capture freshness. Read-only and
stdlib-only — safe to run at any time, never spawns or writes anything.

Pre-install status ("still downloading torch") is the plugin `bin/ultan`
wrapper's job: this module can only run once the tool env exists, so the
wrapper reports install progress whenever the real binary is missing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

from . import __version__, _daemon, _priming


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def _daemon_pid(home: Path) -> int | None:
    try:
        return int((home / "daemon.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _mtime_age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _report_runtime(home: Path) -> bool:
    """Version + deps section. Returns True when the runtime is broken."""
    daemon_bin = _daemon._daemon_bin()  # pyright: ignore[reportPrivateUsage]  # intra-package
    print(f"ultan {__version__} (python {sys.version.split()[0]})")
    print(f"home: {home}")
    retrieval = importlib.util.find_spec("bm25") is not None
    print(f"[retrieval] extra (daemon + search stack): {'installed' if retrieval else 'MISSING'}")
    if daemon_bin.exists():
        print(f"daemon binary: {daemon_bin}")
    else:
        print("daemon binary: MISSING — install is incomplete; reinstall the ultan tool")
    return not retrieval or not daemon_bin.exists()


def _daemon_phase(home: Path) -> dict[str, Any] | None:
    """The daemon's lifecycle flag ({phase, pid, since}), or None."""
    try:
        raw: Any = json.loads((home / "daemon.state").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cast("dict[str, Any]", raw) if isinstance(raw, dict) else None


def _report_daemon(home: Path) -> tuple[bool, bool, bool]:
    """Process + socket + REAL priming probe. Returns (alive, socket_ok, priming_ok),
    where priming_ok means the daemon itself answered a real priming request
    (not that the socket merely accepts connections)."""
    pid = _daemon_pid(home)
    alive = pid is not None and _daemon._pid_alive(pid)  # pyright: ignore[reportPrivateUsage]
    print(f"daemon process: {'running (pid ' + str(pid) + ')' if alive else 'not running'}")
    phase = _daemon_phase(home)
    if phase is not None and alive and phase.get("pid") == pid:
        print(f"daemon phase: {phase.get('phase')} (since {phase.get('since')})")

    # Running daemon code vs what's installed now — surfaces a daemon left on old
    # code after an update (it restarts on the next session; see _daemon).
    running_ver = phase.get("version") if phase is not None else None
    installed_ver = _daemon._installed_daemon_version()  # pyright: ignore[reportPrivateUsage]
    if alive and running_ver:
        if installed_ver and running_ver != installed_ver:
            print(
                f"daemon code: {running_ver} — STALE (installed {installed_ver}); "
                "restarts on the next session"
            )
        else:
            print(f"daemon code: {running_ver}")

    socket_ok = _daemon._socket_answering()  # pyright: ignore[reportPrivateUsage]  # intra-package
    if socket_ok:
        print("priming socket: answering")
    elif alive:
        print(
            "priming socket: not answering yet — daemon is WARMING UP "
            f"(watch {home / 'daemon-spawn.log'})"
        )
    else:
        print("priming socket: absent (daemon lazy-starts on the next prompt)")

    # Send a REAL priming request — the socket accepting a connection (above)
    # does NOT mean the priming handler works: a hung RPC (serve-one-then-
    # deadlock) still accepts connects but never answers. probe_daemon does not
    # fall back to the lexical scan, so None here == the daemon didn't respond.
    priming_ok = False
    if socket_ok:
        t0 = time.monotonic()
        resp = _priming.probe_daemon("doctor health probe: packaging daemon memory")
        dt_ms = (time.monotonic() - t0) * 1000.0
        if resp is not None and resp.get("ok"):
            priming_ok = True
            print(
                f"priming probe: daemon answered ({dt_ms:.0f}ms client / "
                f"{resp.get('took_ms')}ms server, lane={resp.get('lane')})"
            )
        else:
            print(
                f"priming probe: DAEMON DID NOT ANSWER in {dt_ms:.0f}ms — the priming "
                "RPC is hung; the hook falls back to the lexical scan (priming DEGRADED)"
            )
    return alive, socket_ok, priming_ok


def _report_storage(home: Path) -> None:
    """Capture stream + library + spawn-stamp section."""
    events = home / "events.jsonl"
    age = _mtime_age(events)
    if age is None:
        print("capture stream: events.jsonl missing — no hook has captured anything yet")
    else:
        size = events.stat().st_size
        print(f"capture stream: events.jsonl last write {_fmt_age(age)} ({size} bytes)")

    kdir = home / "knowledge"
    if kdir.is_dir():
        n = sum(1 for _ in kdir.rglob("*.md"))
        print(f"knowledge: {n} markdown files")
    else:
        print("knowledge: no library yet (first curated memory creates it)")

    stamp_age = _mtime_age(home / ".daemon-spawn-attempt")
    if stamp_age is not None:
        print(f"last daemon spawn attempt: {_fmt_age(stamp_age)}")


def run() -> int:
    home = _daemon._home()  # pyright: ignore[reportPrivateUsage]  # intra-package
    broken = _report_runtime(home)
    alive, socket_ok, priming_ok = _report_daemon(home)
    _report_storage(home)

    if broken:
        verdict = "BROKEN — runtime deps missing (see above)"
    elif priming_ok:
        verdict = "HEALTHY"
    elif socket_ok:
        verdict = (
            "DEGRADED — daemon up but the priming RPC isn't answering; the hook falls back "
            "to the in-process lexical scan (see 'priming probe' above)"
        )
    elif alive:
        verdict = "WARMING — daemon is loading models; priming uses the lexical fallback meanwhile"
    else:
        verdict = (
            "IDLE — daemon down; it lazy-starts on the next prompt (lexical fallback meanwhile)"
        )
    print(f"verdict: {verdict}")
    return 1 if broken else 0
