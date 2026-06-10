#!/usr/bin/env python3
"""Thin client for the agent-mem daemon RPC socket.

Auto-detects whether the argument is a wikilink/path (→ fetch_entry)
or a free-text query (→ bm25_search). Renders the response as
markdown to stdout. No deps beyond stdlib.
"""
from __future__ import annotations

import json
import os
import re
import socket
import struct
import sys
import time
from pathlib import Path

_HOME = Path(os.environ.get("AGENT_MEM_HOME") or (Path.home() / ".agent-mem"))
SOCKET_PATH = _HOME / "priming.sock"
# Default plugin data dir ("ultan" plugin @ "ultan" marketplace) — where the
# background installer leaves its lock/pid while provisioning.
_PLUGIN_DATA = Path(
    os.environ.get("CLAUDE_PLUGIN_DATA")
    or (Path.home() / ".claude" / "plugins" / "data" / "ultan-ultan")
)
PATH_CHARS_RE = re.compile(r"^[a-zA-Z0-9_./-]+$")
RECV_TIMEOUT_S = 5.0

_MSG_WARMING = (
    "# Ultan is starting up — not broken\n\n"
    "The daemon is warming (first start loads the retrieval models; allow 1-3\n"
    "minutes). Retry this search shortly. Live status: run `ultan doctor`.\n\n"
    "_Priming still works during warmup via the lexical fallback, so the\n"
    "session is not memory-blind meanwhile._\n"
)
_MSG_INSTALLING = (
    "# Ultan is still installing — not broken\n\n"
    "The plugin's background installer is provisioning the runtime (torch +\n"
    "models; minutes on a cold cache). Retry once it finishes. Progress:\n"
    "run `ultan doctor`.\n"
)
_MSG_DOWN = (
    "# Ultan daemon not running\n\n"
    f"No socket at `{SOCKET_PATH}` and no startup in progress. With the\n"
    "plugin installed the daemon lazy-starts on the next prompt — send any\n"
    "message and retry. Details: run `ultan doctor`. (From a source\n"
    "checkout: `uv run agent-mem-daemon -v`.)\n"
)


def _pid_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)  # type: ignore[arg-type]
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _startup_message() -> str | None:
    """A friendly explanation when the socket isn't answering, or None when
    the daemon is genuinely down (not installing, not warming). Checks the
    daemon's lifecycle flag first, then spawn/install breadcrumbs."""
    # 1. The daemon's own phase flag (written between pid-acquire and exit).
    try:
        state = json.loads((_HOME / "daemon.state").read_text(encoding="utf-8"))
        if _pid_alive(state.get("pid")):
            return _MSG_WARMING  # alive but socket not serving yet (or restarting)
    except (OSError, ValueError):
        pass
    # 2. A spawn was attempted moments ago (pre-flag window, or older daemon).
    try:
        age = time.time() - (_HOME / ".daemon-spawn-attempt").stat().st_mtime
        if age < 300:
            return _MSG_WARMING
    except OSError:
        pass
    # 3. The plugin's background install is still running.
    if (_PLUGIN_DATA / ".install.pid").exists() or (_PLUGIN_DATA / ".install.lock").exists():
        return _MSG_INSTALLING
    return None


def _looks_like_path(s: str) -> bool:
    s = s.strip()
    if s.startswith("[[") and s.endswith("]]"):
        return True
    if " " in s:
        return False
    if "/" in s and PATH_CHARS_RE.match(s):
        return True
    return False


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n > 0:
        chunk = sock.recv(n)
        if not chunk:
            raise ConnectionError("daemon closed connection mid-message")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def _send_request(req: dict) -> dict:
    body = json.dumps(req).encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(RECV_TIMEOUT_S)
    try:
        sock.connect(str(SOCKET_PATH))
        sock.sendall(struct.pack(">I", len(body)) + body)
        (length,) = struct.unpack(">I", _recv_exact(sock, 4))
        return json.loads(_recv_exact(sock, length).decode("utf-8"))
    finally:
        sock.close()


def _render_fetch(resp: dict) -> str:
    if not resp.get("ok"):
        return f"# Ultan fetch failed\n\nError: {resp.get('error')}\n"
    out = [f"# `{resp['path']}`\n", "## Content\n", resp["content"], ""]
    if resp.get("siblings"):
        out.append("## Sibling entries (same folder)\n")
        for s in resp["siblings"]:
            out.append(f"- `{s}`")
        out.append("")
    if resp.get("subdirs"):
        out.append("## Subfolders\n")
        for d in resp["subdirs"]:
            out.append(f"- `{d}`")
        out.append("")
    if resp.get("parent_readme_excerpt"):
        out.append("## Parent folder README (excerpt)\n")
        out.append(resp["parent_readme_excerpt"])
        out.append("")
    out.append(
        "_To fetch a neighbour: re-invoke this skill with the wikilink "
        "(e.g. `[[<folder>/<file>]]`)._"
    )
    return "\n".join(out)


def _render_search(resp: dict, query: str) -> str:
    if not resp.get("ok"):
        return f"# Ultan search failed\n\nError: {resp.get('error')}\n"
    hits = resp.get("hits", [])
    if not hits:
        return f"# No matches for `{query}`\n"
    out = [f"# Ultan search: `{query}`\n"]
    for h in hits:
        wl = h["path"][:-3] if h["path"].endswith(".md") else h["path"]
        out.append(
            f"- **[[{wl}]]**  _(score={h['score']:.2f})_ — {h['snippet']}"
        )
    out.append("")
    out.append(
        "_To fetch one: re-invoke this skill with the wikilink._"
    )
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: search.py <wikilink-or-query>", file=sys.stderr)
        return 2
    arg = " ".join(sys.argv[1:]).strip()
    if not arg:
        print("usage: search.py <wikilink-or-query>", file=sys.stderr)
        return 2

    if arg.startswith("path:"):
        arg, mode = arg[len("path:"):].strip(), "fetch"
    elif arg.startswith("query:"):
        arg, mode = arg[len("query:"):].strip(), "search"
    elif _looks_like_path(arg):
        mode = "fetch"
    else:
        mode = "search"

    if not SOCKET_PATH.exists():
        print(_startup_message() or _MSG_DOWN, file=sys.stderr)
        return 1

    try:
        if mode == "fetch":
            resp = _send_request({"op": "fetch_entry", "path": arg})
            print(_render_fetch(resp))
        else:
            resp = _send_request({"op": "bm25_search", "query": arg, "k": 8})
            print(_render_search(resp, arg))
    except (ConnectionError, OSError):
        # Socket file present but not answering — stale socket or a daemon
        # mid-restart. Same friendly triage as the missing-socket path.
        print(_startup_message() or _MSG_DOWN, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"# Ultan skill failed\n\nError: {e!r}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
