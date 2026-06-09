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
from pathlib import Path

_HOME = Path(os.environ.get("AGENT_MEM_HOME") or (Path.home() / ".agent-mem"))
SOCKET_PATH = _HOME / "priming.sock"
PATH_CHARS_RE = re.compile(r"^[a-zA-Z0-9_./-]+$")
RECV_TIMEOUT_S = 5.0


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
        print(
            "# Ultan daemon not running\n\n"
            f"Socket missing at `{SOCKET_PATH}`. Start the daemon:\n\n"
            "```\nuv run agent-mem-daemon -v\n```\n",
            file=sys.stderr,
        )
        return 1

    try:
        if mode == "fetch":
            resp = _send_request({"op": "fetch_entry", "path": arg})
            print(_render_fetch(resp))
        else:
            resp = _send_request({"op": "bm25_search", "query": arg, "k": 8})
            print(_render_search(resp, arg))
    except Exception as e:
        print(f"# Ultan skill failed\n\nError: {e!r}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
