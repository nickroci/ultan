#!/usr/bin/env python3
"""Queue a user-asserted memory for the agent-mem Librarian.

Invoked by the `/ultan` Claude Code slash command. Stdlib-only on
purpose so this script has no install/sync requirement.

In the new architecture the Librarian is the active organiser. Rather
than writing directly to disk (which sidesteps the curator entirely),
``/ultan`` appends an event to the daemon's ``events.jsonl`` with
``payload.user_asserted = true``. The Librarian picks it up like any
other turn content and proposes structural actions; the Scholar then
approves or vetoes.

Usage:
    python3 remember.py "the memory text"
    echo "..." | python3 remember.py -

Behaviour:
- The text is appended to events.jsonl as a synthetic UserPromptSubmit
  event, with ``user_asserted=true`` so the Librarian prompt treats it
  as a user-stated rule (high trust).
- Session id is auto-generated (uuid) — these events stand alone, not
  attached to a live agent session.
- Project slug is derived from cwd (git remote → host-owner-repo
  fallback to basename).
- Prints "queued for librarian" on success.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def home() -> Path:
    env = os.environ.get("AGENT_MEM_HOME")
    return Path(env).expanduser().resolve() if env else (Path.home() / ".agent-mem")


def events_path() -> Path:
    return home() / "events.jsonl"


def project_slug(cwd: Path) -> str | None:
    """git remote → host-owner-repo slug; else basename; None if neither."""
    try:
        url = subprocess.check_output(
            ["git", "-C", str(cwd), "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        url = ""
    if url:
        cleaned = re.sub(r"^(?:https?://|git@|ssh://)", "", url)
        cleaned = cleaned.replace(":", "/").removesuffix(".git")
        return re.sub(r"[^a-z0-9]+", "-", cleaned.lower()).strip("-") or None
    base = cwd.name
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or None


def append_event(text: str, *, scope: str | None) -> dict:
    """Append a user-asserted event to events.jsonl and return the dict
    we wrote. Creates the events file and parent dir if missing."""
    h = home()
    h.mkdir(parents=True, exist_ok=True)
    path = events_path()

    session_id = f"ultan-{uuid.uuid4().hex[:8]}"
    now = time.time()
    now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds")

    # We emit two events in one append: UserPromptSubmit (carrying the
    # rule text) then Stop (so the buffer's turn-sealing logic picks it
    # up immediately). Both share the same session_id.
    common = {
        "ts": now_iso,
        "session_id": session_id,
        "cwd": str(Path.cwd()),
    }
    user_ev = {
        **common,
        "type": "UserPromptSubmit",
        "payload": {
            "text": text,
            "role": "user",
            "user_asserted": True,
            "project_slug": scope,
        },
    }
    stop_ev = {
        **common,
        "type": "Stop",
        "payload": {"user_asserted": True},
    }

    # Use O_APPEND so multiple concurrent /ultan invocations don't tear.
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(
            fd,
            (json.dumps(user_ev) + "\n" + json.dumps(stop_ev) + "\n").encode("utf-8"),
        )
    finally:
        os.close(fd)
    return user_ev


def run(text: str, *, globally: bool = False, scope: str | None = None) -> int:
    """Queue a memory for the Librarian. Shared entry point for both the
    ``remember.py`` CLI and the ``ultan remember`` subcommand.

    ``scope`` is an explicit project-slug override; when None (and not
    ``globally``) it's derived from the cwd. ``globally=True`` forces
    cross-project (global) scope regardless of ``scope``.
    """
    text = text.strip()
    if not text:
        print("ultan: empty memory text", file=sys.stderr)
        return 2

    if globally:
        resolved: str | None = None
    elif scope:
        resolved = scope
    else:
        resolved = project_slug(Path.cwd())

    ev = append_event(text, scope=resolved)
    scope_label = f"project:{resolved}" if resolved else "global"
    print(f"queued for librarian [{scope_label}] session={ev['session_id']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Queue a memory for the agent-mem Librarian."
    )
    ap.add_argument("text", nargs="*",
                    help="Memory text (use '-' to read from stdin).")
    ap.add_argument("--global", dest="globally", action="store_true",
                    help="Mark this memory as global (cross-project) scope.")
    ap.add_argument("--scope", help="Explicit project slug override.")
    args = ap.parse_args()

    if args.text == ["-"] or not args.text:
        if sys.stdin.isatty() and not args.text:
            ap.error("no text supplied (pass text as args, or pipe via '-')")
        text = sys.stdin.read()
    else:
        text = " ".join(args.text)

    return run(text, globally=args.globally, scope=args.scope)


if __name__ == "__main__":
    raise SystemExit(main())
