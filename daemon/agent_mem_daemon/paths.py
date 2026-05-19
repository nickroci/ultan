"""Resolve all on-disk paths under ``~/.agent-mem/``.

Centralised so tests can override the root via ``AGENT_MEM_HOME`` and so
the rest of the daemon never hardcodes a path. The hook author (other
agent) reads the same env var, so the contract stays in one file.
"""
from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """Root directory for all agent-mem state.

    Honours ``AGENT_MEM_HOME`` if set (used by tests and by users who
    want a non-default location). Otherwise defaults to
    ``~/.agent-mem``.
    """
    override = os.environ.get("AGENT_MEM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".agent-mem"


def events_path() -> Path:
    """The JSONL event log the hooks append to and the daemon tails.

    This is the contract between the hooks (other agent's scope) and
    the daemon. Both sides must agree on this path. See README §"Event
    schema" for the line format.
    """
    return home() / "events.jsonl"


def pid_path() -> Path:
    return home() / "daemon.pid"


def offset_state_path() -> Path:
    """Where the JSONL tailer persists its last-read offset across daemon
    restarts. See ``ingest._load_offset_state`` / ``_save_offset_state``.
    """
    return home() / "daemon.offset.json"


def log_path() -> Path:
    return home() / "daemon.log"


def pending_nudges_path() -> Path:
    """Where the (future) Scholar will write ratified interrupts.

    Defined here for completeness even though the skeleton doesn't write
    to it yet — the hook author needs to know where to read from.
    """
    return home() / "pending-nudges.md"


def knowledge_dir() -> Path:
    """Root of the knowledge store (``~/.agent-mem/knowledge/``).

    The Librarian and Scholar both read from this tree. The Scholar
    writes here. The BM25 index in ``~/.agent-mem/.bm25.idx`` is keyed
    on this directory's tree (see ``bm25.load_or_build``).
    """
    return home() / "knowledge"


def index_md_path() -> Path:
    """The master catalog the Scholar maintains and the Librarian reads."""
    return knowledge_dir() / "index.md"


def ensure_home() -> Path:
    """Create the home dir if missing; return it."""
    h = home()
    h.mkdir(parents=True, exist_ok=True)
    return h
