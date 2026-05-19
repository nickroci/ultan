"""Path constants and configuration for agent-mem.

Storage is user-global (under ``~/.agent-mem/``), so lessons from project A
inform work in project B. The code itself still lives wherever the user
checked out the repo — only the data moves.

Two roots to keep straight:

- ``CODE_ROOT`` — the checkout (``…/agent-mem/src``). Used as ``cwd`` for the
  Claude Agent SDK and as the ``--directory`` argument to ``uv run``. The
  schema file (``AGENTS.md``) ships with the code, so it lives here.
- ``STORE_DIR`` — the user-global data directory (``~/.agent-mem/``). All
  daily logs, compiled knowledge, state, and logs live here.

Override the store location with the ``AGENT_MEM_HOME`` environment variable
(useful for tests and for users who don't want their HOME polluted).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# ── Roots ──────────────────────────────────────────────────────────────
# Code root: this file lives at <CODE_ROOT>/scripts/config.py
CODE_ROOT = Path(__file__).resolve().parent.parent

# Storage root: user-global, overridable for tests.
_env_home = os.environ.get("AGENT_MEM_HOME")
STORE_DIR = Path(_env_home).expanduser().resolve() if _env_home else Path.home() / ".agent-mem"

# Backwards-compat alias. A handful of upstream call sites reference
# ``ROOT_DIR``; keep the name working but point it at the store. Anything
# that needs the *code* root should import ``CODE_ROOT`` explicitly.
ROOT_DIR = STORE_DIR

# ── Storage layout (under STORE_DIR) ───────────────────────────────────
DAILY_DIR = STORE_DIR / "daily"
KNOWLEDGE_DIR = STORE_DIR / "knowledge"

# Per-PLAN.md §1, knowledge is split into a global tier and per-project
# subdirectories. Phase 0 only creates the global tier on first write;
# per-project dirs are materialised lazily by the daemon/Scholar later.
GLOBAL_DIR = KNOWLEDGE_DIR / "global"
PROJECTS_DIR = KNOWLEDGE_DIR / "projects"

# Phase-0 default write targets for the existing compile.py flow.
# These point at the global tier so the upstream end-of-day compile still
# has a place to put concept/connection articles. Per-project routing is a
# Scholar concern (Phase 2) — see TODO in compile.py.
CONCEPTS_DIR = GLOBAL_DIR / "concepts"
CONNECTIONS_DIR = GLOBAL_DIR / "connections"
QA_DIR = GLOBAL_DIR / "qa"

REPORTS_DIR = STORE_DIR / "reports"
STATE_DIR = STORE_DIR / "state"

INDEX_FILE = KNOWLEDGE_DIR / "index.md"
LOG_FILE = KNOWLEDGE_DIR / "log.md"
STATE_FILE = STATE_DIR / "state.json"

# ── Code locations (under CODE_ROOT) ───────────────────────────────────
SCRIPTS_DIR = CODE_ROOT / "scripts"
HOOKS_DIR = CODE_ROOT / "hooks"
AGENTS_FILE = CODE_ROOT / "AGENTS.md"


def ensure_store_dirs() -> None:
    """Create the user-global storage tree if it doesn't exist yet.

    Cheap and idempotent. Called by hooks and scripts before any write so a
    fresh install just works on the first SessionEnd.
    """
    for d in (
        STORE_DIR,
        DAILY_DIR,
        KNOWLEDGE_DIR,
        GLOBAL_DIR,
        CONCEPTS_DIR,
        CONNECTIONS_DIR,
        QA_DIR,
        PROJECTS_DIR,
        REPORTS_DIR,
        STATE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ── Timezone ───────────────────────────────────────────────────────────
TIMEZONE = "America/Chicago"


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
