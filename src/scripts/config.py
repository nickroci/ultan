"""Path constants and configuration for agent-mem.

Storage is user-global (under ``~/.agent-mem/``), so lessons from project A
inform work in project B. The code itself still lives wherever the user
checked out the repo — only the data moves.

Two roots to keep straight:

- ``CODE_ROOT`` — the checkout (``…/agent-mem/src``). Used as ``cwd`` for the
  Claude Agent SDK and as the ``--directory`` argument to ``uv run``. The
  schema file (``AGENTS.md``) ships with the code, so it lives here. It is
  derived from this file's location, so it's a genuine module constant.
- The user-global data directory (``~/.agent-mem/``) and everything under
  it. This is *runtime state*, not a constant: it depends on the
  ``AGENT_MEM_HOME`` environment variable, which a test (or a late-binding
  caller) may set after this module is imported. So these paths are resolved
  on demand via :func:`get_config`, never frozen at import time.

Override the store location with the ``AGENT_MEM_HOME`` environment variable
(useful for tests and for users who don't want their HOME polluted). Because
:func:`get_config` reads the env var on every call, a test only has to
``monkeypatch.setenv("AGENT_MEM_HOME", ...)`` — there is no module-level
state to reset and no need to reload this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ── Code locations (env-independent; derived from this file's path) ────
# This file lives at <CODE_ROOT>/scripts/config.py.
CODE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = CODE_ROOT / "scripts"
HOOKS_DIR = CODE_ROOT / "hooks"
AGENTS_FILE = CODE_ROOT / "AGENTS.md"


@dataclass(frozen=True)
class StoreConfig:
    """Resolved, user-global storage paths.

    Built fresh from the environment by :func:`get_config`, so callers
    always see paths consistent with the *current* ``AGENT_MEM_HOME`` —
    there is no import-time freeze to work around.

    Per-PLAN.md §1, knowledge is split into a global tier and per-project
    subdirectories. The ``concepts``/``connections``/``qa`` dirs are the
    Phase-0 default write targets for compile.py; per-project routing is a
    Scholar concern (Phase 2).
    """

    store_dir: Path
    daily_dir: Path
    knowledge_dir: Path
    global_dir: Path
    projects_dir: Path
    concepts_dir: Path
    connections_dir: Path
    qa_dir: Path
    reports_dir: Path
    state_dir: Path
    index_file: Path
    log_file: Path
    state_file: Path

    def all_dirs(self) -> tuple[Path, ...]:
        """Every directory in the store tree, in creation-safe order."""
        return (
            self.store_dir,
            self.daily_dir,
            self.knowledge_dir,
            self.global_dir,
            self.concepts_dir,
            self.connections_dir,
            self.qa_dir,
            self.projects_dir,
            self.reports_dir,
            self.state_dir,
        )


def _resolve_store_dir() -> Path:
    """Resolve ``${AGENT_MEM_HOME:-~/.agent-mem}`` from the environment."""
    env_home = os.environ.get("AGENT_MEM_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return Path.home() / ".agent-mem"


def get_config() -> StoreConfig:
    """Resolve all storage paths from the current environment.

    Reads ``AGENT_MEM_HOME`` at call time (falling back to ``~/.agent-mem``),
    so the override is honoured regardless of when it was set. Cheap enough
    to call freely — no caching, so nothing to invalidate in tests.
    """
    store = _resolve_store_dir()
    knowledge = store / "knowledge"
    global_dir = knowledge / "global"
    return StoreConfig(
        store_dir=store,
        daily_dir=store / "daily",
        knowledge_dir=knowledge,
        global_dir=global_dir,
        projects_dir=knowledge / "projects",
        concepts_dir=global_dir / "concepts",
        connections_dir=global_dir / "connections",
        qa_dir=global_dir / "qa",
        reports_dir=store / "reports",
        state_dir=store / "state",
        index_file=knowledge / "index.md",
        log_file=knowledge / "log.md",
        state_file=store / "state" / "state.json",
    )


def ensure_store_dirs() -> None:
    """Create the user-global storage tree if it doesn't exist yet.

    Cheap and idempotent. Called by hooks and scripts before any write so a
    fresh install just works on the first SessionEnd.
    """
    for d in get_config().all_dirs():
        d.mkdir(parents=True, exist_ok=True)


# ── Timezone ───────────────────────────────────────────────────────────
TIMEZONE = "America/Chicago"


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
