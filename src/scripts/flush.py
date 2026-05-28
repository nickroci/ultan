"""
Memory flush agent — extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a ``.md`` file, uses the Claude Agent
SDK to decide what's worth saving, and appends the result to today's daily
log under ``~/.agent-mem/daily/``.

Usage:
    uv run python flush.py <context_file.md> <session_id> [project_slug]

The third argument is optional and defaults to ``"unknown"``. The hook is
expected to pass the slug it derived from the host agent's cwd; without
that, the flush has no way to know which project a session belonged to,
because by the time we're running in the background we've been detached.
"""

from __future__ import annotations

# Recursion prevention: set this BEFORE any imports that might trigger Claude.
# flush.py calls the Agent SDK, which runs Claude Code, which would fire the
# SessionEnd / PreCompact hook again. The hook checks this env var and bails.
import os

os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

# config.py lives next to this file in scripts/. Add scripts/ to sys.path so
# `import config` works whether we're invoked via `uv run python flush.py …`
# or directly.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from config import (  # noqa: E402
    CODE_ROOT,
    ensure_store_dirs,
    get_config,
)
from utils import State  # noqa: E402

# compile.py ships next to this script, so this is a genuine,
# env-independent constant. The store-relative paths (flush state, log
# files, compile state) are resolved on demand via get_config() instead,
# so importing this module doesn't touch the filesystem.
COMPILE_SCRIPT = _SCRIPTS_DIR / "compile.py"


def _setup_logging() -> None:
    """Ensure the store exists, then point logging at ``<store>/flush.log``.

    Called once at the top of :func:`main`. Deliberately kept out of import
    time so that merely importing this module (e.g. from a test) neither
    creates the store tree nor opens a log-file handle in someone's HOME.
    """
    ensure_store_dirs()
    logging.basicConfig(
        filename=str(get_config().store_dir / "flush.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class FlushState(TypedDict, total=False):
    """Persisted state for the most recent flush. Used only for dedup."""

    session_id: str
    timestamp: float


def load_flush_state() -> FlushState:
    flush_state_file = get_config().state_dir / "last-flush.json"
    if flush_state_file.exists():
        try:
            loaded: FlushState = json.loads(flush_state_file.read_text(encoding="utf-8"))
            return loaded
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_flush_state(state: FlushState) -> None:
    state_dir = get_config().state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "last-flush.json").write_text(json.dumps(state), encoding="utf-8")


def append_to_daily_log(content: str, section: str, project_slug: str) -> None:
    """Append content to today's daily log, tagged with the project slug.

    Daily logs are global (one per date) — the project tag lives inside the
    section heading so an end-of-day compile can route lessons to the right
    ``knowledge/projects/<slug>/`` later.
    """
    today = datetime.now(timezone.utc).astimezone()
    daily_dir = get_config().daily_dir
    daily_dir.mkdir(parents=True, exist_ok=True)
    log_path = daily_dir / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n"
            "## Sessions\n\n"
            "## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    entry = f"### {section} ({time_str}) — project:{project_slug}\n\n{content}\n\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


async def run_flush(context: str, project_slug: str) -> str:
    """Use Claude Agent SDK to extract important knowledge from conversation context."""
    prompt = f"""Review the conversation context below and respond with a concise summary
of important items that should be preserved in the daily log. This session came
from project **{project_slug}** — keep project-specific gotchas under "Lessons
Learned" so a later compile can route them correctly.

Do NOT use any tools — just return plain text.

Format your response as a structured daily log entry with these sections:

**Context:** [One line about what the user was working on]

**Key Exchanges:**
- [Important Q&A or discussions]

**Decisions Made:**
- [Any decisions with rationale]

**Lessons Learned:**
- [Gotchas, patterns, or insights discovered]

**Action Items:**
- [Follow-ups or TODOs mentioned]

Skip anything that is:
- Routine tool calls or file reads
- Content that's trivial or obvious
- Trivial back-and-forth or clarification exchanges

Only include sections that have actual content. If nothing is worth saving,
respond with exactly: FLUSH_OK

## Conversation Context

{context}"""

    response = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                # cwd is the code tree, not the store: the SDK process needs
                # access to AGENTS.md / scripts/, not the markdown corpus.
                cwd=str(CODE_ROOT),
                allowed_tools=[],
                max_turns=2,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
            elif isinstance(message, ResultMessage):
                pass
    except Exception as e:
        logging.error("Agent SDK error: %s\n%s", e, traceback.format_exc())
        response = f"FLUSH_ERROR: {type(e).__name__}: {e}"

    return response


COMPILE_AFTER_HOUR = 18  # 6 PM local time


def maybe_trigger_compilation() -> None:
    """If it's past the compile hour and today's log hasn't been compiled, run compile.py."""
    now = datetime.now(timezone.utc).astimezone()
    if now.hour < COMPILE_AFTER_HOUR:
        return

    cfg = get_config()
    today_log = f"{now.strftime('%Y-%m-%d')}.md"
    if cfg.state_file.exists():
        try:
            compile_state: State = json.loads(cfg.state_file.read_text(encoding="utf-8"))
            ingested = compile_state.get("ingested", {})
            if today_log in ingested:
                log_path = cfg.daily_dir / today_log
                if log_path.exists():
                    current_hash = hashlib.sha256(log_path.read_bytes()).hexdigest()[:16]
                    if ingested[today_log].get("hash") == current_hash:
                        return  # log unchanged since last compile
        except (json.JSONDecodeError, OSError):
            pass

    if not COMPILE_SCRIPT.exists():
        return

    logging.info("End-of-day compilation triggered (after %d:00)", COMPILE_AFTER_HOUR)

    cmd = ["uv", "run", "--directory", str(CODE_ROOT), "python", str(COMPILE_SCRIPT)]

    # Spawn detached so flush.py can exit without waiting on the compile.
    # ``creationflags`` is Windows-only, ``start_new_session`` is POSIX-only —
    # branching here keeps the kwargs concretely typed for each platform.
    try:
        log_handle = open(str(cfg.store_dir / "compile.log"), "a")
        if sys.platform == "win32":
            subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(CODE_ROOT),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            )
        else:
            subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(CODE_ROOT),
                start_new_session=True,
            )
    except Exception as e:
        logging.error("Failed to spawn compile.py: %s", e)


def main() -> None:
    _setup_logging()

    if len(sys.argv) < 3:
        logging.error("Usage: %s <context_file.md> <session_id> [project_slug]", sys.argv[0])
        sys.exit(1)

    context_file = Path(sys.argv[1])
    session_id = sys.argv[2]
    project_slug = sys.argv[3] if len(sys.argv) > 3 else "unknown"

    logging.info(
        "flush.py started for session %s (project=%s), context: %s",
        session_id,
        project_slug,
        context_file,
    )

    if not context_file.exists():
        logging.error("Context file not found: %s", context_file)
        return

    # Deduplication: skip if same session was flushed within 60 seconds
    state = load_flush_state()
    if state.get("session_id") == session_id and time.time() - state.get("timestamp", 0) < 60:
        logging.info("Skipping duplicate flush for session %s", session_id)
        context_file.unlink(missing_ok=True)
        return

    # Read pre-extracted context
    context = context_file.read_text(encoding="utf-8").strip()
    if not context:
        logging.info("Context file is empty, skipping")
        context_file.unlink(missing_ok=True)
        return

    logging.info(
        "Flushing session %s (project=%s): %d chars",
        session_id,
        project_slug,
        len(context),
    )

    # Run the LLM extraction
    response = asyncio.run(run_flush(context, project_slug))

    # Append to daily log
    if "FLUSH_OK" in response:
        logging.info("Result: FLUSH_OK")
        append_to_daily_log(
            "FLUSH_OK - Nothing worth saving from this session",
            "Memory Flush",
            project_slug,
        )
    elif "FLUSH_ERROR" in response:
        logging.error("Result: %s", response)
        append_to_daily_log(response, "Memory Flush", project_slug)
    else:
        logging.info("Result: saved to daily log (%d chars)", len(response))
        append_to_daily_log(response, "Session", project_slug)

    # Update dedup state
    save_flush_state({"session_id": session_id, "timestamp": time.time()})

    # Clean up context file
    context_file.unlink(missing_ok=True)

    # End-of-day auto-compilation: if it's past the compile hour and today's
    # log hasn't been compiled yet, trigger compile.py in the background.
    maybe_trigger_compilation()

    logging.info("Flush complete for session %s", session_id)


if __name__ == "__main__":
    main()
