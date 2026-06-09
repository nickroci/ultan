"""Tests for the Tier 3 synchronous PreToolUse blocker.

Mix of pure-Python tests for ``_blockers`` and end-to-end subprocess
tests for the hook script itself (so the stdin/stdout protocol path is
exercised verbatim — that contract is the whole reason the hook exists).

Run from the ``src/`` directory with::

    uv run python -m pytest hooks/ -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from unittest import mock

import _blockers

_THIS_DIR = Path(__file__).resolve().parent
HOOK_SCRIPT = _THIS_DIR / "pre-tool-use.py"


# ── Helpers ────────────────────────────────────────────────────────────


def _seed_library(home: Path) -> Path:
    """Create a minimal ~/.agent-mem layout under ``home`` and return
    the knowledge dir. log.md is touched so the cache sentinel resolves
    cleanly.
    """
    knowledge = home / "knowledge"
    (knowledge / "global" / "concepts").mkdir(parents=True, exist_ok=True)
    (knowledge / "log.md").write_text("# Build Log\n", encoding="utf-8")
    return knowledge


def _write_entry(knowledge: Path, rel: str, body: str) -> Path:
    """Write an entry under knowledge/ and bump log.md to invalidate
    the blocker cache.
    """
    path = knowledge / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")
    # Touch log.md so the cache sentinel mtime advances.
    log_path = knowledge / "log.md"
    log_path.write_text(log_path.read_text() + ".", encoding="utf-8")
    _blockers.clear_cache()
    return path


def _run_hook(stdin_payload: dict, env_overrides: dict) -> subprocess.CompletedProcess:
    env = {}
    for k in ("HOME", "PYTHONPATH", "VIRTUAL_ENV", "PATH", "PYTHONHOME"):
        if k in os.environ:
            env[k] = os.environ[k]
    env.update(env_overrides)
    # Belt-and-braces: ensure recursion guard is OFF unless the test
    # explicitly opts in.
    if "CLAUDE_INVOKED_BY" not in env_overrides:
        env.pop("CLAUDE_INVOKED_BY", None)
    return subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


# ── Pure-Python tests of _blockers ─────────────────────────────────────


def test_no_blockers_in_library_allows_all_tools(tmp_path: Path):
    """Empty library → load_blockers returns [], find_match returns None."""
    knowledge = _seed_library(tmp_path)
    blockers = _blockers.load_blockers(knowledge)
    assert blockers == []
    assert _blockers.find_match(blockers, "Bash", {"command": "rm -rf /"}) is None


def test_bash_command_matches_regex_blocks(tmp_path: Path):
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/no-rm-rf.md",
        """
        ---
        id: no-rm-rf
        severity: block
        title: "Never rm -rf"
        block_triggers:
          - tool: Bash
            pattern: 'rm -rf'
        ---

        Never recursively force-delete; ask the user first.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert len(blockers) == 1
    match = _blockers.find_match(blockers, "Bash", {"command": "rm -rf /tmp/foo"})
    assert match is not None
    assert match.title == "Never rm -rf"
    assert "ask the user" in match.one_line_rule


def test_bash_command_no_match_allows(tmp_path: Path):
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/no-rm-rf.md",
        """
        ---
        severity: block
        title: "Never rm -rf"
        block_triggers:
          - tool: Bash
            pattern: 'rm -rf'
        ---

        Never recursively force-delete.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert _blockers.find_match(blockers, "Bash", {"command": "ls -la"}) is None


def test_edit_file_pattern_blocks(tmp_path: Path):
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/no-prod-env.md",
        r"""
        ---
        severity: block
        title: "Don't edit production.env"
        block_triggers:
          - tool: Edit
            file_pattern: 'production\.env$'
        ---

        Production secrets live elsewhere; do not touch this file.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    match = _blockers.find_match(blockers, "Edit", {"file_path": "/repo/prod/production.env"})
    assert match is not None
    # Non-matching file_path → allow.
    assert _blockers.find_match(blockers, "Edit", {"file_path": "/repo/dev.env"}) is None


def test_other_tool_not_in_triggers_allows(tmp_path: Path):
    """A Bash blocker must not fire on an Edit call."""
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/bash-only.md",
        """
        ---
        severity: block
        title: "Bash only"
        block_triggers:
          - tool: Bash
            pattern: 'sudo'
        ---

        No sudo.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert _blockers.find_match(blockers, "Edit", {"file_path": "/etc/sudoers"}) is None
    assert _blockers.find_match(blockers, "Bash", {"command": "sudo rm /"}) is not None


def test_entries_with_block_triggers_load_as_advise_by_default(tmp_path: Path):
    """Opt-in to PreToolUse checking is by ``block_triggers`` presence.
    Severity defaults to "advise" (FYI, not hard block)."""
    knowledge = _seed_library(tmp_path)
    # Default severity (absent) → advise.
    _write_entry(
        knowledge,
        "global/concepts/regular-lesson.md",
        """
        ---
        title: "Just a normal lesson"
        block_triggers:
          - tool: Bash
            pattern: 'rm -rf'
        ---

        Body.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert len(blockers) == 1
    assert blockers[0].severity == "advise"


def test_entry_without_block_triggers_not_loaded(tmp_path: Path):
    """An entry with no ``block_triggers`` list is not a Blocker at all,
    regardless of severity. Opt-in is by triggers, not severity."""
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/plain.md",
        """
        ---
        title: "Plain lesson"
        ---

        Body.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert blockers == []


def test_severity_block_loads_as_hard_block(tmp_path: Path):
    """``severity: block`` is opt-in for the rare hard-stop case."""
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/dangerous.md",
        """
        ---
        severity: block
        title: "Dangerous"
        block_triggers:
          - tool: Bash
            pattern: 'rm -rf /'
        ---

        Body.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert len(blockers) == 1
    assert blockers[0].severity == "block"


def test_archive_entries_skipped(tmp_path: Path):
    """Archived blockers must not fire interrupts (§1.2 archive policy)."""
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "_archive/concepts/old-blocker.md",
        """
        ---
        severity: block
        title: "Old blocker"
        block_triggers:
          - tool: Bash
            pattern: 'foo'
        ---

        Body.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert blockers == []


def test_blocker_with_invalid_regex_dropped(tmp_path: Path):
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/bad-regex.md",
        """
        ---
        severity: block
        title: "Bad regex"
        block_triggers:
          - tool: Bash
            pattern: '['
        ---

        Body.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    # No usable triggers → blocker is dropped.
    assert blockers == []


def test_cache_hits_skip_rescan(tmp_path: Path):
    """Second call with unchanged sentinel returns cached list without
    re-walking the filesystem.
    """
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/blocker.md",
        """
        ---
        severity: block
        title: "Block"
        block_triggers:
          - tool: Bash
            pattern: 'foo'
        ---

        Body.
        """,
    )
    first = _blockers.load_blockers(knowledge)
    assert len(first) == 1

    # Now patch _scan_blockers and ensure the cached path doesn't call it.
    with mock.patch.object(_blockers, "_scan_blockers") as scan:
        second = _blockers.load_blockers(knowledge)
        scan.assert_not_called()
    # Same object identity from the cache.
    assert second is first


def test_cache_invalidates_on_log_mtime_bump(tmp_path: Path):
    """Touching log.md must invalidate the cache."""
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/first.md",
        """
        ---
        severity: block
        title: "First"
        block_triggers:
          - tool: Bash
            pattern: 'aaa'
        ---

        Body.
        """,
    )
    assert len(_blockers.load_blockers(knowledge)) == 1

    # Add a second blocker; _write_entry bumps log.md mtime.
    _write_entry(
        knowledge,
        "global/concepts/second.md",
        """
        ---
        severity: block
        title: "Second"
        block_triggers:
          - tool: Bash
            pattern: 'bbb'
        ---

        Body.
        """,
    )
    refreshed = _blockers.load_blockers(knowledge)
    assert len(refreshed) == 2


def test_multiple_triggers_per_entry(tmp_path: Path):
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/multi.md",
        r"""
        ---
        severity: block
        title: "Multi"
        block_triggers:
          - tool: Bash
            pattern: 'gcloud.*deploy.*prod'
          - tool: Edit
            file_pattern: 'production\.env$'
        ---

        Production gates.
        """,
    )
    blockers = _blockers.load_blockers(knowledge)
    assert len(blockers) == 1
    assert len(blockers[0].triggers) == 2
    assert (
        _blockers.find_match(blockers, "Bash", {"command": "gcloud foo deploy bar prod"})
        is not None
    )
    assert _blockers.find_match(blockers, "Edit", {"file_path": "x/production.env"}) is not None
    assert _blockers.find_match(blockers, "Write", {"file_path": "x/production.env"}) is None


# ── End-to-end hook subprocess tests ───────────────────────────────────


def test_hook_allows_when_no_blockers(tmp_path: Path):
    """Fresh library, no blockers → hook emits no deny JSON."""
    _seed_library(tmp_path)
    res = _run_hook(
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
        },
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    assert "permissionDecision" not in res.stdout


def test_hook_denies_matching_bash_call(tmp_path: Path):
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/no-prod-deploy.md",
        """
        ---
        id: no-prod-deploy
        severity: block
        title: "Never deploy to prod without approval"
        block_triggers:
          - tool: Bash
            pattern: 'gcloud.*deploy.*prod'
        ---

        Confirm deploy targets verbally before running gcloud against prod.
        """,
    )
    res = _run_hook(
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "gcloud run deploy svc --project=prod-foo"},
        },
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout, "hook must emit a deny JSON when a blocker matches"
    output = json.loads(res.stdout)
    hso = output["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    reason = hso["permissionDecisionReason"]
    assert "no-prod-deploy" in reason
    assert "Confirm deploy targets" in reason


def test_hook_emits_advisory_on_default_severity_match(tmp_path: Path):
    """An entry with ``block_triggers`` but no explicit severity defaults
    to ``advise`` — the hook emits ``additionalContext`` (FYI) instead of
    blocking. The tool proceeds; the agent decides whether to take notice.
    """
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/prefer-uv.md",
        """
        ---
        id: prefer-uv
        title: "Prefer uv over pip"
        block_triggers:
          - tool: Bash
            pattern: 'pip install'
        ---

        Use uv instead of pip for python deps in this repo.
        """,
    )
    res = _run_hook(
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "pip install requests"},
        },
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout, "hook must emit JSON with additionalContext for advise-tier match"
    output = json.loads(res.stdout)
    hso = output["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    # CRITICAL: advise must NOT deny — the user wants notice, not freeze.
    assert "permissionDecision" not in hso
    assert "additionalContext" in hso
    notice = hso["additionalContext"]
    assert "prefer-uv" in notice
    assert "FYI" in notice or "agent decides" in notice
    assert "Use uv instead of pip" in notice


def test_hook_recursion_guard_skips_check_when_env_set(tmp_path: Path):
    """With CLAUDE_INVOKED_BY set, the hook must exit immediately and
    emit nothing — even if a matching blocker exists.
    """
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/no-rm.md",
        """
        ---
        severity: block
        title: "No rm -rf"
        block_triggers:
          - tool: Bash
            pattern: 'rm -rf'
        ---

        Don't.
        """,
    )
    res = _run_hook(
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /tmp/foo"},
        },
        env_overrides={
            "AGENT_MEM_HOME": str(tmp_path),
            "CLAUDE_INVOKED_BY": "scholar",
        },
    )
    assert res.returncode == 0
    assert res.stdout == ""
    # Event stream must also be silent — no events.jsonl write.
    assert not (tmp_path / "events.jsonl").exists()


def test_hook_appends_event_on_deny(tmp_path: Path):
    """Even when denying, the event-stream side fires so the daemon can
    see the blocked call. PostToolUse won't fire for a denied call, so
    this is the only record.
    """
    knowledge = _seed_library(tmp_path)
    _write_entry(
        knowledge,
        "global/concepts/no-force-push.md",
        """
        ---
        severity: block
        title: "No force-push to main"
        block_triggers:
          - tool: Bash
            pattern: 'git push.*--force.*main'
        ---

        Force-push to main destroys history.
        """,
    )
    res = _run_hook(
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "git push --force origin main"},
        },
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    output = json.loads(res.stdout)
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    # Event line was appended.
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "PreToolUse"
    assert event["payload"]["blocked"] is True
    assert "no-force-push" in event["payload"]["blocker_entry"]


def test_hook_appends_event_on_allow(tmp_path: Path):
    """Allowed calls also get an event so the daemon sees every call."""
    _seed_library(tmp_path)
    res = _run_hook(
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "echo hi"},
        },
        env_overrides={"AGENT_MEM_HOME": str(tmp_path)},
    )
    assert res.returncode == 0, res.stderr
    assert "permissionDecision" not in res.stdout
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    event = json.loads(events_path.read_text(encoding="utf-8").strip())
    assert event["type"] == "PreToolUse"
    assert event["payload"]["blocked"] is False
