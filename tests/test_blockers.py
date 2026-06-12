"""Tests for the ported PreToolUse blocker check (``ultan/_blockers.py``).

This is the deterministic Tier-3 interrupt: a knowledge entry that lists
``block_triggers`` in its frontmatter can either deny a matching tool call
(``severity: block``) or attach an FYI note (``severity: advise``, the
default). The behaviour is ported verbatim from the legacy
``src/hooks/_blockers.py`` + ``src/hooks/pre_tool_use.py``; these tests pin
that parity.

Hermetic: every test points ``AGENT_MEM_HOME`` at a tmp dir, clears the
``CLAUDE_INVOKED_BY`` recursion guard, and drops the process-local blocker
cache so a stale scan from a previous test can't leak across.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ultan import _blockers


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    _blockers.clear_cache()
    return tmp_path


def _knowledge(home: Path) -> Path:
    """Create and return ``$HOME/knowledge``."""
    kdir = home / "knowledge"
    kdir.mkdir(parents=True, exist_ok=True)
    return kdir


def _write_entry(kdir: Path, name: str, text: str) -> Path:
    path = kdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# Default knowledge dir resolves to ``$AGENT_MEM_HOME/knowledge`` (the daemon
# path), so ``evaluate`` with no ``kdir`` arg sees the tmp store.


def test_block_match_on_bash_pattern(tmp_path: Path) -> None:
    kdir = _knowledge(tmp_path)
    _write_entry(
        kdir,
        "danger/rm-rf.md",
        "---\n"
        "title: Never rm -rf the root\n"
        "severity: block\n"
        "block_triggers:\n"
        "  - tool: Bash\n"
        "    pattern: 'rm -rf /'\n"
        "---\n"
        "# Never rm -rf /\n"
        "Never run rm -rf on the filesystem root\n",
    )

    decision = _blockers.evaluate("Bash", {"command": "rm -rf / --no-preserve-root"})

    assert decision is not None
    assert decision.severity == "block"
    assert decision.wiki == "danger/rm-rf"
    assert decision.hook_output["permissionDecision"] == "deny"
    reason = decision.hook_output["permissionDecisionReason"]
    assert "[[danger/rm-rf]]" in reason
    assert "Never run rm -rf on the filesystem root" in reason
    assert "Confirm with the user before retrying." in reason


def test_advise_match_on_edit_file_pattern(tmp_path: Path) -> None:
    kdir = _knowledge(tmp_path)
    _write_entry(
        kdir,
        "secrets-care.md",
        "---\n"
        "title: Touching env files\n"
        "severity: advise\n"
        "block_triggers:\n"
        "  - tool: Edit\n"
        "    file_pattern: '\\.env$'\n"
        "---\n"
        "Editing a .env file: double-check you are not committing a secret\n",
    )

    decision = _blockers.evaluate("Edit", {"file_path": "/repo/.env"})

    assert decision is not None
    assert decision.severity == "advise"
    assert decision.wiki == "secrets-care"
    # Advisory never denies — it only attaches additionalContext.
    assert "permissionDecision" not in decision.hook_output
    ctx = decision.hook_output["additionalContext"]
    assert "[[secrets-care]]" in ctx
    assert "double-check you are not committing a secret" in ctx


def test_advise_is_the_default_severity(tmp_path: Path) -> None:
    """An entry with triggers but no ``severity`` defaults to advise (not block)."""
    kdir = _knowledge(tmp_path)
    _write_entry(
        kdir,
        "no-severity.md",
        "---\n"
        "title: No severity declared\n"
        "block_triggers:\n"
        "  - tool: Bash\n"
        "    pattern: 'curl'\n"
        "---\n"
        "Prefer the project HTTP client over raw curl\n",
    )

    decision = _blockers.evaluate("Bash", {"command": "curl https://example.com"})

    assert decision is not None
    assert decision.severity == "advise"
    assert "additionalContext" in decision.hook_output


def test_no_match_passthrough(tmp_path: Path) -> None:
    kdir = _knowledge(tmp_path)
    _write_entry(
        kdir,
        "danger.md",
        "---\n"
        "severity: block\n"
        "block_triggers:\n"
        "  - tool: Bash\n"
        "    pattern: 'rm -rf /'\n"
        "---\n"
        "Never rm -rf root\n",
    )

    # Command does not match the pattern → no decision, tool proceeds.
    assert _blockers.evaluate("Bash", {"command": "ls -la"}) is None


def test_tool_name_mismatch_does_not_fire(tmp_path: Path) -> None:
    """A Bash trigger must not fire on an Edit call even if the regex would
    match the file path, and vice versa — ``trigger.tool`` is exact-match."""
    kdir = _knowledge(tmp_path)
    _write_entry(
        kdir,
        "bash-only.md",
        "---\n"
        "severity: block\n"
        "block_triggers:\n"
        "  - tool: Bash\n"
        "    pattern: 'secret'\n"
        "---\n"
        "Bash rule only\n",
    )

    # Edit call whose file_path contains 'secret' must NOT trip a Bash trigger.
    assert _blockers.evaluate("Edit", {"file_path": "/repo/secret.txt"}) is None
    # And the Bash trigger still fires for an actual Bash call.
    assert _blockers.evaluate("Bash", {"command": "echo secret"}) is not None


def test_entry_without_block_triggers_is_ignored(tmp_path: Path) -> None:
    """Opt-in is the PRESENCE of ``block_triggers``. A plain entry — even one
    that mentions a tool/severity in prose — is never a blocker."""
    kdir = _knowledge(tmp_path)
    _write_entry(
        kdir,
        "plain.md",
        "---\n"
        "title: Just a normal note\n"
        "severity: block\n"  # severity alone, with no triggers, is inert
        "---\n"
        "This note talks about Bash and rm -rf but lists no block_triggers\n",
    )

    assert _blockers.load_blockers(kdir) == []
    assert _blockers.evaluate("Bash", {"command": "rm -rf /"}) is None


def test_claude_invoked_by_recursion_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside a daemon-spawned Agent-SDK subprocess (CLAUDE_INVOKED_BY set),
    ``evaluate`` must short-circuit to None so the Scholar's own Edit/Write
    calls are never blocked — that would deadlock the daemon."""
    kdir = _knowledge(tmp_path)
    _write_entry(
        kdir,
        "danger.md",
        "---\n"
        "severity: block\n"
        "block_triggers:\n"
        "  - tool: Bash\n"
        "    pattern: 'rm -rf /'\n"
        "---\n"
        "Never rm -rf root\n",
    )

    # Sanity: without the guard the rule fires.
    assert _blockers.evaluate("Bash", {"command": "rm -rf /"}) is not None

    monkeypatch.setenv("CLAUDE_INVOKED_BY", "agent_mem_daemon")
    assert _blockers.evaluate("Bash", {"command": "rm -rf /"}) is None


def test_first_match_wins_and_archive_skipped(tmp_path: Path) -> None:
    """``find_match`` returns the first matching blocker, and ``_archive`` /
    admin files are excluded from the scan."""
    kdir = _knowledge(tmp_path)
    # An archived entry that would otherwise match must be skipped entirely.
    _write_entry(
        kdir,
        "_archive/old-rule.md",
        "---\n"
        "severity: block\n"
        "block_triggers:\n"
        "  - tool: Bash\n"
        "    pattern: 'deploy'\n"
        "---\n"
        "Archived deploy rule\n",
    )
    _write_entry(
        kdir,
        "active.md",
        "---\n"
        "title: Deploy guard\n"
        "severity: advise\n"
        "block_triggers:\n"
        "  - tool: Bash\n"
        "    pattern: 'deploy'\n"
        "---\n"
        "Confirm the target environment before deploying\n",
    )

    blockers = _blockers.load_blockers(kdir)
    assert [b.entry_path.name for b in blockers] == ["active.md"]

    decision = _blockers.evaluate("Bash", {"command": "make deploy"})
    assert decision is not None
    assert decision.wiki == "active"
    assert decision.severity == "advise"
