"""Tests for the infra-level path guard in ``llm._make_path_guard``.

The guard is a ``can_use_tool`` callback registered with the SDK. It runs
before every tool call and returns PermissionResultAllow / Deny. We test
it directly (not via the real SDK) — the SDK contract is just the
return shape.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_mem_daemon.llm import _make_path_guard


def _call(guard, tool_name: str, tool_input: dict):
    """Invoke the async guard synchronously and return (behavior, message)."""
    result = asyncio.run(guard(tool_name, tool_input, None))
    return result.behavior, getattr(result, "message", "")


def test_guard_allows_read_inside_boundary(tmp_path: Path):
    (tmp_path / "x.md").write_text("hi", encoding="utf-8")
    guard = _make_path_guard(tmp_path, allow_writes=True)
    behavior, _ = _call(guard, "Read", {"file_path": str(tmp_path / "x.md")})
    assert behavior == "allow"


def test_guard_denies_read_outside_boundary(tmp_path: Path):
    other = tmp_path.parent / "elsewhere"
    other.mkdir(exist_ok=True)
    (other / "secret.md").write_text("nope", encoding="utf-8")
    guard = _make_path_guard(tmp_path, allow_writes=True)
    behavior, message = _call(guard, "Read", {"file_path": str(other / "secret.md")})
    assert behavior == "deny"
    assert "outside" in message


def test_guard_denies_absolute_home_path_when_boundary_is_test_dir(tmp_path: Path):
    # The exact scenario that bit us: cwd=/tmp/ulttest but LLM emits
    # /Users/.../.agent-mem/... — must be denied.
    guard = _make_path_guard(tmp_path, allow_writes=True)
    fake_real = Path("/Users/nicholasholden/.agent-mem/knowledge/index.md")
    behavior, message = _call(guard, "Read", {"file_path": str(fake_real)})
    assert behavior == "deny"
    assert str(tmp_path) in message


def test_guard_blocks_writes_when_allow_writes_false(tmp_path: Path):
    guard = _make_path_guard(tmp_path, allow_writes=False)
    behavior, message = _call(
        guard,
        "Write",
        {
            "file_path": str(tmp_path / "x.md"),
            "content": "hi",
        },
    )
    assert behavior == "deny"
    assert "not allowed" in message


def test_guard_allows_write_when_allow_writes_true(tmp_path: Path):
    guard = _make_path_guard(tmp_path, allow_writes=True)
    behavior, _ = _call(
        guard,
        "Write",
        {
            "file_path": str(tmp_path / "new.md"),
            "content": "hi",
        },
    )
    assert behavior == "allow"


def test_guard_denies_unknown_tools_by_default(tmp_path: Path):
    guard = _make_path_guard(tmp_path, allow_writes=True)
    behavior, message = _call(guard, "Bash", {"command": "ls /etc"})
    assert behavior == "deny"
    assert "not in the allow-list" in message


def test_guard_handles_relative_path(tmp_path: Path):
    # Relative paths resolve against the process cwd; the guard uses
    # resolve() which expands them. The point of this test is to ensure
    # path traversal attempts (e.g. ../../etc/passwd) are rejected even
    # if they happen to start as "knowledge/...".
    guard = _make_path_guard(tmp_path, allow_writes=True)
    traversal = str(tmp_path / ".." / ".." / "etc" / "passwd")
    behavior, _ = _call(guard, "Read", {"file_path": traversal})
    assert behavior == "deny"


def test_guard_allows_glob_without_path(tmp_path: Path):
    # Glob with no explicit ``path`` defaults to cwd — allowed.
    guard = _make_path_guard(tmp_path, allow_writes=True)
    behavior, _ = _call(guard, "Glob", {"pattern": "**/*.md"})
    assert behavior == "allow"


def test_guard_denies_edit_outside(tmp_path: Path):
    guard = _make_path_guard(tmp_path, allow_writes=True)
    behavior, _ = _call(
        guard,
        "Edit",
        {
            "file_path": "/etc/hosts",
            "old_string": "x",
            "new_string": "y",
        },
    )
    assert behavior == "deny"
