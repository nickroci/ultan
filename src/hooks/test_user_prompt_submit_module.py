"""In-process tests for the ``user_prompt_submit`` module.

The existing ``test_user_prompt_submit.py`` exercises everything via
subprocess. Those tests stay as the contract for the actual entrypoint;
this file drives ``user_prompt_submit.main()`` directly so the branch
machinery is countable in coverage (subprocesses don't propagate
coverage data through stdin/stdout).
"""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from pathlib import Path


def _fresh(monkeypatch, home: Path):
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    for mod in ("config", "_events", "_nudges", "_priming_client", "user_prompt_submit"):
        if mod in sys.modules:
            del sys.modules[mod]
    return importlib.import_module("user_prompt_submit")


def _drive(monkeypatch, mod, payload, capsys) -> str:
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    mod.main()
    return capsys.readouterr().out


def test_module_emits_nothing_when_no_nudges_no_priming(tmp_path: Path, monkeypatch, capsys):
    """Empty prompt → no priming attempted, no nudges → no stdout."""
    ups = _fresh(monkeypatch, tmp_path)
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": ""}, capsys)
    assert out.strip() == ""


def test_module_recursion_guard(tmp_path: Path, monkeypatch, capsys):
    ups = _fresh(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    out = _drive(monkeypatch, ups, {"session_id": "s1", "prompt": "x"}, capsys)
    assert out == ""


def test_module_handles_malformed_stdin(tmp_path: Path, monkeypatch, capsys):
    ups = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("not json"))
    ups.main()
    assert capsys.readouterr().out == ""


def test_module_handles_non_dict_stdin(tmp_path: Path, monkeypatch, capsys):
    ups = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("[]"))
    ups.main()
    assert capsys.readouterr().out == ""


def test_module_handles_windows_backslash(tmp_path: Path, monkeypatch, capsys):
    """A stdin payload with a Windows-style backslash sequence parses
    via the regex fallback."""
    ups = _fresh(monkeypatch, tmp_path)
    raw = '{"session_id": "s1", "cwd": "C:\\Users\\x", "prompt": ""}'
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    ups.main()
    # No prompt content → no output. We just verify no exception.
    assert capsys.readouterr().out == ""


def test_module_emits_nudge_with_session_id(tmp_path: Path, monkeypatch, capsys):
    ups = _fresh(monkeypatch, tmp_path)
    (tmp_path / "pending-nudges.md").write_text(
        "---\nid: a1\ncreated: t\nlesson: global/foo\n---\nbody\n",
        encoding="utf-8",
    )
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": "x"}, capsys)
    payload = json.loads(out.strip())
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert "body" in ctx


def test_module_skips_nudges_when_session_id_missing(tmp_path: Path, monkeypatch, capsys):
    """No session_id → no nudge bookkeeping. File stays intact for a
    future session that has one."""
    ups = _fresh(monkeypatch, tmp_path)
    nudges = tmp_path / "pending-nudges.md"
    nudges.write_text(
        "---\nid: a1\ncreated: t\nlesson: global/foo\n---\nbody\n",
        encoding="utf-8",
    )
    _drive(monkeypatch, ups, {"cwd": "/tmp", "prompt": ""}, capsys)
    assert nudges.exists()


def test_module_priming_failure_is_swallowed(tmp_path: Path, monkeypatch, capsys):
    """get_priming should never raise, but if it did, the hook must
    still process nudges and exit cleanly."""
    ups = _fresh(monkeypatch, tmp_path)
    # Swap get_priming for one that raises.
    monkeypatch.setattr(
        ups,
        "get_priming",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": "x"}, capsys)
    # No nudges queued → no output.
    assert out.strip() == ""


def test_module_take_nudges_failure_is_swallowed(tmp_path: Path, monkeypatch, capsys):
    """A crashy nudge file must never crash the host agent."""
    ups = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ups,
        "take_nudges",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": ""}, capsys)
    assert out.strip() == ""


def test_module_current_project_slug_failure_is_swallowed(tmp_path: Path, monkeypatch, capsys):
    """If slug derivation blows up we treat the session as having no
    project context and deliver nudges permissively."""
    ups = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ups,
        "current_project_slug",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
    )
    (tmp_path / "pending-nudges.md").write_text(
        "---\nid: a1\ncreated: t\nlesson: global/foo\n---\nbody\n",
        encoding="utf-8",
    )
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": ""}, capsys)
    payload = json.loads(out.strip())
    assert "body" in payload["hookSpecificOutput"]["additionalContext"]


def test_module_combines_priming_and_nudges(tmp_path: Path, monkeypatch, capsys):
    """When both are present, priming comes first and nudges are
    fenced behind ``## Active nudges``."""
    ups = _fresh(monkeypatch, tmp_path)
    # Stub the priming and nudge calls so we don't depend on a daemon.
    monkeypatch.setattr(ups, "get_priming", lambda *a, **kw: "PRIMING_BLOCK\n")
    (tmp_path / "pending-nudges.md").write_text(
        "---\nid: a1\ncreated: t\nlesson: global/foo\n---\nnudge-body\n",
        encoding="utf-8",
    )
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": "x"}, capsys)
    ctx = json.loads(out.strip())["hookSpecificOutput"]["additionalContext"]
    assert ctx.index("PRIMING_BLOCK") < ctx.index("## Active nudges")
    assert "nudge-body" in ctx


def test_module_priming_only(tmp_path: Path, monkeypatch, capsys):
    ups = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(ups, "get_priming", lambda *a, **kw: "PRIMING_BLOCK")
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": "x"}, capsys)
    ctx = json.loads(out.strip())["hookSpecificOutput"]["additionalContext"]
    assert ctx == "PRIMING_BLOCK"


def test_module_non_string_prompt_is_treated_as_empty(tmp_path: Path, monkeypatch, capsys):
    ups = _fresh(monkeypatch, tmp_path)
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": "/tmp", "prompt": 42}, capsys)
    assert out.strip() == ""


def test_module_non_string_cwd_is_normalised(tmp_path: Path, monkeypatch, capsys):
    """A non-string cwd shouldn't crash slug derivation."""
    ups = _fresh(monkeypatch, tmp_path)
    (tmp_path / "pending-nudges.md").write_text(
        "---\nid: a1\ncreated: t\nlesson: global/foo\n---\nbody\n",
        encoding="utf-8",
    )
    out = _drive(monkeypatch, ups, {"session_id": "s1", "cwd": 42, "prompt": ""}, capsys)
    payload = json.loads(out.strip())
    assert "body" in payload["hookSpecificOutput"]["additionalContext"]


def test_emit_additional_context_is_no_op_for_empty(tmp_path: Path, monkeypatch, capsys):
    ups = _fresh(monkeypatch, tmp_path)
    ups._emit_additional_context("")
    assert capsys.readouterr().out == ""
