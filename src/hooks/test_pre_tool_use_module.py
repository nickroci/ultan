"""In-process tests for the ``pre_tool_use`` module.

Existing ``test_pre_tool_use.py`` exercises the hyphenated script
through subprocess for the entrypoint contract; this file drives
``pre_tool_use.main()`` directly so the branch machinery is countable
in coverage.
"""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from pathlib import Path
from textwrap import dedent

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _fresh(monkeypatch, home: Path):
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    return importlib.import_module("pre_tool_use")


def _drive(monkeypatch, mod, payload, capsys) -> str:
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    mod.main()
    return capsys.readouterr().out


def _seed(home: Path) -> Path:
    knowledge = home / "knowledge"
    (knowledge / "global" / "concepts").mkdir(parents=True, exist_ok=True)
    (knowledge / "log.md").write_text("# Build Log\n", encoding="utf-8")
    return knowledge


def _entry(knowledge: Path, rel: str, body: str) -> Path:
    path = knowledge / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")
    log = knowledge / "log.md"
    log.write_text(log.read_text() + ".", encoding="utf-8")
    import _blockers as bm  # use the freshly-loaded module

    bm.clear_cache()
    return path


def test_pre_no_blockers_allows_and_emits_event(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    _seed(tmp_path)
    out = _drive(
        monkeypatch,
        pre,
        {"session_id": "s1", "cwd": "/tmp", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        capsys,
    )
    assert "permissionDecision" not in out
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["type"] == "PreToolUse"
    assert rec["payload"]["blocked"] is False


def test_pre_denies_on_block_severity(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    knowledge = _seed(tmp_path)
    _entry(
        knowledge,
        "global/concepts/no-rm.md",
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
    out = _drive(
        monkeypatch,
        pre,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /tmp/foo"},
        },
        capsys,
    )
    body = json.loads(out)
    hso = body["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "Never recursively force-delete" in hso["permissionDecisionReason"]
    # Event captured the deny.
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["payload"]["blocked"] is True
    assert rec["payload"]["severity"] == "block"


def test_pre_advises_on_default_severity(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    knowledge = _seed(tmp_path)
    _entry(
        knowledge,
        "global/concepts/prefer-uv.md",
        """
        ---
        title: "Prefer uv"
        block_triggers:
          - tool: Bash
            pattern: 'pip install'
        ---

        Use uv instead of pip.
        """,
    )
    out = _drive(
        monkeypatch,
        pre,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "pip install x"},
        },
        capsys,
    )
    body = json.loads(out)
    hso = body["hookSpecificOutput"]
    assert "permissionDecision" not in hso
    assert "additionalContext" in hso
    assert "Use uv" in hso["additionalContext"]


def test_pre_recursion_guard(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    _drive(monkeypatch, pre, {"session_id": "s1", "tool_name": "Bash"}, capsys)
    assert not (tmp_path / "events.jsonl").exists()
    assert capsys.readouterr().out == ""


def test_pre_handles_malformed_stdin(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("not json"))
    pre.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_pre_handles_non_dict_stdin(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("[]"))
    pre.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_pre_handles_non_dict_tool_input(tmp_path: Path, monkeypatch, capsys):
    """A tool_input that arrives as a non-dict should be coerced to ``{}``
    rather than crash the matcher."""
    pre = _fresh(monkeypatch, tmp_path)
    _seed(tmp_path)
    _drive(
        monkeypatch,
        pre,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": "weird string",
        },
        capsys,
    )
    # No crash, event was appended.
    assert (tmp_path / "events.jsonl").exists()


def test_pre_swallows_load_blockers_exception(tmp_path: Path, monkeypatch, capsys):
    """A broken library scan must NEVER crash the host agent."""
    pre = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pre,
        "load_blockers",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _drive(
        monkeypatch,
        pre,
        {"session_id": "s1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        capsys,
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["payload"]["blocked"] is False


def test_pre_handles_windows_backslash(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    _seed(tmp_path)
    raw = (
        '{"session_id": "s1", "cwd": "C:\\Users\\x", '
        '"tool_name": "Edit", "tool_input": {"file_path": "C:\\dev\\x.py"}}'
    )
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    pre.main()
    assert (tmp_path / "events.jsonl").exists()


def test_pre_rel_to_knowledge_falls_back_on_exception(tmp_path: Path, monkeypatch, capsys):
    """If wikilink resolution raises, fall back to the entry stem so we
    still emit a deny reason."""
    pre = _fresh(monkeypatch, tmp_path)
    knowledge = _seed(tmp_path)
    _entry(
        knowledge,
        "global/concepts/no-rm.md",
        """
        ---
        severity: block
        title: "x"
        block_triggers:
          - tool: Bash
            pattern: 'rm -rf'
        ---

        Body.
        """,
    )
    monkeypatch.setattr(
        pre,
        "rel_to_knowledge",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
    )
    out = _drive(
        monkeypatch,
        pre,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
        },
        capsys,
    )
    body = json.loads(out)
    hso = body["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    # Fallback name is the entry stem.
    assert "no-rm" in hso["permissionDecisionReason"]


def test_pre_advise_with_rel_to_knowledge_exception(tmp_path: Path, monkeypatch, capsys):
    pre = _fresh(monkeypatch, tmp_path)
    knowledge = _seed(tmp_path)
    _entry(
        knowledge,
        "global/concepts/prefer-uv.md",
        """
        ---
        title: "x"
        block_triggers:
          - tool: Bash
            pattern: 'pip install'
        ---

        Use uv.
        """,
    )
    monkeypatch.setattr(
        pre,
        "rel_to_knowledge",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
    )
    out = _drive(
        monkeypatch,
        pre,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "pip install x"},
        },
        capsys,
    )
    body = json.loads(out)
    assert "additionalContext" in body["hookSpecificOutput"]


def test_pre_appends_event_with_inline_rule_text_missing(tmp_path: Path, monkeypatch, capsys):
    """When the entry body has no usable rule text, the deny reason
    falls back to ``(no inline rule text)``."""
    pre = _fresh(monkeypatch, tmp_path)
    knowledge = _seed(tmp_path)
    # Only headings in the body → no extractable rule.
    _entry(
        knowledge,
        "global/concepts/empty-body.md",
        """
        ---
        severity: block
        title: "x"
        block_triggers:
          - tool: Bash
            pattern: 'foo'
        ---

        # heading only
        """,
    )
    out = _drive(
        monkeypatch,
        pre,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "foo"},
        },
        capsys,
    )
    body = json.loads(out)
    assert "(no inline rule text)" in body["hookSpecificOutput"]["permissionDecisionReason"]


def test_pre_knowledge_dir_uses_default_when_unset(monkeypatch):
    """When AGENT_MEM_HOME is unset, the knowledge dir pre_tool_use reads
    (now centralised in config.get_config) resolves to
    ``~/.agent-mem/knowledge``."""
    monkeypatch.delenv("AGENT_MEM_HOME", raising=False)
    from config import get_config

    assert get_config().knowledge_dir == Path.home() / ".agent-mem" / "knowledge"
