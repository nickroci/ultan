"""Tests for the SessionStart hook (knowledge-base context injection)."""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _fresh(monkeypatch, home: Path):
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    for mod in ("config", "_events", "session_start"):
        if mod in sys.modules:
            del sys.modules[mod]
    return importlib.import_module("session_start")


def _drive(monkeypatch, mod, payload: dict, capsys) -> str:
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    mod.main()
    return capsys.readouterr().out


def test_session_start_injects_context(tmp_path: Path, monkeypatch, capsys):
    ss = _fresh(monkeypatch, tmp_path)
    # Seed an index.md so the hook has something to inject.
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir(parents=True, exist_ok=True)
    (knowledge / "index.md").write_text("# Index\n- entry-a", encoding="utf-8")
    out = _drive(monkeypatch, ss, {"session_id": "s1", "cwd": "/tmp", "source": "startup"}, capsys)
    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert "## Today" in ctx
    assert "## Current project" in ctx
    assert "## Knowledge Base Index" in ctx
    assert "entry-a" in ctx
    # Event fires too.
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["type"] == "SessionStart"
    assert rec["payload"]["source"] == "startup"


def test_session_start_recursion_guard(tmp_path: Path, monkeypatch, capsys):
    ss = _fresh(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    out = _drive(monkeypatch, ss, {"session_id": "s1"}, capsys)
    assert out == ""


def test_session_start_handles_empty_stdin(tmp_path: Path, monkeypatch, capsys):
    """No JSON in stdin → hook still injects context using os.getcwd()."""
    ss = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO(""))
    ss.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "additionalContext" in payload["hookSpecificOutput"]


def test_session_start_handles_malformed_stdin(tmp_path: Path, monkeypatch, capsys):
    """Bad JSON => hook_input falls back to empty dict, context still
    emitted (with unknown source)."""
    ss = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("not json"))
    ss.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "additionalContext" in payload["hookSpecificOutput"]


def test_session_start_handles_non_dict_stdin(tmp_path: Path, monkeypatch, capsys):
    ss = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("[1, 2]"))
    ss.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "additionalContext" in payload["hookSpecificOutput"]


def test_session_start_no_index_file_says_empty(tmp_path: Path, monkeypatch, capsys):
    ss = _fresh(monkeypatch, tmp_path)
    out = _drive(monkeypatch, ss, {"session_id": "s1", "cwd": "/tmp"}, capsys)
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "empty - no articles compiled" in ctx


def test_session_start_includes_recent_log_when_present(tmp_path: Path, monkeypatch, capsys):
    ss = _fresh(monkeypatch, tmp_path)
    daily = tmp_path / "daily"
    daily.mkdir()
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    (daily / f"{today}.md").write_text("today's log line", encoding="utf-8")
    out = _drive(monkeypatch, ss, {"session_id": "s1", "cwd": "/tmp"}, capsys)
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "today's log line" in ctx


def test_session_start_truncates_oversized_context(tmp_path: Path, monkeypatch, capsys):
    ss = _fresh(monkeypatch, tmp_path)
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    # 30k char index — should be cut.
    (knowledge / "index.md").write_text("x" * 30000, encoding="utf-8")
    out = _drive(monkeypatch, ss, {"session_id": "s1", "cwd": "/tmp"}, capsys)
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "...(truncated)" in ctx


def test_session_start_caps_log_lines(tmp_path: Path, monkeypatch, capsys):
    """get_recent_log returns at most MAX_LOG_LINES lines."""
    ss = _fresh(monkeypatch, tmp_path)
    daily = tmp_path / "daily"
    daily.mkdir()
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    big_log = "\n".join([f"line-{i}" for i in range(100)])
    (daily / f"{today}.md").write_text(big_log, encoding="utf-8")
    out = _drive(monkeypatch, ss, {"session_id": "s1", "cwd": "/tmp"}, capsys)
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    # The first lines (line-0 through line-69) should be trimmed.
    assert "line-0\n" not in ctx
    assert "line-99" in ctx


def test_session_start_subprocess_end_to_end(isolated_home: Path, hook_runner):
    res = hook_runner(
        "session-start.py",
        {"session_id": "s1", "cwd": "/tmp", "source": "startup"},
        env={"AGENT_MEM_HOME": str(isolated_home)},
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert "additionalContext" in payload["hookSpecificOutput"]


def test_session_start_session_bucket_failure_is_swallowed(tmp_path: Path, monkeypatch, capsys):
    """If session_bucket raises, the hook must still emit context."""
    ss = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(ss, "session_bucket", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    out = _drive(monkeypatch, ss, {"session_id": "s1", "cwd": "/tmp"}, capsys)
    payload = json.loads(out)
    assert "additionalContext" in payload["hookSpecificOutput"]
