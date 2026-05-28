"""Tests for the PostToolUse hook — append a tool-call event to the
daemon's stream.
"""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _fresh_post(monkeypatch, home: Path):
    monkeypatch.setenv("AGENT_MEM_HOME", str(home))
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    return importlib.import_module("post_tool_use")


def _drive(monkeypatch, mod, payload: dict) -> None:
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(payload)))
    mod.main()


def test_post_emits_event_with_tool_name(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    _drive(
        monkeypatch,
        mod,
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "echo hi"},
            "tool_response": "hi\n",
        },
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["type"] == "PostToolUse"
    assert rec["payload"]["tool"] == "Bash"
    assert rec["payload"]["ok"] is True
    assert "echo hi" in rec["payload"]["content"]


def test_post_marks_error_payload_as_not_ok(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    _drive(
        monkeypatch,
        mod,
        {
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input": {"command": "false"},
            "error": "command failed",
        },
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["payload"]["ok"] is False
    assert "FAILED" in rec["payload"]["summary"]


def test_post_picks_file_path_target(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    _drive(
        monkeypatch,
        mod,
        {
            "session_id": "s1",
            "tool_name": "Edit",
            "tool_input": {"file_path": "/x/y.py", "new_string": "print()"},
            "tool_response": {"content": "ok"},
        },
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert "on /x/y.py" in rec["payload"]["summary"]
    assert rec["payload"]["content"] == "print()"
    assert rec["payload"]["response"] == "ok"


def test_post_handles_string_tool_response(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    _drive(
        monkeypatch,
        mod,
        {
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1\nfile2",
        },
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["payload"]["response"] == "file1\nfile2"


def test_post_handles_missing_tool_name(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    _drive(monkeypatch, mod, {"session_id": "s1", "tool_input": {}})
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert rec["payload"]["tool"] == "unknown"


def test_post_recursion_guard(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "scholar")
    _drive(
        monkeypatch,
        mod,
        {"session_id": "s1", "tool_name": "Bash", "tool_input": {"command": "x"}},
    )
    assert not (tmp_path / "events.jsonl").exists()


def test_post_handles_malformed_stdin(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO("nope"))
    mod.main()  # must not raise
    assert not (tmp_path / "events.jsonl").exists()


def test_post_handles_windows_backslash(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    raw = '{"session_id": "s1", "tool_name": "Edit", "tool_input": {"file_path": "C:\\x\\y"}}'
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    mod.main()
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert "C:" in rec["payload"]["summary"]


def test_post_rejects_non_dict_stdin(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", StringIO('"plain string"'))
    mod.main()
    assert not (tmp_path / "events.jsonl").exists()


def test_post_handles_url_target(tmp_path, monkeypatch):
    mod = _fresh_post(monkeypatch, tmp_path)
    _drive(
        monkeypatch,
        mod,
        {
            "session_id": "s1",
            "tool_name": "WebFetch",
            "tool_input": {"url": "https://x.com"},
        },
    )
    rec = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert "https://x.com" in rec["payload"]["summary"]


def test_post_subprocess_end_to_end(isolated_home: Path, hook_runner):
    res = hook_runner(
        "post-tool-use.py",
        {
            "session_id": "s1",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "ok",
        },
        env={"AGENT_MEM_HOME": str(isolated_home)},
    )
    assert res.returncode == 0, res.stderr
    assert (isolated_home / "events.jsonl").exists()
