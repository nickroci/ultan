"""Tests for path resolution under ``~/.agent-mem``.

Every public function should honour ``AGENT_MEM_HOME`` and fall back to
``~/.agent-mem`` when unset. ``ensure_home`` should create the directory.
"""

from __future__ import annotations

from pathlib import Path

from agent_mem_daemon import paths


def test_home_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    assert paths.home() == tmp_path


def test_home_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MEM_HOME", raising=False)
    h = paths.home()
    # Should land under the user's HOME, with the .agent-mem suffix.
    assert h.name == ".agent-mem"
    assert h.parent == Path.home()


def test_home_expands_tilde(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MEM_HOME", "~/some/custom")
    h = paths.home()
    assert str(h).startswith(str(Path.home()))
    assert h.name == "custom"


def test_all_subpaths_anchored_to_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    assert paths.events_path() == tmp_path / "events.jsonl"
    assert paths.pid_path() == tmp_path / "daemon.pid"
    assert paths.offset_state_path() == tmp_path / "daemon.offset.json"
    assert paths.log_path() == tmp_path / "daemon.log"
    assert paths.pending_nudges_path() == tmp_path / "pending-nudges.md"
    assert paths.knowledge_dir() == tmp_path / "knowledge"
    assert paths.index_md_path() == tmp_path / "knowledge" / "index.md"
    assert paths.priming_socket_path() == tmp_path / "priming.sock"
    assert paths.hot_context_path() == tmp_path / "hot-context.md"


def test_ensure_home_creates_directory(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    monkeypatch.setenv("AGENT_MEM_HOME", str(target))
    assert not target.exists()
    returned = paths.ensure_home()
    assert returned == target
    assert target.is_dir()


def test_ensure_home_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    paths.ensure_home()
    # Second call must not raise (parents=True, exist_ok=True semantics).
    paths.ensure_home()
    assert tmp_path.is_dir()
