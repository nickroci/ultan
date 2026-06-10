"""The MCP server start is the plugin's earliest code execution after
/plugin install — _kick_provisioner must fire the background installer
exactly when the runtime is missing, and never otherwise."""

from pathlib import Path

import pytest

from ultan import _mcp


@pytest.fixture()
def spawn_spy(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    class _FakeProc:
        pass

    def _fake_popen(cmd: list[str], **kwargs: object) -> _FakeProc:
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(_mcp.subprocess, "Popen", _fake_popen)
    return calls


def _plugin_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "plugin-root"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "ensure-ultan.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    data = tmp_path / "plugin-data"
    data.mkdir()
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(root))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(data))
    return root, data


def test_kick_fires_installer_when_unprovisioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawn_spy: list[list[str]]
) -> None:
    root, _data = _plugin_dirs(tmp_path, monkeypatch)
    _mcp._kick_provisioner()
    assert spawn_spy == [["bash", str(root / "scripts" / "ensure-ultan.sh")]]


def test_kick_noop_when_already_provisioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawn_spy: list[list[str]]
) -> None:
    _root, data = _plugin_dirs(tmp_path, monkeypatch)
    (data / "bin").mkdir()
    (data / "bin" / "ultan").write_text("#!/bin/bash\n", encoding="utf-8")
    _mcp._kick_provisioner()
    assert spawn_spy == []


def test_kick_noop_without_plugin_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawn_spy: list[list[str]]
) -> None:
    monkeypatch.delenv("CLAUDE_PLUGIN_ROOT", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "nope"))
    _mcp._kick_provisioner()
    assert spawn_spy == []
