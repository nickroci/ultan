"""The ultan-search skill must triage a missing socket honestly: "starting
up" / "still installing" / "down" — never a bare failure with from-source
advice while the first start is loading models for minutes."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_SKILL = Path(__file__).resolve().parent.parent / "skills" / "ultan-search" / "search.py"


@pytest.fixture()
def skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("ultan_search_skill", _SKILL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    home = tmp_path / "agent-mem"
    home.mkdir()
    data = tmp_path / "plugin-data"
    data.mkdir()
    monkeypatch.setattr(mod, "_HOME", home)
    monkeypatch.setattr(mod, "_PLUGIN_DATA", data)
    return mod


def test_warming_when_state_flag_pid_alive(skill) -> None:
    (skill._HOME / "daemon.state").write_text(
        json.dumps({"phase": "warming", "pid": os.getpid(), "since": "x"}),
        encoding="utf-8",
    )
    msg = skill._startup_message()
    assert msg is not None
    assert "starting up" in msg
    assert "ultan doctor" in msg


def test_warming_when_spawn_stamp_is_fresh(skill) -> None:
    (skill._HOME / ".daemon-spawn-attempt").write_text("x", encoding="utf-8")
    msg = skill._startup_message()
    assert msg is not None
    assert "starting up" in msg


def test_installing_when_install_lock_present(skill) -> None:
    (skill._PLUGIN_DATA / ".install.lock").write_text("token", encoding="utf-8")
    msg = skill._startup_message()
    assert msg is not None
    assert "still installing" in msg


def test_none_when_no_startup_signals(skill) -> None:
    assert skill._startup_message() is None


def test_dead_state_flag_does_not_report_warming(skill) -> None:
    """A flag left by a crashed daemon (dead pid) must not claim warming."""
    (skill._HOME / "daemon.state").write_text(
        json.dumps({"phase": "warming", "pid": 99999999, "since": "x"}),
        encoding="utf-8",
    )
    assert skill._startup_message() is None
