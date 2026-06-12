"""The daemon stamps the version of the code it is running into daemon.state, so
the `ultan` spawn path can detect a daemon left on old code after an update and
restart it. ``agent_mem_home`` isolates AGENT_MEM_HOME (opt-in, per conftest) so
the write never touches the user's real ~/.agent-mem."""

from __future__ import annotations

import importlib.metadata as md
import json
import os
from pathlib import Path

from agent_mem_daemon import __main__ as dm


def test_write_daemon_state_records_running_version(agent_mem_home: Path) -> None:
    dm.write_daemon_state("ready")
    state = json.loads((agent_mem_home / "daemon.state").read_text(encoding="utf-8"))
    assert state["phase"] == "ready"
    assert state["pid"] == os.getpid()
    # The version of the code this process is running, read from on-disk metadata
    # at startup (where it equals the loaded code).
    assert state["version"] == md.version("agent-mem-daemon")
