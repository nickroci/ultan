"""The daemon stamps the version of the code it is running into daemon.state, so
the `ultan` spawn path can detect a daemon left on old code after an update and
restart it. (Home is isolated by the autouse fixture in conftest.)"""

from __future__ import annotations

import importlib.metadata as md
import json
import os

from agent_mem_daemon import __main__ as dm
from agent_mem_daemon.paths import daemon_state_path


def test_write_daemon_state_records_running_version() -> None:
    dm.write_daemon_state("ready")
    state = json.loads(daemon_state_path().read_text(encoding="utf-8"))
    assert state["phase"] == "ready"
    assert state["pid"] == os.getpid()
    # The version of the code this process is running, read from on-disk metadata
    # at startup (where it equals the loaded code).
    assert state["version"] == md.version("agent-mem-daemon")
