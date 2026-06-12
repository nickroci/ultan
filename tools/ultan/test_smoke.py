"""Smoke tests for the agent-mem-tools modules.

Two jobs: (1) give the tools/ultan CI matrix job a non-empty collection
(pytest exits 5 when it collects nothing, which would fail the job), and
(2) pin the laziness contract — importing these modules must NOT pull the
heavy SDK/retrieval stacks, because `ultan advisor`/`ultan remember` import
them inside the CLI dispatch and anything heavy at module level would slow
every `ultan` invocation.
"""

from __future__ import annotations

import subprocess
import sys

_FORBIDDEN = ("torch", "sentence_transformers", "transformers", "claude_agent_sdk")


def test_modules_import_without_heavy_deps() -> None:
    script = (
        "import sys\n"
        "import advisor, remember\n"
        f"bad = [m for m in {_FORBIDDEN!r} if m in sys.modules]\n"
        "assert not bad, f'heavy imports leaked to module level: {bad}'\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
