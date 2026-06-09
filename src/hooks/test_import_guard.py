"""Guardrail: the legacy hook hot path must import FAST and pull NO heavy deps.

Mirror of the root tests/test_hook_import.py, for the from-source deployment
mode (/ultan-install wires settings.json straight at these modules). The
UserPromptSubmit chain (user_prompt_submit → _events/_hookutil/_nudges/
_priming_client/scope, plus `aliases` from agent-mem-search) is stdlib-only
today — but agent-mem-search's sibling flat modules (bm25, embeddings) sit one
stray import away from rank_bm25/torch, and the SDK is installed in this venv,
so a regression here would pass CI silently without this guard.

Measured in a clean subprocess so the result is independent of whatever the
test runner itself has already imported.
"""

from __future__ import annotations

import subprocess
import sys

# The heavy ML stack, the retrieval libs reachable through agent-mem-search's
# flat namespace, and the Agent SDK — none belong on the per-turn hook path.
_FORBIDDEN = (
    "torch",
    "sentence_transformers",
    "transformers",
    "sklearn",
    "numpy",
    "rank_bm25",
    "claude_agent_sdk",
)

# Generous ceiling — the (stdlib-only) hook modules import in milliseconds.
# The ceiling exists to catch a regression, not to micro-tune.
_MAX_IMPORT_S = 0.6


def test_legacy_hook_path_imports_fast_and_heavy_free() -> None:
    script = (
        "import time, sys\n"
        "t = time.perf_counter()\n"
        "import user_prompt_submit  # the legacy user-prompt-submit hot path\n"
        "dt = time.perf_counter() - t\n"
        f"bad = [m for m in {_FORBIDDEN!r} if m in sys.modules]\n"
        "print(round(dt, 4))\n"
        "print(','.join(bad))\n"
        f"assert not bad, f'forbidden imports leaked into the hook path: {{bad}}'\n"
        f"assert dt < {_MAX_IMPORT_S}, f'hook import too slow: {{dt:.3f}}s (ceiling {_MAX_IMPORT_S}s)'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"legacy hook import guard failed:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
