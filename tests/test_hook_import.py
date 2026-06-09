"""Guardrail: the `ultan hook` hot path must import FAST and pull NO heavy
ML deps.

The UserPromptSubmit hook runs as a fresh process every turn under a ~2s
budget; importing torch / sentence-transformers (~5s cold) would blow it. If
someone wires a heavy import into the hook path (directly or transitively),
this test fails loudly.

We measure in a clean subprocess so the result is independent of whatever the
test runner itself has already imported.
"""

from __future__ import annotations

import subprocess
import sys

# Modules that must NEVER be imported by the hook hot path — the heavy ML stack
# plus the MCP SDK (lazy-imported only by `ultan mcp`, never by the hooks).
_FORBIDDEN = ("torch", "sentence_transformers", "transformers", "sklearn", "numpy", "mcp")

# Generous ceiling — importing the (stdlib-only) hook modules should be a few
# milliseconds. The ceiling exists to catch a regression, not to micro-tune.
_MAX_IMPORT_S = 0.6


def test_hook_path_imports_fast_and_torch_free() -> None:
    script = (
        "import time, sys\n"
        "t = time.perf_counter()\n"
        "import ultan._hooks  # the user-prompt-submit hot path\n"
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
        f"hook import guard failed:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
