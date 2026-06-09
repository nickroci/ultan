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

import json
import os
import subprocess
import sys
import time

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


# Generous wall-clock ceiling for the FULL `ultan hook user-prompt-submit`
# entrypoint (interpreter start + argparse dispatch + daemon-down lexical
# fallback). A heavy import sneaking in would blow ~5s past this; a healthy run
# is well under a second. The ceiling catches a regression, not micro-tuning.
_MAX_ENTRYPOINT_S = 2.0


def test_hook_entrypoint_runs_fast_and_torch_free(tmp_path) -> None:
    """End-to-end speed guard: the real `ultan hook user-prompt-submit` path
    (argparse dispatch + stdin parse + daemon-down fallback + priming) must run
    fast and pull no heavy ML deps.

    Hermetic: a throwaway AGENT_MEM_HOME with an empty knowledge tree and a
    pre-touched spawn-attempt stamp, so ensure_running() degrades to the lexical
    fallback instead of spawning a real daemon during the test.
    """
    home = tmp_path / "agent-mem"
    (home / "knowledge").mkdir(parents=True)
    (home / ".daemon-spawn-attempt").touch()  # suppress daemon spawn (backoff)

    script = (
        "import sys\n"
        "import ultan.__main__ as m\n"
        "sys.argv = ['ultan', 'hook', 'user-prompt-submit']\n"
        "rc = m.main()\n"
        f"bad = [x for x in {_FORBIDDEN!r} if x in sys.modules]\n"
        "sys.stderr.write('FORBIDDEN_LEAKED=' + ','.join(bad))\n"
        "sys.exit(rc if not bad else 97)\n"
    )
    env = {**os.environ, "AGENT_MEM_HOME": str(home)}
    payload = json.dumps({"prompt": "what is the forgetting design", "session_id": "t"})

    t = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
    )
    dt = time.perf_counter() - t

    assert proc.returncode == 0, (
        f"hook entrypoint failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert dt < _MAX_ENTRYPOINT_S, (
        f"hook entrypoint too slow: {dt:.3f}s (ceiling {_MAX_ENTRYPOINT_S}s)\n"
        f"stderr={proc.stderr!r}"
    )
    # If it emitted anything, it must be the well-formed hook JSON envelope.
    out = proc.stdout.strip()
    if out:
        data = json.loads(out)
        assert "hookSpecificOutput" in data
