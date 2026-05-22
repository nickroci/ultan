"""Stop hook — append a turn-boundary marker to the daemon's input stream.

Fires at the end of every assistant turn. The daemon uses Stop events
solely as turn boundaries (PLAN §7 item 5): everything since the
previous Stop, including this one, belongs to one turn. The Librarian
runs after each Stop the daemon receives.

Payload is empty by design — turn boundary only, no content. Full
transcript content stays in the transcript file the daemon's future
Librarian will open on demand.

Latency target: < 5 ms.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

_THIS_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _CODE_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _events import HookPayload, append_event  # noqa: E402


def main() -> None:
    # Recursion guard. Stop fires at the end of every assistant turn —
    # including SDK-spawned subagent turns from flush.py. Without this
    # guard every flush.py invocation would fire an extra Stop event.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    try:
        raw_input = sys.stdin.read()
        parsed: Any
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw_input)
            parsed = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError):
        return

    if not isinstance(parsed, dict):
        return
    hook_input: HookPayload = cast("HookPayload", parsed)

    append_event("Stop", hook_input, payload={})


if __name__ == "__main__":  # pragma: no cover
    main()
