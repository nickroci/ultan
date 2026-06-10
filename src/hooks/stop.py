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

import os

from _events import append_event
from _hookutil import parse_stdin


def main() -> None:
    # Recursion guard. Stop fires at the end of every assistant turn —
    # including SDK-spawned subagent turns from flush.py. Without this
    # guard every flush.py invocation would fire an extra Stop event.
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return

    hook_input = parse_stdin()
    if hook_input is None:
        return

    append_event("Stop", hook_input, payload={})


if __name__ == "__main__":  # pragma: no cover
    main()
