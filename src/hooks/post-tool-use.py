"""PostToolUse hook entrypoint — thin shim into :mod:`post_tool_use`."""

from __future__ import annotations

from post_tool_use import main

if __name__ == "__main__":  # pragma: no cover
    main()
