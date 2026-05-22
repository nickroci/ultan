"""PreToolUse hook entrypoint — thin shim into :mod:`pre_tool_use`.

Claude Code invokes this hyphenated path verbatim per its settings.json
convention; the importable logic lives one module over because Python
identifiers can't contain hyphens. See ``pre_tool_use.py`` for the docstring.
"""

from __future__ import annotations

from pre_tool_use import main

if __name__ == "__main__":  # pragma: no cover
    main()
