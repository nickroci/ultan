"""SessionEnd hook entrypoint — thin shim into :mod:`session_end`."""

from __future__ import annotations

from session_end import main

if __name__ == "__main__":  # pragma: no cover
    main()
