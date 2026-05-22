"""SessionStart hook entrypoint — thin shim into :mod:`session_start`."""

from __future__ import annotations

from session_start import main

if __name__ == "__main__":  # pragma: no cover
    main()
