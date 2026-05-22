"""PreCompact hook entrypoint — thin shim into :mod:`pre_compact`."""

from __future__ import annotations

from pre_compact import main

if __name__ == "__main__":  # pragma: no cover
    main()
