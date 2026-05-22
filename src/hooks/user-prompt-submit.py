"""UserPromptSubmit hook entrypoint — thin shim into :mod:`user_prompt_submit`."""

from __future__ import annotations

from user_prompt_submit import main

if __name__ == "__main__":  # pragma: no cover
    main()
