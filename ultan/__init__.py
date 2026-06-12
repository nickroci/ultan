"""Ultan — thin CLI wrapper package (entry-level install)."""


def __getattr__(name: str) -> str:
    # Single source of truth for the version: the installed package metadata
    # (derived from pyproject at build time), so it can never drift from the
    # released dist the way a hardcoded literal did. Computed lazily via PEP 562
    # so `import ultan` — and the per-turn `ultan hook` hot path — never pay the
    # importlib.metadata lookup; only `ultan --version` / `ultan doctor` touch it.
    if name == "__version__":
        from importlib.metadata import (  # noqa: PLC0415 — lazy by design
            PackageNotFoundError,
            version,
        )

        try:
            return version("ultan")
        except PackageNotFoundError:
            # A source tree that was never installed as a dist (rare; the tool is
            # always `uv tool install`ed in practice).
            return "0+unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
