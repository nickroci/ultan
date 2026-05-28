"""Shared LLM-call exception type for the curator agents.

The Librarian and the Scholar are now Pydantic AI agents (see
``librarian_agent`` / ``scholar_agent``); the run wrappers live in
``_agent_common.run_agent_to_output``. The old Claude-Agent-SDK call paths
(``run_librarian_call`` / ``run_scholar_call``), their streaming
``query`` driver, and the pre-write path / wikilink permission guard have
been removed — the typed agents enforce boundaries via per-output
validators + ``ModelRetry``, and the deterministic executor is the only
writer.

This module is kept as the single home for :class:`LLMTimeout` so the
agent modules and ``scholar`` / ``librarian`` orchestrators all raise and
catch the one wall-clock-budget exception type without importing each
other.
"""

from __future__ import annotations


class LLMTimeout(RuntimeError):
    """Raised when an agent run exceeds its wall-clock budget."""
