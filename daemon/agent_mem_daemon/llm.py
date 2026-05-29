"""Shared model-call constants for the Librarian and Scholar.

The roles no longer call the Claude Agent SDK through this module. They run
through ``typed_agent.run_typed`` (which talks to ``claude_agent_sdk`` —
the subscription backend, never the metered API) and apply their decisions
deterministically. What survives here is the small, shared surface the role
modules still import:

- :class:`LLMTimeout` — raised when a role's wall-clock budget is exceeded;
  the caller logs and drops the batch.
- the model + timeout constants the roles pin.

The old ``run_librarian_call`` / ``run_scholar_call`` SDK wrappers, the
free-text JSON drain, and the ``can_use_tool`` path guard are gone: the
model writes nothing to disk (a deterministic executor is the only writer),
so there is no write-tool to guard, and the typed shim handles the streaming
SDK call.
"""

from __future__ import annotations

# Models. The Librarian is the recall tier — Sonnet (Haiku under-extracted
# even textbook user preferences in live testing). The Scholar is the
# precision gatekeeper — Opus.
LIBRARIAN_MODEL = "claude-sonnet-4-6"
SCHOLAR_MODEL = "claude-opus-4-7"

# Both roles run on the daemon's background loop — never on the hot path of
# a user-facing agent turn. Generous timeouts are fine; we'd rather wait than
# drop a packet to a transient slow call.
LIBRARIAN_TIMEOUT_S = 600.0
SCHOLAR_TIMEOUT_S = 600.0


class LLMTimeout(RuntimeError):
    """A role's model call exceeded its wall-clock budget."""
