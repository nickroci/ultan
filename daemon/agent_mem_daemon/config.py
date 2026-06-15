"""Single source of truth for the curator's model identities and the daemon's
shared wall-clock budgets.

Bump a model here and both roles — and any other daemon caller — move together;
there are no scattered per-module copies to miss. Role-specific tuning that
legitimately differs (the Scholar's vs the Librarian's ``OUTPUT_RETRIES``)
stays in the role modules; only the genuinely-shared knobs live here.

The CLI advisor (``tools/ultan/advisor.py``) is a separate, loosely-coupled
package that reaches the daemon only via a runtime ``sys.path`` shim, so it
keeps its own copy of the model names — keep them in sync with the values
below.
"""

from __future__ import annotations

# Models. The Librarian is the recall tier (Sonnet); the Scholar is the
# precision gatekeeper (Opus). Bump these to roll the whole curator forward.
LIBRARIAN_MODEL = "claude-sonnet-4-6"
SCHOLAR_MODEL = "claude-opus-4-8"

# Per-role wall-clock budget for one agent run. Both daemon roles run on the
# background loop (never the hot path of a user turn), so generous timeouts are
# fine — we'd rather wait than drop a packet to a transient slow call.
# The Librarian budget is large because LIBRARIAN_MAX_TURNS lets it do deep,
# multi-step library research; a genuine 100-turn curation pass can legitimately
# run many minutes. Stalls are caught fast by LIBRARIAN_FIRST_PROGRESS_S (below),
# NOT by this cap, so a big number here costs nothing on the failure path.
LIBRARIAN_TIMEOUT_S = 1800.0
SCHOLAR_TIMEOUT_S = 600.0

# Librarian agentic depth. The recall pass does multi-step library research
# (bm25 + embedding search, read_entry to dedup, reorg planning); 100 turns lets
# it go deep on a busy session. The Scholar keeps the shim default (20) — it is
# the precision gatekeeper, not the explorer.
LIBRARIAN_MAX_TURNS = 100

# Stall watchdog + retry for the agent calls. Observed failure mode: the spawned
# agent subprocess occasionally produces no first message and sits until the
# wall-clock budget ($0 cost, exactly the timeout, independent of input size) — a
# startup/transport stall, not slow work. We mirror what Claude Code itself does:
# detect a stalled stream FAST (no first message within FIRST_PROGRESS_S), kill
# the subprocess, and retry — rather than block a worker slot for the full
# budget. Legit runs stream within seconds, so they never trip the watchdog.
# Retry is safe for both roles: the Librarian only proposes, and the Scholar's
# writes happen in the deterministic executor AFTER the agent call returns — a
# stalled/timed-out call produces no decisions, so nothing is half-applied.
LIBRARIAN_FIRST_PROGRESS_S = 90.0
LIBRARIAN_MAX_ATTEMPTS = 3
SCHOLAR_FIRST_PROGRESS_S = 90.0
SCHOLAR_MAX_ATTEMPTS = 3
