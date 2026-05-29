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
LIBRARIAN_TIMEOUT_S = 600.0
SCHOLAR_TIMEOUT_S = 600.0
