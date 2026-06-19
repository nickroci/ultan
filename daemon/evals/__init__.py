"""Behavioural evals for the curator agents (Librarian / Scholar).

These are NOT unit tests. Each eval seeds a throwaway knowledge library (a copy
of the synthetic cooking corpus under ``evals/corpus/knowledge``), drives a
*real* curator agent through the Claude Code subscription (``claude_agent_sdk``
via ``agent_mem_daemon`` — never the metered API), and asserts the agent does
the obviously-right thing in a clear situation (recognise a duplicate instead
of writing a second copy of a rule; capture a clearly-novel in-domain fact).

They are pytest tests, but live OUTSIDE ``tests/`` on purpose: ``pytest``
(``testpaths = ["tests"]``) must not collect them, because every case spends
subscription quota and takes ~1 minute. Run them deliberately::

    cd daemon && uv run --frozen pytest evals/ --no-cov          # run all
    cd daemon && uv run --frozen pytest evals/ --co -q           # list, no calls
    cd daemon && uv run --frozen pytest evals/ --no-cov -k dedupe

The pre-push hook in ``.pre-commit-config.yaml`` runs them only when a curator
prompt/schema file changes. See ``evals/README.md``.
"""

from __future__ import annotations
