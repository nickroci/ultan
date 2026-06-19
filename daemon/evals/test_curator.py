"""Curator agent evals, as pytest tests.

These make REAL agent calls through the Claude Code subscription
(``claude_agent_sdk`` — never the metered API), so they are slow (~1 min each)
and cost subscription quota. They live OUTSIDE ``tests/`` so a normal
``pytest`` run (``testpaths = ["tests"]``) never collects them; run them
deliberately::

    cd daemon && uv run --frozen pytest evals/ --no-cov
    cd daemon && uv run --frozen pytest evals/ --no-cov -k dedupe   # one case
    cd daemon && uv run --frozen pytest evals/ --co -q               # list, no calls

``--no-cov`` because the daemon's 90%-coverage gate would otherwise fail on an
eval-only run. The pre-push hook (``ultan-agent-evals``) runs the first form.

A wrong answer FAILS (blocks the push). An infra hiccup (timeout / transport /
not authenticated) SKIPS — a flaky environment shouldn't wedge a push. Set
``ULTAN_SKIP_EVALS=1`` to skip the whole module.
"""

from __future__ import annotations

import os

import pytest

from .cases import CASES, EvalCase
from .harness import EvalInfraError, run_case, sdk_available

pytestmark = [
    pytest.mark.skipif(
        not sdk_available(),
        reason="claude_agent_sdk not importable — no subscription runtime here",
    ),
    pytest.mark.skipif(
        bool(os.environ.get("ULTAN_SKIP_EVALS")),
        reason="ULTAN_SKIP_EVALS set",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_curator_case(case: EvalCase) -> None:
    try:
        result = run_case(case)
    except EvalInfraError as exc:
        pytest.skip(f"agent infra error (not a regression): {exc}")

    verdict = case.check(result.proposals)
    assert verdict.passed, (
        f"{verdict.detail}  [cost ${result.cost_usd:.3f}, {result.latency_s:.0f}s]"
    )
