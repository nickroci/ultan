"""Seeding + execution for the curator evals.

Seeds a throwaway library by copying the on-disk corpus
(``evals/corpus/knowledge``) into a temp dir, then faithfully reproduces what
``librarian.scan`` does to build the agent prompt
(``librarian_prompt.build_librarian_user_message`` from a buffer snapshot + a
library snapshot) and drives the lower ``run_librarian_agent`` seam directly.

Why the lower seam: ``scan`` swallows every error into an empty packet, which
an "expected a proposal" scorer can't tell apart from a real "proposed
nothing" answer. ``run_librarian_agent`` *raises* on timeout/transport/parse
failure, so an infra hiccup surfaces as :class:`EvalInfraError` — which the
pytest layer turns into a *skip* (not a failure), so a flaky network never
blocks a push, while a genuine wrong answer (a failed assertion) does.

The agent is hard-scoped to the temp library: the research tools are rooted at
the path passed here and refuse to read outside it, and ``run_typed`` runs the
curator with ``setting_sources=[]`` so it has no filesystem/Read/Bash tools —
it can never reach the user's real ``~/.agent-mem``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import EvalCase

# The on-disk seed library copied into each run's temp dir.
CORPUS_KNOWLEDGE = Path(__file__).parent / "corpus" / "knowledge"


class EvalInfraError(RuntimeError):
    """The agent run failed for an infrastructure reason (timeout, transport,
    auth, no valid result) rather than producing a wrong answer. The pytest
    layer skips on this so a flaky environment doesn't block a push."""


@dataclass(frozen=True)
class RunResult:
    proposals: list[dict[str, Any]]
    cost_usd: float
    latency_s: float


def sdk_available() -> bool:
    """True iff ``claude_agent_sdk`` is importable — the runtime backing the
    subscription agent call. Uses ``find_spec`` (not an actual import) so the
    probe costs nothing and leaves no unused import to placate the linters."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("claude_agent_sdk") is not None


# ── Seeding ──────────────────────────────────────────────────────────────────


def seed_library(root: Path) -> Path:
    """Copy the corpus into ``root/knowledge`` and return that path. The corpus
    is the single source of the seeded library — edit the markdown under
    ``evals/corpus/knowledge`` to change what the agent sees."""
    knowledge = root / "knowledge"
    shutil.copytree(CORPUS_KNOWLEDGE, knowledge)
    return knowledge


def _build_snapshot(session_id: str, exchanges: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Build a buffer snapshot in the shape ``librarian.scan`` consumes — one
    sealed turn per exchange, each with a user/assistant event and a Stop
    marker. Mirrors ``tests/test_librarian._payload_snap``."""
    turns: list[dict[str, Any]] = []
    for role, text in exchanges:
        event_type = "UserPromptSubmit" if role == "user" else "PostToolUse"
        payload: dict[str, Any] = {"text": text, "role": role}
        turns.append(
            {
                "started_at": 1.0,
                "sealed_at": 2.0,
                "events": [
                    {"ts": 1.0, "type": event_type, "cwd": "/repo/ultan-eval", "payload": payload},
                    {"ts": 2.0, "type": "Stop", "cwd": "/repo/ultan-eval", "payload": {}},
                ],
            }
        )
    return {"session_id": session_id, "turns": turns}


# ── Running one case ─────────────────────────────────────────────────────────


def _run_librarian(snapshot: dict[str, Any], knowledge: Path) -> tuple[list[dict[str, Any]], float]:
    """Assemble the Librarian prompt exactly as ``librarian.scan`` does and run
    the real agent. Returns ``(proposals, cost_usd)``. Raises on agent/transport
    failure (the caller maps that to :class:`EvalInfraError`)."""
    from agent_mem_daemon import librarian  # noqa: PLC0415
    from agent_mem_daemon import librarian_prompt as lp  # noqa: PLC0415

    formatted_buffer, flat = lp.buffer_to_prompt_text(snapshot)
    prompt = lp.build_librarian_user_message(
        project_slug=lp.derive_project_bucket(snapshot),
        rolling_buffer=formatted_buffer,
        library_snapshot=lp.build_library_snapshot(knowledge),
        applies_when_table=lp.build_applies_when_table(knowledge),
        repair_tasks=lp.format_repair_tasks([]),
    )
    user_asserted = sum(1 for turn in flat if turn[4])
    proposal, cost_usd = librarian.run_librarian_agent(
        prompt, knowledge, user_asserted_turns=user_asserted
    )
    dumped: dict[str, Any] = proposal.model_dump()
    proposals: list[dict[str, Any]] = list(dumped.get("proposals") or [])
    return proposals, float(cost_usd or 0.0)


def run_case(case: EvalCase) -> RunResult:
    """Seed a throwaway library, run the agent once, and return its proposals.

    Raises :class:`EvalInfraError` if the agent call fails (timeout / transport
    / no valid result) — that's not a wrong answer, so the caller skips rather
    than fails."""
    with tempfile.TemporaryDirectory(prefix="ultan-eval-") as tmp:
        knowledge = seed_library(Path(tmp))
        snapshot = _build_snapshot(f"eval-{case.name}", case.exchanges)
        prev_home = os.environ.get("AGENT_MEM_HOME")
        os.environ["AGENT_MEM_HOME"] = str(Path(tmp))
        start = time.perf_counter()
        try:
            proposals, cost_usd = _run_librarian(snapshot, knowledge)
        except Exception as exc:  # noqa: BLE001 — any agent failure is infra, not a wrong answer
            raise EvalInfraError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            _restore_env("AGENT_MEM_HOME", prev_home)
        return RunResult(proposals, cost_usd, time.perf_counter() - start)


def _restore_env(key: str, prev: str | None) -> None:
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev
