"""Declarative eval cases + their scorers.

A case is pure data: a short rolling buffer to feed the Librarian and a
``check`` predicate over the proposals it emits. The seeded library is the
on-disk corpus under ``evals/corpus/knowledge`` (a synthetic cooking library,
so a fixture can never overlap with — or be confused for — the user's real
knowledge base); the harness copies it into a throwaway temp dir per run. This
module only describes *what* the right answer looks like.

Scorers are deliberately tolerant — they assert the one behaviour that
matters and ignore incidental variation, so a non-deterministic model doesn't
flake a pre-push gate. Both cases were chosen because the agent gives the right
answer *reliably* (measured over repeated real runs), and they're mutually
constraining — the first fails an agent that over-writes, the second an agent
that never writes:

- ``dedupe_pasta_salt`` asserts a NEGATIVE — don't write a second copy of a
  rule the corpus already holds (``salt-pasta-water``). Negatives are robust;
  the agent would have to actively misjudge an exact restatement to fail.
- ``novel_cpp_for_ai`` asserts a POSITIVE — a clearly-novel, *in-domain*
  technical fact (a new C++ for AI) the agent must capture. In-domain novelty
  fired 5/5 in measurement, vs a single miss on out-of-domain trivia, so this
  is a genuine surprise/salience test (not a forced write) that holds up as a
  gate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    detail: str


# A scorer takes the list of proposal dicts (``LibrarianProposal.proposals``,
# already validated + ``model_dump``-ed) and returns pass/fail + a one-line why.
Check = Callable[[list[dict[str, Any]]], CheckResult]


@dataclass(frozen=True)
class EvalCase:
    name: str
    description: str
    # (role, text) exchanges; role is "user" or "assistant".
    exchanges: tuple[tuple[str, str], ...]
    check: Check


# ── Scorer helpers ───────────────────────────────────────────────────────────


def _writes(proposals: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in proposals if p.get("action") == "write_entry"]


def _duplicate_writes(
    proposals: Sequence[dict[str, Any]], *, term: str, existing_id: str, existing_path: str
) -> list[dict[str, Any]]:
    """``write_entry`` proposals that are a FRESH duplicate of existing
    knowledge about ``term`` — i.e. a new file covering the same ground. A
    write is NOT a duplicate (and is excluded) when it is dedup-aware: tagged
    ``reinforces``/``drift``/``used_helpfully``, or it links / overwrites the
    existing entry. Those are the legitimate ways to touch a known rule."""
    out: list[dict[str, Any]] = []
    for p in _writes(proposals):
        path = str(p.get("path") or "")
        if term not in f"{path} {p.get('body', '')}".lower():
            continue
        if p.get("salience_signal") in ("reinforces", "drift", "used_helpfully"):
            continue
        existing_ref = str(p.get("existing_entry") or "").lower()
        if existing_id in existing_ref or existing_path in existing_ref:
            continue
        if path == existing_path:
            continue
        out.append(p)
    return out


def _check_dedupe_pasta_salt(proposals: list[dict[str, Any]]) -> CheckResult:
    # References the corpus entry evals/corpus/knowledge/global/cooking/salt-pasta-water.md
    offenders = _duplicate_writes(
        proposals,
        term="pasta",
        existing_id="salt-pasta-water",
        existing_path="global/cooking/salt-pasta-water.md",
    )
    if offenders:
        paths = ", ".join(str(p.get("path") or "?") for p in offenders)
        return CheckResult(
            False,
            f"created {len(offenders)} new write_entry duplicating the existing "
            f"pasta rule ({paths})",
        )
    return CheckResult(
        True,
        f"recognised the existing rule — no duplicate write ({len(proposals)} proposal(s) total)",
    )


def _check_proposed_a_write(proposals: list[dict[str, Any]]) -> CheckResult:
    writes = _writes(proposals)
    if writes:
        return CheckResult(True, f"captured the novel fact ({len(writes)} write_entry)")
    actions = ", ".join(sorted({str(p.get("action") or "?") for p in proposals})) or "none"
    return CheckResult(
        False,
        f"a clearly-novel in-domain fact should be filed as a write_entry; got actions: {actions}",
    )


# ── The cases ────────────────────────────────────────────────────────────────


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        name="dedupe_pasta_salt",
        description=(
            "User restates a rule the corpus already holds (salt pasta water "
            "generously). The Librarian must recognise the duplicate, not write "
            "a second copy."
        ),
        exchanges=(
            (
                "user",
                "Reminder when you cook for me: always salt the pasta water "
                "generously — it should taste like the sea.",
            ),
            (
                "assistant",
                "Understood — I'll salt the pasta water generously, like seawater.",
            ),
        ),
        check=_check_dedupe_pasta_salt,
    ),
    EvalCase(
        name="novel_cpp_for_ai",
        description=(
            "User states a genuinely novel, in-domain technical fact (a new C++ "
            "for AI workloads). The Librarian must recognise the surprise and "
            "propose capturing it."
        ),
        exchanges=(
            (
                "user",
                "Heads up for how we work together: they just released C+++, a "
                "brand-new version of C++ built specifically for AI workloads, and "
                "we are standardising all of our services on it going forward.",
            ),
            (
                "assistant",
                "Understood — I will assume we are standardising on C+++ for our AI services now.",
            ),
        ),
        check=_check_proposed_a_write,
    ),
)


def case_by_name(name: str) -> EvalCase | None:
    for case in CASES:
        if case.name == name:
            return case
    return None
