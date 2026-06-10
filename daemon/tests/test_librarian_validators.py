"""Boundary-validator + run-wrapper tests for the typed Librarian (``librarian.py``).

Mirrors ``test_scholar_validators.py``: the path/body validator helpers raise
:class:`typed_agent.ModelRetry`, and ``run_librarian_agent`` is tested by
stubbing ``librarian.run_typed`` — no model call. The Librarian's bar is
deliberately lower than the Scholar's (recall over precision): well-formed
paths + parseable bodies only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agent_mem_daemon import librarian
from agent_mem_daemon._schemas import LibrarianProposal
from agent_mem_daemon.librarian import LibrarianDeps
from agent_mem_daemon.typed_agent import ModelRetry, TypedResult

from .conftest import scholar_entry_body, seed_scholar_tree


def _proposal(proposals: List[Dict[str, Any]]) -> LibrarianProposal:
    return LibrarianProposal.model_validate({"proposals": proposals, "interrupts": []})


def _deps(k: Path) -> LibrarianDeps:
    return LibrarianDeps(knowledge_dir=k)


def _wf_body() -> str:
    """A body with parseable frontmatter."""
    return scholar_entry_body("x")


# ── validate_proposal: well-formed passes ────────────────────────────────────


def test_validate_proposal_passes_for_wellformed(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": _wf_body(),
                "reasoning": "r",
            }
        ]
    )
    assert librarian.validate_proposal(_deps(k), p) is p


def test_validate_proposal_allows_empty_body(tmp_path: Path) -> None:
    """A thin write with an empty body is left to the Scholar — not a defect."""
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [{"action": "write_entry", "path": "global/python/thin.md", "body": "", "reasoning": "r"}]
    )
    librarian.validate_proposal(_deps(k), p)  # no raise


# ── validate_proposal: malformed paths ───────────────────────────────────────


def test_validate_proposal_rejects_absolute_path(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    p = _proposal([{"action": "write_entry", "path": "/etc/x.md", "body": "", "reasoning": "r"}])
    with pytest.raises(ModelRetry) as exc:
        librarian.validate_proposal(_deps(k), p)
    assert "RELATIVE" in str(exc.value)


def test_validate_proposal_rejects_escape_path(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    p = _proposal([{"action": "write_entry", "path": "../escape.md", "body": "", "reasoning": "r"}])
    with pytest.raises(ModelRetry) as exc:
        librarian.validate_proposal(_deps(k), p)
    assert "OUTSIDE" in str(exc.value)


def test_validate_proposal_rejects_non_md_path(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [{"action": "write_entry", "path": "global/python/new", "body": "", "reasoning": "r"}]
    )
    with pytest.raises(ModelRetry) as exc:
        librarian.validate_proposal(_deps(k), p)
    assert "must end in '.md'" in str(exc.value)


# ── validate_proposal: body frontmatter ──────────────────────────────────────


def test_validate_proposal_rejects_unparseable_body(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [
            {
                "action": "write_entry",
                "path": "global/python/new.md",
                "body": "no frontmatter at all, just prose",
                "reasoning": "r",
            }
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        librarian.validate_proposal(_deps(k), p)
    assert "frontmatter" in str(exc.value)


# ── _entry_target_paths: every action kind's path fields ─────────────────────


def test_validate_proposal_covers_all_entry_path_kinds(tmp_path: Path) -> None:
    """merge / move / archive / deprecate / add_wikilink exercise the path-pair
    extraction branches; all paths are well-formed, so no raise."""
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [
            {
                "action": "merge_entries",
                "source_paths": ["global/python/a.md", "global/python/b.md"],
                "target_path": "global/python/m.md",
                "target_body": "",
                "reasoning": "r",
            },
            {
                "action": "move_entry",
                "from_path": "global/python/a.md",
                "to_path": "global/db/a.md",
                "reasoning": "r",
            },
            {"action": "archive_entry", "path": "global/python/b.md", "reasoning": "r"},
            {
                "action": "deprecate_entry",
                "path": "global/python/c.md",
                "superseded_by": "global/python/m.md",
                "reasoning": "r",
            },
            {
                "action": "add_wikilink",
                "from_path": "global/python/a.md",
                "to_path": "global/python/b.md",
                "reasoning": "r",
            },
        ]
    )
    librarian.validate_proposal(_deps(k), p)  # all well-formed → no raise


# ── abstract_entries: parent + child path validation, parent body ────────────


def test_validate_proposal_abstract_entries_wellformed(tmp_path: Path) -> None:
    """A well-formed abstraction (parent .md, child .md paths, parseable
    parent body) passes the Librarian boundary."""
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [
            {
                "action": "abstract_entries",
                "child_paths": ["global/python/a.md", "global/js/b.md"],
                "parent_path": "global/conventions/likes-tooling.md",
                "parent_title": "Likes tooling",
                "parent_body": _wf_body(),
                "reasoning": "r",
            }
        ]
    )
    librarian.validate_proposal(_deps(k), p)  # no raise


def test_validate_proposal_abstract_entries_rejects_bad_child_path(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [
            {
                "action": "abstract_entries",
                "child_paths": ["global/python/a.md", "global/js/b"],  # no .md
                "parent_path": "global/conventions/likes-tooling.md",
                "parent_title": "Likes tooling",
                "parent_body": "",
                "reasoning": "r",
            }
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        librarian.validate_proposal(_deps(k), p)
    assert "child_paths" in str(exc.value) and "must end in '.md'" in str(exc.value)


def test_validate_proposal_abstract_entries_rejects_unparseable_parent_body(
    tmp_path: Path,
) -> None:
    k = seed_scholar_tree(tmp_path)
    p = _proposal(
        [
            {
                "action": "abstract_entries",
                "child_paths": ["global/python/a.md", "global/js/b.md"],
                "parent_path": "global/conventions/likes-tooling.md",
                "parent_title": "Likes tooling",
                "parent_body": "just prose, no frontmatter",
                "reasoning": "r",
            }
        ]
    )
    with pytest.raises(ModelRetry) as exc:
        librarian.validate_proposal(_deps(k), p)
    assert "frontmatter" in str(exc.value)


# ── run_librarian_agent (stub run_typed; no model call) ──────────────────────


def test_run_librarian_agent_returns_proposal_and_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k = seed_scholar_tree(tmp_path)
    proposal = _proposal([])

    async def fake_run_typed(*_a: Any, **_kw: Any) -> TypedResult:
        return TypedResult(output=proposal, cost_usd=0.002, attempts=1)

    monkeypatch.setattr(librarian, "run_typed", fake_run_typed)
    out, cost = librarian.run_librarian_agent("prompt", k, timeout_s=5.0)
    assert out is proposal
    assert cost == 0.002


def test_run_librarian_agent_timeout_maps_to_llmtimeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    k = seed_scholar_tree(tmp_path)

    async def slow_run_typed(*_a: Any, **_kw: Any) -> TypedResult:
        await asyncio.sleep(10.0)
        raise AssertionError("should have timed out")  # pragma: no cover

    monkeypatch.setattr(librarian, "run_typed", slow_run_typed)
    with pytest.raises(librarian.LLMTimeout):
        librarian.run_librarian_agent("prompt", k, timeout_s=0.05)


def test_run_librarian_agent_sets_recursion_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Curator SDK calls MUST carry CLAUDE_INVOKED_BY so the spawned Claude's
    hooks bail — otherwise the daemon ingests its own calls and loops."""
    k = seed_scholar_tree(tmp_path)
    captured: Dict[str, Any] = {}

    async def fake_run_typed(*_a: Any, **kw: Any) -> TypedResult:
        captured.update(kw)
        return TypedResult(output=_proposal([]), cost_usd=0.0, attempts=1)

    monkeypatch.setattr(librarian, "run_typed", fake_run_typed)
    librarian.run_librarian_agent("prompt", k, timeout_s=5.0)
    assert captured.get("env", {}).get("CLAUDE_INVOKED_BY") == "agent_mem_daemon"


# ── validate_user_asserted_filed: /ultan memories must be filed ───────────────


def test_user_asserted_scan_with_no_proposals_is_bounced(tmp_path: Path) -> None:
    """The prompt instructs the Librarian to FILE [USER-ASSERTED] turns, not
    veto them — yet a model observably returned zero proposals for one,
    silently discarding an explicit /ultan memory. The boundary validator
    enforces the contract with a corrective ModelRetry."""
    k = seed_scholar_tree(tmp_path)
    deps = LibrarianDeps(knowledge_dir=k, user_asserted_turns=1)
    with pytest.raises(ModelRetry) as exc:
        librarian.validate_user_asserted_filed(deps, _proposal([]))
    assert "USER-ASSERTED" in str(exc.value)


def test_user_asserted_scan_with_a_proposal_passes(tmp_path: Path) -> None:
    k = seed_scholar_tree(tmp_path)
    deps = LibrarianDeps(knowledge_dir=k, user_asserted_turns=2)
    p = _proposal(
        [
            {
                "action": "write_entry",
                "path": "global/python/asserted.md",
                "body": "",
                "reasoning": "user asked to keep this",
            }
        ]
    )
    assert librarian.validate_user_asserted_filed(deps, p) is p


def test_scan_without_user_asserted_turns_may_veto_everything(tmp_path: Path) -> None:
    """An ordinary scan is free to propose nothing — the validator only arms
    when the transcript carried [USER-ASSERTED] turns."""
    k = seed_scholar_tree(tmp_path)
    deps = LibrarianDeps(knowledge_dir=k)  # user_asserted_turns defaults to 0
    out = librarian.validate_user_asserted_filed(deps, _proposal([]))
    assert out.proposals == []
