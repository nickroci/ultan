"""Scholar orchestration tests for the Pydantic-AI gatekeeper architecture.

No real LLM calls — we monkeypatch ``scholar.run_scholar_agent`` to return
a canned, already-validated ``ScholarDecisions`` (the agent's typed output),
mirroring what the live agent returns after its validators pass.
``AGENT_MEM_HOME`` is redirected to a tmp dir per test so the
InvocationRecord audit writes don't leak.

Coverage:
  - Empty packet list → no-op.
  - All-empty packets → no agent call.
  - Happy path: returned actions are applied by the executor (files land
    on disk, index/log maintained) and nudges propagate.
  - No-actions path → no files written.
  - Agent exception / timeout → swallowed, batch dropped.
  - Heterogeneous session ids → session_id="batch" on the audit row.
  - Invariants checker runs after the agent call and surfaces violations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from agent_mem_daemon import scholar
from agent_mem_daemon._schemas import ScholarDecisions
from agent_mem_daemon.runs import InvocationRecord

from .conftest import scholar_entry_body


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    yield tmp_path


def _packet(
    session_id: str,
    *,
    proposals=None,
    interrupts=None,
    cwd: str = "/repo",
):
    return {
        "session_id": session_id,
        "cwd": cwd,
        "proposals": proposals or [],
        "interrupts": interrupts or [],
    }


def _decisions(
    actions: Optional[List[Dict[str, Any]]] = None,
    interrupts: Optional[List[Dict[str, Any]]] = None,
) -> ScholarDecisions:
    """Build a validated ScholarDecisions the way the live agent would."""
    return ScholarDecisions.model_validate(
        {"actions": actions or [], "interrupts_processed": interrupts or []}
    )


def _make_canned(decisions: ScholarDecisions, cost: float = 0.01):
    """Stub for ``run_scholar_agent`` — returns (decisions, cost)."""

    def _stub(prompt, knowledge_dir, *, timeout_s):
        return decisions, cost

    return _stub


# ── Manual-parsing path is gone (whole daemon) ────────────────────────


def test_scholar_path_has_no_json_repair_or_blob_extraction():
    """Both curators migrated off the hand-scraped-JSON parser. No module in
    the Scholar path may reference ``json_repair`` / ``extract_json_blob`` /
    ``_response_parser``."""
    import inspect

    from agent_mem_daemon import scholar as scholar_mod
    from agent_mem_daemon import scholar_agent, scholar_executor, scholar_prompt

    forbidden = ("json_repair", "extract_json_blob", "repair_json", "_response_parser")
    for module in (scholar_mod, scholar_agent, scholar_executor, scholar_prompt):
        src = inspect.getsource(module)
        for token in forbidden:
            assert token not in src, f"{token!r} still present in {module.__name__}"


# ── Pre-flight short-circuits ─────────────────────────────────────────


def test_review_empty_packet_list_is_noop(monkeypatch, caplog):
    calls = []

    def _fail(*a, **kw):
        calls.append(1)
        raise AssertionError("SDK must not be called")

    monkeypatch.setattr(scholar, "run_scholar_agent", _fail)
    scholar.review([])
    assert calls == []


def test_review_skips_when_all_packets_empty(monkeypatch):
    sdk_calls: List[int] = []

    def _fail(*a, **kw):
        sdk_calls.append(1)
        raise AssertionError("SDK must not be called for empty batch")

    monkeypatch.setattr(scholar, "run_scholar_agent", _fail)
    scholar.review([_packet("s1"), _packet("s2"), _packet("s3")])
    assert sdk_calls == []


def test_review_runs_even_with_high_recorded_cost(monkeypatch):
    """The daily cost cap was removed; the Scholar must run regardless
    of accumulated spend."""
    from agent_mem_daemon import runs

    runs.save_cost_state(
        {
            "today": runs._today_iso(),
            "today_usd": 999.99,
            "lifetime_usd": 999.99,
        }
    )

    sdk_calls: List[int] = []

    def _stub(prompt, knowledge_dir, *, timeout_s):
        sdk_calls.append(1)
        return _decisions(), 0.01

    monkeypatch.setattr(scholar, "run_scholar_agent", _stub)
    scholar.review(
        [
            _packet(
                "s1", proposals=[{"action": "archive_entry", "path": "a.md", "reasoning": "stale"}]
            )
        ]
    )
    assert sdk_calls == [1]


# ── Approve path ──────────────────────────────────────────────────────


def _valid_entry_body(id_: str, *, scope: str = "global") -> str:
    return scholar_entry_body(id_, scope=scope)


def test_review_happy_path_applies_actions_and_nudges(monkeypatch, tmp_path):
    # The agent APPROVED one of the two proposals (returns just that action),
    # plus one approved + one vetoed interrupt.
    decisions = _decisions(
        actions=[
            {
                "action": "write_entry",
                "path": "global/tooling/x.md",
                "body": _valid_entry_body("x"),
                "reasoning": "r1",
            }
        ],
        interrupts=[
            {
                "lesson_id": "factory-pattern-for-apis",
                "lesson_path": "global/tooling/factory-pattern-for-apis",
                "action": "approve",
                "text": "Memory: factory pattern for new APIs.",
                "reason": "active design",
            },
            {
                "lesson_id": "no-mock-db",
                "lesson_path": "global/testing/no-mock-db",
                "action": "veto",
                "reason": "reading not writing",
            },
        ],
    )
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(decisions, cost=0.03))

    records_finalised: List[InvocationRecord] = []
    orig_finalise = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records_finalised.append(self)
        orig_finalise(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    packets = [
        _packet(
            "s1",
            proposals=[
                {
                    "action": "write_entry",
                    "path": "global/tooling/x.md",
                    "body": _valid_entry_body("x"),
                    "reasoning": "r1",
                },
                {
                    "action": "write_entry",
                    "path": "global/tooling/y.md",
                    "body": _valid_entry_body("y"),
                    "reasoning": "r2",
                },
            ],
            interrupts=[
                {"lesson_id": "factory-pattern-for-apis"},
                {"lesson_id": "no-mock-db"},
            ],
        )
    ]
    scholar.review(packets)

    assert len(records_finalised) == 1
    rec = records_finalised[0]
    assert rec.role == "scholar"
    assert rec.session_id == "s1"
    assert rec.parsed_ok is True
    assert rec.cost_usd == pytest.approx(0.03)
    assert rec.decisions.get("packets_in") == 1
    assert rec.decisions.get("proposals_in") == 2
    assert rec.decisions.get("interrupts_in") == 2
    # Executor applied exactly the one approved action.
    assert rec.decisions.get("actions_applied") == 1
    assert rec.decisions.get("write_entry") == 1
    assert rec.decisions.get("nudge") == 1
    assert rec.decisions.get("interrupt-veto") == 1
    assert rec.decisions.get("nudges_written") == 1

    # The approved entry landed on disk and is catalogued in index.md.
    written = tmp_path / "knowledge" / "global" / "tooling" / "x.md"
    assert written.exists()
    index_md = (tmp_path / "knowledge" / "index.md").read_text(encoding="utf-8")
    assert "[[global/tooling/x]]" in index_md
    # The vetoed proposal was NOT written.
    assert not (tmp_path / "knowledge" / "global" / "tooling" / "y.md").exists()

    nudges_path = tmp_path / "pending-nudges.md"
    assert nudges_path.exists()
    body = nudges_path.read_text(encoding="utf-8")
    assert "factory pattern for new APIs" in body
    assert "no-mock-db" not in body  # vetoed

    runs_dir = tmp_path / "runs"
    jsonl_files = list(runs_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1
    line = jsonl_files[0].read_text(encoding="utf-8").splitlines()[-1]
    parsed_line = json.loads(line)
    assert parsed_line["role"] == "scholar"
    assert parsed_line["parsed_ok"] is True


# ── Veto-everything path (no returned actions) ────────────────────────


def test_review_no_actions_nothing_written(monkeypatch, tmp_path):
    """The agent vetoed every proposal → empty ``actions``. No nudge file,
    no entries written, and no per-action counters on the audit row."""
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(_decisions(), cost=0.02))

    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    packets = [
        _packet(
            "s-veto",
            proposals=[
                {
                    "action": "write_entry",
                    "path": "global/a.md",
                    "body": _valid_entry_body("a"),
                    "reasoning": "r",
                },
                {
                    "action": "write_entry",
                    "path": "global/b.md",
                    "body": _valid_entry_body("b"),
                    "reasoning": "r",
                },
            ],
        )
    ]
    scholar.review(packets)

    rec = records[0]
    assert rec.decisions.get("actions_applied", 0) == 0
    # No nudge file.
    assert not (tmp_path / "pending-nudges.md").exists()
    # No knowledge entries — the agent returned no actions.
    assert not (tmp_path / "knowledge" / "global" / "a.md").exists()
    assert not (tmp_path / "knowledge" / "global" / "b.md").exists()


# ── Invariants integration ────────────────────────────────────────────


def test_review_runs_invariants_check_after_sdk(monkeypatch, tmp_path):
    """The invariants checker runs post-SDK. If the existing tree has
    violations (e.g. a folder with too many entries), those land in
    the decisions counter."""
    # Seed an obviously broken tree: 6 entries in one dir, no README.
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    for i in range(6):
        (k / "global" / "tooling" / f"e{i}.md").write_text(
            "---\nid: e{}\n---\n# x\n".format(i),
            encoding="utf-8",
        )

    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(_decisions(), cost=0.01))

    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )

    assert records[0].decisions.get("invariant_violations", 0) >= 1


def test_review_repairs_phantom_index_row(monkeypatch, tmp_path):
    """End-to-end: the review pipeline self-heals a phantom index.md row
    (the live `some-fake-project` case) so no broken-wikilink violation
    survives into the post-write invariants check."""
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    (k / "README.md").write_text("# Knowledge\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# Global\n", encoding="utf-8")
    (k / "global" / "python" / "README.md").write_text("# Python\n", encoding="utf-8")
    (k / "global" / "python" / "use-uv.md").write_text(
        "---\nid: use-uv\ntype: lesson\nscope: global\nstatus: provisional\n"
        "confidence: 0.7\napplies-when: |\n  x\nkeywords: [a, b, c]\n"
        'title: "use-uv"\ncreated: 2026-05-19\nupdated: 2026-05-19\n'
        "fired: 0\nfired-helpful: 0\nsources:\n  - manual\n---\n\n# uv\n\nUse uv always for python.\n",
        encoding="utf-8",
    )
    phantom = (
        "| [[projects/some-fake-project/security/no-secrets-in-env-example]] "
        "| project:some-fake-project | provisional | 0.85 | s | env.example | "
        "session:hooktest-6AB94685 | 2026-05-19 |\n"
    )
    (k / "index.md").write_text(
        "# Knowledge Index\n\n| Article | Scope |\n|---|---|\n"
        "| [[global/python/use-uv]] | global |\n" + phantom,
        encoding="utf-8",
    )

    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(_decisions(), cost=0.01))

    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )

    index_after = (k / "index.md").read_text(encoding="utf-8")
    assert "some-fake-project" not in index_after  # phantom self-healed
    assert "[[global/python/use-uv]]" in index_after  # real row survived
    assert records[0].decisions.get("wikilinks_repaired", 0) >= 1
    # The repair ran BEFORE the invariants check, so no broken-wikilink
    # violation should remain on the audit row.
    assert "invariant_violations" not in records[0].decisions


def test_review_invariants_clean_does_not_log_violation(monkeypatch, tmp_path):
    """No violations → no ``invariant_violations`` key on the audit row."""
    k = tmp_path / "knowledge"
    k.mkdir()
    # Empty tree — invariants checker returns [].
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(_decisions(), cost=0.01))

    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert "invariant_violations" not in records[0].decisions


# ── Heterogeneous session ids ─────────────────────────────────────────


def test_review_heterogeneous_session_id_marked_batch(monkeypatch):
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(_decisions()))
    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    scholar.review(
        [
            _packet(
                "s1", proposals=[{"action": "archive_entry", "path": "a.md", "reasoning": "r"}]
            ),
            _packet(
                "s2", proposals=[{"action": "archive_entry", "path": "b.md", "reasoning": "r"}]
            ),
        ]
    )
    assert records[0].session_id == "batch"


# ── Agent failure paths ───────────────────────────────────────────────


def test_review_agent_failure_marks_record_unparsed(monkeypatch, tmp_path, caplog):
    """When the agent run raises (e.g. validators exhausted the retry budget
    and Pydantic AI gave up), the batch is dropped: no files, parsed_ok stays
    False, and the record is finalised with an error."""

    def _raise(prompt, knowledge_dir, *, timeout_s):
        raise RuntimeError("output retries exhausted")

    monkeypatch.setattr(scholar, "run_scholar_agent", _raise)
    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    caplog.set_level(logging.WARNING)
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "a.md", "reasoning": "r"}])]
    )
    assert len(records) == 1
    assert records[0].parsed_ok is False
    assert records[0].error is not None
    assert not (tmp_path / "pending-nudges.md").exists()


def test_review_sdk_exception_is_swallowed(monkeypatch, tmp_path, caplog):
    def _raise(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(scholar, "run_scholar_agent", _raise)
    caplog.set_level(logging.WARNING)
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "a.md", "reasoning": "r"}])]
    )


def test_review_sdk_timeout_is_swallowed(monkeypatch, tmp_path, caplog):
    from agent_mem_daemon.llm import LLMTimeout

    def _timeout(*a, **kw):
        raise LLMTimeout("too slow")

    monkeypatch.setattr(scholar, "run_scholar_agent", _timeout)
    caplog.set_level(logging.WARNING)
    scholar.review([_packet("s1", interrupts=[{"lesson_id": "x"}])])
    assert any("timeout" in rec.message.lower() for rec in caplog.records)


# ── End-to-end via mocked LLM: turn → proposal → approval ──────────


def test_review_batch_session_with_no_session_ids_marked_batch(monkeypatch):
    """If all packets lack a session_id, the audit row gets ``session_id="batch"``."""
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(_decisions()))
    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    # Packets with empty session_id — the helper hits the ``if not sids`` branch.
    scholar.review(
        [
            {
                "session_id": "",
                "proposals": [{"action": "archive_entry", "path": "x.md", "reasoning": "r"}],
                "interrupts": [],
            }
        ]
    )
    assert records[0].session_id == "batch"


def test_review_reinforcement_counter_pass_exception_is_swallowed(monkeypatch, caplog):
    """If apply_reinforcement_counters raises, the Scholar logs and
    proceeds with the SDK call."""
    from agent_mem_daemon import scholar_prompt

    def _boom(*a, **kw):
        raise RuntimeError("simulated reinforcement failure")

    monkeypatch.setattr(scholar_prompt, "apply_reinforcement_counters", _boom)
    monkeypatch.setattr(
        scholar,
        "run_scholar_agent",
        _make_canned(_decisions()),
    )

    caplog.set_level(logging.WARNING)
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert any("reinforcement-counter pass raised" in r.message for r in caplog.records)


def test_review_reinforcement_changes_logged(monkeypatch, tmp_path):
    """When the reinforcement helper reports changes, the audit row
    records the count and each change is logged."""
    from agent_mem_daemon import scholar_prompt

    def _has_changes(packets, kdir):
        return ["bumped: a", "bumped: b"]

    monkeypatch.setattr(scholar_prompt, "apply_reinforcement_counters", _has_changes)
    monkeypatch.setattr(
        scholar,
        "run_scholar_agent",
        _make_canned(_decisions()),
    )

    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert records[0].decisions.get("reinforcement_bumps") == 2


def test_review_nudge_file_append_exception_swallowed(monkeypatch, caplog):
    from agent_mem_daemon import scholar_prompt

    decisions = _decisions(
        interrupts=[{"lesson_id": "x", "lesson_path": "x", "action": "approve", "text": "y"}]
    )
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(decisions))

    def _boom(parsed):
        raise RuntimeError("simulated nudge append failure")

    monkeypatch.setattr(scholar_prompt, "append_nudges_from_response", _boom)
    caplog.set_level(logging.WARNING)
    scholar.review(
        [
            _packet(
                "s1",
                proposals=[
                    {"action": "write_entry", "path": "x.md", "body": "x", "reasoning": "r"}
                ],
                interrupts=[{"lesson_id": "x"}],
            )
        ]
    )
    assert any("nudge-file append failed" in r.message for r in caplog.records)


def test_review_reconcile_readmes_exception_swallowed(monkeypatch, caplog):
    from agent_mem_daemon import scholar_prompt

    monkeypatch.setattr(
        scholar,
        "run_scholar_agent",
        _make_canned(_decisions()),
    )

    def _boom(kdir):
        raise RuntimeError("simulated reconcile failure")

    monkeypatch.setattr(scholar_prompt, "reconcile_readmes", _boom)
    caplog.set_level(logging.WARNING)
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert any("README reconciliation raised" in r.message for r in caplog.records)


def test_review_reconcile_readmes_results_recorded(monkeypatch):
    from agent_mem_daemon import scholar_prompt

    monkeypatch.setattr(
        scholar,
        "run_scholar_agent",
        _make_canned(_decisions()),
    )
    monkeypatch.setattr(
        scholar_prompt, "reconcile_readmes", lambda kdir: ["a updated", "b updated"]
    )
    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert records[0].decisions.get("readmes_reconciled") == 2


def test_review_priming_exception_swallowed(monkeypatch, caplog):
    from agent_mem_daemon import priming

    monkeypatch.setattr(
        scholar,
        "run_scholar_agent",
        _make_canned(_decisions()),
    )

    def _boom(*a, **kw):
        raise RuntimeError("simulated priming failure")

    monkeypatch.setattr(priming, "refresh_hot_context", _boom)
    caplog.set_level(logging.WARNING)
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert any("priming refresh raised" in r.message for r in caplog.records)


def test_review_invariants_exception_swallowed(monkeypatch, caplog):
    from agent_mem_daemon import scholar_prompt

    monkeypatch.setattr(
        scholar,
        "run_scholar_agent",
        _make_canned(_decisions()),
    )

    def _boom(kdir):
        raise RuntimeError("simulated invariants failure")

    monkeypatch.setattr(scholar_prompt, "check_invariants_detailed", _boom)
    caplog.set_level(logging.WARNING)
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert any("invariants check raised" in r.message for r in caplog.records)


def test_review_all_empty_priming_exception_swallowed(monkeypatch, caplog):
    """The all-empty branch also refreshes priming; if that raises, the
    Scholar logs and returns cleanly."""
    from agent_mem_daemon import priming

    def _boom(*a, **kw):
        raise RuntimeError("simulated priming failure (empty path)")

    monkeypatch.setattr(priming, "refresh_hot_context", _boom)
    caplog.set_level(logging.WARNING)
    # All-empty packets — should hit the empty-path branch, then catch
    # the priming exception.
    scholar.review([_packet("s1"), _packet("s2")])
    assert any("priming refresh raised (empty-packet path)" in r.message for r in caplog.records)


def test_end_to_end_proposal_approved_writes_entry_to_knowledge_dir(
    monkeypatch,
    tmp_path,
):
    """End-to-end: the agent returns a ``write_entry`` action; the daemon
    EXECUTOR writes the file (the model no longer does). We assert the file
    lands on disk, the audit row reflects the applied action, and the
    invariants check stays clean."""
    target_path = tmp_path / "knowledge" / "global" / "tooling" / "stub-the-factory.md"
    target_path.parent.mkdir(parents=True)
    # README so the invariants checker stays clean.
    (tmp_path / "knowledge" / "README.md").write_text("# k\n", encoding="utf-8")
    (tmp_path / "knowledge" / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (tmp_path / "knowledge" / "global" / "tooling" / "README.md").write_text(
        "# tooling\n",
        encoding="utf-8",
    )

    full_entry = (
        "---\nid: stub-the-factory\ntype: lesson\nscope: global\n"
        "status: provisional\nconfidence: 0.7\n"
        "applies-when: |\n  testing a service with a factory\n"
        "keywords: [test, stub, factory]\n"
        'title: "Stub the factory"\n'
        "created: 2026-05-19\nupdated: 2026-05-19\n"
        "fired: 0\nfired-helpful: 0\nsources:\n  - manual\n"
        "---\n\n# Stub the factory\n\nBody sentence long enough to pass.\n"
    )

    decisions = _decisions(
        actions=[
            {
                "action": "write_entry",
                "path": "global/tooling/stub-the-factory.md",
                "body": full_entry,
                "reasoning": "buffer turn [2]",
            }
        ]
    )
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(decisions, cost=0.01))

    records: List[InvocationRecord] = []
    orig = scholar.runs.InvocationRecord.finalise

    def _spy(self):
        records.append(self)
        orig(self)

    monkeypatch.setattr(scholar.runs.InvocationRecord, "finalise", _spy)

    scholar.review(
        [
            _packet(
                "s1",
                proposals=[
                    {
                        "action": "write_entry",
                        "path": "global/tooling/stub-the-factory.md",
                        "body": full_entry,
                        "reasoning": "buffer turn [2]",
                    }
                ],
            )
        ]
    )

    # The executor — not the model — wrote the file.
    assert target_path.exists()
    assert target_path.read_text(encoding="utf-8") == full_entry
    assert records[0].decisions.get("actions_applied") == 1
    assert records[0].decisions.get("write_entry") == 1
    # No invariant violations introduced.
    assert records[0].decisions.get("invariant_violations", 0) == 0


# ── Integrity-repair escalation (broken wikilink → Librarian pipeline) ─


def _seed_library_with_broken_link(k: Path, *, broken: str = "global/ghost/never-existed") -> Path:
    """Build a minimal invariant-clean tree, then add ONE entry whose body
    contains an unresolvable wikilink the deterministic pass can't fix.
    Returns the entry path. The link is left for the escalation path."""
    (k / "global" / "python").mkdir(parents=True)
    (k / "README.md").write_text("# Knowledge\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# Global\n", encoding="utf-8")
    (k / "global" / "python" / "README.md").write_text("# Python\n", encoding="utf-8")
    entry = k / "global" / "python" / "narr.md"
    entry.write_text(
        "---\nid: narr\ntype: lesson\nscope: global\nstatus: provisional\n"
        "confidence: 0.7\napplies-when: |\n  x\nkeywords: [a, b, c]\n"
        'title: "narr"\ncreated: 2026-05-19\nupdated: 2026-05-19\n'
        "fired: 0\nfired-helpful: 0\nsources:\n  - manual\n---\n\n"
        f"# narr\n\nThis references [[{broken}]] in a sentence we keep.\n",
        encoding="utf-8",
    )
    return entry


@pytest.fixture(autouse=True)
def _fresh_repair_queue():
    """Isolate the process-global repair queue between escalation tests."""
    from agent_mem_daemon import repair_queue

    repair_queue.reset_queue()
    yield
    repair_queue.reset_queue()


def _canned_review(monkeypatch):
    monkeypatch.setattr(scholar, "run_scholar_agent", _make_canned(_decisions(), cost=0.01))


def test_review_escalates_unresolvable_wikilink(monkeypatch, tmp_path):
    """The trigger: a broken link the deterministic pass can't resolve is
    enqueued as a repair task (in-flight) AND left broken on disk."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    entry = _seed_library_with_broken_link(k)
    _canned_review(monkeypatch)

    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )

    q = repair_queue.get_queue()
    # The issue is now queued + in-flight (this review carried NO incoming
    # fingerprints, so nothing was cleared).
    assert q.pending_count() == 1
    assert q.inflight_count() == 1
    fp = ("broken_wikilink", "global/python/narr.md", "global/ghost/never-existed")
    drained = q.drain_pending()
    assert [t.fingerprint for t in drained] == [fp]
    # Left broken on disk so the next detection can re-fire.
    assert "[[global/ghost/never-existed]]" in entry.read_text(encoding="utf-8")


def test_inflight_guard_blocks_concurrent_duplicate(monkeypatch, tmp_path):
    """Two consecutive reviews that BOTH detect the same still-broken link
    (no incoming fingerprints to clear) must escalate it exactly once — the
    in-flight marker blocks the duplicate."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    _seed_library_with_broken_link(k)
    _canned_review(monkeypatch)

    base = [
        _packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])
    ]
    scholar.review(base)  # first detection: enqueues
    scholar.review(base)  # second detection: marker still in-flight → skip

    q = repair_queue.get_queue()
    assert q.inflight_count() == 1
    assert q.pending_count() == 1  # NOT 2 — the duplicate was rejected


def test_review_releases_marker_then_reescalates(monkeypatch, tmp_path):
    """Re-escalation across detections: a review that CARRIES the incoming
    fingerprint releases it on conclusion; the next detection of the still-
    broken link re-escalates (no max-attempts cap, no permanent give-up)."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    _seed_library_with_broken_link(k)
    _canned_review(monkeypatch)

    fp = ("broken_wikilink", "global/python/narr.md", "global/ghost/never-existed")

    # Pre-load the queue as if a prior pass escalated this issue, then
    # simulate the Librarian having drained it (so it's in-flight, attached
    # to the packet now reaching the Scholar).
    q = repair_queue.get_queue()
    q.enqueue(
        repair_queue.RepairTask(
            kind="broken_wikilink",
            file="global/python/narr.md",
            target="global/ghost/never-existed",
            context="ctx",
        )
    )
    q.drain_pending()  # Librarian took it; marker stays in-flight
    assert q.inflight_count() == 1

    # The Scholar reviews the packet that carried this fingerprint. Its
    # proposal does NOT fix the link (canned empty decisions), so the
    # deterministic repair re-detects it — but the marker is still in-flight
    # during the review, so re-detection is SUPPRESSED. The finally then
    # releases the incoming marker.
    packet = _packet(
        "s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}]
    )
    packet["repair_fingerprints"] = [list(fp)]  # JSON-ish shape (list)
    scholar.review([packet])

    # Marker released; nothing re-queued during that same review.
    assert q.inflight_count() == 0
    assert q.pending_count() == 0

    # The NEXT detection (fresh review, no incoming fingerprints) re-fires.
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])]
    )
    assert q.inflight_count() == 1
    assert q.pending_count() == 1


def test_marker_released_even_when_review_short_circuits_empty(monkeypatch, tmp_path):
    """A packet that carries fingerprints but has no proposals/interrupts
    hits the all-empty early return — the markers must STILL be released
    (else a drained-then-errored Librarian leaks its escalation forever)."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    k.mkdir()  # empty tree — no detection happens
    _canned_review(monkeypatch)

    fp = ("broken_wikilink", "global/python/narr.md", "global/ghost/x")
    q = repair_queue.get_queue()
    q.enqueue(
        repair_queue.RepairTask(kind="broken_wikilink", file=fp[1], target=fp[2], context="c")
    )
    q.drain_pending()
    assert q.inflight_count() == 1

    # Empty packet (no proposals, no interrupts) but carrying the fingerprint.
    empty = {
        "session_id": "s1",
        "proposals": [],
        "interrupts": [],
        "repair_fingerprints": [list(fp)],
    }
    scholar.review([empty])

    assert q.inflight_count() == 0  # released despite the empty early-return


# ── Generalised escalation: over-cap dirs and bad frontmatter ─────────


def _entry_frontmatter(id_: str) -> str:
    return (
        f"---\nid: {id_}\ntype: lesson\nscope: global\nstatus: provisional\n"
        "confidence: 0.7\napplies-when: |\n  x\nkeywords: [a, b, c]\n"
        f'title: "{id_}"\ncreated: 2026-05-19\nupdated: 2026-05-19\n'
        "fired: 0\nfired-helpful: 0\nsources:\n  - manual\n---\n\n"
        f"# {id_}\n\nA real body sentence that is clearly long enough.\n"
    )


def _seed_overcap_dir(k: Path) -> None:
    """A flat dir with 6 well-formed entries — over the 5-entry cap. The
    only invariant tripped is the over-cap one (so the escalation queue
    holds exactly one task)."""
    (k / "global" / "python").mkdir(parents=True)
    (k / "README.md").write_text("# Knowledge\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# Global\n", encoding="utf-8")
    (k / "global" / "python" / "README.md").write_text("# Python\n", encoding="utf-8")
    for i in range(6):
        (k / "global" / "python" / f"e{i}.md").write_text(
            _entry_frontmatter(f"e{i}"), encoding="utf-8"
        )


def _seed_bad_frontmatter(k: Path) -> None:
    """A tree clean except for one entry whose frontmatter is missing
    required fields."""
    (k / "global" / "python").mkdir(parents=True)
    (k / "README.md").write_text("# Knowledge\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# Global\n", encoding="utf-8")
    (k / "global" / "python" / "README.md").write_text("# Python\n", encoding="utf-8")
    (k / "global" / "python" / "broken.md").write_text(
        "---\nid: broken\nscope: global\n---\n\n# broken\n\nA real body sentence here.\n",
        encoding="utf-8",
    )


def _archive_packet() -> dict:
    return _packet("s1", proposals=[{"action": "archive_entry", "path": "x.md", "reasoning": "r"}])


def test_review_escalates_overcap_dir(monkeypatch, tmp_path):
    """An over-cap directory the Scholar didn't rebalance is enqueued as an
    ``overcap_dir`` repair task (in-flight), targeting the directory."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    _seed_overcap_dir(k)
    _canned_review(monkeypatch)

    scholar.review([_archive_packet()])

    q = repair_queue.get_queue()
    assert q.pending_count() == 1
    assert q.inflight_count() == 1
    drained = q.drain_pending()
    assert [t.fingerprint for t in drained] == [
        (repair_queue.KIND_OVERCAP_DIR, "global/python", "global/python")
    ]


def test_review_escalates_bad_frontmatter(monkeypatch, tmp_path):
    """An entry with bad frontmatter is enqueued as a ``bad_frontmatter``
    repair task (in-flight), targeting the entry path."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    _seed_bad_frontmatter(k)
    _canned_review(monkeypatch)

    scholar.review([_archive_packet()])

    q = repair_queue.get_queue()
    assert q.pending_count() == 1
    assert q.inflight_count() == 1
    drained = q.drain_pending()
    fp = drained[0].fingerprint
    assert fp == (
        repair_queue.KIND_BAD_FRONTMATTER,
        "global/python/broken.md",
        "global/python/broken.md",
    )


def test_inflight_guard_blocks_overcap_duplicate(monkeypatch, tmp_path):
    """Two consecutive reviews both detecting the same over-cap dir escalate
    it exactly once — the in-flight marker blocks the duplicate (no
    max-attempts cap, the SAME guard wikilinks use)."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    _seed_overcap_dir(k)
    _canned_review(monkeypatch)

    scholar.review([_archive_packet()])  # first detection: enqueues
    scholar.review([_archive_packet()])  # second: marker in-flight → skip

    q = repair_queue.get_queue()
    assert q.inflight_count() == 1
    assert q.pending_count() == 1  # NOT 2


def test_inflight_guard_blocks_bad_frontmatter_duplicate(monkeypatch, tmp_path):
    """Same in-flight discipline for bad frontmatter."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    _seed_bad_frontmatter(k)
    _canned_review(monkeypatch)

    scholar.review([_archive_packet()])
    scholar.review([_archive_packet()])

    q = repair_queue.get_queue()
    assert q.inflight_count() == 1
    assert q.pending_count() == 1


def test_overcap_marker_released_then_reescalates(monkeypatch, tmp_path):
    """A review carrying the over-cap fingerprint releases it on conclusion;
    the next detection re-escalates (never give up)."""
    from agent_mem_daemon import repair_queue

    k = tmp_path / "knowledge"
    _seed_overcap_dir(k)
    _canned_review(monkeypatch)

    fp = (repair_queue.KIND_OVERCAP_DIR, "global/python", "global/python")
    q = repair_queue.get_queue()
    q.enqueue(repair_queue.RepairTask(kind=fp[0], file=fp[1], target=fp[2], context="c"))
    q.drain_pending()  # Librarian took it; marker stays in-flight
    assert q.inflight_count() == 1

    # Scholar reviews the packet carrying the fingerprint. Re-detection is
    # suppressed while the marker is in-flight; the finally then releases it.
    packet = _archive_packet()
    packet["repair_fingerprints"] = [list(fp)]
    scholar.review([packet])
    assert q.inflight_count() == 0
    assert q.pending_count() == 0

    # Next detection re-fires.
    scholar.review([_archive_packet()])
    assert q.inflight_count() == 1
    assert q.pending_count() == 1
