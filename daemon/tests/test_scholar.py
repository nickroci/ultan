"""Scholar orchestration tests for the new gatekeeper architecture.

No real LLM calls — we monkeypatch ``llm.run_scholar_call`` to return
canned responses. ``AGENT_MEM_HOME`` is redirected to a tmp dir per
test so the InvocationRecord audit writes don't leak.

Coverage:
  - Empty packet list → no-op.
  - All-empty packets → no SDK call.
  - Happy path: approval flow populates decisions counters and
    propagates nudges to pending-nudges.md.
  - Veto path: SDK returns all vetoes → no files written, decisions
    counters reflect veto count.
  - Malformed JSON → does not crash, parsed_ok=False.
  - SDK exception / timeout → swallowed, batch dropped.
  - Heterogeneous session ids → session_id="batch" on the audit row.
  - Invariants checker runs after the SDK call and surfaces violations
    in the decisions counter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import pytest

from agent_mem_daemon import scholar
from agent_mem_daemon.runs import InvocationRecord


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


def _make_canned(response_text: str, cost: float = 0.01):
    def _stub(prompt, *, cwd, timeout_s):
        return response_text, cost

    return _stub


# ── Pre-flight short-circuits ─────────────────────────────────────────


def test_review_empty_packet_list_is_noop(monkeypatch, caplog):
    calls = []

    def _fail(*a, **kw):
        calls.append(1)
        raise AssertionError("SDK must not be called")

    monkeypatch.setattr(scholar, "run_scholar_call", _fail)
    scholar.review([])
    assert calls == []


def test_review_skips_when_all_packets_empty(monkeypatch):
    sdk_calls: List[int] = []

    def _fail(*a, **kw):
        sdk_calls.append(1)
        raise AssertionError("SDK must not be called for empty batch")

    monkeypatch.setattr(scholar, "run_scholar_call", _fail)
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

    def _stub(prompt, *, cwd, timeout_s):
        sdk_calls.append(1)
        return json.dumps({"decisions": [], "interrupts_processed": []}), 0.01

    monkeypatch.setattr(scholar, "run_scholar_call", _stub)
    scholar.review(
        [
            _packet(
                "s1", proposals=[{"action": "archive_entry", "path": "a.md", "reasoning": "stale"}]
            )
        ]
    )
    assert sdk_calls == [1]


# ── Approve path ──────────────────────────────────────────────────────


def test_review_happy_path_approve_populates_decisions(monkeypatch, tmp_path):
    canned = {
        "decisions": [
            {"action_index": 0, "decision": "approve", "veto_reason": ""},
            {"action_index": 1, "decision": "veto", "veto_reason": "thin evidence"},
        ],
        "interrupts_processed": [
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
    }
    monkeypatch.setattr(
        scholar,
        "run_scholar_call",
        _make_canned(json.dumps(canned), cost=0.03),
    )

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
                    "body": "x",
                    "reasoning": "r1",
                },
                {
                    "action": "write_entry",
                    "path": "global/tooling/y.md",
                    "body": "y",
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
    assert rec.decisions.get("approve") == 1
    assert rec.decisions.get("veto") == 1
    assert rec.decisions.get("nudge") == 1
    assert rec.decisions.get("interrupt-veto") == 1
    assert rec.decisions.get("nudges_written") == 1

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


# ── Veto-everything path ──────────────────────────────────────────────


def test_review_all_vetoes_no_files_written(monkeypatch, tmp_path):
    """Canned ScholarReview vetoes every proposal. No nudge file,
    no entries written, and the audit row records veto counts."""
    canned = {
        "decisions": [
            {"action_index": 0, "decision": "veto", "veto_reason": "thin evidence"},
            {"action_index": 1, "decision": "veto", "veto_reason": "duplicate"},
        ],
        "interrupts_processed": [],
    }
    monkeypatch.setattr(
        scholar,
        "run_scholar_call",
        _make_canned(json.dumps(canned), cost=0.02),
    )

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
                {"action": "write_entry", "path": "global/a.md", "body": "x", "reasoning": "r"},
                {"action": "write_entry", "path": "global/b.md", "body": "y", "reasoning": "r"},
            ],
        )
    ]
    scholar.review(packets)

    rec = records[0]
    assert rec.decisions.get("veto") == 2
    assert rec.decisions.get("approve", 0) == 0
    # No nudge file.
    assert not (tmp_path / "pending-nudges.md").exists()
    # No knowledge entries — the canned response did not invoke any Write tools.
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

    canned = {
        "decisions": [{"action_index": 0, "decision": "veto", "veto_reason": "n/a"}],
        "interrupts_processed": [],
    }
    monkeypatch.setattr(
        scholar,
        "run_scholar_call",
        _make_canned(json.dumps(canned), cost=0.01),
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

    assert records[0].decisions.get("invariant_violations", 0) >= 1


def test_review_invariants_clean_does_not_log_violation(monkeypatch, tmp_path):
    """No violations → no ``invariant_violations`` key on the audit row."""
    k = tmp_path / "knowledge"
    k.mkdir()
    # Empty tree — invariants checker returns [].
    canned = {"decisions": [], "interrupts_processed": []}
    monkeypatch.setattr(
        scholar,
        "run_scholar_call",
        _make_canned(json.dumps(canned), cost=0.01),
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
    assert "invariant_violations" not in records[0].decisions


# ── Heterogeneous session ids ─────────────────────────────────────────


def test_review_heterogeneous_session_id_marked_batch(monkeypatch):
    canned = {"decisions": [], "interrupts_processed": []}
    monkeypatch.setattr(
        scholar,
        "run_scholar_call",
        _make_canned(json.dumps(canned)),
    )
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


# ── Malformed JSON ────────────────────────────────────────────────────


def test_review_malformed_json_does_not_crash(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        scholar,
        "run_scholar_call",
        _make_canned("definitely not json at all"),
    )
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
    assert not (tmp_path / "pending-nudges.md").exists()
    assert any("parse" in rec.message.lower() for rec in caplog.records)


def test_review_sdk_exception_is_swallowed(monkeypatch, tmp_path, caplog):
    def _raise(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(scholar, "run_scholar_call", _raise)
    caplog.set_level(logging.WARNING)
    scholar.review(
        [_packet("s1", proposals=[{"action": "archive_entry", "path": "a.md", "reasoning": "r"}])]
    )


def test_review_sdk_timeout_is_swallowed(monkeypatch, tmp_path, caplog):
    from agent_mem_daemon.llm import LLMTimeout

    def _timeout(*a, **kw):
        raise LLMTimeout("too slow")

    monkeypatch.setattr(scholar, "run_scholar_call", _timeout)
    caplog.set_level(logging.WARNING)
    scholar.review([_packet("s1", interrupts=[{"lesson_id": "x"}])])
    assert any("timeout" in rec.message.lower() for rec in caplog.records)


# ── End-to-end via mocked LLM: turn → proposal → approval ──────────


def test_review_batch_session_with_no_session_ids_marked_batch(monkeypatch):
    """If all packets lack a session_id, the audit row gets ``session_id="batch"``."""
    canned = {"decisions": [], "interrupts_processed": []}
    monkeypatch.setattr(scholar, "run_scholar_call", _make_canned(json.dumps(canned)))
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
        "run_scholar_call",
        _make_canned(json.dumps({"decisions": [], "interrupts_processed": []})),
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
        "run_scholar_call",
        _make_canned(json.dumps({"decisions": [], "interrupts_processed": []})),
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

    canned = {
        "decisions": [{"action_index": 0, "decision": "approve", "veto_reason": ""}],
        "interrupts_processed": [
            {
                "lesson_id": "x",
                "lesson_path": "x",
                "action": "approve",
                "text": "y",
                "reason": "ok",
            }
        ],
    }
    monkeypatch.setattr(scholar, "run_scholar_call", _make_canned(json.dumps(canned)))

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
        "run_scholar_call",
        _make_canned(json.dumps({"decisions": [], "interrupts_processed": []})),
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
        "run_scholar_call",
        _make_canned(json.dumps({"decisions": [], "interrupts_processed": []})),
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
        "run_scholar_call",
        _make_canned(json.dumps({"decisions": [], "interrupts_processed": []})),
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
        "run_scholar_call",
        _make_canned(json.dumps({"decisions": [], "interrupts_processed": []})),
    )

    def _boom(kdir):
        raise RuntimeError("simulated invariants failure")

    monkeypatch.setattr(scholar_prompt, "check_invariants", _boom)
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
    """Integration-flavoured: the Scholar's SDK call would normally use
    the Write tool to create the file. We simulate that by writing the
    file ourselves from the stub, then assert it survives review +
    invariants. This pins the *control flow* — the file ends up where
    we say it does and the audit row reflects an approve."""
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
        "---\n\n# Stub the factory\nBody.\n"
    )

    def _stub_writes_file(prompt, *, cwd, timeout_s):
        # Simulate the Scholar's Write tool call.
        target_path.write_text(full_entry, encoding="utf-8")
        canned = {
            "decisions": [{"action_index": 0, "decision": "approve", "veto_reason": ""}],
            "interrupts_processed": [],
        }
        return json.dumps(canned), 0.01

    monkeypatch.setattr(scholar, "run_scholar_call", _stub_writes_file)

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

    assert target_path.exists()
    assert records[0].decisions.get("approve") == 1
    # No invariant violations introduced.
    assert records[0].decisions.get("invariant_violations", 0) == 0
