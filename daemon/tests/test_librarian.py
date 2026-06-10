"""End-to-end tests for librarian.scan — orchestration only.

The typed agent run is monkey-patched (we stub
``librarian.run_librarian_agent`` to return a typed
``LibrarianProposal`` + cost). We pin:
- Empty buffer → empty packet, no agent run.
- Stubbed agent → populated EvidencePacket with the ``proposals`` shape.
- Agent timeout / exception → empty packet, never propagated.
- The audit transcript ends up in the AGENT_MEM_HOME tmp dir.
- User-asserted payload flag propagates through the prompt.
- Integrity-repair tasks drain into the prompt + fingerprints ride out.

The agent wiring itself (typed output, ModelRetry on malformed proposals,
read-only tools) is exercised in ``test_librarian_validation.py`` and
``test_agent_research.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_mem_daemon import librarian, repair_queue
from agent_mem_daemon._schemas import LibrarianProposal
from agent_mem_daemon.llm import LLMTimeout


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Pin AGENT_MEM_HOME to a tmp dir."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh_repair_queue():
    """Isolate the process-global repair queue so a leftover escalation
    from another module can't leak a ``repair_fingerprints`` key into these
    packets (and so the drain tests start empty)."""
    repair_queue.reset_queue()
    yield
    repair_queue.reset_queue()


def _proposal(obj: dict) -> LibrarianProposal:
    """Build a validated LibrarianProposal from a plain dict."""
    return LibrarianProposal.model_validate(obj)


def _stub_agent(monkeypatch, *, proposal: LibrarianProposal, cost: float = 0.0, captured=None):
    """Patch the agent runner to return ``(proposal, cost)`` and capture the
    prompt + knowledge_dir it was called with."""

    def _run(prompt, knowledge_dir, *, timeout_s=600.0, user_asserted_turns=0):
        if captured is not None:
            captured["prompt"] = prompt
            captured["knowledge_dir"] = knowledge_dir
        return proposal, cost

    monkeypatch.setattr(librarian, "run_librarian_agent", _run)


def _payload_snap(session_id: str, exchanges: list[tuple[str, str]]):
    """Build a buffer snapshot from a list of (role, text) pairs."""
    turns = []
    for role, text in exchanges:
        typ = "UserPromptSubmit" if role == "user" else "PostToolUse"
        ev = {
            "ts": 1.0,
            "type": typ,
            "cwd": "/repo/acme-widget-svc",
            "payload": {"text": text, "role": role},
        }
        turns.append(
            {
                "events": [ev, {"ts": 2.0, "type": "Stop", "cwd": "/repo", "payload": {}}],
                "started_at": 1.0,
                "sealed_at": 2.0,
            }
        )
    return {
        "session_id": session_id,
        "cwd": "/repo/acme-widget-svc",
        "ended": False,
        "turns": turns,
    }


# ── empty buffer short-circuits ───────────────────────────────────────


def test_scan_empty_buffer_returns_empty_packet_no_llm(home, monkeypatch):
    called = {"n": 0}

    def _run(prompt, knowledge_dir, *, timeout_s=600.0, user_asserted_turns=0):
        called["n"] += 1
        return _proposal({"proposals": [], "interrupts": []}), 0.0

    monkeypatch.setattr(librarian, "run_librarian_agent", _run)

    packet = librarian.scan({"session_id": "s1", "turns": []})
    assert packet["session_id"] == "s1"
    assert packet["proposals"] == []
    assert packet["interrupts"] == []
    assert called["n"] == 0


# ── happy path: stubbed agent → populated packet ──────────────────────


def _seed_knowledge(home: Path):
    """Drop a single confirmed entry + an index.md."""
    k = home / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "global" / "tooling" / "factory-pattern-for-apis.md").write_text(
        "---\n"
        "id: factory-pattern-for-apis\n"
        "scope: global\n"
        "status: confirmed\n"
        "applies-when: |\n"
        "  designing or building any new API\n"
        "  decisions about how clients construct service objects\n"
        "keywords: [factory, paradigm, api, construction]\n"
        "---\n\n"
        "# Use factory pattern\nFactories let us swap implementations.\n",
        encoding="utf-8",
    )
    (k / "index.md").write_text(
        "# Knowledge Base Index\n\n"
        "| Article | Summary | Updated |\n"
        "|---|---|---|\n"
        "| [[global/tooling/factory-pattern-for-apis]] | Use factory pattern | 2026-05-19 |\n",
        encoding="utf-8",
    )
    (k / "README.md").write_text("# agent-mem knowledge\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# Global lessons\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# Tooling\n", encoding="utf-8")


def test_scan_happy_path_proposals_and_interrupts(home, monkeypatch):
    _seed_knowledge(home)

    captured = {}

    proposal = _proposal(
        {
            "proposals": [
                {
                    "action": "write_entry",
                    "path": "global/tooling/stub-the-factory.md",
                    "body": (
                        "---\nid: stub-the-factory\ntype: lesson\nscope: global\n"
                        "status: provisional\nconfidence: 0.7\n"
                        "applies-when: |\n  writing tests against a service that has a factory\n"
                        "keywords: [testing, stub, factory]\n"
                        'title: "Stub the factory not the service"\n'
                        "created: 2026-05-19\nupdated: 2026-05-19\n"
                        "fired: 0\nfired-helpful: 0\nsources:\n  - daily/2026-05-19.md\n"
                        "---\n\n# Stub the factory\nBody.\n"
                    ),
                    "reasoning": "buffer turn [2] said 'Stub the factory, not the service.'",
                }
            ],
            "interrupts": [
                {
                    "lesson_id": "factory-pattern-for-apis",
                    "lesson_path": "global/tooling/factory-pattern-for-apis.md",
                    "matching_applies_when": "designing or building any new API",
                    "evidence": [
                        {
                            "turn_id": 1,
                            "role": "user",
                            "quote": "wiring up the new ReportingService",
                        }
                    ],
                    "match_score": 0.92,
                    "librarian_confidence": 0.88,
                }
            ],
        }
    )
    _stub_agent(monkeypatch, proposal=proposal, cost=0.0012, captured=captured)

    snap = _payload_snap(
        "sess-123",
        [
            (
                "user",
                "I'm wiring up the new ReportingService. Should I just call new ReportingService(db, cache)?",
            ),
            ("assistant", "Use a factory. Stub the factory, not the service."),
        ],
    )
    packet = librarian.scan(snap)

    assert packet["session_id"] == "sess-123"
    assert len(packet["proposals"]) == 1
    assert packet["proposals"][0]["action"] == "write_entry"
    assert packet["proposals"][0]["path"] == "global/tooling/stub-the-factory.md"
    assert len(packet["interrupts"]) == 1
    assert packet["interrupts"][0]["lesson_id"] == "factory-pattern-for-apis"

    # The Librarian's knowledge_dir must be the knowledge directory.
    assert captured["knowledge_dir"] == home / "knowledge"

    # Prompt contents: project slug, buffer text, library snapshot tree,
    # applies-when row, the in-process research tools.
    p = captured["prompt"]
    assert p is not None
    assert "acme-widget-svc" in p
    assert "ReportingService" in p
    assert "## Tree" in p
    assert "global/" in p
    assert "tooling/" in p
    assert "factory-pattern-for-apis | global | designing or building any new API" in p
    # The Librarian researches dedup with its own bm25_search tool.
    assert "bm25_search" in p


def test_scan_user_asserted_event_propagates_to_prompt(home, monkeypatch):
    """A user-asserted payload (from /ultan) must show up in the prompt
    with the [USER-ASSERTED] marker."""
    captured = {}
    _stub_agent(
        monkeypatch, proposal=_proposal({"proposals": [], "interrupts": []}), captured=captured
    )

    snap = {
        "session_id": "ultan-abc",
        "cwd": "/repo/x",
        "turns": [
            {
                "started_at": 1.0,
                "sealed_at": 2.0,
                "events": [
                    {
                        "ts": 1.0,
                        "type": "UserPromptSubmit",
                        "cwd": "/repo/x",
                        "payload": {
                            "text": "always wrap upstream errors at the boundary",
                            "role": "user",
                            "user_asserted": True,
                        },
                    },
                    {"ts": 2.0, "type": "Stop", "cwd": "/repo/x", "payload": {}},
                ],
            }
        ],
    }
    librarian.scan(snap)
    p = captured["prompt"]
    assert p is not None
    assert "[USER-ASSERTED]" in p
    assert "always wrap upstream errors" in p


def test_scan_writes_audit_jsonl(home, monkeypatch):
    _stub_agent(monkeypatch, proposal=_proposal({"proposals": [], "interrupts": []}))
    snap = _payload_snap("sess-audit", [("user", "Always wrap upstream errors.")])
    librarian.scan(snap)

    runs_dir = home / "runs"
    jsonl_files = list(runs_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1
    lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["role"] == "librarian"
    assert row["session_id"] == "sess-audit"
    assert row["parsed_ok"] is True
    assert row["decisions"]["proposals"] == 0
    # `seeds_proposed` is gone — no regex pre-pass anymore; the
    # Librarian uses its own BM25 tool when it wants dedup checks.
    assert "seeds_proposed" not in row["decisions"]


# ── agent exception → empty packet, never propagates ──────────────────


def test_scan_agent_exception_swallowed(home, monkeypatch):
    def boom(prompt, knowledge_dir, *, timeout_s=600.0, user_asserted_turns=0):
        raise RuntimeError("provider down")

    monkeypatch.setattr(librarian, "run_librarian_agent", boom)
    snap = _payload_snap("s-err", [("user", "Always wrap upstream errors.")])
    packet = librarian.scan(snap)
    assert packet["proposals"] == []
    assert packet["interrupts"] == []

    row = json.loads((home / "runs").glob("*.jsonl").__next__().read_text().strip())
    assert row["parsed_ok"] is False
    assert row["decisions"].get("llm_error") == 1


def test_scan_agent_timeout_swallowed(home, monkeypatch):
    def slow(prompt, knowledge_dir, *, timeout_s=600.0, user_asserted_turns=0):
        raise LLMTimeout("exceeded 600s")

    monkeypatch.setattr(librarian, "run_librarian_agent", slow)
    snap = _payload_snap("s-to", [("user", "Always wrap upstream errors.")])
    packet = librarian.scan(snap)
    assert packet["proposals"] == []
    assert packet["interrupts"] == []
    row = json.loads((home / "runs").glob("*.jsonl").__next__().read_text().strip())
    assert row["decisions"].get("llm_timeout") == 1


# ── EvidencePacket shape contract ─────────────────────────────────────


def test_evidence_packet_has_only_expected_top_level_keys(home, monkeypatch):
    """The packet's top-level keys must be exactly the TypedDict's set,
    so the scheduler can rely on the contract."""
    _stub_agent(
        monkeypatch,
        proposal=_proposal(
            {
                "proposals": [
                    {"action": "archive_entry", "path": "global/x/old.md", "reasoning": "old"}
                ],
                "interrupts": [],
            }
        ),
    )
    snap = _payload_snap("sess-shape", [("user", "Always X.")])
    packet = librarian.scan(snap)
    assert set(packet.keys()) <= {"session_id", "proposals", "interrupts"}


# ── Integrity-repair task routing ─────────────────────────────────────


def _queue_task(target: str = "global/ghost/missing") -> tuple:
    task = repair_queue.RepairTask(
        kind=repair_queue.KIND_BROKEN_WIKILINK,
        file="global/python/foo.md",
        target=target,
        context="…[[ghost]]…",
    )
    repair_queue.get_queue().enqueue(task)
    return task.fingerprint


def test_scan_drains_repair_task_into_prompt_and_attaches_fingerprint(home, monkeypatch):
    """A pending repair task is drained, rendered into the prompt, and its
    fingerprint is attached to the emitted packet so the Scholar can later
    release the in-flight marker."""
    fp = _queue_task()
    captured = {}
    _stub_agent(
        monkeypatch, proposal=_proposal({"proposals": [], "interrupts": []}), captured=captured
    )

    snap = _payload_snap("s-rep", [("user", "Always X.")])
    packet = librarian.scan(snap)

    assert captured["prompt"] is not None
    assert "INTEGRITY-REPAIR TASKS" in captured["prompt"]
    assert "target: global/ghost/missing" in captured["prompt"]
    assert packet.get("repair_fingerprints") == [fp]
    # Drained — no longer pending, but still in-flight until the Scholar
    # concludes the review.
    assert repair_queue.get_queue().pending_count() == 0
    assert repair_queue.get_queue().inflight_count() == 1


def test_scan_attaches_fingerprints_even_on_agent_error(home, monkeypatch):
    """Leak fix: a Librarian that drained a task but then errored must STILL
    hand the fingerprint to the Scholar (on the error/empty packet) so the
    marker can be released — otherwise it stays in-flight forever."""
    fp = _queue_task()

    def boom(prompt, knowledge_dir, *, timeout_s=600.0, user_asserted_turns=0):
        raise RuntimeError("provider down")

    monkeypatch.setattr(librarian, "run_librarian_agent", boom)
    snap = _payload_snap("s-rep-err", [("user", "Always X.")])
    packet = librarian.scan(snap)

    assert packet["proposals"] == []
    assert packet.get("repair_fingerprints") == [fp]


def test_scan_runs_agent_for_repair_task_even_with_empty_buffer(home, monkeypatch):
    """A pending repair task is itself a reason to invoke the agent — the
    empty-buffer short-circuit must NOT skip it."""
    fp = _queue_task()
    called = {"n": 0}

    def _run(prompt, knowledge_dir, *, timeout_s=600.0, user_asserted_turns=0):
        called["n"] += 1
        return _proposal({"proposals": [], "interrupts": []}), 0.0

    monkeypatch.setattr(librarian, "run_librarian_agent", _run)
    packet = librarian.scan({"session_id": "s-empty", "turns": []})

    assert called["n"] == 1  # agent ran despite empty buffer
    assert packet.get("repair_fingerprints") == [fp]


def test_scan_no_repair_tasks_attaches_no_fingerprints(home, monkeypatch):
    _stub_agent(monkeypatch, proposal=_proposal({"proposals": [], "interrupts": []}))
    snap = _payload_snap("s-clean", [("user", "Always X.")])
    packet = librarian.scan(snap)
    assert "repair_fingerprints" not in packet
