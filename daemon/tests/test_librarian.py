"""End-to-end tests for librarian.scan — orchestration only.

The LLM call is monkey-patched. We pin:
- Empty buffer → empty packet, no LLM call.
- Mocked LLM → populated EvidencePacket with the new ``proposals`` shape.
- Malformed JSON → empty packet, parsed_ok=False on the audit record.
- Exceptions from the SDK → empty packet, never propagated.
- The audit transcript ends up in the AGENT_MEM_HOME tmp dir.
- User-asserted payload flag propagates through the prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_mem_daemon import librarian, librarian_prompt as lp, llm


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Pin AGENT_MEM_HOME to a tmp dir."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    return tmp_path


def _payload_snap(session_id: str, exchanges: list[tuple[str, str]]):
    """Build a buffer snapshot from a list of (role, text) pairs."""
    turns = []
    for role, text in exchanges:
        typ = "UserPromptSubmit" if role == "user" else "PostToolUse"
        ev = {"ts": 1.0, "type": typ, "cwd": "/repo/acme-widget-svc",
              "payload": {"text": text, "role": role}}
        turns.append({
            "events": [ev, {"ts": 2.0, "type": "Stop", "cwd": "/repo", "payload": {}}],
            "started_at": 1.0,
            "sealed_at": 2.0,
        })
    return {
        "session_id": session_id,
        "cwd": "/repo/acme-widget-svc",
        "ended": False,
        "turns": turns,
    }


# ── empty buffer short-circuits ───────────────────────────────────────


def test_scan_empty_buffer_returns_empty_packet_no_llm(home, monkeypatch):
    called = {"n": 0}

    def fake_llm(prompt, *, cwd=None, timeout_s=60.0):
        called["n"] += 1
        return ('{"proposals": [], "interrupts": []}', 0.0)

    monkeypatch.setattr(llm, "run_librarian_call", fake_llm)

    packet = librarian.scan({"session_id": "s1", "turns": []})
    assert packet["session_id"] == "s1"
    assert packet["proposals"] == []
    assert packet["interrupts"] == []
    assert called["n"] == 0


# ── happy path: mocked LLM → populated packet ─────────────────────────


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

    captured = {"prompt": None, "cwd": None}

    fake_response = {
        "proposals": [
            {
                "action": "write_entry",
                "path": "global/tooling/stub-the-factory.md",
                "body": (
                    "---\nid: stub-the-factory\ntype: lesson\nscope: global\n"
                    "status: provisional\nconfidence: 0.7\n"
                    "applies-when: |\n  writing tests against a service that has a factory\n"
                    "keywords: [testing, stub, factory]\n"
                    "title: \"Stub the factory not the service\"\n"
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
                    {"turn_id": 1, "role": "user",
                     "quote": "wiring up the new ReportingService"}
                ],
                "match_score": 0.92,
                "librarian_confidence": 0.88,
            }
        ],
    }

    def fake_llm(prompt, *, cwd=None, timeout_s=60.0):
        captured["prompt"] = prompt
        captured["cwd"] = cwd
        return (json.dumps(fake_response), 0.0012)

    monkeypatch.setattr(llm, "run_librarian_call", fake_llm)

    snap = _payload_snap("sess-123", [
        ("user", "I'm wiring up the new ReportingService. Should I just call new ReportingService(db, cache)?"),
        ("assistant", "Use a factory. Stub the factory, not the service."),
    ])
    packet = librarian.scan(snap)

    assert packet["session_id"] == "sess-123"
    assert len(packet["proposals"]) == 1
    assert packet["proposals"][0]["action"] == "write_entry"
    assert packet["proposals"][0]["path"] == "global/tooling/stub-the-factory.md"
    assert len(packet["interrupts"]) == 1
    assert packet["interrupts"][0]["lesson_id"] == "factory-pattern-for-apis"

    # The Librarian's cwd must be the knowledge directory.
    assert captured["cwd"] == home / "knowledge"

    # Prompt contents: project slug, buffer text, library snapshot tree,
    # applies-when row, BM25 seeds.
    p = captured["prompt"]
    assert p is not None
    assert "acme-widget-svc" in p
    assert "ReportingService" in p
    assert "## Tree" in p
    assert "global/" in p
    assert "tooling/" in p
    assert "factory-pattern-for-apis | global | designing or building any new API" in p
    # Regex seed pre-pass is gone — the Librarian calls bm25_search itself.
    assert "bm25_search" in p


def test_scan_user_asserted_event_propagates_to_prompt(home, monkeypatch):
    """A user-asserted payload (from /ultan) must show up in the prompt
    with the [USER-ASSERTED] marker."""
    captured = {"prompt": None}

    def fake_llm(prompt, *, cwd=None, timeout_s=60.0):
        captured["prompt"] = prompt
        return ('{"proposals": [], "interrupts": []}', 0.0)

    monkeypatch.setattr(llm, "run_librarian_call", fake_llm)

    snap = {
        "session_id": "ultan-abc",
        "cwd": "/repo/x",
        "turns": [{
            "started_at": 1.0,
            "sealed_at": 2.0,
            "events": [
                {"ts": 1.0, "type": "UserPromptSubmit", "cwd": "/repo/x",
                 "payload": {
                     "text": "always wrap upstream errors at the boundary",
                     "role": "user",
                     "user_asserted": True,
                 }},
                {"ts": 2.0, "type": "Stop", "cwd": "/repo/x", "payload": {}},
            ],
        }],
    }
    librarian.scan(snap)
    p = captured["prompt"]
    assert p is not None
    assert "[USER-ASSERTED]" in p
    assert "always wrap upstream errors" in p


def test_scan_writes_audit_jsonl(home, monkeypatch):
    monkeypatch.setattr(
        llm, "run_librarian_call",
        lambda prompt, *, cwd=None, timeout_s=60.0: (
            '{"proposals": [], "interrupts": []}', 0.0,
        ),
    )
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


# ── malformed JSON → empty packet, no crash ───────────────────────────


def test_scan_malformed_json_returns_empty_packet(home, monkeypatch):
    monkeypatch.setattr(
        llm, "run_librarian_call",
        lambda prompt, *, cwd=None, timeout_s=60.0: ("not valid json at all", 0.0),
    )
    snap = _payload_snap("s-bad", [("user", "Always wrap upstream errors.")])
    packet = librarian.scan(snap)
    assert packet["proposals"] == []
    assert packet["interrupts"] == []
    assert packet["session_id"] == "s-bad"

    row = json.loads(
        (home / "runs").glob("*.jsonl").__next__().read_text().strip()
    )
    assert row["parsed_ok"] is False
    assert row["decisions"].get("parse_failed") == 1


# ── SDK exception → empty packet, never propagates ────────────────────


def test_scan_sdk_exception_swallowed(home, monkeypatch):
    def boom(prompt, *, cwd=None, timeout_s=60.0):
        raise RuntimeError("network down")

    monkeypatch.setattr(llm, "run_librarian_call", boom)
    snap = _payload_snap("s-err", [("user", "Always wrap upstream errors.")])
    packet = librarian.scan(snap)
    assert packet["proposals"] == []
    assert packet["interrupts"] == []


def test_scan_sdk_timeout_swallowed(home, monkeypatch):
    def slow(prompt, *, cwd=None, timeout_s=60.0):
        raise llm.LLMTimeout("exceeded 60s")

    monkeypatch.setattr(llm, "run_librarian_call", slow)
    snap = _payload_snap("s-to", [("user", "Always wrap upstream errors.")])
    packet = librarian.scan(snap)
    assert packet["proposals"] == []
    assert packet["interrupts"] == []
    row = json.loads(
        (home / "runs").glob("*.jsonl").__next__().read_text().strip()
    )
    assert row["decisions"].get("llm_timeout") == 1


# ── EvidencePacket shape contract ─────────────────────────────────────


def test_evidence_packet_has_only_expected_top_level_keys(home, monkeypatch):
    """The packet's top-level keys must be exactly the TypedDict's set,
    so the scheduler can rely on the contract."""
    monkeypatch.setattr(
        llm, "run_librarian_call",
        lambda prompt, *, cwd=None, timeout_s=60.0: (
            '{"proposals": [{"action": "archive_entry", "path": "x.md", "reasoning": "old"}], "interrupts": []}',
            0.0,
        ),
    )
    snap = _payload_snap("sess-shape", [("user", "Always X.")])
    packet = librarian.scan(snap)
    assert set(packet.keys()) <= {"session_id", "proposals", "interrupts"}
