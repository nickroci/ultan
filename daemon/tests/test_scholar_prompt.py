"""Tests for the Scholar prompt assembly + JSON parsing + nudge writing
+ hierarchy invariants checker."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent_mem_daemon import scholar_prompt

# ── prompt assembly ───────────────────────────────────────────────────


def test_build_prompt_includes_proposals_and_library_snapshot():
    packets = [
        {
            "session_id": "s1",
            "cwd": "/repo/a",
            "proposals": [
                {
                    "action": "write_entry",
                    "path": "global/tooling/uv-basics.md",
                    "body": "---\nid: uv-basics\n---\n# uv basics\n",
                    "reasoning": "buffer turn [2] said 'always uv sync'",
                }
            ],
            "interrupts": [],
        }
    ]
    prompt = scholar_prompt.build_prompt(
        packets,
        now=datetime(2026, 5, 19, 15, 0, 0, tzinfo=timezone.utc),
        library_snapshot="## Tree\n```\nglobal/\n```",
    )
    assert "global/tooling/uv-basics.md" in prompt
    assert "2026-05-19T15:00:00" in prompt
    assert "## Tree" in prompt
    # Veto language and final-output language must survive into the rendered prompt.
    assert "VETO" in prompt
    assert "FINAL OUTPUT" in prompt
    # Action vocabulary must be present.
    for action in (
        "write_entry",
        "merge_entries",
        "split_folder",
        "archive_entry",
        "add_wikilink",
        "update_readme",
        "move_entry",
        "update_entry",
    ):
        assert action in prompt, f"action {action!r} missing from Scholar prompt"


def test_build_prompt_annotates_action_indices():
    """Each proposal in the rendered JSON carries an ``_action_index``
    so the Scholar can reference them in its final response."""
    packets = [
        {
            "session_id": "s1",
            "proposals": [
                {"action": "archive_entry", "path": "a.md", "reasoning": "r1"},
                {"action": "archive_entry", "path": "b.md", "reasoning": "r2"},
            ],
            "interrupts": [],
        },
        {
            "session_id": "s2",
            "proposals": [
                {"action": "archive_entry", "path": "c.md", "reasoning": "r3"},
            ],
            "interrupts": [],
        },
    ]
    prompt = scholar_prompt.build_prompt(
        packets,
        library_snapshot="(empty)",
    )
    # Flat indexing across packets: 0, 1, 2.
    assert '"_action_index": 0' in prompt
    assert '"_action_index": 1' in prompt
    assert '"_action_index": 2' in prompt


def test_build_prompt_handles_empty_proposals():
    packets = [{"session_id": "s1", "proposals": [], "interrupts": []}]
    prompt = scholar_prompt.build_prompt(packets, library_snapshot="(empty)")
    assert "s1" in prompt


# ── response parsing ──────────────────────────────────────────────────


def test_parse_response_plain_json():
    text = json.dumps(
        {
            "decisions": [
                {"action_index": 0, "decision": "approve", "veto_reason": ""},
                {"action_index": 1, "decision": "veto", "veto_reason": "thin evidence"},
            ],
            "interrupts_processed": [],
        }
    )
    parsed, ok = scholar_prompt.parse_response(text)
    assert ok is True
    assert parsed is not None
    assert parsed["decisions"][0]["decision"] == "approve"
    assert parsed["decisions"][1]["decision"] == "veto"
    assert parsed["decisions"][1]["veto_reason"] == "thin evidence"


def test_parse_response_fenced_json():
    payload = {"decisions": [], "interrupts_processed": []}
    text = "Some reasoning here.\n\n```json\n" + json.dumps(payload) + "\n```"
    parsed, ok = scholar_prompt.parse_response(text)
    assert ok is True
    assert parsed == payload


def test_parse_response_trailing_object_fallback():
    text = 'Prose blathering ...\n{"decisions": [], "interrupts_processed": []}'
    parsed, ok = scholar_prompt.parse_response(text)
    assert ok is True
    assert parsed is not None
    assert parsed["decisions"] == []


def test_parse_response_malformed():
    parsed, ok = scholar_prompt.parse_response("definitely not json {oops")
    assert ok is False
    assert parsed is None


def test_parse_response_empty():
    parsed, ok = scholar_prompt.parse_response("")
    assert ok is False
    assert parsed is None


def test_parse_response_unknown_decision_rejected_via_default():
    """Pydantic literal: any decision other than 'approve'/'veto' falls
    back to the default ('veto'). Better to drop than to silently
    approve."""
    text = json.dumps(
        {
            "decisions": [{"action_index": 0, "decision": "maybe"}],
            "interrupts_processed": [],
        }
    )
    parsed, ok = scholar_prompt.parse_response(text)
    # Should reject because Literal["approve","veto"] doesn't accept
    # "maybe" — Pydantic raises ValidationError.
    assert ok is False


# ── decisions summary ────────────────────────────────────────────────


def test_summarise_decisions_counts_actions():
    parsed = {
        "decisions": [
            {"action_index": 0, "decision": "approve"},
            {"action_index": 1, "decision": "approve"},
            {"action_index": 2, "decision": "veto"},
            {"action_index": 3, "decision": "veto"},
        ],
        "interrupts_processed": [
            {"action": "approve"},
            {"action": "veto"},
            {"action": "veto"},
        ],
    }
    counters = scholar_prompt.summarise_decisions(parsed)
    assert counters == {
        "approve": 2,
        "veto": 2,
        "nudge": 1,
        "interrupt-veto": 2,
    }


def test_summarise_decisions_handles_none_and_garbage():
    assert scholar_prompt.summarise_decisions(None) == {}
    assert scholar_prompt.summarise_decisions({"decisions": "nope"}) == {}
    assert scholar_prompt.summarise_decisions({}) == {}


# ── nudge file writing + round-trip ──────────────────────────────────


def test_append_nudges_writes_blocks(tmp_path: Path):
    target = tmp_path / "pending-nudges.md"
    parsed = {
        "decisions": [],
        "interrupts_processed": [
            {
                "lesson_id": "factory-pattern-for-apis",
                "lesson_path": "global/tooling/factory-pattern-for-apis",
                "action": "approve",
                "text": "Memory: use factory pattern for new APIs.",
                "reason": "good match",
            },
            {
                "lesson_id": "no-mock-db",
                "lesson_path": "global/testing/no-mock-db",
                "action": "veto",
                "reason": "not actionable",
            },
            {
                "lesson_id": "another",
                "lesson_path": "global/conventions/another",
                "action": "approve",
                "text": "Memory: another nudge text here.",
                "reason": "ok",
            },
        ],
    }
    written = scholar_prompt.append_nudges_from_response(
        parsed,
        now=datetime(2026, 5, 19, 15, 0, 0, tzinfo=timezone.utc),
        path=target,
    )
    assert len(written) == 2
    body = target.read_text(encoding="utf-8")
    assert "Memory: use factory pattern for new APIs." in body
    assert "Memory: another nudge text here." in body
    assert "not actionable" not in body
    parsed_back = scholar_prompt.parse_nudges_file(body)
    assert len(parsed_back) == 2
    assert parsed_back[0]["lesson"] == "global/tooling/factory-pattern-for-apis"
    assert parsed_back[0]["text"] == "Memory: use factory pattern for new APIs."
    assert parsed_back[1]["lesson"] == "global/conventions/another"


def test_append_nudges_appends_not_overwrites(tmp_path: Path):
    target = tmp_path / "pending-nudges.md"
    target.write_text(
        "---\nid: pre\ncreated: 2026-01-01T00:00:00+00:00\nlesson: pre/lesson\n---\nPre-existing nudge.\n",
        encoding="utf-8",
    )
    parsed = {
        "interrupts_processed": [
            {
                "lesson_id": "x",
                "lesson_path": "x/path",
                "action": "approve",
                "text": "Newly written nudge.",
            }
        ],
    }
    scholar_prompt.append_nudges_from_response(parsed, path=target)
    body = target.read_text(encoding="utf-8")
    assert "Pre-existing nudge." in body
    assert "Newly written nudge." in body


def test_append_nudges_skips_malformed_approval(tmp_path: Path):
    target = tmp_path / "pending-nudges.md"
    parsed = {
        "interrupts_processed": [
            {"lesson_id": "x", "action": "approve"},
            {"lesson_id": "y", "lesson_path": "y/path", "action": "approve"},
        ],
    }
    written = scholar_prompt.append_nudges_from_response(parsed, path=target)
    assert written == []
    assert not target.exists() or target.read_text(encoding="utf-8") == ""


def test_append_nudges_with_no_approvals_writes_nothing(tmp_path: Path):
    target = tmp_path / "pending-nudges.md"
    parsed = {
        "interrupts_processed": [
            {"lesson_id": "x", "lesson_path": "x", "action": "veto", "reason": "nope"},
        ],
    }
    written = scholar_prompt.append_nudges_from_response(parsed, path=target)
    assert written == []
    assert not target.exists()


def test_parse_nudges_file_empty():
    assert scholar_prompt.parse_nudges_file("") == []
    assert scholar_prompt.parse_nudges_file("   \n  \n") == []


def test_parse_nudges_file_roundtrip_with_three_blocks():
    body = (
        "---\nid: a1\ncreated: 2026-05-19T00:00:00+00:00\nlesson: l/a\n---\nfirst nudge\n"
        "---\nid: b2\ncreated: 2026-05-19T00:00:01+00:00\nlesson: l/b\n---\nsecond nudge\n"
        "---\nid: c3\ncreated: 2026-05-19T00:00:02+00:00\nlesson: l/c\n---\nthird nudge\n"
    )
    parsed = scholar_prompt.parse_nudges_file(body)
    assert [n["id"] for n in parsed] == ["a1", "b2", "c3"]
    assert [n["text"] for n in parsed] == ["first nudge", "second nudge", "third nudge"]


# ── hierarchy invariants ─────────────────────────────────────────────


def _valid_frontmatter(*, id_: str, scope: str = "global") -> str:
    return (
        "---\n"
        f"id: {id_}\n"
        "type: lesson\n"
        f"scope: {scope}\n"
        "status: provisional\n"
        "confidence: 0.7\n"
        "applies-when: |\n"
        "  designing things\n"
        "keywords: [a, b, c]\n"
        f'title: "{id_}"\n'
        "created: 2026-05-19\n"
        "updated: 2026-05-19\n"
        "fired: 0\n"
        "fired-helpful: 0\n"
        "sources:\n"
        "  - manual\n"
        "---\n\n# title\n\nThe rule and a sentence explaining why it applies in practice.\n"
    )


def test_check_invariants_clean_tree(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics"),
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert out == [], f"expected clean, got: {out}"


def test_check_invariants_missing_readme(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics"),
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    # Both the global/ dir and global/tooling/ dir lack READMEs.
    assert any(v == "missing README.md in global/tooling/" for v in out), out
    assert any(v == "missing README.md in global/" for v in out), out


def test_check_invariants_too_many_entries(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    for i in range(6):
        (k / "global" / "tooling" / f"e{i}.md").write_text(
            _valid_frontmatter(id_=f"e{i}"),
            encoding="utf-8",
        )
    out = scholar_prompt.check_invariants(k)
    assert any("global/tooling/" in v and "6 entry" in v for v in out)


def test_check_invariants_broken_wikilink(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics") + "See [[global/missing/no-such-entry]]\n",
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert any("broken wikilink" in v and "no-such-entry" in v for v in out)


def test_check_invariants_archive_wikilink_allowed(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics") + "Old: [[_archive/global/tooling/old]]\n",
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    # Archive links are intentional per AGENTS.md §1.2 — should NOT
    # be flagged.
    assert not any("_archive" in v for v in out)


def test_check_invariants_daily_wikilink_allowed(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics") + "Source: [[daily/2026-05-19]]\n",
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert out == []


def test_check_invariants_missing_frontmatter_fields(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    (k / "global" / "tooling" / "incomplete.md").write_text(
        "---\nid: incomplete\nscope: global\n---\n# x\n",
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert any("missing frontmatter fields" in v and "incomplete.md" in v for v in out)


def test_check_invariants_excludes_archive_tree(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "_archive" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics"),
        encoding="utf-8",
    )
    # Archive entry with bad frontmatter — must NOT be flagged.
    (k / "_archive" / "tooling" / "old.md").write_text(
        "no frontmatter here",
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert out == []


def test_check_invariants_broken_wikilink_in_readme(tmp_path: Path):
    # Regression: the old check excluded README.md from wikilink scanning,
    # which let `[[error-handling-exception-preservation]]` slip past in
    # a live run. README/index/log links MUST be validated.
    k = tmp_path / "knowledge"
    (k / "projects" / "daemon").mkdir(parents=True)
    (k / "README.md").write_text("# root\n[[daemon/]]\n", encoding="utf-8")
    (k / "projects" / "README.md").write_text(
        "# projects\n[[daemon/]]\n",
        encoding="utf-8",
    )
    (k / "projects" / "daemon" / "README.md").write_text(
        "# daemon\n[[error-handling-exception-preservation]]\n",  # broken!
        encoding="utf-8",
    )
    (k / "projects" / "daemon" / "error-handling.md").write_text(
        _valid_frontmatter(id_="error-handling", scope="project:daemon"),
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert any(
        "broken wikilink" in v and "error-handling-exception-preservation" in v for v in out
    ), f"expected broken-wikilink error, got: {out}"


def test_check_invariants_sibling_wikilink_resolves(tmp_path: Path):
    # README links to a sibling entry by its bare slug (no folder prefix).
    # That's a common shape and should resolve.
    k = tmp_path / "knowledge"
    (k / "projects" / "daemon").mkdir(parents=True)
    (k / "README.md").write_text("# root\n", encoding="utf-8")
    (k / "projects" / "README.md").write_text("# projects\n", encoding="utf-8")
    (k / "projects" / "daemon" / "README.md").write_text(
        "# daemon\n[[error-handling]]\n",
        encoding="utf-8",
    )
    (k / "projects" / "daemon" / "error-handling.md").write_text(
        _valid_frontmatter(id_="error-handling", scope="project:daemon"),
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert out == [], f"sibling wikilink should resolve, got: {out}"


def test_check_invariants_empty_body_flagged(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    # Frontmatter is complete but body is just a one-word heading.
    (k / "global" / "tooling" / "stub.md").write_text(
        _valid_frontmatter(id_="stub").rsplit("---", 1)[0] + "---\n\n# x\n",
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert any("empty or trivial" in v and "stub.md" in v for v in out), out


def test_check_invariants_scope_path_mismatch(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "projects" / "daemon").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "projects" / "README.md").write_text("# p\n", encoding="utf-8")
    (k / "projects" / "daemon" / "README.md").write_text("# d\n", encoding="utf-8")
    # Entry says scope=global but lives under projects/daemon/.
    (k / "projects" / "daemon" / "misfiled.md").write_text(
        _valid_frontmatter(id_="misfiled", scope="global"),
        encoding="utf-8",
    )
    out = scholar_prompt.check_invariants(k)
    assert any("scope/path mismatch" in v and "misfiled" in v for v in out), out


def test_reconcile_creates_missing_readmes(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    (k / "global" / "python" / "use-uv.md").write_text(
        _valid_frontmatter(id_="use-uv"),
        encoding="utf-8",
    )
    # No READMEs anywhere.
    changes = scholar_prompt.reconcile_readmes(k)
    # Should have created READMEs at: knowledge/, global/, global/python/
    assert (k / "README.md").exists()
    assert (k / "global" / "README.md").exists()
    assert (k / "global" / "python" / "README.md").exists()
    assert len(changes) == 3


def test_reconcile_updates_children_section(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    (k / "global" / "python" / "use-uv.md").write_text(
        _valid_frontmatter(id_="use-uv"),
        encoding="utf-8",
    )
    # README already exists with stale content (no children listed yet).
    (k / "global" / "python" / "README.md").write_text(
        "# Python\n\nPython-related lessons.\n",
        encoding="utf-8",
    )
    (k / "global" / "README.md").write_text("# Global\n", encoding="utf-8")
    (k / "README.md").write_text("# Knowledge\n", encoding="utf-8")

    scholar_prompt.reconcile_readmes(k)
    py_readme = (k / "global" / "python" / "README.md").read_text(encoding="utf-8")
    # Prose preserved.
    assert "Python-related lessons." in py_readme
    # Auto-section added with the entry.
    assert "<!-- ULTAN:children (auto) -->" in py_readme
    assert "[[global/python/use-uv]]" in py_readme
    # Parent listing includes the python folder.
    g_readme = (k / "global" / "README.md").read_text(encoding="utf-8")
    assert "[[global/python/]]" in g_readme


def test_reconcile_is_idempotent(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    (k / "global" / "python" / "use-uv.md").write_text(
        _valid_frontmatter(id_="use-uv"),
        encoding="utf-8",
    )
    first = scholar_prompt.reconcile_readmes(k)
    second = scholar_prompt.reconcile_readmes(k)
    assert first  # something happened first time
    assert second == []  # nothing should change second time


def test_reconcile_preserves_prose_outside_markers(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    (k / "global" / "python" / "use-uv.md").write_text(
        _valid_frontmatter(id_="use-uv"),
        encoding="utf-8",
    )
    # README with prose, no markers yet — reconcile must append, not erase.
    (k / "global" / "python" / "README.md").write_text(
        "# Python\n\nHand-written notes\nthat must survive.\n",
        encoding="utf-8",
    )
    (k / "global" / "README.md").write_text("# G\n", encoding="utf-8")
    (k / "README.md").write_text("# K\n", encoding="utf-8")
    scholar_prompt.reconcile_readmes(k)
    py_readme = (k / "global" / "python" / "README.md").read_text(encoding="utf-8")
    assert "Hand-written notes" in py_readme
    assert "that must survive." in py_readme
    assert "[[global/python/use-uv]]" in py_readme


def test_reconcile_removes_archived_entry_from_listing(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    (k / "global" / "python" / "use-uv.md").write_text(
        _valid_frontmatter(id_="use-uv"),
        encoding="utf-8",
    )
    scholar_prompt.reconcile_readmes(k)  # initial state with use-uv listed
    # Archive the entry by moving it.
    (k / "_archive" / "global" / "python").mkdir(parents=True)
    (k / "_archive" / "global" / "python" / "use-uv.md").write_text("", encoding="utf-8")
    (k / "global" / "python" / "use-uv.md").unlink()
    scholar_prompt.reconcile_readmes(k)
    py_readme = (k / "global" / "python" / "README.md").read_text(encoding="utf-8")
    assert "[[global/python/use-uv]]" not in py_readme
    # Empty folder still has a marker section.
    assert "<!-- ULTAN:children (auto) -->" in py_readme


def test_check_invariants_missing_dir_returns_empty(tmp_path: Path):
    out = scholar_prompt.check_invariants(tmp_path / "nope")
    assert out == []


# ── reinforcement counter ────────────────────────────────────────────


def test_apply_reinforcement_bumps_counter_on_existing_entry(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    entry = k / "global" / "tooling" / "use-uv.md"
    entry.write_text(_valid_frontmatter(id_="use-uv"), encoding="utf-8")

    packets = [
        {
            "session_id": "s1",
            "proposals": [
                {
                    "action": "update_entry",
                    "salience_signal": "reinforces",
                    "existing_entry": "global/tooling/use-uv.md",
                    "path": "global/tooling/use-uv.md",
                    "new_body": "...",
                    "reasoning": "user reaffirmed",
                }
            ],
        }
    ]

    changes = scholar_prompt.apply_reinforcement_counters(packets, k)
    assert any("global/tooling/use-uv.md" in c for c in changes)

    import yaml as _yaml

    text = entry.read_text(encoding="utf-8")
    fm = _yaml.safe_load(text.split("---", 2)[1])
    assert fm.get("reinforced") == 1
    assert fm.get("last_reinforced")  # date stamped


def test_apply_reinforcement_dedupes_within_batch(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    entry = k / "global" / "tooling" / "use-uv.md"
    entry.write_text(_valid_frontmatter(id_="use-uv"), encoding="utf-8")

    # Two proposals both reinforce the same entry in one batch.
    packets = [
        {
            "session_id": "s1",
            "proposals": [
                {
                    "action": "update_entry",
                    "salience_signal": "reinforces",
                    "existing_entry": "global/tooling/use-uv.md",
                    "path": "global/tooling/use-uv.md",
                    "new_body": "",
                    "reasoning": "",
                },
                {
                    "action": "update_entry",
                    "salience_signal": "reinforces",
                    "existing_entry": "global/tooling/use-uv.md",
                    "path": "global/tooling/use-uv.md",
                    "new_body": "",
                    "reasoning": "",
                },
            ],
        }
    ]

    changes = scholar_prompt.apply_reinforcement_counters(packets, k)
    assert len(changes) == 1  # deduped

    import yaml as _yaml

    fm = _yaml.safe_load(entry.read_text(encoding="utf-8").split("---", 2)[1])
    assert fm.get("reinforced") == 1  # not 2


def test_apply_reinforcement_rejects_path_traversal(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global").mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("---\nid: x\n---\n", encoding="utf-8")

    packets = [
        {
            "session_id": "s1",
            "proposals": [
                {
                    "action": "update_entry",
                    "salience_signal": "reinforces",
                    "existing_entry": "../outside.md",
                    "path": "../outside.md",
                    "new_body": "",
                    "reasoning": "",
                }
            ],
        }
    ]

    changes = scholar_prompt.apply_reinforcement_counters(packets, k)
    assert changes == []
    # Outside file must NOT have been touched.
    assert outside.read_text(encoding="utf-8") == "---\nid: x\n---\n"


def test_apply_reinforcement_skips_non_reinforces_signals(tmp_path: Path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    entry = k / "global" / "tooling" / "use-uv.md"
    entry.write_text(_valid_frontmatter(id_="use-uv"), encoding="utf-8")

    packets = [
        {
            "session_id": "s1",
            "proposals": [
                {
                    "action": "write_entry",
                    "salience_signal": "novel",
                    "path": "x",
                    "body": "",
                    "reasoning": "",
                },
                {
                    "action": "deprecate_entry",
                    "salience_signal": "contradicts",
                    "existing_entry": "global/tooling/use-uv.md",
                    "path": "global/tooling/use-uv.md",
                    "superseded_by": "x",
                    "reasoning": "",
                },
            ],
        }
    ]

    changes = scholar_prompt.apply_reinforcement_counters(packets, k)
    assert changes == []
    assert "reinforced" not in entry.read_text(encoding="utf-8")


# ── Secrets-redaction invariant ──────────────────────────────────────


def test_build_prompt_includes_no_literal_secrets_invariant():
    """Hierarchy invariant #6 must surface in the assembled prompt
    so the Scholar has explicit veto criteria for credential leaks."""
    prompt = scholar_prompt.build_prompt([], library_snapshot="(empty)")
    assert "NO LITERAL SECRETS" in prompt
    assert "contains-secret" in prompt
    for pattern in ("API key", "ghp_", "AKIA", "sk-", "BEGIN ... PRIVATE KEY"):
        assert pattern in prompt, f"secrets invariant missing pattern: {pattern!r}"
