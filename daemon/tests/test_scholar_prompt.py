"""Tests for the Scholar prompt assembly + decision accounting + nudge
writing + hierarchy invariants checker.

The Scholar now returns a typed, validated ``ScholarDecisions`` (no JSON
scrape), so the parsing tests are gone; ``summarise_decisions`` and
``append_nudges_from_response`` consume that model directly."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent_mem_daemon import scholar_prompt
from agent_mem_daemon._schemas import ScholarDecisions

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
    # Veto language and the typed-output language must survive into the prompt.
    assert "VETO" in prompt
    assert "ScholarDecisions" in prompt
    # The action enumeration (Librarian's full vocabulary) is still inlined.
    for action in (
        "write_entry",
        "merge_entries",
        "archive_entry",
        "move_entry",
        "update_entry",
    ):
        assert action in prompt, f"action {action!r} missing from Scholar prompt"


def test_build_prompt_annotates_action_indices():
    """Each proposal in the rendered JSON carries an ``_action_index`` so the
    Scholar can refer to the Librarian's proposals while verifying them."""
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


def test_build_prompt_embeds_scholar_decisions_schema():
    """The output schema inlined into the prompt is the ScholarDecisions
    model's JSON Schema (actions + interrupts_processed), not the old
    approve/veto ScholarReview shape."""
    prompt = scholar_prompt.build_prompt(
        [{"session_id": "s1", "proposals": [], "interrupts": []}],
        library_snapshot="(empty)",
    )
    assert "interrupts_processed" in prompt
    assert '"actions"' in prompt


def test_build_prompt_documents_abstract_entries_gate():
    """The Scholar prompt names abstract_entries in its vocabulary and carries
    the aha/abstraction gate with its GOOD + MUST-VETO examples."""
    prompt = scholar_prompt.build_prompt(
        [{"session_id": "s1", "proposals": [], "interrupts": []}],
        library_snapshot="(empty)",
    )
    assert "abstract_entries" in prompt
    assert "ABSTRACTION GATE" in prompt
    assert "linting across languages" in prompt  # the GOOD example
    assert "likes good code" in prompt  # a MUST-VETO example


# ── decision accounting (typed-output era) ────────────────────────────


def _decisions(actions: List[Dict[str, Any]], interrupts: List[Dict[str, Any]]) -> ScholarDecisions:
    return ScholarDecisions.model_validate({"actions": actions, "interrupts_processed": interrupts})


def test_summarise_decisions_counts_actions_and_interrupts():
    decisions = _decisions(
        [
            {"action": "archive_entry", "path": "a.md", "reasoning": "r"},
            {"action": "archive_entry", "path": "b.md", "reasoning": "r"},
        ],
        [
            {"action": "approve", "lesson_path": "l/a", "text": "t"},
            {"action": "veto"},
            {"action": "veto"},
        ],
    )
    counters = scholar_prompt.summarise_decisions(decisions)
    assert counters["actions_applied"] == 2
    assert counters["archive_entry"] == 2
    assert counters["nudge"] == 1
    assert counters["interrupt-veto"] == 2


def test_summarise_decisions_empty():
    assert scholar_prompt.summarise_decisions(_decisions([], [])) == {}


# ── nudge file writing + round-trip ──────────────────────────────────


def test_append_nudges_writes_blocks(tmp_path: Path):
    target = tmp_path / "pending-nudges.md"
    decisions = _decisions(
        [],
        [
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
    )
    written = scholar_prompt.append_nudges_from_response(
        decisions,
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
        "---\nid: pre\ncreated: 2026-01-01T00:00:00+00:00\nlesson: pre/lesson\n"
        "---\nPre-existing nudge.\n",
        encoding="utf-8",
    )
    decisions = _decisions(
        [],
        [
            {
                "lesson_id": "x",
                "lesson_path": "x/path",
                "action": "approve",
                "text": "Newly written nudge.",
            }
        ],
    )
    scholar_prompt.append_nudges_from_response(decisions, path=target)
    body = target.read_text(encoding="utf-8")
    assert "Pre-existing nudge." in body
    assert "Newly written nudge." in body


def test_append_nudges_skips_malformed_approval(tmp_path: Path):
    target = tmp_path / "pending-nudges.md"
    decisions = _decisions(
        [],
        [
            {"lesson_id": "x", "action": "approve"},
            {"lesson_id": "y", "lesson_path": "y/path", "action": "approve"},
        ],
    )
    written = scholar_prompt.append_nudges_from_response(decisions, path=target)
    assert written == []
    assert not target.exists() or target.read_text(encoding="utf-8") == ""


def test_append_nudges_with_no_approvals_writes_nothing(tmp_path: Path):
    target = tmp_path / "pending-nudges.md"
    decisions = _decisions(
        [],
        [{"lesson_id": "x", "lesson_path": "x", "action": "veto", "reason": "nope"}],
    )
    written = scholar_prompt.append_nudges_from_response(decisions, path=target)
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


# ── check_invariants_detailed: structured violations for escalation ───


def test_check_invariants_detailed_projects_to_string_list(tmp_path: Path):
    # The string-list contract is exactly the .message of each detailed
    # violation (same order).
    from agent_mem_daemon import scholar_prompt as sp

    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics"), encoding="utf-8"
    )
    detailed = sp.check_invariants_detailed(k)
    assert [v.message for v in detailed] == sp.check_invariants(k)


def test_check_invariants_detailed_overcap_is_escalatable(tmp_path: Path):
    from agent_mem_daemon import repair_queue
    from agent_mem_daemon import scholar_prompt as sp

    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    for i in range(6):
        (k / "global" / "tooling" / f"e{i}.md").write_text(
            _valid_frontmatter(id_=f"e{i}"), encoding="utf-8"
        )
    overcap = [
        v for v in sp.check_invariants_detailed(k) if v.repair_kind == repair_queue.KIND_OVERCAP_DIR
    ]
    assert len(overcap) == 1
    v = overcap[0]
    assert v.file == "global/tooling"
    assert v.target == "global/tooling"
    # The context lists the entries so the Librarian needn't re-walk the tree.
    assert "e0.md" in v.context and "e5.md" in v.context
    task = v.to_repair_task()
    assert task is not None
    assert task.kind == repair_queue.KIND_OVERCAP_DIR


def test_check_invariants_detailed_bad_frontmatter_is_escalatable(tmp_path: Path):
    from agent_mem_daemon import repair_queue
    from agent_mem_daemon import scholar_prompt as sp

    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("# k\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# g\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text("# t\n", encoding="utf-8")
    (k / "global" / "tooling" / "incomplete.md").write_text(
        "---\nid: incomplete\nscope: global\n---\n# x\n", encoding="utf-8"
    )
    bad = [
        v
        for v in sp.check_invariants_detailed(k)
        if v.repair_kind == repair_queue.KIND_BAD_FRONTMATTER
    ]
    assert len(bad) == 1
    v = bad[0]
    assert v.file == "global/tooling/incomplete.md"
    assert v.target == "global/tooling/incomplete.md"
    assert v.to_repair_task() is not None


def test_check_invariants_detailed_display_only_kinds_have_no_task(tmp_path: Path):
    # README/wikilink/scope/empty-body violations are display-only: no
    # repair task (handled by reconciler / wikilink-repair / not
    # independently actionable).
    from agent_mem_daemon import scholar_prompt as sp

    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    # No READMEs anywhere AND a broken wikilink — both display-only.
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        _valid_frontmatter(id_="uv-basics") + "See [[global/missing/no-such]]\n",
        encoding="utf-8",
    )
    detailed = sp.check_invariants_detailed(k)
    readme_or_link = [
        v for v in detailed if "README" in v.message or "broken wikilink" in v.message
    ]
    assert readme_or_link  # we produced some
    assert all(v.repair_kind is None for v in readme_or_link)
    assert all(v.to_repair_task() is None for v in readme_or_link)


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


def test_build_prompt_exempts_repair_proposals_from_novelty():
    """Repair-originated proposals must be verify-and-execute, NOT judged
    for novelty/dedupe. The prompt must say so explicitly and tie the
    exemption to the packet's ``repair_fingerprints`` marker."""
    prompt = scholar_prompt.build_prompt([], library_snapshot="(empty)")
    assert "INTEGRITY-REPAIR PROPOSALS" in prompt
    assert "repair_fingerprints" in prompt
    assert "VERIFY-AND-RETURN" in prompt
    # The salience filter must tell the Scholar to skip repair proposals.
    assert "SKIP it" in prompt or "SKIP this section" in prompt or "bypass this filter" in prompt
    # And it must spell out that "not novel"/"duplicate" are not valid veto
    # reasons for a repair.
    assert "NEVER valid veto reasons for a repair proposal" in prompt


# ── repair_broken_wikilinks (post-write self-healing) ─────────────────


_INDEX_HEADER = (
    "# Knowledge Index\n\n"
    "| Article | Scope | Status | Conf | Summary | Applies-when | From | Updated |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _seed_repair_tree(tmp_path: Path) -> Path:
    """A minimal, invariant-clean tree with one real entry, ready for the
    repair tests to dirty with phantom/broken links."""
    k = tmp_path / "knowledge"
    (k / "global" / "python").mkdir(parents=True)
    (k / "README.md").write_text("# Knowledge\n", encoding="utf-8")
    (k / "global" / "README.md").write_text("# Global\n", encoding="utf-8")
    (k / "global" / "python" / "README.md").write_text("# Python\n", encoding="utf-8")
    (k / "global" / "python" / "use-uv.md").write_text(
        _valid_frontmatter(id_="use-uv"),
        encoding="utf-8",
    )
    return k


def test_repair_removes_phantom_index_row(tmp_path: Path):
    # Reproduces the live `some-fake-project` phantom: a catalog row in
    # index.md that points at an entry which was never written.
    k = _seed_repair_tree(tmp_path)
    phantom = (
        "| [[projects/some-fake-project/security/no-secrets-in-env-example]] "
        "| project:some-fake-project | provisional | 0.85 "
        "| Never put real secrets in env.example | editing env.example files "
        "| session:hooktest-6AB94685 | 2026-05-19 |\n"
    )
    real_row = "| [[global/python/use-uv]] | global | provisional | 0.7 | uv | x | y | z |\n"
    (k / "index.md").write_text(_INDEX_HEADER + real_row + phantom, encoding="utf-8")

    # Pre-condition: the invariant fires on the phantom.
    pre = scholar_prompt.check_invariants(k)
    assert any("some-fake-project" in v for v in pre)

    changes = scholar_prompt.repair_broken_wikilinks(k)
    assert any("phantom row" in c for c in changes)

    index_after = (k / "index.md").read_text(encoding="utf-8")
    assert "some-fake-project" not in index_after  # phantom row gone
    assert "[[global/python/use-uv]]" in index_after  # real row preserved
    assert _INDEX_HEADER.splitlines()[0] in index_after  # header preserved

    # Post-condition: broken-wikilink violations drop to zero.
    post = scholar_prompt.check_invariants(k)
    assert not any("broken wikilink" in v for v in post)


def test_repair_resolves_body_link_by_unique_leaf(tmp_path: Path):
    # A broken wikilink in an entry body whose leaf matches exactly one
    # existing entry is rewritten to that entry's canonical path.
    k = _seed_repair_tree(tmp_path)
    entry = k / "global" / "python" / "linked.md"
    entry.write_text(
        _valid_frontmatter(id_="linked") + "\nSee [[global/wrongdir/use-uv]] for details.\n",
        encoding="utf-8",
    )
    changes = scholar_prompt.repair_broken_wikilinks(k)
    assert any("rewrote" in c and "use-uv" in c for c in changes)
    body = entry.read_text(encoding="utf-8")
    assert "[[global/python/use-uv]]" in body
    assert "[[global/wrongdir/use-uv]]" not in body
    assert not any("broken wikilink" in v for v in scholar_prompt.check_invariants(k))


def test_repair_neutralises_unresolvable_body_link(tmp_path: Path):
    # A broken body link with no unique leaf match is neutralised to plain
    # text — the broken edge goes away but surrounding prose survives.
    k = _seed_repair_tree(tmp_path)
    entry = k / "global" / "python" / "narr.md"
    entry.write_text(
        _valid_frontmatter(id_="narr")
        + "\nThis references [[global/ghost/never-existed]] in a sentence.\n",
        encoding="utf-8",
    )
    changes = scholar_prompt.repair_broken_wikilinks(k)
    assert any("neutralised" in c for c in changes)
    body = entry.read_text(encoding="utf-8")
    assert "[[global/ghost/never-existed]]" not in body
    # Surrounding prose preserved; leaf name kept as readable text.
    assert "in a sentence." in body
    assert "never-existed" in body
    assert not any("broken wikilink" in v for v in scholar_prompt.check_invariants(k))


def test_repair_neutralise_preserves_alias_text(tmp_path: Path):
    k = _seed_repair_tree(tmp_path)
    entry = k / "global" / "python" / "alias.md"
    entry.write_text(
        _valid_frontmatter(id_="alias") + "\nSee [[global/ghost/gone|the old guide]] now.\n",
        encoding="utf-8",
    )
    scholar_prompt.repair_broken_wikilinks(k)
    body = entry.read_text(encoding="utf-8")
    assert "the old guide" in body
    assert "[[" not in body.split("# title", 1)[-1]


def test_repair_is_idempotent_and_noop_when_clean(tmp_path: Path):
    k = _seed_repair_tree(tmp_path)
    real_row = "| [[global/python/use-uv]] | global | provisional | 0.7 | uv | x | y | z |\n"
    (k / "index.md").write_text(_INDEX_HEADER + real_row, encoding="utf-8")
    # Clean tree → no changes.
    assert scholar_prompt.repair_broken_wikilinks(k) == []
    # Add a phantom, repair once, then a second pass is a no-op.
    phantom = "| [[projects/x/y/ghost]] | project:x | provisional | 0.5 | g | a | b | c |\n"
    (k / "index.md").write_text(_INDEX_HEADER + real_row + phantom, encoding="utf-8")
    first = scholar_prompt.repair_broken_wikilinks(k)
    assert first
    assert scholar_prompt.repair_broken_wikilinks(k) == []


def test_repair_missing_dir_returns_empty(tmp_path: Path):
    assert scholar_prompt.repair_broken_wikilinks(tmp_path / "nope") == []


def test_repair_index_prose_link_neutralised_not_row_deleted(tmp_path: Path):
    # A broken link in index.md PROSE (not a table row) must be neutralised
    # by the body path, never trigger whole-line deletion of narrative.
    k = _seed_repair_tree(tmp_path)
    (k / "index.md").write_text(
        "# Knowledge Index\n\nSee also [[global/ghost/missing]] for context.\n\n"
        "| Article | Scope |\n|---|---|\n"
        "| [[global/python/use-uv]] | global |\n",
        encoding="utf-8",
    )
    changes = scholar_prompt.repair_broken_wikilinks(k)
    assert any("neutralised" in c for c in changes)
    index_after = (k / "index.md").read_text(encoding="utf-8")
    assert "See also" in index_after  # prose line survived
    assert "[[global/ghost/missing]]" not in index_after
    assert "[[global/python/use-uv]]" in index_after  # real row intact
    assert not any("broken wikilink" in v for v in scholar_prompt.check_invariants(k))


def test_repair_leaf_resolver_ambiguous_returns_none(tmp_path: Path):
    # Two entries share the same leaf — the resolver must NOT guess, so the
    # broken link is neutralised rather than mis-rewritten.
    k = _seed_repair_tree(tmp_path)
    (k / "global" / "git").mkdir(parents=True)
    (k / "global" / "git" / "README.md").write_text("# Git\n", encoding="utf-8")
    (k / "global" / "git" / "use-uv.md").write_text(
        _valid_frontmatter(id_="use-uv"), encoding="utf-8"
    )
    entry = k / "global" / "python" / "amb.md"
    entry.write_text(
        _valid_frontmatter(id_="amb") + "\nLink to [[global/nowhere/use-uv]].\n",
        encoding="utf-8",
    )
    changes = scholar_prompt.repair_broken_wikilinks(k)
    # Ambiguous leaf → neutralised, not rewritten.
    assert any("neutralised" in c and "amb.md" in c for c in changes)
    body = entry.read_text(encoding="utf-8")
    assert "[[global/nowhere/use-uv]]" not in body


def test_repair_skips_archive_and_log(tmp_path: Path):
    k = _seed_repair_tree(tmp_path)
    # log.md may legitimately quote dead paths — must not be rewritten.
    (k / "log.md").write_text("see [[global/ghost/dead]] (archived action)\n", encoding="utf-8")
    # _archive links are intentional and resolve as valid.
    (k / "_archive").mkdir()
    (k / "_archive" / "old.md").write_text("[[global/ghost/dead]]\n", encoding="utf-8")
    changes = scholar_prompt.repair_broken_wikilinks(k)
    assert changes == []
    assert "[[global/ghost/dead]]" in (k / "log.md").read_text(encoding="utf-8")


def test_repair_index_row_prefix_does_not_delete_valid_longer_row(tmp_path: Path):
    # A broken target that is a PREFIX of a valid entry's link must not
    # trigger deletion of a row whose only link is the longer, valid one.
    # E.g. broken ``global/python/use`` vs valid ``[[global/python/use-uv]]`` —
    # a naive substring match on ``[[global/python/use`` would wrongly nuke
    # the valid row. The match must be on the full ``]]``/``|`` token boundary.
    k = _seed_repair_tree(tmp_path)
    valid_row = "| [[global/python/use-uv]] | global | provisional | 0.7 | uv | x | y | z |\n"
    phantom_row = "| [[global/python/use]] | global | provisional | 0.5 | g | a | b | c |\n"
    (k / "index.md").write_text(_INDEX_HEADER + valid_row + phantom_row, encoding="utf-8")

    changes = scholar_prompt.repair_broken_wikilinks(k)
    assert any("phantom row" in c for c in changes)
    index_after = (k / "index.md").read_text(encoding="utf-8")
    # The valid longer-named entry's row must survive.
    assert "[[global/python/use-uv]]" in index_after
    # Only the genuinely broken prefix row is removed.
    assert "[[global/python/use]]" not in index_after
    assert not any("broken wikilink" in v for v in scholar_prompt.check_invariants(k))


# ── escalation hook: on_unresolved leaves the link broken ─────────────


def test_unresolvable_link_escalated_is_left_broken(tmp_path: Path):
    # When an on_unresolved escalator takes ownership of a link the
    # deterministic pass can't resolve, the link must be LEFT BROKEN on
    # disk (not neutralised) so the next pass can re-detect and re-escalate.
    k = _seed_repair_tree(tmp_path)
    entry = k / "global" / "python" / "narr.md"
    entry.write_text(
        _valid_frontmatter(id_="narr")
        + "\nThis references [[global/ghost/never-existed]] in a sentence.\n",
        encoding="utf-8",
    )
    seen: list[tuple[str, str, str]] = []

    def _escalator(rel_file: str, target: str, context: str) -> bool:
        seen.append((rel_file, target, context))
        return True  # owned by escalation

    changes = scholar_prompt.repair_broken_wikilinks(k, on_unresolved=_escalator)

    # The callback fired with the file, broken target, and a context snippet.
    assert seen == [("global/python/narr.md", "global/ghost/never-existed", seen[0][2])]
    assert "[[global/ghost/never-existed]]" in seen[0][2]  # context captured
    # The link is still on disk (detectable next pass), NOT neutralised.
    body = entry.read_text(encoding="utf-8")
    assert "[[global/ghost/never-existed]]" in body
    assert any("escalated" in c for c in changes)
    # And the invariant still fires — that's the intended "keep detecting".
    assert any("broken wikilink" in v for v in scholar_prompt.check_invariants(k))


def test_escalator_declining_falls_back_to_neutralise(tmp_path: Path):
    # If the escalator returns False (declines ownership), the historical
    # neutralise-as-stopgap behaviour still applies.
    k = _seed_repair_tree(tmp_path)
    entry = k / "global" / "python" / "narr.md"
    entry.write_text(
        _valid_frontmatter(id_="narr") + "\nSee [[global/ghost/gone]] now.\n",
        encoding="utf-8",
    )

    changes = scholar_prompt.repair_broken_wikilinks(k, on_unresolved=lambda *_: False)
    body = entry.read_text(encoding="utf-8")
    assert "[[global/ghost/gone]]" not in body  # neutralised
    assert any("neutralised" in c for c in changes)


def test_resolvable_link_never_escalates(tmp_path: Path):
    # A link the deterministic pass CAN resolve (unique leaf match) is
    # rewritten and never offered to the escalator.
    k = _seed_repair_tree(tmp_path)
    entry = k / "global" / "python" / "linked.md"
    entry.write_text(
        _valid_frontmatter(id_="linked") + "\nSee [[global/wrongdir/use-uv]] now.\n",
        encoding="utf-8",
    )
    calls: list[str] = []
    scholar_prompt.repair_broken_wikilinks(
        k, on_unresolved=lambda _f, t, _c: calls.append(t) or True
    )
    assert calls == []  # resolvable → no escalation
    assert "[[global/python/use-uv]]" in entry.read_text(encoding="utf-8")
