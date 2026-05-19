"""Pure-function tests for librarian_prompt — buffer flattening, seed
extraction, BM25 attachment, library snapshot, prompt assembly, and
JSON response parsing.

No SDK calls. No I/O outside tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from agent_mem_daemon import librarian_prompt as lp


# ── flatten_buffer / format_rolling_buffer ────────────────────────────


def _ev(typ: str, payload: dict | None = None):
    return {"ts": 1.0, "type": typ, "cwd": "/repo", "payload": payload or {}}


def _snap(turns_events):
    return {
        "session_id": "s1",
        "cwd": "/repo/acme-widget-svc",
        "ended": False,
        "turns": [
            {"events": evs, "started_at": 0.0, "sealed_at": 1.0}
            for evs in turns_events
        ],
    }


def test_flatten_buffer_assigns_monotonic_turn_ids_and_skips_seal_events():
    snap = _snap([
        [_ev("UserPromptSubmit", {"text": "hello"}),
         _ev("PostToolUse", {"tool": "Read", "arguments": {"path": "x"}}),
         _ev("Stop")],
        [_ev("UserPromptSubmit", {"text": "follow up"}),
         _ev("SessionEnd")],
    ])
    flat = lp.flatten_buffer(snap)
    # Stop / SessionEnd dropped; PostToolUse synthesised to "Read(path)".
    assert [t for t, _r, _x, _u in flat] == [1, 2, 3]
    roles = [r for _t, r, _x, _u in flat]
    assert roles[0] == "user"
    assert roles[1] == "assistant"
    assert roles[2] == "user"
    texts = [x for _t, _r, x, _u in flat]
    assert "hello" in texts
    assert "follow up" in texts
    assert any("Read(" in t for t in texts)
    # None are user-asserted in this snapshot.
    assert all(not u for _t, _r, _x, u in flat)


def test_flatten_buffer_user_asserted_flag_propagates():
    snap = _snap([
        [_ev("UserPromptSubmit", {"text": "always wrap errors", "user_asserted": True})]
    ])
    flat = lp.flatten_buffer(snap)
    assert len(flat) == 1
    assert flat[0][3] is True


def test_flatten_buffer_empty_when_no_turns():
    assert lp.flatten_buffer({"turns": []}) == []
    assert lp.flatten_buffer({}) == []


def test_format_rolling_buffer_renders_expected_format():
    flat = [
        (1, "user", "wire up the new ReportingService", False),
        (2, "assistant", "use a factory", False),
    ]
    s = lp.format_rolling_buffer(flat)
    assert s.splitlines() == [
        "[1] [user] wire up the new ReportingService",
        "[2] [assistant] use a factory",
    ]


def test_format_rolling_buffer_marks_user_asserted():
    flat = [(1, "user", "always wrap errors", True)]
    s = lp.format_rolling_buffer(flat)
    assert "[USER-ASSERTED]" in s
    assert "[1] [user] [USER-ASSERTED] always wrap errors" == s


def test_format_rolling_buffer_compacts_whitespace():
    flat = [(1, "user", "line\nbreak  with\tspaces", False)]
    s = lp.format_rolling_buffer(flat)
    assert s == "[1] [user] line break with spaces"


def test_format_rolling_buffer_empty():
    assert "(empty" in lp.format_rolling_buffer([])


# ── extract_seed_phrases ──────────────────────────────────────────────


def test_seed_extraction_finds_imperatives_and_directives():
    txt = dedent("""
        We should wrap upstream errors at the boundary.
        Always stub the factory, not the service.
        Never construct services directly from controllers.
        The fix is to convert errors to domain types.
        The gotcha is sqlite lacks Postgres constraint types.
        Don't mock the database in tests.
        Use a factory pattern for new APIs.
    """).strip()
    seeds = lp.extract_seed_phrases(txt)
    text = " || ".join(seeds).lower()
    assert "should wrap upstream" in text
    assert "always stub the factory" in text
    assert "never construct services" in text
    assert "the fix is" in text
    assert "the gotcha is" in text
    assert "mock the database" in text
    assert any("factory pattern" in s.lower() for s in seeds)


def test_seed_extraction_dedupes_substrings():
    txt = "Always wrap upstream errors at the boundary. Always wrap upstream errors at the boundary."
    seeds = lp.extract_seed_phrases(txt)
    assert len(seeds) == 1


def test_seed_extraction_returns_empty_on_plain_text():
    txt = "I renamed utils_v2.py to utils.py and updated the three import sites."
    seeds = lp.extract_seed_phrases(txt)
    assert seeds == []


def test_seed_extraction_caps_at_max():
    txt = " ".join([f"Always do thing-{i} for reasons." for i in range(30)])
    seeds = lp.extract_seed_phrases(txt, max_seeds=5)
    assert len(seeds) == 5


# ── attach_bm25_hits / format_bm25_seeds ──────────────────────────────


class _StubIndex:
    def __init__(self, knowledge_dir: Path, results_for: dict):
        self.knowledge_dir = knowledge_dir
        self._results = results_for

    def search(self, query: str, k: int = 10):
        for needle, hits in self._results.items():
            if needle in query.lower():
                return hits[:k]
        return []


def test_attach_bm25_hits_renders_relative_paths(tmp_path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    target = kdir / "global" / "tooling" / "factory-pattern-for-apis.md"
    target.parent.mkdir(parents=True)
    target.write_text("dummy", encoding="utf-8")

    idx = _StubIndex(kdir, {"factory": [(target, 14.8, "snippet")]})
    out = lp.attach_bm25_hits(
        ["Use a factory: ReportingServiceFactory.create(...)"],
        idx,
        knowledge_dir=kdir,
    )
    assert len(out) == 1
    assert out[0]["seed"].startswith("Use a factory")
    hits = out[0]["hits"]
    assert len(hits) == 1
    assert hits[0]["entry_id"] == "factory-pattern-for-apis"
    assert hits[0]["path"] == "global/tooling/factory-pattern-for-apis.md"
    assert hits[0]["score"] == 14.8


def test_attach_bm25_hits_handles_none_index():
    out = lp.attach_bm25_hits(["seed1", "seed2"], None)
    assert [e["seed"] for e in out] == ["seed1", "seed2"]
    assert all(e["hits"] == [] for e in out)


def test_format_bm25_seeds_empty_message():
    assert "regex extractor found no candidate seeds" in lp.format_bm25_seeds([])


def test_format_bm25_seeds_includes_hits_and_no_hit_marker():
    s = lp.format_bm25_seeds([
        {"seed": "use a factory", "hits": [
            {"entry_id": "factory-pattern-for-apis", "score": 14.8,
             "path": "global/tooling/factory-pattern-for-apis.md"},
        ]},
        {"seed": "renamed file", "hits": []},
    ])
    assert 'seed: "use a factory"' in s
    assert "hit 1: entry_id=factory-pattern-for-apis" in s
    assert "score=14.8" in s
    assert "(no hits)" in s


# ── read_index_md / build_applies_when_table ──────────────────────────


def test_read_index_md_missing_returns_sentinel(tmp_path):
    out = lp.read_index_md(tmp_path / "knowledge")
    assert "no entries" in out


def test_read_index_md_returns_file_contents(tmp_path):
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "index.md").write_text(
        "# Knowledge Base Index\n\n| Article | ... |\n", encoding="utf-8"
    )
    out = lp.read_index_md(k)
    assert "Knowledge Base Index" in out


def _write_entry(kdir: Path, rel: str, status: str, applies_when: list[str], scope: str = "global"):
    p = kdir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "---\n"
        f"id: {Path(rel).stem}\n"
        f"scope: {scope}\n"
        f"status: {status}\n"
        "applies-when: |\n"
        + "".join(f"  {ln}\n" for ln in applies_when)
        + "---\n\n# title\n"
    )
    p.write_text(body, encoding="utf-8")


def test_build_applies_when_table_only_confirmed(tmp_path):
    k = tmp_path / "knowledge"
    k.mkdir()
    _write_entry(k, "global/tooling/factory.md", "confirmed",
                 ["designing or building any new API",
                  "decisions about how clients construct service objects"])
    _write_entry(k, "global/tooling/draft.md", "provisional",
                 ["this should not appear"])
    _write_entry(k, "_archive/tooling/old.md", "confirmed",
                 ["never surface archived entries"])
    out = lp.build_applies_when_table(k)
    lines = out.splitlines()
    assert any("factory | global | designing or building any new API" in ln for ln in lines)
    assert any("decisions about how clients" in ln for ln in lines)
    assert not any("draft" in ln for ln in lines)
    assert not any("archived" in ln for ln in lines)


def test_build_applies_when_table_missing_dir(tmp_path):
    assert "empty" in lp.build_applies_when_table(tmp_path / "nope")


# ── build_library_snapshot ────────────────────────────────────────────


def test_library_snapshot_missing_dir(tmp_path):
    out = lp.build_library_snapshot(tmp_path / "nope")
    # Even missing dir produces something usable for the prompt.
    assert "empty" in out.lower() or "no entries" in out.lower() or "no catalog" in out.lower()


def test_library_snapshot_renders_tree_and_readmes(tmp_path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "projects" / "demo").mkdir(parents=True)
    (k / "README.md").write_text(
        "# agent-mem knowledge\n\nThe top-level index.\n", encoding="utf-8"
    )
    (k / "global" / "README.md").write_text(
        "# Global\nCross-project lessons.\n", encoding="utf-8"
    )
    (k / "global" / "tooling" / "README.md").write_text(
        "# Tooling\nNotes about toolchains.\n", encoding="utf-8"
    )
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        "---\nid: uv-basics\n---\n# uv basics\n", encoding="utf-8"
    )
    (k / "projects" / "README.md").write_text(
        "# Projects\nPer-repo entries.\n", encoding="utf-8"
    )
    (k / "index.md").write_text(
        "# Knowledge Base Index\n\n| Article | Scope |\n", encoding="utf-8"
    )
    out = lp.build_library_snapshot(k)
    # Tree must show the structure.
    assert "## Tree" in out
    assert "global/" in out
    assert "tooling/" in out
    assert "uv-basics.md" in out
    assert "projects/" in out
    # Root README excerpt.
    assert "agent-mem knowledge" in out
    # Top-level folder READMEs.
    assert "Cross-project lessons." in out
    assert "Per-repo entries." in out
    # index.md excerpt.
    assert "Knowledge Base Index" in out


def test_library_snapshot_truncates_to_budget(tmp_path):
    k = tmp_path / "knowledge"
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "README.md").write_text("x" * 5000, encoding="utf-8")
    out = lp.build_library_snapshot(k, max_chars=500)
    assert len(out) <= 500 + 100  # room for the truncation marker
    assert "truncated" in out.lower()


def test_library_snapshot_excludes_archive(tmp_path):
    k = tmp_path / "knowledge"
    (k / "_archive" / "tooling").mkdir(parents=True)
    (k / "_archive" / "tooling" / "old.md").write_text(
        "---\nid: old\n---\n# old\n", encoding="utf-8"
    )
    (k / "global" / "tooling").mkdir(parents=True)
    (k / "global" / "tooling" / "live.md").write_text(
        "---\nid: live\n---\n# live\n", encoding="utf-8"
    )
    out = lp.build_library_snapshot(k)
    assert "live.md" in out
    assert "old.md" not in out


# ── derive_project_slug ───────────────────────────────────────────────


def test_derive_project_slug_from_cwd():
    snap = {"cwd": "/repos/acme-widget-svc/"}
    assert lp.derive_project_slug(snap) == "acme-widget-svc"


def test_derive_project_slug_from_explicit_field():
    snap = {"cwd": "/x", "project_slug": "Acme/Widget Svc"}
    assert lp.derive_project_slug(snap) == "acme-widget-svc"


def test_derive_project_slug_from_event_payload():
    snap = {"turns": [{"events": [
        {"type": "PostToolUse", "payload": {"project_slug": "my-proj"}}
    ]}]}
    assert lp.derive_project_slug(snap) == "my-proj"


def test_derive_project_slug_unknown_when_nothing():
    assert lp.derive_project_slug({}) == "unknown"


# ── load_prompt_template / assemble_prompt ────────────────────────────


def test_template_contains_expected_placeholders():
    t = lp.load_prompt_template()
    for needle in (
        "{{PROJECT_SLUG}}",
        "{{ROLLING_BUFFER}}",
        "{{LIBRARY_SNAPSHOT}}",
        "{{APPLIES_WHEN_TABLE}}",
    ):
        assert needle in t, f"missing placeholder: {needle}"
    # BM25_SEEDS used to be a placeholder; it's now exposed as an MCP
    # tool the Librarian invokes itself. Regression guard:
    assert "{{BM25_SEEDS}}" not in t
    assert "bm25_search" in t  # the Librarian must be told about the tool


def test_template_describes_all_action_types():
    """The Librarian must know about every legal action type by name.

    Now generated from ``_schemas.py`` via the ``{{ACTION_TYPES}}``
    placeholder, so we assert against the assembled prompt rather than
    the raw template.
    """
    p = lp.assemble_prompt(
        project_slug="x",
        rolling_buffer="(empty)",
        library_snapshot="(empty)",
        applies_when_table="(empty)",
    )
    for action in (
        "write_entry", "update_entry", "merge_entries",
        "move_entry", "archive_entry", "update_readme",
        "add_wikilink", "split_folder",
    ):
        assert action in p, f"action type {action!r} missing from assembled prompt"


def test_assemble_prompt_substitutes_all_placeholders():
    p = lp.assemble_prompt(
        project_slug="acme-widget-svc",
        rolling_buffer="[1] [user] hi",
        library_snapshot="## Tree\n```\nglobal/\n```",
        applies_when_table="factory | global | designing or building any new API",
    )
    assert "{{" not in p, "unsubstituted placeholder remains"
    assert "acme-widget-svc" in p
    assert "[1] [user] hi" in p
    assert "global/" in p
    assert "factory | global | designing or building any new API" in p


# ── parse_librarian_json / normalise_packet ───────────────────────────


def test_parse_clean_proposal_json():
    obj = {
        "proposals": [
            {
                "action": "write_entry",
                "path": "global/tooling/foo.md",
                "body": "---\nid: foo\n---\n# foo\n",
                "reasoning": "buffer turn [3] said 'always foo'",
            }
        ],
        "interrupts": [],
    }
    out = lp.parse_librarian_json(json.dumps(obj))
    assert out is not None
    assert len(out["proposals"]) == 1
    p = out["proposals"][0]
    assert p["action"] == "write_entry"
    assert p["path"] == "global/tooling/foo.md"
    assert p["reasoning"].startswith("buffer turn")


def test_parse_proposal_with_all_action_types():
    """Each action type validates and round-trips through the parser."""
    obj = {
        "proposals": [
            {"action": "write_entry", "path": "a.md", "body": "x", "reasoning": "r"},
            {"action": "update_entry", "path": "b.md", "new_body": "x", "reasoning": "r"},
            {"action": "merge_entries", "source_paths": ["c.md", "d.md"],
             "target_path": "c.md", "target_body": "x", "reasoning": "r"},
            {"action": "move_entry", "from_path": "e.md", "to_path": "f.md", "reasoning": "r"},
            {"action": "archive_entry", "path": "g.md", "reasoning": "r"},
            {"action": "update_readme", "folder_path": "global", "new_body": "x", "reasoning": "r"},
            {"action": "add_wikilink", "from_path": "h.md", "to_path": "i.md",
             "context": "see also", "reasoning": "r"},
            {"action": "split_folder", "folder_path": "global/big",
             "into": {"sub1": ["a.md", "b.md"]}, "reasoning": "r"},
        ],
        "interrupts": [],
    }
    out = lp.parse_librarian_json(json.dumps(obj))
    assert out is not None
    assert len(out["proposals"]) == 8
    assert [p["action"] for p in out["proposals"]] == [
        "write_entry", "update_entry", "merge_entries", "move_entry",
        "archive_entry", "update_readme", "add_wikilink", "split_folder",
    ]


def test_parse_proposal_unknown_action_rejected():
    obj = {"proposals": [{"action": "summon_demon", "reasoning": "x"}], "interrupts": []}
    out = lp.parse_librarian_json(json.dumps(obj))
    # Unknown discriminator must reject the whole response.
    assert out is None


def test_parse_json_with_code_fences():
    obj = {"proposals": [], "interrupts": []}
    text = "```json\n" + json.dumps(obj) + "\n```"
    out = lp.parse_librarian_json(text)
    assert out is not None
    assert out["proposals"] == []
    assert out["interrupts"] == []


def test_parse_json_with_prefix_prose():
    obj = {"proposals": [], "interrupts": []}
    text = "Here is my output:\n\n" + json.dumps(obj)
    out = lp.parse_librarian_json(text)
    assert out is not None
    assert out["proposals"] == []


def test_parse_malformed_returns_none():
    assert lp.parse_librarian_json("{not json") is None
    assert lp.parse_librarian_json("") is None
    assert lp.parse_librarian_json("   ") is None


def test_parse_non_object_returns_none():
    assert lp.parse_librarian_json(json.dumps([1, 2, 3])) is None


def test_normalise_packet_extracts_proposals_and_interrupts():
    parsed = {
        "proposals": [
            {"action": "write_entry", "path": "a.md"},
            "not a dict",
        ],
        "interrupts": [{"lesson_id": "x"}],
    }
    out = lp.normalise_packet(parsed)
    assert len(out["proposals"]) == 1
    assert out["proposals"][0]["path"] == "a.md"
    assert out["interrupts"] == [{"lesson_id": "x"}]


def test_normalise_packet_tolerates_missing_keys():
    out = lp.normalise_packet({})
    assert out == {"proposals": [], "interrupts": []}


def test_normalise_packet_accepts_interrupt_candidates_alias():
    out = lp.normalise_packet({"interrupt_candidates": [{"lesson_id": "x"}]})
    assert out["interrupts"] == [{"lesson_id": "x"}]
