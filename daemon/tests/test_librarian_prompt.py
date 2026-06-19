"""Pure-function tests for librarian_prompt — buffer flattening, recency
cap, library snapshot, project-bucket derivation, and prompt assembly
(including the integrity-repair section).

No agent calls. No I/O outside tmp_path. The typed-output validation that
the old JSON parser used to cover now lives in ``test_librarian_agent``.
"""

from __future__ import annotations

from pathlib import Path

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
            {"events": evs, "started_at": 0.0, "sealed_at": 1.0, "turn_seq": i}
            for i, evs in enumerate(turns_events, start=1)
        ],
    }


def test_flatten_buffer_assigns_monotonic_turn_ids_and_skips_seal_events():
    snap = _snap(
        [
            [
                _ev("UserPromptSubmit", {"text": "hello"}),
                _ev("PostToolUse", {"tool": "Read", "arguments": {"path": "x"}}),
                _ev("Stop"),
            ],
            [_ev("UserPromptSubmit", {"text": "follow up"}), _ev("SessionEnd")],
        ]
    )
    flat = lp.flatten_buffer(snap)
    # Stop / SessionEnd dropped; PostToolUse synthesised to "Read(path)".
    assert [t for t, _s, _r, _x, _u in flat] == [1, 2, 3]
    # turn_seq is the OWNING turn's stable id: the two events in turn 1
    # share seq 1; the single event in turn 2 has seq 2.
    assert [s for _t, s, _r, _x, _u in flat] == [1, 1, 2]
    roles = [r for _t, _s, r, _x, _u in flat]
    assert roles[0] == "user"
    assert roles[1] == "assistant"
    assert roles[2] == "user"
    texts = [x for _t, _s, _r, x, _u in flat]
    assert "hello" in texts
    assert "follow up" in texts
    assert any("Read(" in t for t in texts)
    # None are user-asserted in this snapshot.
    assert all(not u for _t, _s, _r, _x, u in flat)


def test_flatten_buffer_turn_seq_defaults_to_zero_when_absent():
    # Legacy snapshots / fixtures without a turn_seq field must not crash;
    # the seq falls back to 0 (the un-citable sentinel).
    snap = {
        "session_id": "s1",
        "turns": [{"events": [_ev("UserPromptSubmit", {"text": "hi"})], "sealed_at": 1.0}],
    }
    flat = lp.flatten_buffer(snap)
    assert len(flat) == 1
    assert flat[0][1] == 0


def test_flatten_buffer_user_asserted_flag_propagates():
    snap = _snap([[_ev("UserPromptSubmit", {"text": "always wrap errors", "user_asserted": True})]])
    flat = lp.flatten_buffer(snap)
    assert len(flat) == 1
    assert flat[0][4] is True


def test_flatten_buffer_renders_recall_surfaced_event():
    # The hook emits a `Surfaced` event (role="recall") recording what the
    # instant triggers showed; the Librarian must see it inline with the turn
    # so it can spot recall gaps. It renders like any text-bearing turn line.
    snap = _snap(
        [
            [
                _ev("UserPromptSubmit", {"text": "how do I make money from BTC"}),
                _ev(
                    "Surfaced",
                    {"role": "recall", "content": "instant-recall surfaced: [[a/b]], [[c/d]]"},
                ),
            ]
        ]
    )
    flat = lp.flatten_buffer(snap)
    roles = [r for _id, _seq, r, _t, _u in flat]
    assert "recall" in roles
    recall_text = next(t for _id, _seq, r, t, _u in flat if r == "recall")
    assert "[[a/b]]" in recall_text and "[[c/d]]" in recall_text


def test_system_prompt_includes_recall_gap_signal():
    s = lp.librarian_system_prompt()
    assert "RECALL-GAP" in s
    assert "PRECISION GATE" in s  # the keyword-pollution safeguard must survive


def test_flatten_buffer_empty_when_no_turns():
    assert lp.flatten_buffer({"turns": []}) == []
    assert lp.flatten_buffer({}) == []


def test_format_rolling_buffer_renders_expected_format():
    flat = [
        (1, 1, "user", "wire up the new ReportingService", False),
        (2, 2, "assistant", "use a factory", False),
    ]
    s = lp.format_rolling_buffer(flat)
    assert s.splitlines() == [
        "[1] (turn_seq=1) [user] wire up the new ReportingService",
        "[2] (turn_seq=2) [assistant] use a factory",
    ]


def test_format_rolling_buffer_marks_user_asserted():
    flat = [(1, 4, "user", "always wrap errors", True)]
    s = lp.format_rolling_buffer(flat)
    assert "[USER-ASSERTED]" in s
    assert "[1] (turn_seq=4) [user] [USER-ASSERTED] always wrap errors" == s


def test_format_rolling_buffer_compacts_whitespace():
    flat = [(1, 1, "user", "line\nbreak  with\tspaces", False)]
    s = lp.format_rolling_buffer(flat)
    assert s == "[1] (turn_seq=1) [user] line break with spaces"


def test_format_rolling_buffer_empty():
    assert "(empty" in lp.format_rolling_buffer([])


# ── cap_buffer_to_recent / buffer_to_prompt_text recency cap ──────────


def test_cap_buffer_keeps_all_when_under_budget():
    flat = [(i, i, "user", f"turn {i}", False) for i in range(1, 6)]
    capped = lp.cap_buffer_to_recent(flat, max_chars=10_000)
    assert capped == flat


def test_cap_buffer_drops_oldest_turns_over_budget():
    # 10 turns of ~50 chars of text each. A 200-char budget should keep
    # only the most-recent few and drop the oldest.
    flat = [(i, i, "user", "x" * 50, False) for i in range(1, 11)]
    capped = lp.cap_buffer_to_recent(flat, max_chars=200)
    assert len(capped) < len(flat)
    # The kept turns are the MOST RECENT, in original (oldest-first) order.
    ids = [t for t, _s, _r, _x, _u in capped]
    assert ids == sorted(ids)
    assert ids[-1] == 10  # newest turn always retained
    assert ids[0] > 1  # at least the oldest turn was dropped


def test_cap_buffer_always_keeps_at_least_most_recent_turn():
    # A single turn far larger than the budget must still survive — an
    # empty buffer would defeat the Librarian entirely.
    flat = [(1, 1, "user", "x" * 1000, False)]
    capped = lp.cap_buffer_to_recent(flat, max_chars=10)
    assert capped == flat


def test_cap_buffer_empty_input():
    assert lp.cap_buffer_to_recent([], max_chars=100) == []


def test_cap_buffer_logs_dropped_count(caplog):
    flat = [(i, i, "user", "x" * 50, False) for i in range(1, 11)]
    with caplog.at_level("INFO", logger="agent_mem_daemon.librarian_prompt"):
        lp.cap_buffer_to_recent(flat, max_chars=200)
    assert any("truncated to recency budget" in r.message for r in caplog.records)


def test_cap_buffer_no_log_when_nothing_dropped(caplog):
    flat = [(1, 1, "user", "short", False)]
    with caplog.at_level("INFO", logger="agent_mem_daemon.librarian_prompt"):
        lp.cap_buffer_to_recent(flat, max_chars=10_000)
    assert not any("truncated" in r.message for r in caplog.records)


def test_buffer_to_prompt_text_bounds_formatted_block():
    # Build a snapshot with many large turns, then prove the formatted
    # rolling-buffer block honours the char cap — this is what actually
    # bounds `prompt_chars` in librarian.scan.
    big_turns = [[_ev("UserPromptSubmit", {"text": "y" * 500})] for _ in range(50)]
    snap = _snap(big_turns)
    budget = 2_000
    formatted, flat = lp.buffer_to_prompt_text(snap, max_chars=budget)
    # The formatted block is bounded (allow a small per-line newline slop).
    assert len(formatted) <= budget + 200
    # Without the cap the same snapshot would render far larger.
    uncapped = lp.format_rolling_buffer(lp.flatten_buffer(snap))
    assert len(uncapped) > budget * 5
    # The returned flat is the SAME capped window the model saw, newest-kept.
    assert len(flat) < 50
    assert flat[-1][0] == 50


def test_buffer_to_prompt_text_default_cap_is_module_constant():
    # The default budget must be the named, tunable module constant.
    assert lp.ROLLING_BUFFER_MAX_CHARS == lp.ROLLING_BUFFER_BUDGET_TOKENS * 4


# ── read_index_md / build_applies_when_table ──────────────────────────


def test_read_index_md_missing_returns_sentinel(tmp_path):
    out = lp.read_index_md(tmp_path / "knowledge")
    assert "no entries" in out


def test_read_index_md_returns_file_contents(tmp_path):
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "index.md").write_text("# Knowledge Base Index\n\n| Article | ... |\n", encoding="utf-8")
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
        "applies-when: |\n" + "".join(f"  {ln}\n" for ln in applies_when) + "---\n\n# title\n"
    )
    p.write_text(body, encoding="utf-8")


def test_build_applies_when_table_only_confirmed(tmp_path):
    k = tmp_path / "knowledge"
    k.mkdir()
    _write_entry(
        k,
        "global/tooling/factory.md",
        "confirmed",
        [
            "designing or building any new API",
            "decisions about how clients construct service objects",
        ],
    )
    _write_entry(k, "global/tooling/draft.md", "provisional", ["this should not appear"])
    _write_entry(k, "_archive/tooling/old.md", "confirmed", ["never surface archived entries"])
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
    (k / "global" / "README.md").write_text("# Global\nCross-project lessons.\n", encoding="utf-8")
    (k / "global" / "tooling" / "README.md").write_text(
        "# Tooling\nNotes about toolchains.\n", encoding="utf-8"
    )
    (k / "global" / "tooling" / "uv-basics.md").write_text(
        "---\nid: uv-basics\n---\n# uv basics\n", encoding="utf-8"
    )
    (k / "projects" / "README.md").write_text("# Projects\nPer-repo entries.\n", encoding="utf-8")
    (k / "index.md").write_text("# Knowledge Base Index\n\n| Article | Scope |\n", encoding="utf-8")
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
    snap = {
        "turns": [{"events": [{"type": "PostToolUse", "payload": {"project_slug": "my-proj"}}]}]
    }
    assert lp.derive_project_slug(snap) == "my-proj"


def test_derive_project_slug_unknown_when_nothing():
    assert lp.derive_project_slug({}) == "unknown"


def test_derive_project_bucket_uses_session_bucket_resolver(tmp_path, monkeypatch):
    """``derive_project_bucket`` must route through
    ``aliases.session_bucket`` so the daemon's path generation stays in
    sync with the hook side. We point AGENT_MEM_HOME at tmp_path, lay
    down a real bucket dir + a git repo, and verify the resolver picks
    the bucket name (not the slug) for the LLM prompt."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    # On-disk bucket named "agent-mem", git repo with that basename,
    # session slug from the (hypothetical) git remote.
    (tmp_path / "knowledge" / "projects" / "agent-mem").mkdir(parents=True)
    repo = tmp_path / "agent-mem"
    repo.mkdir()
    (repo / ".git").mkdir()

    snap = {"cwd": str(repo), "project_slug": "github.com-nickroci-ultan"}
    assert lp.derive_project_bucket(snap) == "agent-mem"


def test_derive_project_bucket_falls_back_to_slug_without_cwd():
    """Older snapshots / weird inputs without cwd should still produce
    a usable value — return the (slugified) slug rather than crashing.
    ``derive_project_slug`` flattens non-alphanum to dashes, hence
    ``github.com-...`` becomes ``github-com-...``."""
    snap = {"project_slug": "github.com-nickroci-ultan"}
    assert lp.derive_project_bucket(snap) == "github-com-nickroci-ultan"


# ── system prompt / user message split (prompt-cache refactor) ────────
#
# The big static instructions now live in a byte-stable ``system_prompt``
# (the cacheable prefix); the five per-scan dynamic blocks (project slug,
# rolling buffer, library snapshot, applies-when table, repair tasks) move
# to the first user message. The combined view (system + user) is what the
# model effectively sees, so legacy content assertions check that.


def test_template_contains_dynamic_placeholders():
    # The raw template still carries the two STATIC schema placeholders
    # (substituted into the system prompt). The five DYNAMIC placeholders
    # have been removed — their data now rides in the user message.
    t = lp.load_prompt_template()
    for gone in (
        "{{PROJECT_SLUG}}",
        "{{ROLLING_BUFFER}}",
        "{{LIBRARY_SNAPSHOT}}",
        "{{APPLIES_WHEN_TABLE}}",
        "{{REPAIR_TASKS}}",
    ):
        assert gone not in t, f"dynamic placeholder should be gone from template: {gone}"
    for static in ("{{ACTION_TYPES}}", "{{RESPONSE_SHAPE}}"):
        assert static in t, f"static placeholder missing from template: {static}"
    # BM25_SEEDS used to be a placeholder; it's now an in-process research
    # tool the Librarian invokes itself. Regression guard:
    assert "{{BM25_SEEDS}}" not in t
    assert "bm25_search" in t  # the Librarian must be told about the tool


def test_system_prompt_is_byte_stable():
    # The whole point of the refactor: the system prompt must be identical
    # on every call so the engine's prefix cache hits across scans.
    assert lp.librarian_system_prompt() == lp.librarian_system_prompt()


def test_system_prompt_has_no_placeholders_or_dynamic_markers():
    sp = lp.librarian_system_prompt()
    # No leftover template placeholders.
    assert "{{" not in sp and "}}" not in sp
    # The instructions survived.
    assert "You are the Librarian" in sp
    assert "THE SALIENCE TEST" in sp
    # The dynamic data tags are described (referenced) but NOT filled with
    # data — the data lives only in the user message. The opening data
    # lines must NOT appear in the static prefix.
    assert "slug:" not in sp
    assert "<rolling_buffer>\n" not in sp  # the data block opener, not prose


def test_system_prompt_describes_all_action_types():
    """The Librarian must know about every legal action type by name. The
    ACTION_TYPES table lives in the (static) system prompt."""
    sp = lp.librarian_system_prompt()
    for action in (
        "write_entry",
        "update_entry",
        "merge_entries",
        "move_entry",
        "archive_entry",
        "update_readme",
        "add_wikilink",
        "split_folder",
        "abstract_entries",
    ):
        assert action in sp, f"action type {action!r} missing from system prompt"


def test_system_prompt_documents_abstract_entries_aha_gate():
    """The Librarian system prompt must carry the four-part aha gate plus the
    GOOD and MUST-REJECT examples so the model only proposes genuine
    abstractions."""
    sp = lp.librarian_system_prompt()
    assert "AHA GATE" in sp
    assert "Predictive lift" in sp
    assert "linting across languages" in sp  # the GOOD example
    assert "likes good code" in sp  # a MUST-REJECT example


def test_user_message_carries_all_dynamic_blocks():
    msg = lp.build_librarian_user_message(
        project_slug="acme-widget-svc",
        rolling_buffer="[1] [user] hi",
        library_snapshot="## Tree\n```\nglobal/\n```",
        applies_when_table="factory | global | designing or building any new API",
    )
    # Each block present, tagged, with its data.
    assert "<project_context>\nslug: acme-widget-svc\n</project_context>" in msg
    assert "<rolling_buffer>\n[1] [user] hi\n</rolling_buffer>" in msg
    assert "<library_snapshot>\n## Tree\n```\nglobal/\n```\n</library_snapshot>" in msg
    assert (
        "<applies_when_table>\nfactory | global | designing or building any new API\n"
        "</applies_when_table>" in msg
    )
    # No instructions leaked into the user message — it is data only.
    assert "You are the Librarian" not in msg
    assert "THE SALIENCE TEST" not in msg


def test_user_message_empty_value_fallbacks():
    # Preserve the old assemble_prompt fallbacks: "unknown"/"(empty)"/sentinel.
    msg = lp.build_librarian_user_message(
        project_slug="",
        rolling_buffer="",
        library_snapshot="",
        applies_when_table="",
    )
    assert "slug: unknown" in msg
    assert "<rolling_buffer>\n(empty)\n</rolling_buffer>" in msg
    assert "<library_snapshot>\n(empty)\n</library_snapshot>" in msg
    assert "<applies_when_table>\n(empty)\n</applies_when_table>" in msg
    assert lp._NO_REPAIR_TASKS in msg


# ── format_repair_tasks / repair-task prompt block ────────────────────


def test_format_repair_tasks_empty_returns_sentinel():
    out = lp.format_repair_tasks([])
    assert out == lp._NO_REPAIR_TASKS
    assert "none" in out.lower()


def test_format_repair_tasks_renders_each_field():
    from agent_mem_daemon import repair_queue

    tasks = [
        repair_queue.RepairTask(
            kind=repair_queue.KIND_BROKEN_WIKILINK,
            file="global/python/foo.md",
            target="global/ghost/bar",
            context="…see [[global/ghost/bar]] for details…",
        )
    ]
    out = lp.format_repair_tasks(tasks)
    assert "kind: broken_wikilink" in out
    assert "file: global/python/foo.md" in out
    assert "target: global/ghost/bar" in out
    assert "context: …see [[global/ghost/bar]] for details…" in out


def test_user_message_renders_repair_tasks_block():
    from agent_mem_daemon import repair_queue

    tasks = [
        repair_queue.RepairTask(
            kind=repair_queue.KIND_BROKEN_WIKILINK,
            file="global/python/foo.md",
            target="global/ghost/bar",
            context="ctx",
        )
    ]
    # The dedicated highest-priority SECTION (instructions) lives in the
    # system prompt; the rendered TASK data lives in the user message.
    sp = lp.librarian_system_prompt()
    assert "INTEGRITY-REPAIR TASKS" in sp
    assert "update_entry" in sp  # one of the prescribed repair actions
    msg = lp.build_librarian_user_message(
        project_slug="x",
        rolling_buffer="(empty)",
        library_snapshot="(empty)",
        applies_when_table="(empty)",
        repair_tasks=lp.format_repair_tasks(tasks),
    )
    assert "<repair_tasks>" in msg
    assert "target: global/ghost/bar" in msg


def test_user_message_repair_tasks_defaults_to_sentinel():
    # Callers that don't escalate anything need not pass repair_tasks.
    msg = lp.build_librarian_user_message(
        project_slug="x",
        rolling_buffer="(empty)",
        library_snapshot="(empty)",
        applies_when_table="(empty)",
    )
    assert lp._NO_REPAIR_TASKS in msg


def test_format_repair_tasks_renders_overcap_and_bad_frontmatter():
    from agent_mem_daemon import repair_queue

    tasks = [
        repair_queue.RepairTask(
            kind=repair_queue.KIND_OVERCAP_DIR,
            file="global/python",
            target="global/python",
            context="6 entries in global/python/ (cap 5): a, b, c, d, e, f",
        ),
        repair_queue.RepairTask(
            kind=repair_queue.KIND_BAD_FRONTMATTER,
            file="global/python/x.md",
            target="global/python/x.md",
            context="missing frontmatter fields in global/python/x.md: keywords",
        ),
    ]
    out = lp.format_repair_tasks(tasks)
    assert "kind: overcap_dir" in out
    assert "kind: bad_frontmatter" in out
    assert "file: global/python" in out


def test_prompt_template_dispatches_on_repair_kind():
    # The INTEGRITY-REPAIR section must teach the Librarian all three kinds
    # and the EXISTING action it should propose for each.
    tmpl = lp.load_prompt_template()
    assert "kind: broken_wikilink" in tmpl
    assert "kind: overcap_dir" in tmpl
    assert "kind: bad_frontmatter" in tmpl
    # Over-cap → split_folder / move_entry rebalance.
    assert "split_folder" in tmpl
    assert "move_entry" in tmpl
    # Bad frontmatter → update_entry that re-serialises valid frontmatter.
    assert "re-serialises valid YAML" in tmpl
    # Repairs are integrity fixes, not salience judgments.
    assert "salience_signal: null" in tmpl


def test_user_message_renders_overcap_task_block():
    from agent_mem_daemon import repair_queue

    tasks = [
        repair_queue.RepairTask(
            kind=repair_queue.KIND_OVERCAP_DIR,
            file="global/python",
            target="global/python",
            context="6 entries in global/python/ (cap 5): a, b, c, d, e, f",
        )
    ]
    msg = lp.build_librarian_user_message(
        project_slug="x",
        rolling_buffer="(empty)",
        library_snapshot="(empty)",
        applies_when_table="(empty)",
        repair_tasks=lp.format_repair_tasks(tasks),
    )
    assert "kind: overcap_dir" in msg
    # The split_folder repair guidance lives in the (static) system prompt.
    assert "split_folder" in lp.librarian_system_prompt()


# ── Secrets-redaction guidance ───────────────────────────────────────


def test_system_prompt_includes_secrets_redaction_rule():
    """Ground rule 7 must surface in the system prompt — covers the
    common credential patterns so the Librarian can't claim ignorance."""
    sp = lp.librarian_system_prompt()
    assert "NEVER quote secrets or credentials" in sp
    # Pin a handful of the named patterns so a future prompt edit
    # can't silently drop them.
    for pattern in ("API keys", "ghp_", "AKIA", "sk-", "BEGIN ... PRIVATE KEY"):
        assert pattern in sp, f"secrets rule missing pattern: {pattern!r}"
