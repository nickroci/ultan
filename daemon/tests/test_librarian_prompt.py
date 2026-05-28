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
        "turns": [{"events": evs, "started_at": 0.0, "sealed_at": 1.0} for evs in turns_events],
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
    snap = _snap([[_ev("UserPromptSubmit", {"text": "always wrap errors", "user_asserted": True})]])
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


# ── cap_buffer_to_recent / buffer_to_prompt_text recency cap ──────────


def test_cap_buffer_keeps_all_when_under_budget():
    flat = [(i, "user", f"turn {i}", False) for i in range(1, 6)]
    capped = lp.cap_buffer_to_recent(flat, max_chars=10_000)
    assert capped == flat


def test_cap_buffer_drops_oldest_turns_over_budget():
    # 10 turns of ~50 chars of text each. A 200-char budget should keep
    # only the most-recent few and drop the oldest.
    flat = [(i, "user", "x" * 50, False) for i in range(1, 11)]
    capped = lp.cap_buffer_to_recent(flat, max_chars=200)
    assert len(capped) < len(flat)
    # The kept turns are the MOST RECENT, in original (oldest-first) order.
    ids = [t for t, _r, _x, _u in capped]
    assert ids == sorted(ids)
    assert ids[-1] == 10  # newest turn always retained
    assert ids[0] > 1  # at least the oldest turn was dropped


def test_cap_buffer_always_keeps_at_least_most_recent_turn():
    # A single turn far larger than the budget must still survive — an
    # empty buffer would defeat the Librarian entirely.
    flat = [(1, "user", "x" * 1000, False)]
    capped = lp.cap_buffer_to_recent(flat, max_chars=10)
    assert capped == flat


def test_cap_buffer_empty_input():
    assert lp.cap_buffer_to_recent([], max_chars=100) == []


def test_cap_buffer_logs_dropped_count(caplog):
    flat = [(i, "user", "x" * 50, False) for i in range(1, 11)]
    with caplog.at_level("INFO", logger="agent_mem_daemon.librarian_prompt"):
        lp.cap_buffer_to_recent(flat, max_chars=200)
    assert any("truncated to recency budget" in r.message for r in caplog.records)


def test_cap_buffer_no_log_when_nothing_dropped(caplog):
    flat = [(1, "user", "short", False)]
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
    # BM25_SEEDS used to be a placeholder; it's now an in-process research
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
        "write_entry",
        "update_entry",
        "merge_entries",
        "move_entry",
        "archive_entry",
        "update_readme",
        "add_wikilink",
        "split_folder",
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


def test_assemble_prompt_renders_repair_tasks_block():
    from agent_mem_daemon import repair_queue

    tasks = [
        repair_queue.RepairTask(
            kind=repair_queue.KIND_BROKEN_WIKILINK,
            file="global/python/foo.md",
            target="global/ghost/bar",
            context="ctx",
        )
    ]
    p = lp.assemble_prompt(
        project_slug="x",
        rolling_buffer="(empty)",
        library_snapshot="(empty)",
        applies_when_table="(empty)",
        repair_tasks=lp.format_repair_tasks(tasks),
    )
    # The dedicated highest-priority section and the rendered task land in
    # the prompt, and the Librarian is told to fix it with an EXISTING action.
    assert "INTEGRITY-REPAIR TASKS" in p
    assert "target: global/ghost/bar" in p
    assert "update_entry" in p  # one of the prescribed repair actions
    assert "{{" not in p


def test_assemble_prompt_repair_tasks_defaults_to_sentinel():
    # Callers that don't escalate anything need not pass repair_tasks.
    p = lp.assemble_prompt(
        project_slug="x",
        rolling_buffer="(empty)",
        library_snapshot="(empty)",
        applies_when_table="(empty)",
    )
    assert lp._NO_REPAIR_TASKS in p
    assert "{{" not in p


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


def test_assemble_prompt_renders_overcap_task_block():
    from agent_mem_daemon import repair_queue

    tasks = [
        repair_queue.RepairTask(
            kind=repair_queue.KIND_OVERCAP_DIR,
            file="global/python",
            target="global/python",
            context="6 entries in global/python/ (cap 5): a, b, c, d, e, f",
        )
    ]
    p = lp.assemble_prompt(
        project_slug="x",
        rolling_buffer="(empty)",
        library_snapshot="(empty)",
        applies_when_table="(empty)",
        repair_tasks=lp.format_repair_tasks(tasks),
    )
    assert "kind: overcap_dir" in p
    assert "split_folder" in p
    assert "{{" not in p


# ── Secrets-redaction guidance ───────────────────────────────────────


def test_assemble_prompt_includes_secrets_redaction_rule():
    """Ground rule 7 must surface in the assembled prompt — covers the
    common credential patterns so the Librarian can't claim ignorance."""
    p = lp.assemble_prompt(
        project_slug="x",
        rolling_buffer=[],
        library_snapshot="(empty)",
        applies_when_table="(none)",
    )
    assert "NEVER quote secrets or credentials" in p
    # Pin a handful of the named patterns so a future prompt edit
    # can't silently drop them.
    for pattern in ("API keys", "ghp_", "AKIA", "sk-", "BEGIN ... PRIVATE KEY"):
        assert pattern in p, f"secrets rule missing pattern: {pattern!r}"
