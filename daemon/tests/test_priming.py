"""Tests for the Tier-1 ambient-priming refresher.

The module under test (``agent_mem_daemon.priming``) runs after every
Scholar batch. It BM25-searches the library against a flattened view of
the recent buffer, boosts by each entry's ``reinforced`` counter, and
writes the top-K most-relevant entries to ``~/.agent-mem/hot-context.md``
within a small char budget. The UserPromptSubmit hook injects the file
contents as ``additionalContext`` for the next turn.

These tests pin the file's output shape, the boost interaction, the
char-budget trimming behaviour, idempotence, the empty-input path, and
the atomic-write guarantee.

The BM25 backend is the real ``bm25`` package from ``agent-mem-search``
(declared as a path dep in ``daemon/pyproject.toml``). We let it run
against on-disk fixtures rather than mocking — keeps the test honest.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest

from agent_mem_daemon import priming

from .conftest import build_library, render_entry, write_entry

# ── Fixture helpers ───────────────────────────────────────────────────

# This module renders entries with a slightly different default body than
# the shared default ("...The rule applies when..."). Bind that template
# once so the renderer stays single-sourced in conftest while these
# fixtures keep their exact, BM25-/rerank-sensitive bytes.
_PRIMING_BODY_TEMPLATE = "Body for {id_}. The rule applies when {applies_when}."

# Longer multi-line body for the python/uv seed entry — engineered so the
# relevance test scores the python/uv entry highest by BM25.
_PRIMING_UV_BODY = (
    "Always use uv for python package management. Never pip. "
    "uv is faster, deterministic, and the project standard. "
    "Install dependencies with `uv add`, run scripts with `uv run`."
)

_entry = functools.partial(render_entry, default_body_template=_PRIMING_BODY_TEMPLATE)
_write = write_entry


def _seed_library(root: Path) -> Path:
    """Build an 8-entry library across two top-level folders.

    The buffer text used in the relevance test ("python uv package
    manager") is engineered to score the python/uv entry highest by
    BM25.
    """
    return build_library(
        root,
        include_type_hints=True,
        uv_body=_PRIMING_UV_BODY,
        default_body_template=_PRIMING_BODY_TEMPLATE,
    )


@pytest.fixture(autouse=True)
def _isolate_bm25_index(tmp_path, monkeypatch):
    """Force BM25 to live in tmp so concurrent tests don't share
    pickled indices (``bm25.load_or_build`` writes ``.bm25.idx``
    next to the knowledge dir's parent — which IS tmp_path here)."""
    monkeypatch.setenv("AGENT_MEM_HOME", str(tmp_path))
    yield


# ── tests ─────────────────────────────────────────────────────────────


def test_extract_buffer_text_concatenates_proposals_and_interrupts():
    packets = [
        {
            "session_id": "s1",
            "proposals": [
                {
                    "action": "write_entry",
                    "reasoning": "the user said to use uv",
                    "body": "Always use uv for python.",
                    "path": "global/python/use-uv.md",
                },
            ],
            "interrupts": [
                {
                    "lesson_id": "abc",
                    "evidence": [{"turn_id": 1, "role": "user", "quote": "we ship with uv"}],
                    "matching_applies_when": "installing python deps",
                },
            ],
        },
    ]
    text = priming.extract_buffer_text(packets)
    assert "use uv for python" in text
    assert "we ship with uv" in text
    assert "installing python deps" in text
    # Pure-id fields should not leak in.
    assert "abc" not in text
    assert "write_entry" not in text


def test_extract_buffer_text_handles_empty_input():
    assert priming.extract_buffer_text([]) == ""
    assert priming.extract_buffer_text([{}]) == ""
    assert priming.extract_buffer_text([{"proposals": [], "interrupts": []}]) == ""


def test_refresh_hot_context_writes_top_k_entries(tmp_path):
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"

    priming.refresh_hot_context(
        k,
        rolling_buffer_text=(
            "we were debugging a python package install. "
            "tried pip and it broke. switching to uv for the python deps."
        ),
        out_path=out,
        top_k=5,
    )

    body = out.read_text(encoding="utf-8")
    assert body.startswith("## Ultan — your library says")
    assert "Wikilinks resolve to real entries" in body
    assert "ultan-search" in body
    # The python/uv entry should be the strongest match.
    assert "[[global/python/use-uv-not-pip]]" in body
    # 1-to-top_k bullets — the rerank stage filters candidates below its
    # relevance floor, so a tightly-targeted query (this one is python+uv
    # only) can legitimately surface fewer than top_k matches. Pre-rerank
    # this test asserted exactly 5; with rerank, "quality > quantity."
    bullet_lines = [line for line in body.splitlines() if line.startswith("- [[")]
    assert 1 <= len(bullet_lines) <= 5


def test_refresh_respects_char_budget(tmp_path):
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"

    # Coherent English buffer — the cross-encoder reranker needs sentence
    # structure to score candidates, not bare keyword soup. The buffer
    # touches both python/uv and git/rebase entries so multiple matches
    # land above the rerank floor and the budget actually has to trim.
    priming.refresh_hot_context(
        k,
        rolling_buffer_text=(
            "we were debugging python package installs, switching between pip and uv, "
            "and ran into trouble rebasing a feature branch onto main after a force-push. "
            "ruff also flagged some imports that needed reordering."
        ),
        out_path=out,
        top_k=20,  # ask for far more than fits
        char_budget=700,
    )

    body = out.read_text(encoding="utf-8")
    assert len(body) <= 700, f"output {len(body)} chars exceeds budget"
    # The header MUST still appear — trimming should drop bullets, not
    # the framing.
    assert body.startswith("## Ultan — your library says")


def test_refresh_reinforced_counter_boosts_rank(tmp_path):
    """With two entries scoring similarly on BM25, the one with a
    larger ``reinforced`` counter should rise to the top — that's the
    "user reasserted this so it weighs more" semantics.

    BM25Okapi clamps IDF to zero for any term whose document frequency
    is >= N/2, so we need enough decoy documents in the corpus for the
    target tokens to remain informative. The seeded library already has
    8 entries spread across disjoint vocabularies (python vs git), so
    adding the two target entries on top gives plenty of headroom for
    the shared "uv tool" tokens to score above zero.
    """
    k = _seed_library(tmp_path)

    # Two near-identical entries (same tokens), but one has reinforced=10.
    # They share enough rare-in-corpus vocabulary ("vinglex", "qwoxlate")
    # to score nonzero BM25 against the buffer.
    _write(
        k / "global" / "weak-entry.md",
        _entry(
            id_="weak-entry",
            title="Weak entry about vinglex",
            applies_when="working with vinglex tools",
            keywords=["vinglex", "qwoxlate"],
            body="The vinglex framework and qwoxlate runtime.",
        ),
    )
    _write(
        k / "global" / "strong-entry.md",
        _entry(
            id_="strong-entry",
            title="Strong entry about vinglex",
            applies_when="working with vinglex tools",
            keywords=["vinglex", "qwoxlate"],
            reinforced=10,
            body="Another take on the vinglex framework and qwoxlate runtime.",
        ),
    )

    out = tmp_path / "hot-context.md"
    priming.refresh_hot_context(
        k,
        rolling_buffer_text="vinglex qwoxlate tools",
        out_path=out,
        top_k=2,
    )

    body = out.read_text(encoding="utf-8")
    lines = [line for line in body.splitlines() if line.startswith("- [[")]
    assert len(lines) == 2
    # Strong entry must come first because reinforcement boosted it.
    assert "strong-entry" in lines[0]
    assert "weak-entry" in lines[1]
    # And its bullet should carry the (×10) marker.
    assert "(×10)" in lines[0]
    # The weak one has no reinforced count, so no parenthetical.
    assert "(×" not in lines[1]


def _write_useful_pair(k: Path) -> None:
    """Two entries identical except for their fired / fired-helpful counts:
    ``proven`` was relied on 8/10 surfaces, ``ignored`` 0/10."""
    (k / "global").mkdir(parents=True, exist_ok=True)
    _write(
        k / "global" / "proven.md",
        _entry(
            id_="proven",
            title="Proven entry",
            applies_when="working with foo",
            keywords=["foo"],
            fired=10,
            fired_helpful=8,
        ),
    )
    _write(
        k / "global" / "ignored.md",
        _entry(
            id_="ignored",
            title="Ignored entry",
            applies_when="working with foo",
            keywords=["foo"],
            fired=10,
            fired_helpful=0,
        ),
    )


def test_usefulness_breaks_tie_between_equal_rerank_candidates(tmp_path):
    """When two candidates score identically, the one actually relied on
    (higher fired-helpful/fired) wins the tiebreak."""
    k = tmp_path / "knowledge"
    _write_useful_pair(k)
    # Identical rerank scores — only the usefulness tiebreaker can order them.
    hits = [
        (k / "global" / "ignored.md", 3.0),
        (k / "global" / "proven.md", 3.0),
    ]
    ranked = priming._boost_with_reinforcement(hits)
    order = [p.stem for p, _score, _r in ranked]
    assert order == ["proven", "ignored"]


def test_usefulness_does_not_override_a_clear_rerank_gap(tmp_path):
    """The tiebreaker is gentle: a clearly stronger rerank score wins even
    when that entry is chronically ignored and the weaker one is maximally
    useful. Usefulness can only break near-ties, not flip real relevance."""
    k = tmp_path / "knowledge"
    _write_useful_pair(k)
    # proven (useful) gets the weak rerank score; ignored gets a strong one.
    hits = [
        (k / "global" / "proven.md", 2.0),
        (k / "global" / "ignored.md", 5.0),
    ]
    ranked = priming._boost_with_reinforcement(hits)
    assert ranked[0][0].stem == "ignored"


def test_refresh_idempotent(tmp_path):
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"
    buffer = "python uv package install"

    priming.refresh_hot_context(k, buffer, out, top_k=3)
    first_body = out.read_text(encoding="utf-8")
    first_mtime = out.stat().st_mtime_ns

    # Force the kernel to advance the mtime granularity threshold.
    # Some filesystems have 1s resolution; bump deliberately.
    os.utime(out, (out.stat().st_atime - 5, out.stat().st_mtime - 5))
    pinned_mtime = out.stat().st_mtime_ns

    priming.refresh_hot_context(k, buffer, out, top_k=3)
    second_body = out.read_text(encoding="utf-8")
    second_mtime = out.stat().st_mtime_ns

    # Byte-identical output is the strict idempotence guarantee.
    assert first_body == second_body
    # No write should have happened on the no-op call — the pinned
    # backdated mtime should survive untouched.
    assert second_mtime == pinned_mtime, (
        f"second refresh rewrote the file (mtime changed from "
        f"{pinned_mtime} to {second_mtime}); expected idempotent skip"
    )
    # Sanity: the first call DID write (or we're testing nothing).
    assert first_mtime != pinned_mtime


def test_refresh_empty_buffer_clears_existing_file(tmp_path):
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"

    # Seed a non-empty hot-context.
    priming.refresh_hot_context(k, "python uv install", out, top_k=2)
    assert out.read_text(encoding="utf-8").startswith("## Ultan — your library says")

    # Empty buffer on the next batch — stale priming must be cleared
    # (we don't want the agent primed by something no longer relevant).
    priming.refresh_hot_context(k, "", out, top_k=2)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


def test_refresh_empty_buffer_no_existing_file_skips(tmp_path):
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"
    assert not out.exists()

    # Empty buffer + no existing file — function must return silently
    # without creating an empty file. (The hook treats missing and
    # empty the same, but not writing at all is the cleaner default.)
    result = priming.refresh_hot_context(k, "", out, top_k=2)
    assert result is None
    assert not out.exists()


def test_refresh_no_matches_clears_existing_file(tmp_path):
    """Buffer has content but no BM25 hit lands in the library. Stale
    priming must be cleared so we don't keep injecting yesterday's
    entries."""
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"

    # Seed.
    priming.refresh_hot_context(k, "python uv install", out, top_k=2)
    assert out.read_text(encoding="utf-8").strip()

    # Buffer text contains only tokens guaranteed not to match the
    # library — random latinate strings.
    priming.refresh_hot_context(
        k,
        rolling_buffer_text="zxcvbnm qwertyuiop asdfghjkl",
        out_path=out,
        top_k=2,
    )
    assert out.read_text(encoding="utf-8") == ""


def test_refresh_atomic_write_failure_leaves_original_intact(tmp_path, monkeypatch):
    """If the write fails partway through, the previously-written file
    must remain readable and unchanged. The function must NOT propagate
    the exception — Scholar's caller wraps it but defensive coding here
    matters too."""
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"

    # First successful write.
    priming.refresh_hot_context(k, "python uv install", out, top_k=2)
    original = out.read_text(encoding="utf-8")
    assert original

    # Break ``os.replace`` so the atomic rename always fails. The temp
    # file gets created (mkstemp succeeds) and written, but the final
    # swap raises. ``_atomic_write`` should re-raise, ``refresh_hot_context``
    # should catch and log, and the original file should be untouched.
    real_replace = os.replace
    call_log = {"n": 0}

    def _boom(src, dst):
        if str(dst) == str(out):
            call_log["n"] += 1
            raise OSError("simulated atomic-write failure")
        return real_replace(src, dst)

    monkeypatch.setattr(priming.os, "replace", _boom)

    # Refresh with DIFFERENT buffer text so a successful write would
    # produce different bytes (proves the original survived rather than
    # being coincidentally identical).
    priming.refresh_hot_context(
        k,
        rolling_buffer_text="git rebase branch merge commit history",
        out_path=out,
        top_k=2,
    )

    # The failure must have been triggered.
    assert call_log["n"] >= 1
    # File still exists and matches the pre-failure content.
    assert out.exists()
    assert out.read_text(encoding="utf-8") == original

    # And no orphaned tmp files left lying around.
    leftover = list(tmp_path.glob("hot-context.md.*.tmp"))
    assert leftover == [], f"leftover tmp files: {leftover}"


def test_refresh_never_raises_on_garbage_inputs(tmp_path):
    """Defensive smoke: weird inputs (None, non-strings) must not blow
    up the caller."""
    out = tmp_path / "hot-context.md"
    # Non-existent knowledge dir.
    priming.refresh_hot_context(tmp_path / "does-not-exist", "x", out)
    # Non-string buffer.
    priming.refresh_hot_context(
        tmp_path,
        None,  # type: ignore[arg-type]
        out,
    )
    # No assertion needed — the test passes if no exception escapes.


def test_bullet_format_includes_title_hook_and_count(tmp_path):
    """Wire-shape pin: ``- [[path]] (×N) — Title — applies-when hook``
    for reinforced; ``- [[path]] — Title — hook`` for unreinforced.
    Tested directly so a formatting refactor can't silently change the
    on-disk shape.

    Uses realistic English content — the cross-encoder reranker scores
    based on natural-language semantic match, so synthetic nonsense
    words ("vinglex", "qwoxlate") would fall below the rerank floor and
    never reach this assertion. The query and the two layered entries
    are written so each scores positively against its targeted entry.
    """
    k = _seed_library(tmp_path)

    reinforced_path = k / "global" / "pin-docker-versions.md"
    _write(
        reinforced_path,
        _entry(
            id_="pin-docker-versions",
            title="Pin docker image versions",
            applies_when="writing a Dockerfile or docker-compose service",
            keywords=["docker", "image", "version", "pin"],
            reinforced=4,
        ),
    )
    plain_path = k / "global" / "use-ripgrep.md"
    _write(
        plain_path,
        _entry(
            id_="use-ripgrep",
            title="Prefer ripgrep over grep",
            applies_when="searching through a code repository",
            keywords=["ripgrep", "grep", "search"],
        ),
    )

    out = tmp_path / "hot-context.md"
    priming.refresh_hot_context(
        k,
        rolling_buffer_text=(
            "we need to pin our docker image versions in the Dockerfile, "
            "and use ripgrep when searching through the code repository."
        ),
        out_path=out,
        top_k=2,
    )
    body = out.read_text(encoding="utf-8")

    assert (
        "[[global/pin-docker-versions]] (×4) — Pin docker image versions — "
        "writing a Dockerfile or docker-compose service" in body
    )
    assert (
        "[[global/use-ripgrep]] — Prefer ripgrep over grep — "
        "searching through a code repository" in body
    )


def test_scope_bonus_prefers_current_project_then_global(tmp_path):
    """Three entries with identical RRF scores but different scopes:
    current-project gets the +0.020 bump, global gets +0.005, other
    project gets nothing. Ordering after _boost_with_reinforcement must
    reflect that."""
    k = tmp_path / "knowledge"
    cur_path = k / "projects" / "myproj" / "entry.md"
    other_path = k / "projects" / "otherproj" / "entry.md"
    global_path = k / "global" / "entry.md"
    _write(cur_path, _entry(id_="cur", title="Cur", applies_when="x", keywords=["x"]))
    _write(other_path, _entry(id_="oth", title="Oth", applies_when="x", keywords=["x"]))
    _write(global_path, _entry(id_="glb", title="Glb", applies_when="x", keywords=["x"]))

    # Tied RRF score so only the scope bonus orders them.
    hits = [(other_path, 0.05), (global_path, 0.05), (cur_path, 0.05)]
    ranked = priming._boost_with_reinforcement(hits, knowledge_dir=k, current_project_slug="myproj")
    paths = [str(p.relative_to(k)) for p, _, _ in ranked]
    assert paths == [
        "projects/myproj/entry.md",
        "global/entry.md",
        "projects/otherproj/entry.md",
    ]


def test_scope_bonus_uses_alias_map_to_resolve_bucket_slug(tmp_path):
    """A bucket named ``agent-mem`` whose canonical slug is declared in
    project-aliases.json as ``github.com-nickroci-ultan`` should win the
    current-project boost when that session slug arrives. The autouse
    fixture points AGENT_MEM_HOME at tmp_path, so the alias file lives
    at tmp_path/project-aliases.json for the duration of the test."""
    k = tmp_path / "knowledge"
    cur_path = k / "projects" / "agent-mem" / "entry.md"
    other_path = k / "projects" / "vol-predictor" / "entry.md"
    _write(cur_path, _entry(id_="cur", title="Cur", applies_when="x", keywords=["x"]))
    _write(other_path, _entry(id_="oth", title="Oth", applies_when="x", keywords=["x"]))

    aliases_file = tmp_path / "project-aliases.json"
    aliases_file.write_text('{"agent-mem": "github.com-nickroci-ultan"}', encoding="utf-8")

    hits = [(other_path, 0.05), (cur_path, 0.05)]
    ranked = priming._boost_with_reinforcement(
        hits, knowledge_dir=k, current_project_slug="github.com-nickroci-ultan"
    )
    paths = [str(p.relative_to(k)) for p, _, _ in ranked]
    assert paths == ["projects/agent-mem/entry.md", "projects/vol-predictor/entry.md"]


def test_alias_helpers_default_to_identity_when_empty():
    """No alias file -> ``{}`` map -> bucket maps to itself. The
    canonical resolver lives in tools/search/aliases.py (shared with
    the hook side); we re-export it through priming so this test
    pins the daemon-visible behaviour."""
    assert priming.load_aliases(Path("/nonexistent/path/that/does/not/exist")) == {}
    assert priming.bucket_canonical_slug("agent-mem", {}) == "agent-mem"
    assert priming.bucket_canonical_slug(None, {}) is None
    assert (
        priming.bucket_canonical_slug("agent-mem", {"agent-mem": "github.com-nickroci-ultan"})
        == "github.com-nickroci-ultan"
    )


def test_scope_bonus_off_when_slug_missing(tmp_path):
    """Without a current_project_slug, only the global bonus applies —
    current-project and other-project entries get nothing, so the global
    one floats above ties."""
    k = tmp_path / "knowledge"
    other_path = k / "projects" / "otherproj" / "entry.md"
    global_path = k / "global" / "entry.md"
    _write(other_path, _entry(id_="oth", title="Oth", applies_when="x", keywords=["x"]))
    _write(global_path, _entry(id_="glb", title="Glb", applies_when="x", keywords=["x"]))

    hits = [(other_path, 0.05), (global_path, 0.05)]
    ranked = priming._boost_with_reinforcement(hits, knowledge_dir=k, current_project_slug=None)
    paths = [str(p.relative_to(k)) for p, _, _ in ranked]
    assert paths == ["global/entry.md", "projects/otherproj/entry.md"]


def test_idempotence_skips_write_after_initial(tmp_path, monkeypatch):
    """Strict version of the idempotence guarantee: count actual writes
    through the atomic-write path. Second call must not invoke it."""
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"

    write_count = {"n": 0}
    real_atomic = priming._atomic_write

    def _counting(path, content):
        write_count["n"] += 1
        return real_atomic(path, content)

    monkeypatch.setattr(priming, "_atomic_write", _counting)

    priming.refresh_hot_context(k, "python uv install", out, top_k=2)
    assert write_count["n"] == 1
    priming.refresh_hot_context(k, "python uv install", out, top_k=2)
    assert write_count["n"] == 1, "second identical call must be a no-op write"
