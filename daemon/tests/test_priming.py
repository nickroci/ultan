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

import os
from pathlib import Path
from typing import List, Optional

import pytest

from agent_mem_daemon import priming

# ── Fixture helpers ───────────────────────────────────────────────────


def _entry(
    *,
    id_: str,
    title: str,
    applies_when: str,
    keywords: List[str],
    reinforced: Optional[int] = None,
    body: str = "",
    scope: str = "global",
) -> str:
    """Render a valid library entry as a YAML-frontmattered markdown
    string. Matches the schema enforced by
    ``scholar_prompt._REQUIRED_FRONTMATTER_FIELDS``."""
    lines = [
        "---",
        f"id: {id_}",
        "type: lesson",
        f"scope: {scope}",
        "status: provisional",
        "confidence: 0.7",
        "applies-when: |",
    ]
    for line in applies_when.splitlines():
        lines.append(f"  {line}")
    lines.append("keywords: [" + ", ".join(keywords) + "]")
    lines.append(f'title: "{title}"')
    lines.append("created: 2026-05-19")
    lines.append("updated: 2026-05-19")
    lines.append("fired: 0")
    lines.append("fired-helpful: 0")
    if reinforced is not None:
        lines.append(f"reinforced: {reinforced}")
    lines.append("sources:")
    lines.append("  - manual")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    body_text = body or f"Body for {id_}. The rule applies when {applies_when}."
    lines.append(body_text)
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_library(root: Path) -> Path:
    """Build an 8-entry library across two top-level folders.

    The buffer text used in the relevance test ("python uv package
    manager") is engineered to score the python/uv entry highest by
    BM25.
    """
    k = root / "knowledge"
    _write(k / "README.md", "# knowledge root\n")
    _write(k / "global" / "README.md", "# global\n")
    _write(k / "global" / "python" / "README.md", "# python\n")
    _write(k / "global" / "git" / "README.md", "# git\n")

    _write(
        k / "global" / "python" / "use-uv-not-pip.md",
        _entry(
            id_="use-uv-not-pip",
            title="Always use uv for python",
            applies_when="installing python deps or running scripts",
            keywords=["python", "uv", "pip", "packaging"],
            body=(
                "Always use uv for python package management. Never pip. "
                "uv is faster, deterministic, and the project standard. "
                "Install dependencies with `uv add`, run scripts with `uv run`."
            ),
        ),
    )
    _write(
        k / "global" / "python" / "ruff-format.md",
        _entry(
            id_="ruff-format",
            title="Format python with ruff",
            applies_when="formatting python files",
            keywords=["python", "ruff", "format"],
        ),
    )
    _write(
        k / "global" / "python" / "type-hints.md",
        _entry(
            id_="type-hints",
            title="Always use type hints",
            applies_when="writing python functions",
            keywords=["python", "types", "mypy"],
        ),
    )
    _write(
        k / "global" / "git" / "no-force-push.md",
        _entry(
            id_="no-force-push",
            title="Never force-push to main",
            applies_when="pushing to git remotes",
            keywords=["git", "push", "remote"],
        ),
    )
    _write(
        k / "global" / "git" / "small-commits.md",
        _entry(
            id_="small-commits",
            title="Prefer small commits",
            applies_when="committing changes",
            keywords=["git", "commits", "history"],
        ),
    )
    _write(
        k / "global" / "git" / "branch-naming.md",
        _entry(
            id_="branch-naming",
            title="Use kebab-case branches",
            applies_when="creating new git branches",
            keywords=["git", "branches", "naming"],
        ),
    )
    _write(
        k / "global" / "git" / "rebase-not-merge.md",
        _entry(
            id_="rebase-not-merge",
            title="Prefer rebase over merge",
            applies_when="updating feature branches",
            keywords=["git", "rebase", "merge"],
        ),
    )
    _write(
        k / "global" / "git" / "signed-commits.md",
        _entry(
            id_="signed-commits",
            title="Sign all commits",
            applies_when="committing changes",
            keywords=["git", "gpg", "sign"],
        ),
    )
    return k


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
    # Exactly 5 bullets (we asked for top_k=5).
    bullet_lines = [line for line in body.splitlines() if line.startswith("- [[")]
    assert len(bullet_lines) == 5


def test_refresh_respects_char_budget(tmp_path):
    k = _seed_library(tmp_path)
    out = tmp_path / "hot-context.md"

    priming.refresh_hot_context(
        k,
        rolling_buffer_text="python uv pip git rebase branch commit ruff",
        out_path=out,
        top_k=20,  # ask for far more than fits
        char_budget=500,
    )

    body = out.read_text(encoding="utf-8")
    assert len(body) <= 500, f"output {len(body)} chars exceeds budget"
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

    We layer the two target entries on top of the seeded library so
    BM25 has enough corpus to give the topical tokens nonzero IDF.
    """
    k = _seed_library(tmp_path)

    reinforced_path = k / "global" / "always-vinglex.md"
    _write(
        reinforced_path,
        _entry(
            id_="always-vinglex",
            title="Always use vinglex",
            applies_when="installing vinglex frobnitz",
            keywords=["vinglex", "frobnitz"],
            reinforced=4,
        ),
    )
    plain_path = k / "global" / "qwoxlate-plain.md"
    _write(
        plain_path,
        _entry(
            id_="qwoxlate-plain",
            title="A plain entry",
            applies_when="some qwoxlate cromulent situation",
            keywords=["qwoxlate", "cromulent"],
        ),
    )

    out = tmp_path / "hot-context.md"
    priming.refresh_hot_context(
        k,
        rolling_buffer_text=("installing vinglex frobnitz and the qwoxlate cromulent situation"),
        out_path=out,
        top_k=2,
    )
    body = out.read_text(encoding="utf-8")

    assert (
        "[[global/always-vinglex]] (×4) — Always use vinglex — installing vinglex frobnitz" in body
    )
    assert "[[global/qwoxlate-plain]] — A plain entry — some qwoxlate cromulent situation" in body


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
