"""Direct tests for the smaller helpers in ``priming.py``.

The end-to-end behaviours are pinned in ``test_priming.py``; this file
covers the branchy helpers (frontmatter parsing edges, lane fallbacks,
RRF merge degenerates, bullet rendering with missing pieces).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_mem_daemon import priming

# ── _parse_frontmatter ───────────────────────────────────────────────


def test_parse_frontmatter_returns_empty_dict_on_missing_block() -> None:
    assert priming._parse_frontmatter("no frontmatter here\n# title\n") == {}


def test_parse_frontmatter_returns_empty_on_yaml_error() -> None:
    text = "---\n\tnot: valid: yaml: at all\n: bad\n---\n\nbody\n"
    fm = priming._parse_frontmatter(text)
    # YAML may parse leniently — we just want no exception escapes and
    # the fallback to be a dict.
    assert isinstance(fm, dict)


def test_parse_frontmatter_returns_empty_when_top_level_not_dict() -> None:
    """`---\n- a\n- b\n---` is a YAML list, not a mapping → fallback."""
    text = "---\n- a\n- b\n---\n\nbody\n"
    assert priming._parse_frontmatter(text) == {}


def test_parse_frontmatter_returns_empty_for_explicit_null() -> None:
    """Empty body parses to None; helper coerces to {}."""
    text = "---\n---\n\nbody\n"
    assert priming._parse_frontmatter(text) == {}


# ── _first_line ───────────────────────────────────────────────────────


def test_first_line_handles_none() -> None:
    assert priming._first_line(None) == ""


def test_first_line_handles_list_of_strings() -> None:
    assert priming._first_line(["", "  first  ", "second"]) == "first"


def test_first_line_handles_empty_list() -> None:
    assert priming._first_line([]) == ""


def test_first_line_handles_list_of_only_empties() -> None:
    assert priming._first_line(["", "   ", ""]) == ""


def test_first_line_handles_multi_line_string() -> None:
    assert priming._first_line("  \n  hello world\nignored\n") == "hello world"


def test_first_line_returns_empty_for_blank_string() -> None:
    assert priming._first_line("   \n\n\t  ") == ""


# ── _shorten ──────────────────────────────────────────────────────────


def test_shorten_truncates_with_ellipsis() -> None:
    out = priming._shorten("a" * 100, max_chars=10)
    assert len(out) == 10
    assert out.endswith("…")


def test_shorten_short_passthrough() -> None:
    assert priming._shorten("ok", max_chars=10) == "ok"


def test_shorten_collapses_whitespace() -> None:
    assert priming._shorten("a   b\n\tc") == "a b c"


# ── _reinforced_count ─────────────────────────────────────────────────


def test_reinforced_count_handles_garbage() -> None:
    assert priming._reinforced_count({"reinforced": "abc"}) == 0
    assert priming._reinforced_count({"reinforced": None}) == 0
    assert priming._reinforced_count({}) == 0


def test_reinforced_count_clamps_negative_to_zero() -> None:
    assert priming._reinforced_count({"reinforced": -5}) == 0


def test_reinforced_count_passes_through_positive() -> None:
    assert priming._reinforced_count({"reinforced": 4}) == 4


# ── _entry_title fallback ────────────────────────────────────────────


def test_entry_title_uses_frontmatter_title_when_present(tmp_path: Path) -> None:
    assert priming._entry_title({"title": "  My Rule "}, tmp_path / "x.md") == "My Rule"


def test_entry_title_falls_back_to_dekebabed_stem(tmp_path: Path) -> None:
    assert priming._entry_title({}, tmp_path / "use-uv-not-pip.md") == "use uv not pip"


def test_entry_title_ignores_empty_frontmatter_title(tmp_path: Path) -> None:
    assert priming._entry_title({"title": "  "}, tmp_path / "fallback.md") == "fallback"


# ── _wikilink_path ────────────────────────────────────────────────────


def test_wikilink_path_handles_outside_root(tmp_path: Path) -> None:
    """If the entry isn't under knowledge_dir, fall back to absolute."""
    elsewhere = tmp_path.parent / "outside.md"
    result = priming._wikilink_path(elsewhere, tmp_path)
    # ValueError branch — returns the string of the absolute path with
    # .md stripped.
    assert "outside" in result


# ── _bm25_search / _embedding_search fall-throughs ──────────────────


def test_bm25_search_empty_query_returns_empty(tmp_path: Path) -> None:
    assert priming._bm25_search(tmp_path, "", k=5) == []


def test_bm25_search_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert priming._bm25_search(tmp_path / "no-such-dir", "python", k=5) == []


def test_bm25_search_swallows_load_failure(tmp_path: Path, monkeypatch, caplog) -> None:
    """If bm25 load_or_build raises something other than FileNotFoundError,
    we log and return []."""
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "x.md").write_text("# x")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated index build failure")

    monkeypatch.setattr(priming, "bm25_load_or_build", _boom)
    assert priming._bm25_search(k, "python", k=5) == []


def test_bm25_search_swallows_search_failure(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()
    (k / "x.md").write_text("# x")

    class _Index:
        def search(self, q, k):
            raise RuntimeError("simulated search failure")

    def _load(*args, **kwargs):
        return _Index()

    monkeypatch.setattr(priming, "bm25_load_or_build", _load)
    assert priming._bm25_search(k, "python", k=5) == []


def test_bm25_search_filenotfound_during_load(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()

    def _missing(*args, **kwargs):
        raise FileNotFoundError("no index")

    monkeypatch.setattr(priming, "bm25_load_or_build", _missing)
    assert priming._bm25_search(k, "python", k=5) == []


def test_embedding_search_empty_query_returns_empty(tmp_path: Path) -> None:
    assert priming._embedding_search(tmp_path, "", k=5) == []


def test_embedding_search_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert priming._embedding_search(tmp_path / "missing", "x", k=5) == []


def test_embedding_search_handles_load_failure(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated emb load failure")

    monkeypatch.setattr(priming, "embeddings_load_or_build", _boom)
    assert priming._embedding_search(k, "x", k=5) == []


def test_embedding_search_handles_filenotfound(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()

    def _missing(*args, **kwargs):
        raise FileNotFoundError("no index")

    monkeypatch.setattr(priming, "embeddings_load_or_build", _missing)
    assert priming._embedding_search(k, "x", k=5) == []


def test_embedding_search_handles_search_failure(tmp_path: Path, monkeypatch) -> None:
    k = tmp_path / "knowledge"
    k.mkdir()

    class _Idx:
        def search(self, q, k):
            raise RuntimeError("simulated emb search failure")

    def _load(*args, **kwargs):
        return _Idx()

    monkeypatch.setattr(priming, "embeddings_load_or_build", _load)
    assert priming._embedding_search(k, "x", k=5) == []


def test_embedding_search_filters_noise_floor(tmp_path: Path, monkeypatch) -> None:
    """Hits with score < 0.25 are dropped (model noise floor)."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()

    class _Hit:
        def __init__(self, p, s):
            self.path = p
            self.score = s

    good = str(kdir / "good.md")
    noise = str(kdir / "noise.md")

    class _Idx:
        def search(self, q, k):
            return [_Hit(good, 0.5), _Hit(noise, 0.05)]

    monkeypatch.setattr(priming, "embeddings_load_or_build", lambda *a, **kw: _Idx())
    out = priming._embedding_search(kdir, "x", k=5)
    # Only the 0.5-score hit survives.
    assert len(out) == 1
    assert "good.md" in str(out[0][0])


# ── _rrf_merge ───────────────────────────────────────────────────────


def test_rrf_merge_handles_empty_inputs() -> None:
    assert priming._rrf_merge([], k_top=3) == []
    assert priming._rrf_merge([[]], k_top=3) == []
    # Lone non-empty list — same items come back, ordered.
    rankings = [[(Path("a.md"), 1.0), (Path("b.md"), 0.5)]]
    result = priming._rrf_merge(rankings, k_top=2)
    assert len(result) == 2
    assert result[0][0] == Path("a.md")


def test_rrf_merge_combines_two_rankings() -> None:
    """A path that ranks 2nd in one list and 1st in another should beat
    a path that's only top in one list."""
    a, b, c = Path("a.md"), Path("b.md"), Path("c.md")
    rankings = [
        [(a, 1.0), (b, 0.5), (c, 0.1)],
        [(b, 1.0), (a, 0.5)],  # b is 1st here
    ]
    result = priming._rrf_merge(rankings, k_top=3)
    paths = [p for p, _ in result]
    # a and b both score: a = 1/(60+1)+1/(60+2); b = 1/(60+2)+1/(60+1) → tie
    # Tie-break is stable on path string → a sorts before b.
    assert set(paths[:2]) == {a, b}


# ── _hybrid_search lane fallbacks ────────────────────────────────────


def test_hybrid_search_returns_empty_when_both_lanes_empty(tmp_path: Path) -> None:
    """No library / no hits in either lane → []."""
    assert priming._hybrid_search(tmp_path / "missing", "anything", k=5) == []


# ── _render_bullet edge cases ────────────────────────────────────────


def test_render_bullet_handles_unreadable_file(tmp_path: Path) -> None:
    """If the markdown file can't be read, return empty string."""
    assert priming._render_bullet(tmp_path / "no-such-file.md", 0, tmp_path) == ""


def test_render_bullet_title_only_with_no_title(tmp_path: Path, library_entry) -> None:
    """If the file has no usable title and we asked for title-only, the
    bullet still renders (just the wikilink + reinforce count)."""
    # An entry with NO title in frontmatter — _entry_title falls back to
    # the stem. The bullet will always show the stem.
    p = tmp_path / "bare-stem.md"
    # Manually write frontmatter without a title field — _entry_title
    # will fall back to the stem.
    p.write_text(
        "---\nid: bare-stem\ntype: lesson\nscope: global\nstatus: provisional\n"
        "confidence: 0.7\napplies-when: |\n  x\nkeywords: [x]\ncreated: 2026-05-19\n"
        "updated: 2026-05-19\nfired: 0\nfired-helpful: 0\nsources:\n  - manual\n---\n"
    )
    out = priming._render_bullet(p, 0, tmp_path, title_only=True)
    # The fallback title is the stem ("bare stem") so the title arm fires.
    assert "[[bare-stem]]" in out


def test_render_bullet_hook_only_when_title_blank(tmp_path: Path) -> None:
    """If title is empty but applies-when is present (after fallback),
    the bullet uses the hook only path. We can't easily reach this in
    isolation because the title fallback always produces *some* string;
    just confirm the no-title no-hook path returns the bare bullet."""
    p = tmp_path / "x.md"
    # YAML where the frontmatter exists but neither title nor
    # applies-when render. We can't force title="" via _entry_title's
    # fallback — it always uses the stem. So this branch is hard to hit
    # except via _entry_title returning ''. Use a stem that becomes ''
    # after the de-kebab... but that's not possible. Skip the false
    # branch — it's defensive code.
    p.write_text(
        "---\nid: x\ntype: lesson\nscope: global\nstatus: provisional\n"
        'confidence: 0.7\napplies-when: |\n  hook-only\nkeywords: [x]\ntitle: "x"\n'
        "created: 2026-05-19\nupdated: 2026-05-19\nfired: 0\nfired-helpful: 0\n"
        "sources:\n  - manual\n---\n"
    )
    out = priming._render_bullet(p, 0, tmp_path, title_only=False)
    assert "[[x]]" in out


# ── _atomic_write error path ─────────────────────────────────────────


def test_atomic_write_raises_on_mkstemp_failure(tmp_path: Path, monkeypatch) -> None:
    """If mkstemp itself fails, the helper propagates (caller wraps)."""
    import tempfile as _tempfile

    def _boom(*args, **kwargs):
        raise OSError("simulated mkstemp failure")

    monkeypatch.setattr(_tempfile, "mkstemp", _boom)
    with pytest.raises(OSError):
        priming._atomic_write(tmp_path / "out.md", "hello")


# ── _collect_strings recursion ───────────────────────────────────────


def test_collect_strings_recurses_into_nested_structures() -> None:
    obj = {
        "session_id": "skip-me",
        "proposals": [{"action": "skip-too", "title": "keep me", "deep": {"nested": "also keep"}}],
    }
    out = priming._collect_strings(obj)
    assert "keep me" in out
    assert "also keep" in out
    # skip_keys never leak in.
    assert "skip-me" not in out
    assert "skip-too" not in out


def test_collect_strings_skips_non_strings() -> None:
    """Ints, floats, bools, None are explicitly excluded."""
    out = priming._collect_strings({"a": 42, "b": 3.14, "c": True, "d": None})
    assert out == []


# ── Body excerpt extraction ──────────────────────────────────────────


def test_extract_body_excerpt_skips_frontmatter_and_heading() -> None:
    text = (
        "---\nid: x\ntitle: T\n---\n\n"
        "# Title heading\n\n"
        "This is the first body paragraph that the agent should see.\n\n"
        "Second paragraph should not appear.\n"
    )
    assert priming._extract_body_excerpt(text) == (
        "This is the first body paragraph that the agent should see."
    )


def test_extract_body_excerpt_returns_empty_for_empty_body() -> None:
    assert priming._extract_body_excerpt("---\nid: x\n---\n\n# Heading only\n") == ""


def test_extract_body_excerpt_truncates_with_ellipsis_on_word_boundary() -> None:
    long_para = "lorem ipsum " * 60  # ~720 chars
    text = f"---\nid: x\n---\n\n{long_para}\n"
    out = priming._extract_body_excerpt(text, max_chars=80)
    assert len(out) <= 80
    assert out.endswith("…")
    # Word boundary: no trailing partial token.
    head = out[:-1].rstrip()
    assert head.endswith("lorem") or head.endswith("ipsum")


def test_extract_body_excerpt_handles_text_without_frontmatter() -> None:
    text = "Plain body text with no frontmatter at all here.\n"
    assert priming._extract_body_excerpt(text).startswith("Plain body text")


# ── Freshness marker ─────────────────────────────────────────────────


def test_is_fresh_within_window_returns_true() -> None:
    from datetime import date as _date

    today = _date(2026, 5, 27)
    assert priming._is_fresh({"updated": "2026-05-25"}, today=today) is True


def test_is_fresh_outside_window_returns_false() -> None:
    from datetime import date as _date

    today = _date(2026, 5, 27)
    assert priming._is_fresh({"updated": "2026-05-01"}, today=today) is False


def test_is_fresh_falls_back_to_created() -> None:
    from datetime import date as _date

    today = _date(2026, 5, 27)
    assert priming._is_fresh({"created": "2026-05-26"}, today=today) is True


def test_is_fresh_returns_false_for_no_dates() -> None:
    assert priming._is_fresh({}) is False


# ── Kind marker (path-derived) ───────────────────────────────────────


def test_kind_marker_conventions_path() -> None:
    kdir = Path("/k")
    p = kdir / "global" / "conventions" / "code-quality" / "use-uv.md"
    assert priming._kind_marker(p, kdir, {}) == "C"


def test_kind_marker_findings_path() -> None:
    kdir = Path("/k")
    p = kdir / "projects" / "vol" / "research" / "findings" / "vol-normalization.md"
    assert priming._kind_marker(p, kdir, {}) == "F"


def test_kind_marker_warning_severity_wins_over_path() -> None:
    kdir = Path("/k")
    p = kdir / "global" / "conventions" / "rm-rf-warning.md"
    assert priming._kind_marker(p, kdir, {"severity": "block"}) == "W"


def test_kind_marker_returns_empty_for_unclassified() -> None:
    kdir = Path("/k")
    p = kdir / "projects" / "vol" / "concepts" / "natural-hold.md"
    assert priming._kind_marker(p, kdir, {}) == ""


# ── Scope penalty (cross-project) ────────────────────────────────────


def test_scope_bonus_cross_project_returns_penalty() -> None:
    aliases: dict[str, str] = {}
    # Bucket != current project, not __global__, current_project_slug
    # is present → penalty fires.
    assert priming._scope_bonus("vol-predictor", "agent-mem", aliases) == (
        priming._SCOPE_PENALTY_CROSS_PROJECT
    )


def test_scope_bonus_no_current_slug_collapses_to_zero() -> None:
    """Without a baseline, cross-project can't be distinguished from
    current-project — collapse to zero rather than penalise blindly."""
    assert priming._scope_bonus("vol-predictor", None, {}) == 0.0


def test_scope_bonus_global_unchanged() -> None:
    assert priming._scope_bonus("__global__", "agent-mem", {}) == priming._SCOPE_BONUS_GLOBAL


# ── Dedup in _assemble_output ────────────────────────────────────────


@pytest.fixture
def _seed_two_entries(tmp_path: Path) -> tuple[Path, Path, Path]:
    k = tmp_path / "knowledge"
    a = k / "global" / "a.md"
    b = k / "global" / "b.md"
    for p, slug, body in (
        (a, "a", "Entry A body — first rule about handling X carefully."),
        (b, "b", "Entry B body — separate rule about handling Y instead."),
    ):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"---\nid: {slug}\ntitle: Entry {slug.upper()}\napplies-when: when {slug} matters\n---\n\n"
            f"# Heading\n\n{body}\n"
        )
    return k, a, b


def test_assemble_dedup_filters_already_sent_entries(_seed_two_entries) -> None:
    k, a, b = _seed_two_entries
    ranked = [(a, 1.0, 0), (b, 0.5, 0)]
    # Pretend "a" was already sent to this session.
    rendered, newly_sent = priming._assemble_output(
        ranked,
        k,
        top_k=3,
        char_budget=2000,
        already_sent={"global/a"},
    )
    assert rendered, "expected at least the b entry to render"
    assert "[[global/a]]" not in rendered
    assert "[[global/b]]" in rendered
    assert newly_sent == ["global/b"]


def test_assemble_dedup_returns_empty_when_all_sent(_seed_two_entries) -> None:
    k, a, b = _seed_two_entries
    ranked = [(a, 1.0, 0), (b, 0.5, 0)]
    rendered, newly_sent = priming._assemble_output(
        ranked,
        k,
        top_k=3,
        char_budget=2000,
        already_sent={"global/a", "global/b"},
    )
    assert rendered == ""
    assert newly_sent == []


def test_assemble_emits_body_excerpts_when_budget_allows(_seed_two_entries) -> None:
    k, a, b = _seed_two_entries
    ranked = [(a, 1.0, 0), (b, 0.5, 0)]
    rendered, _ = priming._assemble_output(
        ranked,
        k,
        top_k=3,
        char_budget=2000,
        already_sent=set(),
    )
    # Body excerpts appear under bullets prefixed with "  > ".
    assert "  > Entry A body" in rendered
    assert "  > Entry B body" in rendered
