"""Behavioural regression tests for ``scripts/compile.py``.

Covers the deterministic file-selection helpers (``_resolve_target`` and
``_select_files``). The SDK-driven ``compile_daily_log`` / ``main`` paths
are deliberately not exercised here — they hit the Claude Agent SDK.

Namespaces are built directly rather than via ``argparse`` so each test
controls exactly the three flags ``_select_files`` reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import compile as compile_mod
import pytest
import utils


def _ns(*, file: str | None = None, all: bool = False, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(file=file, all=all, dry_run=dry_run)


def _make_daily(home: Path, name: str, content: str = "log\n") -> Path:
    daily = home / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    path = daily / name
    path.write_text(content, encoding="utf-8")
    return path


# ── _resolve_target ──────────────────────────────────────────────────


def test_resolve_target_absolute_path(store_home: Path) -> None:
    log = _make_daily(store_home, "2026-05-01.md")
    assert compile_mod._resolve_target(str(log)) == log


def test_resolve_target_basename_under_daily_dir(store_home: Path) -> None:
    log = _make_daily(store_home, "2026-05-02.md")
    # A bare relative path resolves to the daily dir by basename.
    assert compile_mod._resolve_target("2026-05-02.md") == log


def test_resolve_target_under_store_root(store_home: Path) -> None:
    # A file living at the store root (not under daily/) is found via the
    # second fallback branch.
    rooted = store_home / "rooted.md"
    rooted.write_text("x\n", encoding="utf-8")
    assert compile_mod._resolve_target("rooted.md") == rooted


def test_resolve_target_missing_exits(store_home: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        compile_mod._resolve_target("does-not-exist.md")
    assert exc.value.code == 1


# ── _select_files ────────────────────────────────────────────────────


def test_select_files_all_returns_every_log(store_home: Path) -> None:
    a = _make_daily(store_home, "2026-05-01.md")
    b = _make_daily(store_home, "2026-05-02.md")
    selected = compile_mod._select_files(_ns(all=True), utils.load_state())
    assert selected == [a, b]


def test_select_files_default_returns_only_changed(store_home: Path) -> None:
    compiled = _make_daily(store_home, "2026-05-01.md", "already compiled\n")
    changed = _make_daily(store_home, "2026-05-02.md", "stale content\n")

    # Record `compiled` in state with its CURRENT hash; record `changed`
    # with a stale hash so it shows as needing recompilation.
    state = utils.load_state()
    state["ingested"] = {
        compiled.name: {"hash": utils.file_hash(compiled), "compiled_at": "t", "cost_usd": 0.0},
        changed.name: {"hash": "0000000000000000", "compiled_at": "t", "cost_usd": 0.0},
    }
    selected = compile_mod._select_files(_ns(), state)
    assert selected == [changed]


def test_select_files_default_includes_never_compiled(store_home: Path) -> None:
    fresh = _make_daily(store_home, "2026-05-03.md")
    selected = compile_mod._select_files(_ns(), utils.load_state())
    assert selected == [fresh]


def test_select_files_file_flag_resolves_one(store_home: Path) -> None:
    _make_daily(store_home, "2026-05-01.md")
    target = _make_daily(store_home, "2026-05-02.md")
    # --file wins over --all, and resolves via _resolve_target.
    selected = compile_mod._select_files(_ns(file="2026-05-02.md", all=True), utils.load_state())
    assert selected == [target]
