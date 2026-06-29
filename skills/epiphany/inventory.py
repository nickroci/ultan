#!/usr/bin/env python3
"""Compact inventory of the Ultan knowledge library, for the /epiphany skill.

Walks ``$AGENT_MEM_HOME/knowledge`` (default ``~/.agent-mem/knowledge``) and
scrapes each entry's title/type/keywords from frontmatter, then prints a compact
map grouped by region. This is the *territory* a roaming agent partitions when
hunting for a far-pair connection.

Pure filesystem walk, stdlib only — no daemon dependency, so it works even while
the daemon is still warming. Any ``python3`` runs it.

Usage:
  inventory.py                  full map, grouped by region
  inventory.py --regions        region names + entry counts only (cheap overview)
  inventory.py --region PREFIX   only entries whose path starts with PREFIX
                                 (e.g. ``projects/bq-data`` or ``global/concepts``)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

HOME = Path(os.environ.get("AGENT_MEM_HOME") or (Path.home() / ".agent-mem"))
ROOT = HOME / "knowledge"
MAX_KW = 6  # keywords shown per entry, to keep the map compact


def _scrape(path: Path) -> tuple[str | None, str | None, list[str]]:
    """Cheap frontmatter scrape → (title, type, keywords). No YAML dep."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None, []
    if not text.startswith("---"):
        return None, None, []
    title = typ = None
    kws: list[str] = []
    for line in text.splitlines()[1:]:
        if line.strip() == "---":
            break
        if line.startswith("title:"):
            title = line[len("title:") :].strip().strip("\"'")
        elif line.startswith("type:"):
            typ = line[len("type:") :].strip().strip("\"'")
        elif line.startswith("keywords:"):
            raw = line[len("keywords:") :].strip()
            if raw.startswith("[") and raw.endswith("]"):
                kws = [k.strip().strip("\"'") for k in raw[1:-1].split(",") if k.strip()]
    return title, typ, kws


def _region(rel: str, depth: int) -> str:
    """First ``depth`` path components define a region. depth=2 →
    ``projects/vol-predictor`` (one project = one region); depth=3 →
    ``projects/vol-predictor/model`` (partition *within* a project by subsystem)."""
    parts = rel.split("/")
    if len(parts) > depth:
        return "/".join(parts[:depth])
    # Shallow entry: group by its parent folder (or the bare name at the root).
    return "/".join(parts[:-1]) or parts[0]


def _collect(prefix: str | None, depth: int) -> list[tuple[str, str, str, str, list[str]]]:
    rows: list[tuple[str, str, str, str, list[str]]] = []
    if not ROOT.is_dir():
        sys.stderr.write(f"No knowledge library at {ROOT}\n")
        return rows
    for p in sorted(ROOT.rglob("*.md")):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("_archive/") or p.name in {"README.md", "MEMORY.md"}:
            continue
        ref = rel[:-3]  # drop .md → usable as a wikilink target
        if prefix and not ref.startswith(prefix):
            continue
        title, typ, kws = _scrape(p)
        rows.append((ref, _region(rel, depth), title or p.stem.replace("-", " "), typ or "?", kws))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", action="store_true", help="region names + counts only")
    ap.add_argument("--region", metavar="PREFIX", help="filter to a path prefix")
    ap.add_argument(
        "--depth",
        type=int,
        default=2,
        help="path components that define a region (default 2; use 3 to partition "
        "within one project by subsystem, e.g. --region projects/vol-predictor --depth 3)",
    )
    args = ap.parse_args()

    rows = _collect(args.region, args.depth)
    if not rows:
        print("_(no entries found)_")
        return 0

    by_region: defaultdict[str, list[tuple[str, str, str, list[str]]]] = defaultdict(list)
    for ref, region, title, typ, kws in rows:
        by_region[region].append((ref, title, typ, kws))

    if args.regions:
        print(f"# Ultan territory — {len(rows)} entries across {len(by_region)} regions\n")
        for region in sorted(by_region):
            print(f"- {region} — {len(by_region[region])} entries")
        return 0

    print(f"# Ultan inventory — {len(rows)} entries across {len(by_region)} regions\n")
    for region in sorted(by_region):
        print(f"## {region} ({len(by_region[region])})")
        for ref, title, typ, kws in by_region[region]:
            kw = f"  ·  {', '.join(kws[:MAX_KW])}" if kws else ""
            print(f"- [{typ}] {ref} — {title}{kw}")
        print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # A downstream reader closed early (e.g. ``| head``). Redirect the rest
        # of stdout to devnull so the interpreter's final flush stays quiet.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0)
