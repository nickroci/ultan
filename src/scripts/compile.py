"""
Compile daily conversation logs into structured knowledge articles.

This is the "LLM compiler" — it reads daily logs (source code) and produces
organized knowledge articles (the executable). All paths resolve against the
user-global store at ``~/.agent-mem/``.

Usage:
    uv run python compile.py                    # compile new/changed logs only
    uv run python compile.py --all              # force recompile everything
    uv run python compile.py --file daily/2026-04-01.md  # compile a specific log
    uv run python compile.py --dry-run          # show what would be compiled
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from config import (
    AGENTS_FILE,
    ensure_store_dirs,
    get_config,
    now_iso,
)
from utils import (
    IngestedEntry,
    State,
    file_hash,
    list_raw_files,
    list_wiki_articles,
    load_state,
    read_wiki_index,
    save_state,
)


async def compile_daily_log(log_path: Path, state: State) -> float:
    """Compile a single daily log into knowledge articles.

    Returns the API cost of the compilation.
    """
    cfg = get_config()
    log_content = log_path.read_text(encoding="utf-8")
    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index()

    # Read existing articles for context
    existing_articles_context = ""
    existing: dict[str, str] = {}
    for article_path in list_wiki_articles():
        rel = article_path.relative_to(cfg.knowledge_dir)
        existing[str(rel)] = article_path.read_text(encoding="utf-8")

    if existing:
        parts: list[str] = []
        for rel_path, content in existing.items():
            parts.append(f"### {rel_path}\n```markdown\n{content}\n```")
        existing_articles_context = "\n\n".join(parts)

    timestamp = now_iso()

    # TODO(agent-mem): Phase 0 routes every concept article into the global
    # tier. Per-project routing (writing into knowledge/projects/<slug>/) is
    # a Scholar concern — see PLAN.md §4. The daily log entries are already
    # tagged with `project:<slug>` by flush.py, so a Phase-2 prompt can read
    # those tags and choose the right destination dir. For now we keep
    # everything global so the upstream compile loop still works.
    prompt = f"""You are a knowledge compiler. Your job is to read a daily conversation log
and extract knowledge into structured wiki articles.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## Existing Wiki Articles

{existing_articles_context if existing_articles_context else "(No existing articles yet)"}

## Daily Log to Compile

**File:** {log_path.name}

{log_content}

## Your Task

Read the daily log above and compile it into wiki articles following the schema exactly.

### Rules:

1. **Extract key concepts** - Identify 3-7 distinct concepts worth their own article
2. **Create concept articles** in `knowledge/global/concepts/` - One .md file per concept
   - Use the exact article format from AGENTS.md (YAML frontmatter + sections)
   - Include `sources:` in frontmatter pointing to the daily log file
   - Use `[[global/concepts/slug]]` wikilinks to link to related concepts
   - Write in encyclopedia style - neutral, comprehensive
3. **Create connection articles** in `knowledge/global/connections/` if this log reveals non-obvious
   relationships between 2+ existing concepts
4. **Update existing articles** if this log adds new information to concepts already in the wiki
   - Read the existing article, add the new information, add the source to frontmatter
5. **Update knowledge/index.md** - Add new entries to the table
   - Each entry: `| [[path/slug]] | One-line summary | source-file | {timestamp[:10]} |`
6. **Append to knowledge/log.md** - Add a timestamped entry:
   ```
   ## [{timestamp}] compile | {log_path.name}
   - Source: daily/{log_path.name}
   - Articles created: [[global/concepts/x]], [[global/concepts/y]]
   - Articles updated: [[global/concepts/z]] (if any)
   ```

### File paths:
- Write concept articles to: {cfg.concepts_dir}
- Write connection articles to: {cfg.connections_dir}
- Update index at: {cfg.knowledge_dir / "index.md"}
- Append log at: {cfg.knowledge_dir / "log.md"}

### Quality standards:
- Every article must have complete YAML frontmatter
- Every article must link to at least 2 other articles via [[wikilinks]]
- Key Points section should have 3-5 bullet points
- Details section should have 2+ paragraphs
- Related Concepts section should have 2+ entries
- Sources section should cite the daily log with specific claims extracted
"""

    cost = 0.0

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                # SDK cwd is the store — the knowledge base is what it
                # reads and writes. Absolute paths to cfg.concepts_dir etc.
                # still work; this just makes Glob/Grep land in the right
                # place if the model uses them.
                cwd=str(cfg.store_dir),
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
                permission_mode="acceptEdits",
                max_turns=30,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        pass  # compilation output - LLM writes files directly
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
                print(f"  Cost: ${cost:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0

    # Update state
    rel_path = log_path.name
    ingested = state.setdefault("ingested", {})
    ingested[rel_path] = IngestedEntry(
        hash=file_hash(log_path),
        compiled_at=now_iso(),
        cost_usd=cost,
    )
    state["total_cost"] = state.get("total_cost", 0.0) + cost
    save_state(state)

    return cost


def _resolve_target(file_arg: str) -> Path:
    """Resolve a ``--file`` argument to an absolute log path.

    Tries the argument as-is, then under the store's daily dir, then under
    the store root. Exits with code 1 if none of the candidates exist.
    """
    cfg = get_config()
    target = Path(file_arg)
    if not target.is_absolute():
        target = cfg.daily_dir / target.name
    if not target.exists():
        target = cfg.store_dir / file_arg
    if not target.exists():
        print(f"Error: {file_arg} not found")
        sys.exit(1)
    return target


def _select_files(args: argparse.Namespace, state: State) -> list[Path]:
    """Decide which daily logs to compile this run.

    - ``--file`` → exactly that one (resolved via :func:`_resolve_target`).
    - ``--all`` → every log under the store's daily dir.
    - default → only logs whose content hash has changed since the last
      run (or never been compiled).
    """
    if args.file:
        return [_resolve_target(args.file)]

    all_logs = list_raw_files()
    if args.all:
        return all_logs

    changed: list[Path] = []
    ingested = state.get("ingested", {})
    for log_path in all_logs:
        prev = ingested.get(log_path.name)
        if prev is None or prev.get("hash") != file_hash(log_path):
            changed.append(log_path)
    return changed


def _run_compile_loop(to_compile: list[Path], state: State) -> None:
    """Compile each log sequentially, printing a per-file progress line."""
    total_cost = 0.0
    for i, log_path in enumerate(to_compile, 1):
        print(f"\n[{i}/{len(to_compile)}] Compiling {log_path.name}...")
        cost = asyncio.run(compile_daily_log(log_path, state))
        total_cost += cost
        print("  Done.")

    articles = list_wiki_articles()
    print(f"\nCompilation complete. Total cost: ${total_cost:.2f}")
    print(f"Knowledge base: {len(articles)} articles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile daily logs into knowledge articles")
    parser.add_argument("--all", action="store_true", help="Force recompile all logs")
    parser.add_argument("--file", type=str, help="Compile a specific daily log file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compiled")
    args = parser.parse_args()

    ensure_store_dirs()
    state = load_state()
    to_compile = _select_files(args, state)

    if not to_compile:
        print("Nothing to compile - all daily logs are up to date.")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to compile ({len(to_compile)}):")
    for f in to_compile:
        print(f"  - {f.name}")

    if args.dry_run:
        return

    _run_compile_loop(to_compile, state)


if __name__ == "__main__":
    main()
