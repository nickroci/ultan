# agent-mem — Phase 0 skeleton

This directory is a fork of `prior-art/claude-memory-compiler` (by coleam00),
modified to be the foundation of `agent-mem`. The whole system this is a
piece of is described in `../PLAN.md`. Read that first — it explains the
two-tier (Librarian/Scholar) curator, the daemon, BM25 search, interrupts,
and per-project scoping. None of those are implemented here. **This is just
the skeleton: hooks fire on SessionEnd / PreCompact / SessionStart, a
background process extracts conversation context with the Claude Agent SDK,
and the result lands in a user-global daily log.**

For the original architecture (compiler analogy, schema, lint checks, daily
log format, etc.), see [`prior-art/claude-memory-compiler/README.md`](../prior-art/claude-memory-compiler/README.md)
and [`prior-art/claude-memory-compiler/AGENTS.md`](../prior-art/claude-memory-compiler/AGENTS.md).
This README documents only what is different.

---

## What changed vs upstream

### Storage is user-global

Upstream assumed one knowledge base per project, living alongside the code
(`<repo>/daily/`, `<repo>/knowledge/`, state in `<repo>/scripts/`). agent-mem
puts everything under `~/.agent-mem/` so lessons from project A inform work
in project B. Layout (see `scripts/config.py`):

```
~/.agent-mem/
  daily/                              # raw extracted entries, one file per date
  knowledge/
    index.md                          # master catalog
    log.md                            # append-only build log
    global/
      concepts/                       # Phase-0 compile target
      connections/
      qa/
    projects/<slug>/                  # created lazily; populated in Phase 2
  reports/                            # lint output
  state/
    state.json                        # compile/query stats
    last-flush.json                   # dedup state for flush.py
    session-flush-*.md                # transient context files
  flush.log
  compile.log
```

The store root can be overridden by setting `AGENT_MEM_HOME` (handy for
tests, or if you don't want the dotfile in `$HOME`).

The schema file (`AGENTS.md`) and the scripts themselves stay in the code
checkout — `scripts/config.py` distinguishes `CODE_ROOT` (the checkout) from
`STORE_DIR` (`~/.agent-mem/`). The Agent SDK is given `STORE_DIR` as its
`cwd` for compile/query/lint (so Read/Write/Glob land in the knowledge
tree), and `CODE_ROOT` for `flush.py` (which doesn't touch the corpus
directly).

### Per-project tagging

A new module `scripts/scope.py` exposes `current_project_slug(cwd)`:

- Prefers `git config --get remote.origin.url`, normalised to
  `<host>-<owner>-<repo>` (e.g. `github.com-coleam00-claude-memory-compiler`).
- Falls back to `basename(cwd)`.
- Returns `"unknown"` as a last resort.

The SessionEnd and PreCompact hooks call this with the cwd reported in the
hook's stdin payload (`hook_input["cwd"]`), then pass the slug to `flush.py`
as a third positional argument. `flush.py` tags each daily-log section with
`project:<slug>` so a later compile pass (Scholar's job, Phase 2) can route
lessons into `knowledge/projects/<slug>/`. Phase 0 itself does **not** write
into per-project dirs — `compile.py` still writes everything to
`knowledge/global/`, with a `TODO(agent-mem)` noting the deferred routing.

The SessionStart hook also resolves the slug and injects a "Current project"
note alongside the index, so the active agent knows which subtree's lessons
to weight.

### Hook plumbing

The recursion guard (`CLAUDE_INVOKED_BY`) is preserved verbatim — flush.py
sets it before importing anything, and both hooks bail at the top if they
see it. This is the only thing standing between us and an infinite loop.

Hook commands now use `uv run --directory <agent-mem/src>` so they work from
any cwd. See `_dot_claude_disabled/settings.json` for the template (replace
`AGENT_MEM_SRC` with your absolute checkout path).

### State and logs follow the store

- `state.json`, `last-flush.json`, and transient `session-flush-*` /
  `flush-context-*` files now live in `~/.agent-mem/state/`.
- `flush.log` and `compile.log` live in `~/.agent-mem/`.
- `reports/` (lint output) lives in `~/.agent-mem/reports/`.

The `.gitignore` is simplified accordingly: nothing the runtime writes
should appear under `src/` anymore. If you ever see one of those files
reappear in the checkout, that's a bug.

### Wikilinks under the new layout

Compiled articles now live under `knowledge/global/...`. Wikilinks should
be written as `[[global/concepts/foo]]` rather than the upstream
`[[concepts/foo]]`. `utils.wiki_article_exists` and `count_inbound_links`
accept both forms as a compatibility shim, so any articles you had under
the upstream layout still resolve until the next compile pass migrates
them.

### Cross-platform notes preserved

The Windows-specific bits from upstream (the unescaped-backslash regex
fallback when parsing the hook's stdin, the `CREATE_NO_WINDOW` flag,
avoiding `DETACHED_PROCESS` because it breaks the Agent SDK's subprocess
I/O) are all kept as-is. I haven't tested on Windows; I haven't actively
broken it either.

---

## Install

```bash
# 1. From the agent-mem checkout, sync the venv
cd src
uv sync

# 2. Tell Claude Code about the hooks. Copy the contents of
#    _dot_claude_disabled/settings.json into either:
#      - ~/.claude/settings.json   (user-global — applies to every project)
#      - <project>/.claude/settings.json   (per-project)
#    and replace every occurrence of AGENT_MEM_SRC with the absolute path
#    to this src/ directory.

# 3. Start a Claude Code session, do some work, and end it. The SessionEnd
#    hook fires flush.py in the background. Look for output in:
#      ~/.agent-mem/flush.log         (hook + flush diagnostics)
#      ~/.agent-mem/daily/<date>.md   (the extracted session summary)
```

To override the store location for tests or dotfile-averse setups:

```bash
export AGENT_MEM_HOME=/tmp/agent-mem-test
```

---

## What this skeleton does NOT do

Per PLAN.md, the following are explicitly out of scope for Phase 0 and will
be added by later agents:

- The long-running `agent-mem-daemon` and Unix socket / JSONL-tail transport.
- The Librarian (Haiku-tier candidate extraction) and the Scholar
  (Opus-tier writer / interrupt gate).
- BM25 search, the `agent-mem search` CLI, the `--bm25` / `--index` modes.
- `applies-when`-based live scoring of the rolling buffer.
- Soft interrupts via `pending-nudges.md` / `UserPromptSubmit`.
- The provisional / confirmed lifecycle and `agent-mem review`.

What works today is exactly the upstream loop, repointed at the user-global
store and tagged with the project slug: **SessionEnd → flush → daily log →
(after 6 PM) end-of-day compile → `knowledge/global/`**.

---

## Open notes for later phases

Things I noticed while doing this fork that PLAN.md doesn't currently
capture:

1. **`hook_input["cwd"]` is the source of truth for project scope, not the
   hook process's own cwd.** Claude Code invokes hooks under its own
   working directory, but the agent's project can be different. PLAN.md §7.4
   says "the hook knows the current working directory" without spelling out
   *which* cwd. Use the stdin field.
2. **The Agent SDK's `cwd` is now ambiguous.** `flush.py` points it at
   `CODE_ROOT` (no need for the corpus). `compile.py`, `query.py`, `lint.py`
   point it at `STORE_DIR` (they read/write the corpus). The daemon will
   need to make the same choice per task — worth adding to PLAN.md §1.
3. **`STATE_DIR` is now a real directory under `STORE_DIR`,** not the
   scripts directory as upstream had it. The hooks call `ensure_store_dirs()`
   so a fresh `~/.agent-mem/` materialises on first use — important for the
   daemon too, which will want the same guarantee.
4. **Daily logs are global, project-tagged in the section heading.** PLAN.md
   §1 shows `daily/` at the top level and `knowledge/projects/<slug>/` under
   knowledge. I read that as: raw events are chronological-by-time
   (one-dimensional), compiled lessons are scoped-by-project. Phase 2's
   Scholar reads the section tags in daily logs to route writes. If PLAN
   wanted per-project daily logs instead, this would need to change.
5. **Wikilink form change is a breaking schema bump.** Any pre-existing
   `[[concepts/foo]]` links from an upstream user's KB will resolve via the
   legacy fallback in `utils.wiki_article_exists` for now, but the next
   compile pass will write `[[global/concepts/foo]]` and the mismatch will
   accumulate. A one-shot migration script (rewrite links in-place) should
   land in Phase 1 or 2.
6. **PLAN.md §1 shows `connections/` directly under `knowledge/`** in the
   storage diagram, but contextually means under whichever tier. I put
   `connections/` under both `global/` and `projects/<slug>/` to keep the
   schema uniform. Worth tightening in PLAN.md.
