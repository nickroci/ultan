# Ultan

A personal memory system for coding agents. Built for Claude Code; lives outside any one session, project, or machine.

> *"Show me a man who has read all of the books of one of the major branches of knowledge — say, military history — and I'll show you a man more ignorant than the merest churl. For while he has read, others have written; and the body of available knowledge has grown so much faster than his understanding of it that he is, on balance, less learned at the end of his studies than at their beginning."*
>
> — Master Ultan, *The Shadow of the Torturer*, Gene Wolfe

Named for Master Ultan, the blind librarian of the Citadel in Gene Wolfe's *The Book of the New Sun*. He runs an impossibly vast subterranean library without sight — knows it by where each volume sits, by the shape of what a visitor asks for, by what he has been told over the years. The conceit fits. This system doesn't read your sessions in real time. It remembers what you've explicitly said, what you've revealed through how you work, what you've corrected. Then it surfaces it when the shape of what you're doing matches something it already holds — and when it doesn't, it tells you honestly, instead of guessing.

Ultan watches your conversations as you work, learns your preferences and conventions, and surfaces them when they matter. It's the "remember when you told me to always use uv" that you wish Claude already did natively, except organised, deduplicated, validated, and proactively consulted before the agent interrupts you to ask something you've already answered.

It's your library. On your disk. In plain markdown. You can `ls` it, `cat` it, `git` it.

---

## What it does

- **Captures only what's actually new.** Modelled on how brains encode memory: prediction error gates the write. The curator asks, of every candidate, "would a competent assistant produce this advice unprompted?" If yes — it's already in the model's baseline knowledge, no info added — skip. If no, capture. Three signals trigger a write:
  - **Contradicts** an existing entry — user has changed their mind. Highest priority. Deprecates the old, writes the new.
  - **Novel** — not in the library, not derivable from the model's training (user-specific facts, strict overrides of defaults, idiosyncratic preferences).
  - **Reinforces** — user repeated something we already have. No new entry; daemon bumps a `reinforced` counter on the existing entry to track how often it's reasserted.
- **Two-tier curator with asymmetric bars.** A Librarian (Sonnet) does fast salience detection — low bar, recall-tuned. A Scholar (Opus) deliberates — higher bar, precision-tuned. Mirrors System 1 / System 2, and the dopamine-RPE signal: the Librarian flags anything that might be new; the Scholar verifies "would I, as a more capable model, have produced this anyway?" before approving.
- **Organises a real library, not a flat pile.** Topical hierarchy emerges from content. Every folder has a README. ≤5 entries per directory before splitting. Auto-maintained child listings between marker comments. Wikilinks validate. Frontmatter validates. Scope/path agreement enforced.
- **Three slash commands** wire it into Claude Code without ceremony:
  - `/ultan <text>` — drop something into memory now, no extraction needed.
  - `/ultan-install` — wire the hooks into the current project's `.claude/settings.json`.
  - `/ultan-advisor <question>` — query the library before asking the user a preference question. The advisor finds relevant entries (Sonnet, BM25 + Read), writes a referenced answer (Opus), and clearly distinguishes stored knowledge from its own opinion. *Always cheaper to check than to ask.*
- **Soft nudges, not Clippy.** When the Librarian sees a stored preference that applies to what you're doing, the Scholar decides whether to mention it. Approved nudges land in `pending-nudges.md` and inject as `additionalContext` on your next turn — budgeted to 1/turn, 3/session.
- **Pure markdown store.** No vector DB, no database, no daemons-of-daemons. Everything lives under `~/.agent-mem/` and you can `ls`, `cat`, or `git` it.

---

## Quick start

```bash
# 1. Sync the daemon's deps (uv-managed)
cd daemon && uv sync --extra dev

# 2. Sync the search CLI (separate venv, shared BM25 implementation)
cd ../tools/search && uv sync

# 3. Install the slash commands and hooks template
#    - /ultan, /ultan-install, /ultan-advisor live at ~/.claude/commands/
#    - In any project: `/ultan-install` to wire hooks into <project>/.claude/settings.json
#    - Or `/ultan-install --global` for user-wide

# 4. Start the daemon (foreground; logs to ~/.agent-mem/daemon.log)
cd /path/to/ultan/daemon && uv run agent-mem-daemon -v
#    (nohup, tmux, or a launchd plist if you want it persistent — Phase 4 work.)

# 5. Open Claude Code in any project where you ran /ultan-install and work normally.
#    Entries land under ~/.agent-mem/knowledge/ as the Scholar approves them.
```

To save a memory explicitly: `/ultan never deploy to prod without my explicit OK`.

To ask before asking the user: `/ultan-advisor should I use respx or hand-roll an httpx mock?`.

---

## How it works (the short version)

```
hooks (UserPromptSubmit, PostToolUse, Stop, SessionEnd, …)
    ↓  append JSONL line per event
~/.agent-mem/events.jsonl
    ↓  tailed
agent-mem-daemon
    │  ┌──────────────────────────────────────────────┐
    │  │  TailerThread  →  RollingBuffer per session  │
    │  │                          ↓                   │
    │  │             DebounceScheduler                │
    │  │            (per-session quiet timer)         │
    │  │                          ↓                   │
    │  │  LibrarianPool  (N parallel Sonnet workers)  │
    │  │   - read library snapshot                    │
    │  │   - BM25 search (in-process MCP tool)        │
    │  │   - Read/Glob for verification               │
    │  │   - classify salience (contradicts/novel/    │
    │  │     reinforces) + propose actions            │
    │  │                          ↓                   │
    │  │  daemon: bump `reinforced` counters          │
    │  │   (deterministic, no SDK cost — empirical    │
    │  │   "user mentioned this again" signal)        │
    │  │                          ↓                   │
    │  │  ScholarWorker  (single Opus worker)         │
    │  │   - apply higher-bar salience filter:        │
    │  │     "would I produce this advice unprompted?"│
    │  │   - approve+execute via Write/Edit           │
    │  │   - or veto+drop with reasoning              │
    │  │                          ↓                   │
    │  │  Reconciler (deterministic, post-batch)      │
    │  │   - ensure README at every folder            │
    │  │   - sync auto-managed child listings         │
    │  │   - check wikilinks resolve                  │
    │  │   - check frontmatter validates              │
    │  │   - check scope/path agreement               │
    │  └──────────────────────────────────────────────┘
    ↓
~/.agent-mem/knowledge/  ← your library
```

Design discipline that survived live testing:

- **Path guard at the SDK layer.** A `can_use_tool` callback rejects any tool call whose path resolves outside the knowledge directory. Doesn't trust the prompt to behave; enforces in infrastructure.
- **No silent fix-ups.** The Scholar can only approve-and-execute or veto-and-drop. If the Librarian got the path wrong, the proposal is lost — recurs next session if real. Forces the Librarian to be careful.
- **Schema as single source of truth.** All prompt instructions describing the JSON the LLM should emit are generated from Pydantic models at prompt-assembly time. Change the schema, the prompt updates automatically.
- **Auto-reconciled READMEs.** Every folder's README has a `<!-- ULTAN:children (auto) -->` marker block. The LLM writes prose above; the daemon keeps the listing in sync after every batch. No drift.
- **Streaming-mode SDK calls** so `can_use_tool` works, with a final-`{...}`-block JSON extractor that's robust against tool-call markers preceding the response.
- **Persistent tailer offset** so daemon restarts resume mid-stream instead of seeking to EOF and losing the events that arrived during downtime.

---

## Prior art

Ultan is forked from and inspired by two earlier projects. Heavy credit to both authors:

- **[coleam00/claude-memory-compiler](https://github.com/coleam00/claude-memory-compiler)** — provided the codebase skeleton: hook architecture (`SessionEnd` → flush → daily log → compile), the recursion-guard pattern via `CLAUDE_INVOKED_BY`, and the Karpathy-style knowledge layout (daily, concepts, connections, qa, index, log). The whole writes-via-Claude-Agent-SDK pattern with `permission_mode="acceptEdits"` and the `claude_code` preset is from here, near-verbatim.
- **[mann1x/claude-hooks](https://github.com/mann1x/claude-hooks)** — the daemon pattern, the threaded worker model, and the discipline around fault isolation in parallel pools. We cribbed the `ThreadPoolExecutor` fan-out from their `_parallel.py` and the threading reasoning from `daemon.py` (their explicit choice of stdlib `threading` over asyncio — *"hooks are I/O-bound; GIL releases during socket reads; small ThreadPoolExecutor gives near-linear speedup without the rewrite cost"* — is exactly what made our parallel daemon tractable).

What we deliberately didn't borrow from `claude-hooks`: the Qdrant / pgvector / sqlite-vec providers, the API proxy, the dashboard, the LSP engine. Excellent code, all of it — just not what a personal memory store needs.

---

## Layout

```
agent-mem/
  README.md             ← this file
  daemon/               ← the long-lived event-ingest daemon
    agent_mem_daemon/   ← package
    tests/              ← 177 pytest tests
    pyproject.toml      ← uv-managed
  src/                  ← Phase-0 hook layer (forked from claude-memory-compiler)
    hooks/              ← UserPromptSubmit, PostToolUse, Stop, ...
    scripts/            ← compile / flush / lint / query
    AGENTS.md           ← entry-schema reference
  tools/
    ultan/              ← /ultan, /ultan-install, /ultan-advisor scripts
    search/             ← `agent-mem search` CLI + BM25 indexer (shared library)
  docs/
    LIBRARIAN_PROMPT.md ← Librarian role reference
    SCHOLAR_PROMPT.md   ← Scholar role reference
    SCHEMA.md           ← entry schema reference
```

Storage on disk:

```
~/.agent-mem/
  events.jsonl          ← append-only; hooks write, daemon tails
  daemon.log            ← rotated daemon log
  daemon.pid            ← acquired on start
  daemon.offset.json    ← persistent tailer offset for restart-safe resume
  pending-nudges.md     ← Scholar writes, hook reads + clears + injects
  cost.json             ← running spend tally
  runs/<date>.jsonl     ← per-call audit (cost, duration, decisions, parsed_ok)
  runs/<ts>-<role>-<sid>.md  ← full prompt + response transcripts (7-day TTL)
  knowledge/            ← your library
    README.md
    index.md            ← catalog
    log.md              ← Scholar action log (writes + vetoes + reasoning)
    global/             ← cross-project entries (Librarian organises sub-topics dynamically)
    projects/<slug>/    ← per-repo entries
    _archive/           ← archived entries
```

---

## Status

177 tests passing. Live-tested end-to-end against real Sonnet + Opus calls. Currently a personal dogfood project — not packaged for `pip install`. Expect to clone, `uv sync`, and tune the prompts to your own preferences.

## License

MIT.
