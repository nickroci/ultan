# Ultan

A personal memory system for coding agents. Built for Claude Code; lives outside any one session, project, or machine.

> *"Show me a man who has read all of the books of one of the major branches of knowledge — say, military history — and I'll show you a man more ignorant than the merest churl. For while he has read, others have written; and the body of available knowledge has grown so much faster than his understanding of it that he is, on balance, less learned at the end of his studies than at their beginning."*
>
> — Master Ultan, Gene Wolfe

Ultan watches your conversations as you work, learns your preferences and conventions, and surfaces them when they matter. It's the "remember when you told me to always use uv" that you wish Claude already did natively, except organised, deduplicated, validated, and proactively consulted before the agent interrupts you to ask something you've already answered.

It's your library. On your disk. In plain markdown. You can `ls` it, `cat` it, `git` it.

---

## Design, in one paragraph

Ultan is modelled — deliberately, at the level of the architecture, not as decoration — on how mammalian brains decide what to remember and how to surface it again later. Three ideas drive the whole system:

1. **Surprise gates storage.** Memory is not a transcript. The brain encodes events that violate prediction; novelty and reward-prediction-error are the dopaminergic signals that license hippocampal write (Lisman & Grace, 2005; Schultz, Dayan & Montague, 1997). Ultan's curator does the same: it captures an entry only when the user has told it something a competent assistant would *not* have produced unprompted. No surprise, no write. The three salience signals — **contradicts**, **novel**, **reinforces** — are direct analogues of prediction-error, novelty, and reactivation/consolidation (Sinclair & Barense, 2019).
2. **Two systems, asymmetric bars.** Fast recall-tuned detection, slow precision-tuned deliberation — System 1 and System 2 (Kahneman, 2011). The Librarian (Sonnet) flags candidates aggressively. The Scholar (Opus) verifies them slowly and decides whether to commit. Cheap-and-broad gates expensive-and-narrow, the way the brain's salience network gates the prefrontal cortex.
3. **Three retrieval tiers.** Humans don't query memory uniformly. They get ambient familiarity-driven priming, deliberate hippocampal recollection, and a fast orbital-PFC "stop signal" when the environment matches a stored constraint (Yonelinas, 2002; Aron, Robbins & Poldrack, 2014). Ultan exposes each as a separate mechanism with its own latency budget (§ *Three retrieval tiers*, below).

### References

- Lisman, J.E. & Grace, A.A. (2005). *The Hippocampal-VTA Loop: Controlling the Entry of Information into Long-Term Memory.* Neuron, 46(5), 703–713.
- Schultz, W., Dayan, P. & Montague, P.R. (1997). *A neural substrate of prediction and reward.* Science, 275(5306), 1593–1599.
- Yonelinas, A.P. (2002). *The nature of recollection and familiarity: A review of 30 years of research.* Journal of Memory and Language, 46(3), 441–517.
- Kahneman, D. (2011). *Thinking, Fast and Slow.* Farrar, Straus and Giroux.
- Aron, A.R., Robbins, T.W. & Poldrack, R.A. (2014). *Inhibition and the right inferior frontal cortex: one decade on.* Trends in Cognitive Sciences, 18(4), 177–185.
- Sinclair, A.H. & Barense, M.D. (2019). *Prediction Error and Memory Reactivation: How Incomplete Reminders Drive Reconsolidation.* Trends in Neurosciences, 42(10), 727–739.
- Collins, A.M. & Loftus, E.F. (1975). *A spreading-activation theory of semantic processing.* Psychological Review, 82(6), 407–428.

---

## What it does

- **Surprise gates the write** (see *Design, in one paragraph*, above). The curator asks, of every candidate, "would a competent assistant produce this advice unprompted?" If yes — already in the model's baseline knowledge, no information added by storing it — skip. If no, capture. Three salience signals trigger a write, mapping directly onto prediction-error, novelty, and reactivation:
  - **Contradicts** an existing entry — user has changed their mind. Highest priority. Deprecates the old, writes the new. *(Prediction-error: stored belief was wrong.)*
  - **Novel** — not in the library, not derivable from the model's training (user-specific facts, strict overrides of defaults, idiosyncratic preferences). *(Novelty: no matching trace exists.)*
  - **Reinforces** — user repeated something we already have. No new entry; daemon bumps a `reinforced` counter on the existing entry to track how often it's reasserted. *(Reactivation: existing trace strengthened, not duplicated — Sinclair & Barense, 2019.)*
- **Two-tier curator with asymmetric bars.** The Librarian (Sonnet) does fast salience detection — low bar, recall-tuned. The Scholar (Opus) deliberates — higher bar, precision-tuned. System 1 gates System 2; cheap-and-broad gates expensive-and-narrow.
- **Organises a real library, not a flat pile.** Topical hierarchy emerges from content. Every folder has a README. ≤5 entries per directory before splitting. Auto-maintained child listings between marker comments. Wikilinks validate. Frontmatter validates. Scope/path agreement enforced.
- **Three slash commands** wire it into Claude Code without ceremony:
  - `/ultan <text>` — drop something into memory now, no extraction needed.
  - `/ultan-install` — wire the hooks into the current project's `.claude/settings.json`.
  - `/ultan-advisor <question>` — query the library before asking the user a preference question. The advisor finds relevant entries (Sonnet, BM25 + embeddings + Read), writes a referenced answer (Opus), and clearly distinguishes stored knowledge from its own opinion. *Always cheaper to check than to ask.*
- **Pure markdown store.** No database. The library is `~/.agent-mem/knowledge/` — `ls`, `cat`, `git` it. Two derived indexes alongside (`.bm25.idx` for keyword, `.embeddings.idx` for semantic) auto-rebuild on drift.

### Three retrieval tiers

The agent can't afford to query the library for everything it's about to do, and humans don't either — retrieval is layered. Familiarity-based priming via spreading activation (Collins & Loftus, 1975); explicit hippocampal recollection (Yonelinas, 2002); and the rapid orbital-PFC "stop signal" that interrupts an in-flight action when it conflicts with a stored constraint (Aron, Robbins & Poldrack, 2014). Ultan implements all three:

| Tier | Cognitive analog | Latency | Trigger | What it does |
|---|---|---|---|---|
| **1. Ambient priming** | Familiarity / spreading activation (Collins & Loftus, 1975) | ~0 (already in prompt context) | Daemon refreshes `hot-context.md` after every batch | Top 5 most-relevant entries injected as `additionalContext` on every UserPromptSubmit. Agent has them "in mind" without asking. Hybrid retrieval: BM25 + sentence-transformer embeddings merged via RRF, boosted by `reinforced` counter. ≤500 char budget. |
| **2. Deliberate recall** | Hippocampal recollection (Yonelinas, 2002) | 30-60s | `/ultan-advisor <question>` invocation | Sonnet Librarian searches deeply, Opus Scholar synthesises a referenced answer. The drill-down when the priming snippet isn't enough. |
| **3. Acute notice** | Orbital-PFC stop signal (Aron et al., 2014) | <100ms synchronous | PreToolUse hook on every tool call | Default `advise`: pattern-matches the tool call against entries with `block_triggers`; on match, emits an `additionalContext` FYI — *"📚 Library note: [[X]] applies here"* — and the tool proceeds. **The agent decides.** Opt-in `severity: block` is reserved for genuinely dangerous actions (`rm -rf /`, force-push to main); only then does the hook hard-deny. Like a human noticing a relevant memory mid-action: noticed, not paralysed. |

The Scholar still owns a **soft nudge pipeline** orthogonal to these — when the Librarian sees a stored preference matching the rolling buffer, the Scholar can approve a nudge to `pending-nudges.md` for injection on the next turn. Budget: 1/turn, 3/session.

---

## Quick start

```bash
# 1. Sync the daemon's deps (uv-managed)
cd daemon && uv sync --extra dev

# 2. Sync the search CLI (separate venv, shared BM25 implementation)
cd ../tools/search && uv sync

# 3. Install the slash commands and hooks
#    - /ultan, /ultan-install, /ultan-advisor live at ~/.claude/commands/
#    - `/ultan-install` writes hooks into ~/.claude/settings.json (GLOBAL — every
#      Claude Code project). One daemon per machine serves the whole library
#      across every repo, so global is the recommended default.
#    - `/ultan-install --project` if you'd rather scope to one repo only.

# 4. Start the daemon (foreground; logs to ~/.agent-mem/daemon.log). One per
#    machine — it listens on ~/.agent-mem/priming.sock and answers Tier-1
#    priming requests from every hook on every project.
cd /path/to/ultan/daemon && uv run agent-mem-daemon -v
#    (nohup, tmux, or a launchd plist if you want it persistent — Phase 4 work.)

# 5. Open Claude Code in any project (no per-project setup needed once the
#    hooks are global) and work normally. Entries land under
#    ~/.agent-mem/knowledge/ as the Scholar approves them.
```

When the daemon is running, the ``UserPromptSubmit`` hook makes a sub-100 ms
Unix-socket call into it to get a priming snippet keyed on your *current
prompt* (not the last batch's curation). When the daemon is down, the hook
falls back to a tiny in-process lexical scan so you still see relevant entries.

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

## A note on cost

**Ultan is token-heavy by design.** It trades tokens for richer memory and lower friction. Expect roughly:

- **Librarian (Sonnet)** runs after each session's quiet period (per-session debounce, default 30s). With moderate activity that's ~10-30 invocations per working day per project. Each is a few thousand input tokens (prompt + library snapshot + buffer) plus a few hundred output tokens.
- **Scholar (Opus)** runs in batches — every 3 Librarian packets or 60s, whichever first. Each batch is one Opus call: prompt + accumulated proposals, ~30s wall time, ~$0.20-0.50 per batch on pay-as-you-go pricing.
- **Ambient priming (Tier 1)** is daemon-side BM25 + embeddings, **no LLM cost**, but it injects up to 500 chars into every UserPromptSubmit — call it ~150 tokens of prompt overhead per turn.
- **Advisor (`/ultan-advisor`)** is one Sonnet call (Librarian step) + one Opus call (Scholar synthesis) per invocation. ~$0.30-0.50 each.
- **PreToolUse Tier 3** is pure deterministic regex match, **no LLM cost**, sub-100ms.

For pay-as-you-go API users this adds up — easily $5-20/day for active dogfooding. **If you're on Max/Pro**, the cost lands against your quota rather than your card; the latency cost remains.

If that's too rich for your workflow: turn off the daemon entirely and use just the slash commands (`/ultan`, `/ultan-advisor`) — the curator stops running, but the explicit-write and explicit-query paths still work. You lose ambient priming, automatic extraction, and proactive nudges, but you keep the markdown library and the on-demand advisor.

## Status

225 tests passing across daemon, hooks, and search. Live-tested end-to-end against real Sonnet + Opus calls including the three retrieval tiers, the curator's salience-signal classification, README reconciler, wikilink validator, and the PreToolUse advisory/block hook. Currently a personal dogfood project — not packaged for `pip install`. Expect to clone, `uv sync`, and tune the prompts to your own preferences.

## License

MIT.
