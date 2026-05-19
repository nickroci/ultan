# agent-mem — Plan (v2)

A local, async, curator-driven memory system for coding agents. Built by merging two existing projects and adding the live-watch + two-tier-judge + interrupt loop on top.

---

## 0. Starting point — what we are forking and what we are stealing

### Fork: `prior-art/claude-memory-compiler` (coleam00)

Small, focused, well-aligned. ~8 files. We use this as the **codebase skeleton**.

What we keep verbatim or near-verbatim:
- **Hook architecture.** `hooks/session-end.py` + `hooks/pre-compact.py` reading transcript JSONL from stdin, extracting last-N turns, spawning a background process. Same pattern, no need to redo.
- **Recursion guard.** `CLAUDE_INVOKED_BY` env var prevents the Agent SDK calling Claude Code which fires the hook again. We will hit this exact problem; this is the fix.
- **`flush.py` shape.** Background process that reads pre-extracted context from a temp file and calls the Claude Agent SDK. Our flush will be different in content but the scaffolding stays.
- **`compile.py` Agent SDK pattern.** `system_prompt={"type":"preset","preset":"claude_code"}`, `allowed_tools=["Read","Write","Edit","Glob","Grep"]`, `permission_mode="acceptEdits"`, `max_turns=30`. This is "use CC for all the prompting" already solved.
- **Karpathy-style knowledge layout.** `daily/` (raw), `knowledge/concepts/`, `knowledge/connections/`, `knowledge/qa/`, `knowledge/index.md`, `knowledge/log.md`. Their `AGENTS.md` schema is well thought through; we extend it, we do not replace it.
- **`lint.py` health checks.** Broken links, orphans, contradictions, staleness — already done.

What we change:
- **Storage scope.** Their layout assumes one knowledge base per project (the repo *is* the kb). We want one global store under `~/.agent-mem/` with per-project subdirectories, so lessons from project A inform work in project B.
- **No more end-of-day batch compile.** They compile once at 6 PM. We want continuous extraction (see §1).
- **Add `applies-when` field** to entries — see §2.

### Steal selectively: `prior-art/claude-hooks` (mann1x)

Sprawling. ~hundreds of files, multiple vector backends, dashboards, systemd units, proxies. We do **not** fork this — too much baggage. We read it and steal patterns.

What we take:
- **Daemon pattern.** Their `run_daemon.py` + `claude-hooks-daemon` shows how to run a long-lived local Python process with systemd/launchd/Task Scheduler integration and log rotation. We will need this for the live watcher. **Read for the pattern, write our own.**
- **PostToolUse hook usage.** They wire PostToolUse to run linters on edits. We will wire it to stream tool calls to our daemon for live evaluation against stored lessons.
- **Detached subprocess trick.** They fork dedup-and-store into a "detached subprocess so Stop returns immediately, saving 200–500 ms per turn." We want the same for our curator writes — never block the agent.
- **Hybrid RRF (vector + BM25) idea.** Their `sqlite_vec` provider does RRF over FTS5 BM25 + vector cosine. We will skip the vector half (BM25 only initially — see §3) but the RRF blend code is a useful reference if we add vectors later.
- **Provider abstraction shape.** Their `Provider` ABC (`detect`/`verify`/`recall`/`store`) is overkill for v1, but if we ever want to swap storage backends, this is the right interface.

What we explicitly reject:
- Qdrant, pgvector, sqlite-vec dependencies (BM25 over markdown is enough at our scale).
- The dashboard, the API proxy, the consultants engine, the LSP integration, the OpenWolf coupling, the Caliber pre-commit hooks — all out of scope.
- Systemd units in v1. A `launchd` plist or `nohup` is fine for personal use on macOS.

### Genuinely new — not in either repo

1. **Librarian + Scholar** — the curator is split into two roles. The **Librarian** (cheap model) organises and looks: scans the rolling buffer, scores against existing lessons, gathers candidate-lesson evidence, runs BM25 dedup checks. The Librarian never writes to the store and never decides what reaches the user. The **Scholar** (expensive model) is the only thing that writes: it reviews the Librarian's evidence, makes the final call on whether a candidate becomes a stored lesson, and decides whether a queued interrupt is worth raising to the user.
2. **Live watcher** — daemon scoring in-flight transcripts against an `applies-when` index, not just end-of-session extraction.
3. **Soft-interrupt UX** — batched, judge-approved nudges injected on next agent turn.
4. **BM25 search alongside hierarchy + index** — see §3.
5. **Provisional / confirmed lifecycle.**

---

## 1. Architecture

```
                          ┌────────────────────────────────┐
                          │  Claude Code (the user agent)  │
                          │                                │
  user turn ────►  UserPromptSubmit hook ──────►───────────┼───┐
                          │                                │   │
                          │  PostToolUse hook (per tool) ──┼───┼───►  agent-mem
                          │                                │   │       daemon
                          │  Stop hook (turn end)  ────────┼───┘       (long-lived)
                          │                                │              │
                          │  SessionEnd hook ──────────────┼─────────────►│
                          └────────────────────────────────┘              │
                                                                          │
   pending nudges file ◄────────────────────────────────────────────────┐ │
   (read by next UserPromptSubmit hook)                                 │ │
                                                                        │ │
                                                            ┌───────────▼─▼────────┐
                                                            │  agent-mem daemon    │
                                                            │  - rolling buffer    │
                                                            │  - Librarian (cheap) │
                                                            │  - Scholar (expens.) │
                                                            │  - BM25 dedup        │
                                                            │  - writer (CC SDK)   │
                                                            └──────────┬───────────┘
                                                                       │
                                                                       ▼
                                                  ┌──────────────────────────────────┐
                                                  │ ~/.agent-mem/                    │
                                                  │  knowledge/                      │
                                                  │    index.md                      │
                                                  │    log.md                        │
                                                  │    daily/                        │
                                                  │    global/concepts/              │
                                                  │    global/connections/           │
                                                  │    projects/<slug>/concepts/     │
                                                  │    projects/<slug>/connections/  │
                                                  │    _archive/                     │
                                                  │  .bm25.idx                       │
                                                  │  daemon.log                      │
                                                  │  pending-nudges.md               │
                                                  └──────────────────────────────────┘

(Full schema, frontmatter, project-slug recipe, and legacy-entry migration
policy live in `src/AGENTS.md` — authoritative.)
```

### Components

**1. Hooks (Python, lightweight, no API calls)** — adapted from claude-memory-compiler:
- `UserPromptSubmit`: read pending-nudges file; inject into prompt via `additionalContext`; clear the file.
- `PostToolUse`: append tool call summary to the daemon's input pipe (FIFO or socket).
- `Stop`: signal end-of-turn to daemon; daemon decides whether to write.
- `SessionEnd` / `PreCompact`: extract transcript, hand off final batch.

Hooks do *zero* LLM calls. They are file I/O and IPC only. Same constraint as claude-memory-compiler — keep hook latency < 50ms.

**2. Daemon (`agent-mem-daemon`)** — new. One per user, started by launchd on macOS. Houses the Librarian and the Scholar.
- Reads PostToolUse / Stop events from a Unix socket at `~/.agent-mem/sock`.
- Maintains a rolling buffer of the last N turns per session (default N=20).
- After every Stop event, the **Librarian** runs:
  - **Relevance scan**: is anything in the rolling buffer covered by an existing `confirmed` lesson? If yes, score it and queue an interrupt candidate with evidence (which lesson, which buffer turns matched, similarity score).
  - **Lesson extraction**: are there any new candidate lessons in this turn? Each candidate comes with its supporting turns quoted.
  - **BM25 dedup pass**: for every candidate, search existing entries for near-duplicates and attach the hits to the evidence.
  - The Librarian writes nothing to the knowledge store. It only fills the candidates queue and the interrupts queue.
- Periodically (every M turns or every K candidates queued), the **Scholar** runs:
  - For each candidate lesson: review the Librarian's evidence + dedup hits; decide *write new*, *update existing*, or *discard*; if writing, produce the final markdown via the Claude Agent SDK as a `provisional` entry.
  - For each queued interrupt: decide whether it's actually worth raising to the user; if yes, phrase the nudge; if no, drop it.
  - The Scholar is the only thing that writes to disk and the only thing that decides what the user sees.

**3. CLI (`agent-mem`)** — new. User-facing operations:
- `agent-mem review` — batch-review provisional entries (promote / reject / edit).
- `agent-mem search <query>` — three modes: hierarchy browse, index-led LLM retrieval, BM25 keyword.
- `agent-mem promote <id>` / `demote <id>` / `forget <id>`.
- `agent-mem doctor` — runs lint.py + checks daemon health + reports queue sizes.

**4. Storage** — `~/.agent-mem/knowledge/` (extension of claude-memory-compiler's layout):

```
~/.agent-mem/
  daily/                              # raw extracted lessons by date (immutable)
  knowledge/
    index.md                          # master catalog with applies-when column
    log.md                            # append-only build log
    global/
      concepts/
        factory-pattern-for-apis.md
        no-mock-db.md
      connections/
        paradigms-cross-cutting.md
    projects/
      <repo-slug>/
        concepts/
        connections/
    _archive/                         # demoted / forgotten entries
  .bm25.idx                           # BM25 index, regenerated on write
  daemon.log
  daemon.sock
  pending-nudges.md                   # written by daemon, read by UserPromptSubmit hook
```

---

## 2. Entry schema

Extends claude-memory-compiler's frontmatter with a few fields the curator/judge need:

```markdown
---
id: factory-pattern-for-apis
type: lesson | fact | reference | gotcha
scope: global | project:<slug>
status: provisional | confirmed | stale
confidence: 0.0–1.0                   # judge-assigned
applies-when: |                       # short trigger phrase(s), one per line
  designing or building any new API
  decisions about how clients construct service objects
keywords: [factory, paradigm, api, construction]   # for BM25 fallback
created: 2026-05-19
updated: 2026-05-19
fired: 0                              # how many times an interrupt cited this
fired-helpful: 0                      # how many of those the user kept
sources:
  - daily/2026-05-12.md#L42
related: [[concepts/paradigms-cross-cutting]]
---

# Use factory pattern for new APIs

**Rule:** New service-facing APIs go through a factory, not direct constructor calls from consumers.

**Why:** [...]

**How to apply:** [...]
```

The `applies-when` strings are what the Librarian matches the rolling-buffer text against. Keep them short, declarative, and trigger-shaped. The Scholar writes them; if a lesson interrupts too often or too rarely, the Scholar rewrites them on its next pass through that entry.

---

## 3. Retrieval — three modes, none alone is enough

Karpathy's argument (which claude-memory-compiler bakes in) is: "LLM-over-index beats similarity at small scale." This is correct for **answering known-topic questions**. It fails for **cross-cutting keyword lookups**.

Concrete example: the lesson "use factory pattern for new APIs" is filed under `global/concepts/factory-pattern-for-apis.md`. Its index entry is something like *"Factory pattern for API construction"*. If a user asks "what paradigms do we use for…", the word "paradigm" appears nowhere in the index. The LLM-over-index can't find it. BM25 over the article body can.

So we ship all three:

| Mode | Implementation | When the active agent uses it |
|---|---|---|
| **Hierarchy traversal** | `ls`/`Glob`/`Read` via Claude Code's built-in tools, scoped to `~/.agent-mem/knowledge/` | "Show me everything under code-style" — topic-known browsing |
| **Index-led LLM retrieval** | `agent-mem search --index <q>`: spawns Claude Agent SDK with `index.md` in context, asks "which articles are relevant," reads those in full | Default — questions phrased as topics |
| **BM25 over article bodies** | `agent-mem search --bm25 <q>`: stock BM25 (rank-bm25 or MiniSearch) over full article text. Falls through if index search returns nothing useful | Keyword fishing, paraphrase-tolerant fallback, cross-cutting concepts |

The Librarian also uses BM25 — for dedup detection. When it extracts a new candidate lesson, it BM25-searches existing entries to find near-duplicates and hands the hits to the Scholar as part of the evidence packet.

**No vector DB in v1.** BM25 with proper tokenization (markdown-aware, code-block-aware) covers the paraphrase cases well enough at small scale. If BM25 misses things (e.g., "don't mock the database" vs "use real Postgres"), we add a local sentence-transformer + RRF blend later. Reference: mann1x's sqlite_vec provider does exactly this.

### Pinned BM25 tokenization rules

Surfaced during implementation; pinned here so they don't drift:

1. **YAML frontmatter is stripped from the body before tokenization, except `keywords:` and `applies-when:` which ARE indexed.** Everything else in frontmatter (`id`, `status`, `confidence`, `sources`, `related`, `fired`, `fired-helpful`) is metadata and not searchable.
2. **Code fences are kept verbatim.** Identifiers in code blocks are valuable search terms.
3. **No stemming, no stopword removal in v1.** Lowercase + split on `[^a-z0-9]+` + drop tokens of length < 2. Stemming (Porter) is a deferred optimization for paradigm/paradigms-style misses — adds a dependency, skip for now.
4. **`_archive/`, `index.md`, and `log.md` are excluded from the BM25 corpus.** The catalog files name-drop every concept, which collapses `BM25Okapi` IDF for concept names (it clamps to 0 for any term appearing in ≥ N/2 documents). Including them silently breaks search at small corpus sizes.
5. **Index cache lives at `~/.agent-mem/.bm25.idx`.** Pickled, atomic rename on write, self-healing rebuild if unpickling fails. Rebuilds when any tracked file's mtime advances or the file set changes.

---

## 4. The Librarian and the Scholar

Two roles, one daemon. The split is deliberate: the Librarian does the cheap, high-volume looking-and-sorting work; the Scholar does the expensive, judgment-heavy writing-and-deciding work. **The Librarian never writes to the store and never decides what the user sees.** That separation is the whole point — keep volume cheap, keep judgment expensive, never mix them.

**The Scholar has hard veto over everything the Librarian proposes.** Every Librarian output is a *request*, not an instruction. The Scholar can:
- Refuse to write a candidate as a new entry (it's noise, not a lesson).
- Refuse to merge two entries the Librarian flagged as duplicates (they look similar but actually capture different things).
- Refuse to update an existing entry (the new evidence doesn't materially change what's already stored).
- Refuse to fire a queued interrupt (the user doesn't need to see this now, or ever).
- Demote, rephrase, or rewrite anything before it commits.

If the Scholar discards a Librarian request, that's the end of it — the Librarian doesn't get to retry or escalate. There is no override path. This asymmetry is the invariant that keeps cheap-tier mistakes from ever reaching disk or the user.

```
                rolling buffer (last 20 turns)
                          │
                          ▼
              ┌─────────────────────────┐
              │  LIBRARIAN               │  Haiku-tier
              │  (runs every Stop)       │  via Claude Agent SDK
              │                          │
              │  Organises and looks:    │
              │   - extract candidate    │
              │     lessons + evidence   │
              │   - score buffer against │
              │     each `confirmed`     │
              │     entry's applies-when │
              │   - BM25 dedup hits      │
              │     for each candidate   │
              │                          │
              │  Outputs evidence        │
              │  packets to the          │
              │  candidates queue and    │
              │  interrupts queue.       │
              │  Writes nothing.         │
              └─────────────────────────┘
                          │
                          ▼
              candidates + interrupts queue
                          │
                          ▼
              ┌─────────────────────────┐
              │  SCHOLAR                 │  Opus-tier
              │  (runs every K turns or  │  via Claude Agent SDK
              │   N items queued)        │
              │                          │
              │  Reads what the          │
              │  Librarian assembled.    │
              │  Has hard veto.          │
              │                          │
              │  For each candidate:     │
              │   - write new entry, OR  │
              │   - update existing, OR  │
              │   - merge with existing, │
              │     OR                   │
              │   - VETO (discard, no    │
              │     retry, no escalation)│
              │                          │
              │  For each interrupt:     │
              │   - approve + phrase, OR │
              │   - VETO (drop silently) │
              │                          │
              │  Only the Scholar writes │
              │  to disk. Only the       │
              │  Scholar approves user-  │
              │  facing interrupts.      │
              │  No appeals process.     │
              └─────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
      write entry    queue nudge   discard
      (provisional)  for next turn
```

**Why split the roles this way.** Volume work (per-Stop scanning, dedup, scoring) needs to run cheaply and often, so Haiku. Judgment work (is this really a durable lesson? is this worth interrupting for?) is exactly what cheap models are worst at — that's why it goes to Opus. By forcing all writes and all user-visible decisions through the Scholar, the worst case is "Scholar is too cautious"; we never get the failure mode where Haiku confidently writes nonsense or fires bad interrupts.

**Cost stays bounded.**
- Librarian runs per-Stop. Cheap, high-frequency.
- Scholar runs every K turns *or* when N items have queued (whichever comes first). Bounded.
- Most turns produce zero candidates because the Librarian's first filter is strict, so the Scholar often runs on a near-empty queue and returns fast.
- If the Scholar's queue ever exceeds a configured ceiling, the daemon backs off the Librarian instead of running the Scholar more — we'd rather drop coverage than let cost spiral.

---

## 5. Interrupt UX

Hard requirement: **never block the active agent.** Interrupts are soft.

- Daemon writes ratified interrupts to `~/.agent-mem/pending-nudges.md`.
- The next `UserPromptSubmit` hook reads + clears that file and injects content as `additionalContext`. The agent sees: *"Two relevant lessons from memory: [link to lesson 1], [link to lesson 2]. The user has not been asked. Apply if relevant."*
- The active agent decides what to do. It can ignore the nudge.
- After K injections, the daemon writes a "review batch" notice telling the user `agent-mem review` is worth running.

**Budget:** at most 1 nudge per turn, at most 3 per session. The judge picks the highest-confidence ones if more queue. Without this budget, this becomes Clippy.

**False-positive feedback:** the user can `agent-mem reject <nudge-id>` to mark a nudge as wrong. The judge re-evaluates the lesson's `applies-when` next time it touches that entry; after N rejections, the entry is auto-demoted to `provisional`.

---

## 6. Lesson lifecycle

```
candidate → provisional → confirmed → stale → archived
            (judge wrote)  (user promoted     (auto, after
                            OR fired N        M months no
                            times helpful)    activity)
```

- **provisional** — judge wrote it; not yet used for interrupts; user has not endorsed.
- **confirmed** — eligible for interrupts. Reaches this state via `agent-mem review` or auto-promotion (fired ≥ 3 times with `fired-helpful / fired` ≥ 0.66).
- **stale** — no activity for M months; surfaced in `agent-mem doctor` for user attention.
- **archived** — moved to `_archive/`, removed from indexes, never surfaced again.

---

## 7. Decisions and open questions

### Decided (pinned)

1. **Librarian model.** Haiku via Claude Agent SDK.
2. **Scholar model.** Opus 4.7 via Claude Agent SDK.
3. **Hook → daemon transport: JSONL tail.** Hooks append one line per event to `~/.agent-mem/events.jsonl` (overridable via `AGENT_MEM_HOME`). Daemon tails with mtime + inode tracking, handles rotation/truncation. **Hooks must use O_APPEND** so concurrent writes from parallel Claude Code sessions stay atomic for lines under PIPE_BUF.
4. **Frozen event JSON line schema:**
   ```json
   {"ts": "<ISO-8601 or unix-float>",
    "session_id": "<string, required>",
    "type": "PostToolUse|Stop|SessionEnd|UserPromptSubmit|SessionStart|...",
    "cwd": "<string|null>",
    "payload": {<arbitrary>}}
   ```
   `ts` and `payload` are optional. Unknown `type` values are accepted and logged. Lines terminated with `\n`.
5. **Turn semantics.** A turn ends at every `Stop`. Everything between two `Stop`s (including the trailing Stop) belongs to one turn. `SessionEnd` seals the open turn and marks the session ended — and fires one final Librarian pass (treated like Stop, with `ended: true` in the snapshot).
6. **Frozen stub signatures** (Task #6 must preserve these):
   ```python
   librarian.scan(buffer_snapshot: dict) -> EvidencePacket  # {"session_id","candidates","interrupts"}
   scholar.review(packets: list[EvidencePacket]) -> None
   ```
7. **Scheduler defaults: K=3 Librarian invocations OR M=10 minutes, whichever fires first.** Backpressure ceiling = 20 queued packets — pause Librarian when exceeded.
8. **Per-project slug recipe** lives in a shared `paths.py` (consumed by hooks AND daemon). Recipe per `src/AGENTS.md` §1.1: git remote URL if present → lowercase + strip protocol/host + drop `.git` + non-alphanumeric → `-`; else `basename(cwd)` with the same slugification.
9. **Multiple parallel sessions.** Single daemon, one rolling buffer per `session_id`. The daemon is single-process so no write-side locking inside the daemon; hooks use O_APPEND (see #3).
10. **Nudge rendering.** `additionalContext` via `UserPromptSubmit` for fresh nudges. `SessionStart` for stale-but-relevant carryover. Budget: 1 nudge/turn, 3/session.

### Still open

- **`UserPromptSubmit` events in the daemon stream.** Daemon currently accepts them as in-turn events and the Librarian sees them in the buffer. Confirm this is what we want once Phase 3 (interrupts) lands — alternative is eliding them.
- **What does the Librarian receive on `SessionEnd` vs `Stop`?** Currently identical, only distinguished by `ended: true` in the snapshot. May want SessionEnd to trigger a deeper/broader pass.
- **Counter ownership confirmed: daemon writes `fired` / `fired-helpful`** (empirical facts, not judgment). Scholar writes `confidence`, `applies-when`, content. See `src/AGENTS.md` §2.

---

## 8. Phased build

Each phase is independently shippable and answers a real question.

**Phase 0 — vendor + smoke test (1 day).**
- Fork claude-memory-compiler into `src/`.
- Strip its project-local assumption; relocate storage to `~/.agent-mem/`.
- Run it as-is for a few sessions. Confirm the SessionEnd → flush → daily log → compile loop works on macOS.
- **Kill criterion:** if the hook/SDK plumbing doesn't work on our machine, we have a problem to solve before designing anything new.

**Phase 1 — three-mode search (2 days).**
- `agent-mem search` CLI with `--hierarchy`, `--index`, `--bm25` modes.
- BM25 index regenerated on every knowledge write (rank-bm25, cached in `.bm25.idx`).
- The active agent gets a SessionStart hook injecting the index + a hint about `agent-mem search --bm25` as a fallback tool.
- **Kill criterion:** if Phase 1 + claude-memory-compiler's compile loop already feels useful, the daemon may be unnecessary. Reassess before Phase 2.

**Phase 2 — daemon with Librarian + Scholar, write path only (4 days).**
- Long-running `agent-mem-daemon` listening on `~/.agent-mem/daemon.sock`.
- PostToolUse + Stop hooks stream events to it.
- **Librarian** assembles evidence packets from the rolling buffer: candidate-lesson text quoted from the transcript, BM25 dedup hits against existing entries, candidate "merge into" / "update" suggestions. The Librarian writes nothing to the knowledge store — its output is a structured packet handed to the Scholar.
- **Scholar** is the only thing that touches disk. It reads each packet and decides: write new (as `provisional`), update existing, merge with existing, or discard. All writes go through the Claude Agent SDK.
- No interrupts yet — this phase is purely about getting the write path right.
- `agent-mem review` CLI for batch-promote / reject of provisional entries.
- **Kill criterion:** if the Scholar's writes are mostly garbage even with Opus-tier judgment, the upstream Librarian packets must be the problem — re-prompt the Librarian or drop the curator entirely and keep Phase 1.

**Phase 3 — interrupts (3 days).**
- Librarian relevance-scans rolling buffer against `applies-when` index, queues interrupt candidates with evidence.
- Scholar reviews each queued interrupt and ratifies (or drops) it before anything reaches the user.
- Ratified interrupts → `pending-nudges.md` → `UserPromptSubmit` injection.
- Budget enforcement (1/turn, 3/session).
- `agent-mem reject` feedback loop demotes lessons whose interrupts repeatedly miss.
- **Kill criterion:** if interrupts annoy more than they help after one week of dogfooding even with the Scholar gatekeeping, kill interrupts and ship just the curator + search.

**Phase 4 — hardening (open-ended).**
- launchd plist for daemon startup.
- Log rotation (cribbed from mann1x's `_rotate_if_large`).
- Sentence-transformer + RRF if BM25 misses too many paraphrases.
- Generic protocol for non-Claude-Code hosts (Codex CLI, Cursor).

---

## 9. What success looks like

After 4 weeks of personal dogfooding:

- `~/.agent-mem/knowledge/` has 30–100 entries, browsable in any text editor.
- The user can answer "what does my agent know about $TOPIC?" by `ls`, `agent-mem search`, or just opening `index.md`.
- Fewer than 1 nudge per session on average; > 60% rated useful in `agent-mem review`.
- Measurable: repeated corrections per session drop between week-1 and week-4 baselines.
- If any of the above fails, the markdown store + CLI is still a useful artifact even with the daemon removed.

---

## 10. Open work items

- [ ] §0: read mann1x's `run_daemon.py` and `claude_hooks/daemon.py` end-to-end before designing ours.
- [ ] §1: decide socket vs JSONL-tail for hook→daemon transport. (Default: JSONL-tail.)
- [ ] §2: write a starter `AGENTS.md` for our schema, extending claude-memory-compiler's.
- [ ] §3: pick BM25 library (`rank_bm25` Python — simple, no deps).
- [ ] §4: write the Librarian prompt and the Scholar prompt; calibrate against 1 week of dogfood transcripts.
- [ ] §5: decide nudge rendering (additionalContext format).
