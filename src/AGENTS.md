# AGENTS.md — agent-mem Knowledge Base Schema (Compiler Specification)

> This file is the **single source of truth** for the on-disk schema of the
> `agent-mem` knowledge store. The Claude Agent SDK reads it verbatim as the
> "compiler specification" (see `scripts/compile.py` — `schema = AGENTS_FILE.read_text()`).
> Everything the Librarian and Scholar write must conform to what is described
> here.
>
> Architectural background (the two-tier curator, the daemon, BM25, interrupts)
> lives in `../PLAN.md`. Role prompts live in `../docs/LIBRARIAN_PROMPT.md` and
> `../docs/SCHOLAR_PROMPT.md`. This file is the schema; those documents are
> the workflow.
>
> Adapted from [Andrej Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
> architecture by way of `prior-art/claude-memory-compiler/AGENTS.md`. Instead
> of ingesting external articles, this system compiles knowledge from your own
> AI conversations.

---

## The Compiler Analogy

```
daily/          = source code    (your conversations - the raw material)
LLM             = compiler       (extracts and organises knowledge)
knowledge/      = executable     (structured, queryable knowledge base)
lint            = test suite     (health checks for consistency)
queries         = runtime        (using the knowledge)
```

You do not manually organise your knowledge. You have conversations; the LLM
handles the synthesis, cross-referencing, and maintenance.

---

## 1. Directory layout

The knowledge store lives at `~/.agent-mem/` (overridable via
`AGENT_MEM_HOME`). The canonical tree:

```
~/.agent-mem/
├── daily/                              # raw extracted entries by date (immutable)
│   ├── 2026-05-18.md
│   └── 2026-05-19.md
├── knowledge/
│   ├── index.md                        # master catalog with applies-when column
│   ├── log.md                          # append-only build log
│   ├── global/
│   │   ├── concepts/                   # atomic, cross-project lessons
│   │   │   ├── factory-pattern-for-apis.md
│   │   │   └── no-mock-db.md
│   │   └── connections/                # cross-cutting synthesis between 2+ concepts
│   │       └── paradigms-cross-cutting.md
│   ├── projects/
│   │   └── <slug>/                     # see §1.1 for slug rules
│   │       ├── concepts/
│   │       └── connections/
│   └── _archive/                       # demoted / forgotten entries (read-only)
│       ├── concepts/
│       └── connections/
├── .bm25.idx                           # BM25 index, regenerated on every write
├── daemon.log
├── daemon.sock
├── pending-nudges.md                   # written by the Scholar, read by UserPromptSubmit
└── state.json                          # daemon state (queue sizes, last Scholar run, etc.)
```

### 1.1 Project slug

`<slug>` is computed by the hook (cheap, deterministic, no LLM):

1. If `git -C $cwd config --get remote.origin.url` returns a value, slugify it:
   lowercase, strip protocol/host, drop `.git`, replace non-alphanumeric runs
   with `-`.
   - `git@github.com:acme/widget-svc.git` → `acme-widget-svc`
   - `https://github.com/acme/widget-svc` → `acme-widget-svc`
2. Otherwise fall back to `basename(cwd)`, slugified the same way.

The hook stamps the slug onto every event it streams to the daemon. The Scholar
uses it when deciding `scope:` (see §2) and when picking a write path.

### 1.2 Archive policy

`_archive/` is structurally identical to `global/` and `projects/<slug>/`. An
archived entry keeps its full frontmatter but adds `status: stale` (if
auto-demoted) or stays at its last status with an `archived: YYYY-MM-DD` field
appended (if explicitly forgotten). The BM25 index excludes anything under
`_archive/`. The `index.md` catalog does not list archived entries. Wikilinks
pointing into `_archive/` are intentionally allowed — they are not lint errors
— but no live entry should originate one.

### 1.3 Daily logs (Layer 1 — immutable source)

Daily logs capture what happened in your AI coding sessions. They are the
"raw sources" — append-only, never edited after the fact. Each file:

```markdown
# Daily Log: YYYY-MM-DD

## Sessions

### Session (HH:MM) - Brief Title  (project:<slug>)

**Context:** What the user was working on.

**Key Exchanges:**
- User asked about X, assistant explained Y
- Decided to use Z approach because...
- Discovered that W doesn't work when...

**Decisions Made:**
- Chose library X over Y because...
- Architecture: went with pattern Z

**Lessons Learned:**
- Always do X before Y to avoid...
- The gotcha with Z is that...

**Action Items:**
- [ ] Follow up on X
- [ ] Refactor Y when time permits
```

Each session heading is tagged with `project:<slug>` (the same slug as §1.1) so
a downstream compile pass can route lessons into `global/` vs `projects/<slug>/`.

### 1.4 Knowledge tier (Layer 2 — LLM-owned)

The LLM owns the entire `knowledge/` directory. Humans read it but rarely
edit it directly. The Scholar is the only thing that writes to disk.

### 1.5 This file (Layer 3 — the spec)

AGENTS.md is the schema that tells the LLM how to compile and maintain the
knowledge base. This is the "compiler specification." It is loaded into the
Scholar's and the compiler's prompt verbatim.

---

## 2. Frontmatter — full field reference

Every article under `global/`, `projects/<slug>/`, or `_archive/` carries YAML
frontmatter. The canonical field order is below.

```yaml
---
id: factory-pattern-for-apis            # filename without .md, kebab-case, stable
type: lesson                            # one of: lesson | fact | reference | gotcha
scope: global                           # global | project:<slug>
status: provisional                     # provisional | confirmed | stale
confidence: 0.72                        # Scholar-assigned, 0.0–1.0
applies-when: |                         # trigger phrases, one per line, declarative
  designing or building any new API
  decisions about how clients construct service objects
  refactoring a constructor-heavy service layer
keywords: [factory, paradigm, api, construction, dependency-injection]   # BM25 tokenization aid
title: "Use factory pattern for new APIs"
aliases: [api-factory]
tags: [architecture, paradigm]
created: 2026-05-19
updated: 2026-05-19
fired: 0                                # total times this entry triggered an interrupt that reached the user
fired-helpful: 0                        # of those, how many the user kept (did not reject)
sources:                                # daily/* or transcript anchors
  - "[[daily/2026-05-12]]#L42"
related:
  - "[[global/connections/paradigms-cross-cutting]]"
---
```

### 2.1 Field semantics

| Field | Type | Required | Written by | Read by | Purpose |
|---|---|---|---|---|---|
| `id` | string (kebab-case) | yes | Scholar | everything | Stable identifier. Must equal the filename without `.md`. Never rewritten — if an entry is renamed, write a new one and archive the old. |
| `type` | enum | yes | Scholar | Scholar, CLI | `lesson` (rule of thumb), `fact` (verified statement), `reference` (pointer to external authoritative source), `gotcha` (bug-shaped). Drives wording in nudges. |
| `scope` | enum | yes | Scholar | Librarian, Scholar, hooks | `global` if useful across projects; `project:<slug>` if tied to one repo. The path (`global/...` vs `projects/<slug>/...`) **must** agree with this field — if they disagree, lint reports an error. |
| `status` | enum | yes | Scholar (initial), CLI (promotion) | Librarian, Scholar | `provisional` (Scholar wrote, user has not endorsed, **not eligible for interrupts**), `confirmed` (eligible for interrupts; reached via `agent-mem review` or auto-promotion per PLAN §6), `stale` (no activity for M months, surfaced by `doctor`). |
| `confidence` | float in `[0.0, 1.0]` | yes | Scholar | Scholar, CLI | Scholar's belief the lesson is durable. Used as a tiebreaker when the nudge budget is exceeded. Re-evaluated on every Scholar update. |
| `applies-when` | multi-line string | yes | Scholar | Librarian | Trigger phrases, one per line, short and declarative. The Librarian scores rolling-buffer turns against these. **Style:** name the situation, not the rule. Bad: "use factories". Good: "designing or building any new API". |
| `keywords` | list[string] | yes (≥ 3) | Scholar | BM25 indexer, CLI | Tokenization aid for BM25. Short list (5–10). Include synonyms the body may not contain — e.g. an entry titled "factory pattern" should list `paradigm` if "paradigm" is how the user might phrase the question. |
| `title` | string | yes | Scholar | index.md, CLI | Human-readable headline. Renders in `index.md`. |
| `aliases` | list[string] | optional | Scholar | CLI search | Alternate names. Helps the LLM-over-index retrieval mode pick this entry. |
| `tags` | list[string] | optional | Scholar | CLI, lint | Domain tags. Lighter than `keywords`; intended for browsing, not search. |
| `created` | ISO date | yes | Scholar | lint, CLI | Date the entry was first written. Never changes. |
| `updated` | ISO date | yes | Scholar | lint, CLI | Last write date. Touched on any Scholar update. |
| `fired` | int (≥ 0) | yes | daemon (writer) | Scholar (auto-promote), CLI | Total times an interrupt cited this entry **and reached the user** (i.e. survived Scholar veto and budget). |
| `fired-helpful` | int (≥ 0) | yes | daemon (writer) on `agent-mem reject` non-events / `accept` events | Scholar, CLI | Of the `fired` count, how many the user kept. `fired-helpful / fired` is the promotion signal (PLAN §6: ≥ 0.66 with `fired ≥ 3` auto-promotes; high reject rate auto-demotes). |
| `sources` | list[wikilink] | yes (≥ 1) | Scholar | lint | Where the lesson came from. Daily logs or transcript anchors. |
| `related` | list[wikilink] | optional | Scholar | lint, CLI | Cross-references. Lint checks for missing backlinks. |

### 2.2 Why each new field exists

- **`status`** — gates interrupts. Provisional entries are write-only until the
  user (or the auto-promotion rule) endorses them. Without this, every Scholar
  write would immediately start interrupting, which makes Scholar mistakes
  very expensive.
- **`confidence`** — the Scholar's own uncertainty estimate. The daemon uses
  it to rank when the per-session nudge budget (PLAN §5: 1/turn, 3/session) is
  exceeded.
- **`applies-when`** — the Librarian's matching surface. Distinct from
  `tags`/`title` because those describe the *rule*; `applies-when` describes
  the *situation* in which the rule fires. The Librarian scores the rolling
  buffer against this and *only* this.
- **`keywords`** — BM25 needs help with paraphrase ("paradigm" vs "pattern").
  Listing them here lets the tokenizer find the entry even when the body
  never uses the user's word.
- **`fired` / `fired-helpful`** — the lifecycle signal. Together they drive
  auto-promotion (PLAN §6) and `agent-mem reject` auto-demotion (PLAN §5).
- **`scope`** — separates lessons that belong to one repo from lessons that
  travel. The Librarian uses it as a hard filter: when active in project A,
  lessons with `scope: project:B` are not candidates for interrupts.

### 2.2.1 Optional blocking fields (Tier 3 — synchronous PreToolUse)

Two optional frontmatter fields opt an entry in to the **synchronous
PreToolUse deterministic interrupt** (the only point in Claude Code's
hook contract where the host can block a tool call **before** it runs).
The post-hoc Librarian → Scholar → `pending-nudges.md` pipeline is too
late to stop a destructive action; the PreToolUse hook
(`src/hooks/pre-tool-use.py`) loads the small set of entries flagged
here and pattern-matches them against every tool call.

**Opt-in is by `block_triggers` presence** — any entry that lists
triggers gets loaded by the PreToolUse hook. `severity` controls
**how loud** the response is when a trigger matches:

| Field | Type | Required | Written by | Read by | Purpose |
|---|---|---|---|---|---|
| `block_triggers` | list[dict] | optional — present means PreToolUse will check | Scholar / user | PreToolUse hook | One or more match rules. Two shapes recognised below. |
| `severity` | enum | optional (default `advise`) | Scholar / user | PreToolUse hook | `advise` (FYI — emits `additionalContext`, tool proceeds, agent decides) or `block` (hard stop — emits `permissionDecision: deny`, tool refused). Default is `advise` — like a human noticing a relevant constraint, not paralysed by it. |

**Trigger shapes:**

```yaml
# Default (advise): tool proceeds; agent gets an additionalContext FYI.
block_triggers:
  - tool: Bash
    pattern: 'pip install'           # "use uv instead" FYI
  - tool: Edit
    file_pattern: 'production\.env$' # FYI when touching prod env

# Opt-in hard block (rare — reserve for genuinely dangerous things):
severity: block
block_triggers:
  - tool: Bash
    pattern: 'rm -rf /(?!tmp)'       # blanket rm -rf outside /tmp
  - tool: Bash
    pattern: 'git push.*--force.*main'
```

- `tool: Bash` + `pattern: <regex>` — the regex (Python `re.search`,
  not `re.fullmatch`) is matched against the `command` field of
  `tool_input`. Authors anchor with `^`/`$` themselves if they want
  exact-match semantics.
- `tool: Edit` (or `Write`, `NotebookEdit`) + `file_pattern: <regex>`
  — matched against the `file_path` field of `tool_input`.

A trigger fires only when its `tool` value exactly matches the
PascalCase Claude Code tool name (`Bash`, `Edit`, `Write`,
`NotebookEdit`, ...). Invalid regexes are silently dropped at load
time — better one dead trigger than a crashing hook on every tool
call. An entry with zero usable triggers is dropped entirely.

**On match:**

- `severity: advise` (default) → hook emits
  `{"hookSpecificOutput": {"additionalContext": "📚 Library note (FYI; agent decides): [[<wikilink>]] applies here — <one-line rule>"}}`.
  The tool runs; the agent gets the note as system context and decides
  on its own whether to take notice.
- `severity: block` → hook emits
  `{"hookSpecificOutput": {"permissionDecision": "deny", "permissionDecisionReason": "⚠ Library blocks this action: [[<wikilink>]] - <one-line rule>. Confirm with the user before retrying."}}`.
  The tool is refused; the agent must reconsider.

The one-line rule in both cases is the first non-heading, non-empty
line of the entry body, so author your trigger entries with a
one-sentence rule on the first line.

**Recursion guard:** the PreToolUse hook short-circuits when
`CLAUDE_INVOKED_BY` is set, so the Scholar's own Edit/Write calls
inside the daemon are never blocked. This is what keeps the daemon
from deadlocking on its own writes.

### 2.3 Invariants (enforced by `agent-mem doctor` / lint)

1. `id` equals filename without `.md`.
2. The path agrees with `scope`: `scope: global` ⇒ under `global/`;
   `scope: project:<slug>` ⇒ under `projects/<slug>/`.
3. `status: confirmed` requires `fired-helpful / fired ≥ 0.66` **or** explicit
   user promotion (recorded in `log.md`).
4. `applies-when` has at least one non-blank line.
5. `keywords` has at least 3 entries.
6. `fired-helpful ≤ fired`.
7. `confidence` is in `[0.0, 1.0]`.
8. Every `sources` and `related` link resolves (or points into `_archive/`).

---

## 3. `knowledge/index.md` format

A table listing every knowledge article. The Librarian reads this (table only,
not bodies) to recognise candidates that are already covered; the
index-led-LLM retrieval mode of `agent-mem search` reads it first to pick
which articles to load in full.

```markdown
# Knowledge Base Index

| Article | Scope | Status | Conf | Summary | Applies-when (first) | Compiled From | Updated |
|---|---|---|---|---|---|---|---|
| [[global/concepts/factory-pattern-for-apis]] | global | provisional | 0.72 | Factory pattern for API construction | designing or building any new API | daily/2026-05-12.md | 2026-05-19 |
| [[projects/acme-widget-svc/concepts/no-mock-db]] | project:acme-widget-svc | confirmed | 0.91 | Never mock the database in tests — use a real Postgres container | writing or reviewing a database-touching test | daily/2026-05-14.md | 2026-05-17 |
```

The `Applies-when` column is rendered as the **first** applies-when line for
compactness; the full list lives in the entry's frontmatter. The Scholar
maintains this file on every write.

---

## 4. `knowledge/log.md` format

Append-only chronological record of every Scholar action and (eventually)
every query and lint operation. Every Scholar write appends one block.

```markdown
# Build Log

## [2026-05-19T14:30:00] write | provisional | global/concepts/factory-pattern-for-apis
- Source: daily/2026-05-19.md
- Trigger: Librarian candidate id=cand-2026-05-19-007
- Confidence: 0.72
- Dedup near-hits considered: (none above threshold)

## [2026-05-19T14:30:01] update | global/connections/paradigms-cross-cutting
- Source: daily/2026-05-19.md
- Trigger: Librarian candidate id=cand-2026-05-19-008 (merge into existing)
- Added related: [[global/concepts/factory-pattern-for-apis]]

## [2026-05-19T14:30:02] veto | candidate id=cand-2026-05-19-009
- Reason: thin evidence (one off-hand remark, not a durable rule)
- No file written.

## [2026-05-19T14:30:03] nudge | global/concepts/factory-pattern-for-apis
- Buffer turns matched: assistant turn 14, user turn 15
- Match score: 0.81
- Phrased as: "Memory: factory pattern is the established approach for new APIs here."
- Queued to pending-nudges.md, will fire on next UserPromptSubmit.

## [2026-05-19T14:30:04] interrupt-veto | candidate matched=global/concepts/no-mock-db
- Reason: user is reading, not writing tests — not actionable this turn.

## [2026-05-19T14:30:05] compile | daily/2026-05-19.md
- Articles created: [[global/concepts/factory-pattern-for-apis]]
- Articles updated: [[global/concepts/error-handling]]

## [2026-05-19T14:31:00] query | "What auth patterns do I use?"
- Consulted: [[global/concepts/supabase-auth]], [[global/concepts/nextjs-middleware]]
- Filed to: (none)
```

Action values:

| Action | Meaning |
|---|---|
| `write` | New entry created. |
| `update` | Existing entry modified. |
| `merge` | Two or more existing entries combined; survivor + archived ones each get a log line. |
| `veto` | Candidate discarded; no file written. |
| `nudge` | Interrupt approved and queued to `pending-nudges.md`. |
| `interrupt-veto` | Interrupt candidate dropped. |
| `compile` | A daily log was processed (one or more articles created/updated as a batch). |
| `query` | A user query was answered against the KB. |
| `lint` | A lint pass ran. |

---

## 5. Article formats

### 5.1 Concept articles (`knowledge/global/concepts/` or `knowledge/projects/<slug>/concepts/`)

One article per atomic piece of knowledge. These are facts, patterns,
decisions, preferences, and lessons extracted from your conversations.

```markdown
---
id: <kebab-case-slug>
type: lesson
scope: global
status: provisional
confidence: 0.7
applies-when: |
  <situation 1 in which the rule fires>
  <situation 2 ...>
keywords: [<at least 3>]
title: "Concept Name"
aliases: [alternate-name]
tags: [domain, topic]
created: YYYY-MM-DD
updated: YYYY-MM-DD
fired: 0
fired-helpful: 0
sources:
  - "[[daily/YYYY-MM-DD]]"
related:
  - "[[global/concepts/related-concept]]"
---

# Concept Name

[2-4 sentence core explanation, encyclopedia voice]

## Key Points

- [3–5 self-contained bullets, each carrying its own meaning]

## Details

[Deeper explanation, 2+ paragraphs, encyclopedia-style]

## How to apply

[Concrete recipe — what the agent should actually do when the applies-when
trigger fires. Numbered steps where order matters.]

## When this does **not** apply

[Counter-examples. Lessons without escape hatches over-fire.]

## Related Concepts

- [[global/concepts/related-concept]] - How it connects

## Sources

- [[daily/YYYY-MM-DD]] - Initial discovery
- [[daily/YYYY-MM-DD]] - Updated after debugging session
```

### 5.2 Connection articles (`knowledge/global/connections/` or `knowledge/projects/<slug>/connections/`)

Cross-cutting synthesis linking 2+ concepts. Created when a conversation
reveals a non-obvious relationship.

```markdown
---
id: <kebab-case-slug>
type: reference
scope: global
status: provisional
confidence: 0.7
applies-when: |
  <situation that surfaces this cross-cutting concern>
keywords: [<at least 3>]
title: "Connection: X and Y"
tags: [cross-cutting]
created: YYYY-MM-DD
updated: YYYY-MM-DD
fired: 0
fired-helpful: 0
sources:
  - "[[daily/YYYY-MM-DD]]"
related:
  - "[[global/concepts/concept-x]]"
  - "[[global/concepts/concept-y]]"
---

# Connection: X and Y

## The Connection

[What links these concepts]

## Key Insight

[The non-obvious relationship discovered]

## Evidence

[Specific examples from conversations]

## Related Concepts

- [[global/concepts/concept-x]]
- [[global/concepts/concept-y]]
```

---

## 6. Fully-worked example article

The "factory pattern for APIs" case — chosen because it exercises every new
field, has a real cross-cutting story, and shows how `applies-when` differs
in shape from `title`/`tags`.

**Path:** `~/.agent-mem/knowledge/global/concepts/factory-pattern-for-apis.md`

```markdown
---
id: factory-pattern-for-apis
type: lesson
scope: global
status: provisional
confidence: 0.72
applies-when: |
  designing or building any new API
  decisions about how clients construct service objects
  refactoring a constructor-heavy service layer
  reviewing code that instantiates services with `new Foo(...)` at call sites
keywords: [factory, paradigm, api, construction, dependency-injection, service-locator]
title: "Use factory pattern for new APIs"
aliases: [api-factory, service-factory]
tags: [architecture, paradigm]
created: 2026-05-19
updated: 2026-05-19
fired: 0
fired-helpful: 0
sources:
  - "[[daily/2026-05-12]]#L42"
  - "[[daily/2026-05-19]]#L88"
related:
  - "[[global/connections/paradigms-cross-cutting]]"
---

# Use factory pattern for new APIs

**Rule:** New service-facing APIs go through a factory, not direct constructor calls from consumers.

## Key Points

- Consumers never call `new ServiceX(...)` directly; they call `ServiceX.create(...)` or an injected `ServiceXFactory`.
- The factory owns wiring: config resolution, credential loading, transport selection, retry policy.
- This applies to **service-facing** APIs (the boundary the rest of the codebase consumes). Internal helpers do not need a factory.

## Why

- Decouples consumers from construction-order coupling. When we added a `retryPolicy` argument in 2026-05, every direct-`new` call site had to be touched; factory call sites were unchanged.
- Gives one obvious place to insert test seams. Most service-level tests now stub the factory, not the service.
- Carries forward to multi-tenant: the factory is where the tenancy context attaches.

## How to apply

1. Define `XFactory.create(opts)` returning a fully-wired `X`.
2. Export only the factory and the resulting interface — never the class.
3. If a consumer needs to compose two services, accept the factories, not the instances.

## When this does **not** apply

- Pure-functional modules (no state, no dependencies).
- Test doubles inside a single test file.

## Sources

- [[daily/2026-05-12]] — initial discovery while refactoring `PaymentsService`.
- [[daily/2026-05-19]] — second confirmation while building `NotificationsService`; same construction-coupling problem reappeared.

## Related

- [[global/connections/paradigms-cross-cutting]] — overall "paradigms" cross-cutting note that links factory pattern to other construction-time decisions (DI containers, service locator).
```

Notes on how this entry exercises each field:

- `applies-when` lists four distinct trigger shapes — design, decision-making,
  refactor, review — because the rule fires in all four contexts. The Librarian
  scores any rolling-buffer turn against each line independently and reports
  the max.
- `keywords` includes `paradigm` even though the body says "pattern" — this
  is the cross-cutting case: a BM25 search for "what paradigms do we use"
  must reach this entry.
- `status: provisional` and `fired: 0` are the initial Scholar write.
  Promotion to `confirmed` requires either `agent-mem review` or `fired ≥ 3`
  with `fired-helpful / fired ≥ 0.66`.
- `scope: global` and the path `global/concepts/...` agree — required by
  invariant §2.3(2).

---

## 7. Core operations

### 7.1 Compile (daily/ → knowledge/)

When processing a daily log:

1. Read the daily log file.
2. Read `knowledge/index.md` to understand current knowledge state.
3. Read existing articles that may need updating.
4. For each piece of knowledge found in the log:
   - If an existing concept article covers this topic: **UPDATE** it with new
     information; add the daily log as a source; bump `updated`.
   - If it is a new topic: **CREATE** a new concept article under
     `knowledge/global/concepts/` (Phase 0 routes everything to `global/`;
     per-project routing is a Scholar concern, see PLAN §4).
5. If the log reveals a non-obvious connection between 2+ existing concepts:
   **CREATE** a connection article under `knowledge/global/connections/`.
6. **UPDATE** `knowledge/index.md` with new/modified entries.
7. **APPEND** to `knowledge/log.md` with a `compile` block.

**Rules for writing articles (the Scholar / compiler must obey these):**

- A single daily log may touch 3–10 knowledge articles.
- Prefer updating existing articles over creating near-duplicates.
- Use Obsidian-style `[[wikilinks]]` with full relative paths from
  `knowledge/`. **The current path form is `[[global/concepts/foo]]`, NOT the
  legacy `[[concepts/foo]]`.** The legacy form is recognised by the lint
  compatibility shim but the compiler must emit the new form on every write.
- Write in encyclopedia style — factual, concise, self-contained.
- Every article must have complete YAML frontmatter per §2.
- Every article must link back to its source daily log(s) via `sources:`.
- Every article must link to at least 2 other articles via `[[wikilinks]]`
  (either inline in the body or in the `related` frontmatter list).
- "Key Points" section: 3–5 self-contained bullets.
- "Details" section: 2+ paragraphs.
- "Related Concepts" section: 2+ entries.
- "Sources" section: cite the daily log(s) with specific claims extracted.

### 7.2 Query (Ask the Knowledge Base)

1. Read `knowledge/index.md` (the master catalog).
2. Based on the question, identify 3–10 relevant articles from the index.
3. Read those articles in full.
4. Synthesise an answer with `[[wikilink]]` citations.
5. If `--file-back` is specified: file a follow-up under
   `knowledge/global/concepts/` (or under the appropriate project) and update
   `index.md` and `log.md`.

**Why this works without RAG:** at personal-KB scale (50–500 articles), the
LLM reading a structured index outperforms cosine similarity. The LLM
understands what the question is really asking and selects pages accordingly.
Embeddings find similar words; the LLM finds relevant concepts. For
cross-cutting keyword cases the index fails (the user says "paradigm" and the
index says "pattern"), BM25 over article bodies + `keywords` frontmatter is
the fallback (PLAN §3, mode 3).

### 7.3 Lint (Health Checks)

Seven checks, run periodically:

1. **Broken links** — `[[wikilinks]]` pointing to non-existent articles (links
   into `_archive/` are intentionally allowed; see §1.2).
2. **Orphan pages** — Articles with zero inbound links from other articles.
3. **Orphan sources** — Daily logs that have not been compiled yet.
4. **Stale articles** — Source daily log changed since article was last
   compiled.
5. **Contradictions** — Conflicting claims across articles (requires LLM
   judgment).
6. **Missing backlinks** — A links to B but B does not link back to A.
7. **Sparse articles** — Below 200 words, likely incomplete.

Plus all invariants from §2.3.

Output: a markdown report with severity levels (error, warning, suggestion),
written to `~/.agent-mem/reports/lint-YYYY-MM-DD.md`.

---

## 8. Conventions

- **Wikilinks:** Obsidian-style `[[path/to/article]]` without `.md` extension.
  Paths are relative to `knowledge/`. **Use the new form
  `[[global/concepts/foo]]` or `[[projects/<slug>/concepts/foo]]` — not the
  legacy `[[concepts/foo]]`.**
- **Writing style:** Encyclopedia-style, factual, third-person where
  appropriate.
- **Dates:** ISO 8601 (`YYYY-MM-DD` for dates, full ISO for timestamps in
  `log.md`).
- **File naming:** lowercase, hyphens for spaces
  (e.g. `supabase-row-level-security.md`). The `id` frontmatter field must
  equal the filename without `.md`.
- **Frontmatter:** every article has YAML frontmatter conforming to §2. The
  required fields (`id`, `type`, `scope`, `status`, `confidence`,
  `applies-when`, `keywords`, `title`, `created`, `updated`, `fired`,
  `fired-helpful`, `sources`) must all be present, even if their values are
  defaults.
- **Sources:** always link back to the daily log(s) that contributed to an
  article.

---

## 9. Migration from legacy claude-memory-compiler entries

Existing claude-memory-compiler entries lack `id`, `type`, `scope`, `status`,
`confidence`, `applies-when`, `keywords`, `fired`, `fired-helpful`. When the
daemon first reads a legacy entry it treats it as:

- `id` ← filename stem.
- `type` ← `lesson`.
- `scope` ← `global` (legacy entries are not project-scoped).
- `status` ← `confirmed` (we trust what was already there).
- `confidence` ← `0.80` (default for inherited content).
- `applies-when` ← empty — these entries cannot trigger interrupts until the
  Scholar backfills the field on its next touch.
- `keywords` ← derived from `tags + aliases + title` (Scholar fills properly
  on next touch).
- `fired` / `fired-helpful` ← `0`.

The Scholar backfills missing fields the first time it updates a legacy
entry; no bulk migration runs. This avoids spending Opus turns on entries
that may never be touched again.

Likewise the legacy wikilink form `[[concepts/foo]]` is recognised by the
lint compatibility shim (`utils.wiki_article_exists` accepts both forms), but
every new Scholar write must emit the new form `[[global/concepts/foo]]`.

---

## 10. Lifecycle

```
candidate → provisional → confirmed → stale → archived
            (Scholar       (user promoted        (auto, after
             wrote)          OR fired N times    M months no
                             with                activity)
                             fired-helpful/fired
                             ≥ 0.66)
```

- **provisional** — Scholar wrote it; not yet used for interrupts; user has
  not endorsed.
- **confirmed** — eligible for interrupts. Reaches this state via
  `agent-mem review` or auto-promotion (`fired ≥ 3` and
  `fired-helpful / fired ≥ 0.66`).
- **stale** — no activity for M months; surfaced in `agent-mem doctor` for
  user attention.
- **archived** — moved to `_archive/`, removed from indexes, never surfaced
  again.

Only the Scholar writes `provisional`. Only the user (via `agent-mem review`)
or the auto-promotion rule (computed by the daemon) flips to `confirmed`.
Demotion to `provisional` happens automatically when the user rejects N
nudges from the same lesson (PLAN §5).

---

## 11. State tracking

`~/.agent-mem/state.json` and `~/.agent-mem/state/state.json` track:

- `ingested` — map of daily log filenames to SHA-256 hashes, compilation
  timestamps, and costs.
- `query_count` — total queries run.
- `last_lint` — timestamp of most recent lint.
- `total_cost` — cumulative API cost.

Plus daemon-managed state (queue sizes, last Scholar run, etc.) once Phase 2
lands.

---

## 12. Scaling beyond index-guided retrieval

At ~2,000+ articles / ~2M+ tokens, the index becomes too large for the
context window. At that point, hybrid RRF (BM25 + sentence-transformer) is
the planned next step (PLAN §3, §8 Phase 4). The BM25 half is already in
place once `agent-mem search --bm25` lands in Phase 1.
