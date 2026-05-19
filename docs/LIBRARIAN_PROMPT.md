# Librarian Prompt

The Librarian is the **active organiser** in the two-tier curator. It runs on every `Stop` event via the Claude Agent SDK, Haiku-tier. Given the conversation buffer and a snapshot of the library's current state, it **proposes** structural actions (write, update, merge, move, archive, update README, add wikilink, split folder) that will keep the library well-organised at every step. It writes nothing to disk.

The Scholar (a more capable model, Opus-tier) is the gatekeeper and the only writer. For each proposed action the Librarian emits, the Scholar will either APPROVE (and execute via Write/Edit) or VETO (and drop with a one-sentence reason). **There are no fix-ups** — the Scholar cannot silently rewrite a bad proposal. That forces the Librarian to be careful.

This document specifies:
1. How the daemon assembles the prompt inputs.
2. The action vocabulary the Librarian may propose.
3. The exact prompt text (template, with placeholder anchors).
4. The output schema.
5. Operational notes for the daemon implementer.

---

## 1. Inputs and assembly

The daemon pulls everything the Librarian sees into a single prompt:

| Input | Source | Pre-computed by |
|---|---|---|
| Rolling buffer | last ~20 turns of the active session | daemon (already in memory) |
| Library snapshot | knowledge dir tree + READMEs + index.md excerpt | daemon, on every call |
| `applies-when` table | derived from frontmatter of every `confirmed` entry | daemon, on every call |
| BM25 dedup near-hits | candidate seed phrases (regex pre-pass) → BM25 over the existing tree | daemon |
| Active project slug | from the hook | daemon |

The library snapshot is hard-capped at ~3 KB so a sprawling corpus doesn't push the Librarian's prompt past Haiku's sensible context size. Truncation is signalled inline.

### 1.1 The Librarian has read-only tools

`allowed_tools=["Read", "Glob"]`, `max_turns=10`, `permission_mode` default (no edits possible).

The snapshot is a teaser. When the Librarian wants to verify that an existing entry actually covers a candidate lesson — instead of trusting BM25 — it can call `Read("global/tooling/uv-basics.md")` to fetch the full entry. Same for `Glob` to discover what's in a subfolder. The Librarian's cwd is the knowledge directory so paths are clean.

Recommended budget: at most ~5 tool calls per run. The Librarian is Haiku-tier; we don't want it spinning.

### 1.2 User-asserted turns (`/ultan`)

The `/ultan` slash command appends a synthetic UserPromptSubmit event with `payload.user_asserted = true` to the daemon's `events.jsonl`. The Librarian sees those turns with a `[USER-ASSERTED]` prefix in the rolling buffer block and is told to file them by default — the user has explicitly named the rule.

### 1.3 BM25 pre-computation

Same as before: a regex pre-pass extracts candidate "seed phrases" (always/never/should/the fix is/etc.) from the buffer, runs BM25 over the existing knowledge dir for each, and attaches the top 3 hits. The Librarian uses these as a starting signal:
- Seed with strong hit (score > 8) → probably `update_entry` or `merge_entries` (but Read the target first to confirm).
- Seed with no hits → probably `write_entry`.
- The Librarian is free to propose actions on content the regex missed; recall is the daemon's job, judgement is the Librarian's.

---

## 2. Action vocabulary

These are the only legal `action` values. Each one carries action-specific fields plus a required `reasoning` field.

| Action | Fields | Use when |
|---|---|---|
| `write_entry` | `path`, `body`, `reasoning` | New lesson, no existing entry covers it. |
| `update_entry` | `path`, `new_body`, `reasoning` | Existing entry covers it; replace contents. |
| `merge_entries` | `source_paths` (list), `target_path`, `target_body`, `reasoning` | Multiple existing entries are the same lesson; consolidate. Sources go to `_archive/`. |
| `move_entry` | `from_path`, `to_path`, `reasoning` | Entry belongs in a different folder. |
| `archive_entry` | `path`, `reasoning` | Stale or redundant entry. |
| `update_readme` | `folder_path`, `new_body`, `reasoning` | Folder's README is out of date wrt its contents. |
| `add_wikilink` | `from_path`, `to_path`, `context`, `reasoning` | Two entries are related but not linked. |
| `split_folder` | `folder_path`, `into` (dict subfolder→list of entry paths), `reasoning` | Folder has >5 entries; restructure into subfolders. |

Paths are RELATIVE to `knowledge/`. The hierarchy is dynamic — categories are not hard-coded. Pick names based on content (`tooling`, `architecture`, `security`, `conventions`, `testing`, etc. — invent new ones when they fit).

Top-level dirs are `global/` (cross-project) and `projects/<slug>/` (per-repo).

---

## 3. Hierarchy invariants

The Librarian must keep these true after its proposals are applied. The Scholar VETOES actions that would violate them, so the Librarian should accompany any borderline action with the compensating actions needed:

1. **Every directory has a README.md.** Propose `update_readme` for any folder you create or rearrange.
2. **No flat directory exceeds 5 entry .md files** (excluding README). If a `write_entry` would push a folder to 6, propose a `split_folder` in the same response.
3. **Every wikilink resolves.** Don't propose `add_wikilink` to a path that doesn't exist (or won't exist after your proposals).
4. **Every entry's frontmatter validates.** Required fields: `id`, `type`, `scope`, `status`, `confidence`, `applies-when`, `keywords`, `title`, `created`, `updated`, `fired`, `fired-helpful`, `sources`. See `src/AGENTS.md §2`.

After the Scholar finishes executing, the daemon runs a deterministic invariants check (`scholar_prompt.check_invariants`) and logs a WARNING for each violation that slipped through. The Librarian's prompt should keep violations at zero in steady state.

---

## 4. Operational notes (for the daemon implementer)

- The Librarian's response must be a single JSON object matching `LibrarianProposal` (see `daemon/agent_mem_daemon/_schemas.py`). The daemon validates with Pydantic; a malformed response produces an empty packet (`{proposals: [], interrupts: []}`) and is logged.
- The Librarian's response happens AFTER its tool calls. Its FINAL text message must be the JSON.
- If the rolling buffer has no quotable text (e.g. Stop fired with no new turns), the daemon skips the LLM call entirely.
- The `interrupts` part of the response is unchanged from the previous architecture — Librarian surfaces matches against the applies-when table; Scholar approves/vetoes nudges; approved nudges land in `~/.agent-mem/pending-nudges.md`.
- The daemon never gives the Librarian Write or Edit tools. The Scholar is the only writer.
- One Stop = one Librarian run. No retries. Dropped proposals are gone — the same lesson will resurface in a future session.

---

## 5. Why this design

The previous design split labour as "Librarian extracts evidence; Scholar picks the action." That worked but had a failure mode: the Scholar inherited a noisy queue and ended up doing organisation that the Librarian could have done cheaper. The library also drifted into "huge pile of books" shape because no role was responsible for proactive structural maintenance.

The new design pushes organisation into Haiku where it belongs. The Librarian sees the whole library state on every call and is told its job is to keep it well-organised at every step. The Scholar becomes a precision filter: it reads the proposed action, verifies the claim, and either ratifies it or drops it. The "no fix-ups" rule prevents the Scholar from quietly absorbing Librarian sloppiness — Librarian quality becomes visible in the veto rate.
