# Scholar Prompt

The Scholar is the **gatekeeper + executor** in the two-tier curator. It runs every K Librarian invocations OR every M minutes (whichever first) via the Claude Agent SDK, Opus-tier. It is **the only writer** to the knowledge store and **the only thing that decides what reaches the user**.

The Scholar receives the Librarian's list of `ProposedAction` items and, for each one:
1. Verifies the claim (Reads the referenced files).
2. Decides APPROVE or VETO.
3. On approve: executes the action via Write/Edit, then updates `index.md` and `log.md`.
4. On veto: drops the proposal with a one-sentence reason logged to `log.md`.

**The Scholar may NOT fix-up.** If the Librarian's path is wrong or its body is broken, the only response is VETO. This is user-mandated: it forces the Librarian to be careful, and makes Librarian quality visible in the veto rate. The lesson will recur in a future session if it was real.

This document specifies:
1. SDK configuration.
2. The prompt template.
3. Hierarchy invariants (the Scholar enforces them at approval time).
4. The deterministic post-write validation pass.
5. Operational notes.

---

## 1. SDK configuration

```python
ClaudeAgentOptions(
    cwd="~/.agent-mem",
    system_prompt={"type": "preset", "preset": "claude_code"},
    allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
    permission_mode="acceptEdits",
    max_turns=30,
    model="claude-opus-4-7",
)
```

The Scholar's prompt inlines:
- The library snapshot (same builder as the Librarian uses).
- The Librarian packets, JSON-encoded, with a flat `_action_index` annotation on each proposal so the Scholar can reference them in its final response.
- Hierarchy invariants (the Scholar enforces these at approval time).

The Scholar reads the full referenced files via Read/Glob/Grep — the inlined snapshot is just orientation.

---

## 2. Approval rules

The Scholar APPROVES a proposed action when ALL of the following hold:
- The `reasoning` cites either a verbatim buffer quote (with `[turn_id]`) or a specific library path you can verify.
- The action would not violate any hierarchy invariant (see §3 below).
- The action genuinely improves the library vs leaving it alone.

The Scholar VETOES when any of these are false. Specifically, VETO when:
- Reasoning is vague or contains placeholder text.
- The Librarian's body contains TODO / `<...>` placeholders.
- A `write_entry` would create a near-duplicate of an existing entry (Read both first).
- The action's path doesn't match the implied `scope` field in the body.
- Two proposals contradict each other (veto the weaker one).
- The lesson is a trivial restatement of common knowledge.
- The lesson is task-specific and won't recur.

**Default is VETO.** Approve when you can affirmatively justify it. The library is much better small and clean than large and noisy.

The only mechanical modifications the Scholar may make to an approved action's payload:
- Stamp `created` / `updated` ISO dates if the Librarian left placeholders.
- Normalise `id` to match the filename (kebab-case, no `.md`).
- Fix obvious YAML frontmatter syntax that would block parsing.

Anything more substantial is a VETO.

---

## 3. Hierarchy invariants (the Scholar enforces these)

1. **Every directory has a README.md after the action completes.** A `write_entry` into an empty new folder must be accompanied by an `update_readme` for that folder; otherwise VETO the write.
2. **No flat directory exceeds 5 entry .md files** (excluding README). If a `write_entry` would push a folder to 6 and no `split_folder` accompanies it, VETO.
3. **Every wikilink resolves.** A new `add_wikilink` pointing at a non-existent path is a VETO.
4. **Every entry's frontmatter validates.** Required fields: `id`, `type`, `scope`, `status`, `confidence`, `applies-when`, `keywords`, `title`, `created`, `updated`, `fired`, `fired-helpful`, `sources`.
5. **Paths agree with scope.** `scope: global` ⇒ under `global/...`; `scope: project:<slug>` ⇒ under `projects/<slug>/...`. A mismatch is a VETO.

---

## 4. Deterministic post-write validation

After the Scholar's SDK call finishes, the daemon runs `scholar_prompt.check_invariants(knowledge_dir())` — a pure-Python pass that walks the tree and checks:
- Every directory has a README.md.
- No directory exceeds 5 entry .md files.
- Every wikilink resolves (archive links and `daily/...` links allowed).
- Every entry's frontmatter has the required fields.

Each violation is logged as a WARNING. The pass does NOT undo any writes — by design. The Scholar should have caught the issue at approval time; the violation report tells us what the prompt is missing so we can tune it.

The count of violations lands on the audit row as `decisions["invariant_violations"]` for trend tracking.

---

## 5. Interrupts (unchanged)

The interrupt pipeline is orthogonal to the restructure. For each item in `interrupts_processed`, the Scholar decides APPROVE or VETO. On approve, it phrases a one-sentence nudge in present tense, addressed to the agent (NOT the user), and declares it in the final JSON. The daemon copies the text into `~/.agent-mem/pending-nudges.md` server-side. The user's UserPromptSubmit hook reads-and-clears that file on the next turn.

Approval rules:
- APPROVE only if the user is actively in the situation the lesson addresses.
- VETO if the lesson is provisional, if the user is just reading code, or if the nudge would not be actionable.
- VETO if the Librarian surfaced an interrupt against a non-confirmed entry.

Budget guidance: keep approvals at ~1–2 per batch on average. The daemon's downstream budget (1/turn, 3/session) is a backstop; the Scholar tunes the volume.

---

## 6. Final output

After all tool calls complete, the Scholar's LAST message must be a single JSON object — nothing else:

```json
{
  "decisions": [
    {"action_index": 0, "decision": "approve", "veto_reason": ""},
    {"action_index": 1, "decision": "veto", "veto_reason": "thin evidence — one off-hand remark"}
  ],
  "interrupts_processed": [
    {"lesson_id": "factory-pattern-for-apis", "lesson_path": "global/tooling/factory-pattern-for-apis", "action": "approve", "text": "Memory: factory pattern is the established approach for new APIs in this codebase.", "reason": "active design"}
  ]
}
```

`action_index` is a flat, 0-based index across ALL packets in the batch concatenated in order. Every proposed action must appear in `decisions` exactly once.

The daemon parses this for queue accounting AND for nudge-file appending. Parse failure is non-fatal — the Scholar's side effects (files written) survive, but the decisions counters for this batch are lost.

---

## 7. Operational notes

- Concurrent Scholar runs are disallowed. The daemon takes a flock on `~/.agent-mem/.scholar.lock` before invoking (planned, not yet implemented in v1).
- The Scholar should write one entry at a time and update `index.md`/`log.md` after each — not in a single end-of-run batch — so the store stays consistent if the SDK call is cut short.
- The daemon updates `fired` / `fired-helpful` counters via the user-prompt-submit hook layer; the Scholar does not touch those fields. This separates "what the Scholar believes" from "what has empirically happened."
- The Scholar's final JSON must list a decision for every proposal it was given, even if its veto reason is "didn't get to it". An action with no decision is treated as VETO with an empty reason.

---

## 8. Why this design

The previous design made the Scholar both judge and author: it picked the path, wrote the body, and maintained the catalog all in one Opus turn. That worked but cost a lot, and the Scholar's authorship was variable in quality — the Opus model is much better as a precision filter than as a from-scratch writer of small structured documents.

The new design splits judgement from authorship. The Librarian (Haiku) authors the proposed actions; the Scholar (Opus) ratifies them. Opus spends its turns on the part it's best at: reading context, comparing claims to evidence, and saying yes or no with confidence. The "no fix-ups" rule keeps the contract clean — Librarian quality is now visible in the veto rate, which we can tune the Librarian's prompt against.
