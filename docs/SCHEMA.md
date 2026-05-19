# SCHEMA.md — moved

The on-disk schema for the `agent-mem` knowledge store now lives in
[`../src/AGENTS.md`](../src/AGENTS.md) — the single source of truth.

That file is loaded verbatim by `src/scripts/compile.py` as the compiler
specification passed to the Claude Agent SDK, so keeping the schema there
(rather than duplicated in two places) is what guarantees the running system
and the documented schema do not drift.

Anything that used to live here — directory layout, frontmatter field
reference, invariants, project-slug recipe, fully-worked example, log/index
formats, legacy-entry migration policy — is in `src/AGENTS.md`. Section
numbers from the previous version of this file map as follows:

| Old `docs/SCHEMA.md` section | New `src/AGENTS.md` section |
|---|---|
| §1 Directory layout | §1 |
| §1.1 Project slug | §1.1 |
| §1.2 Archive policy | §1.2 |
| §2 Frontmatter — full field reference | §2 |
| §2.1 Field semantics | §2.1 |
| §2.2 Why each new field exists | §2.2 |
| §2.3 Invariants | §2.3 |
| §3 `knowledge/index.md` format | §3 |
| §4 `knowledge/log.md` format | §4 |
| §5 Fully-worked example article | §6 |
| §6 Migration from claude-memory-compiler entries | §9 |

References elsewhere in the codebase (PLAN.md, role prompts) that still cite
`docs/SCHEMA.md` should be read as citing this redirect — the content they
point at is in `src/AGENTS.md`.
