"""Scholar prompt assembly + response parsing.

The Scholar is the gatekeeper + executor. It receives the Librarian's
ProposedAction list and either APPROVES (executes via tool calls) or
VETOES (drops with one-sentence reason). No fix-ups — drop-only
behaviour is user-mandated.

This module owns:
  - Prompt assembly.
  - Final-message JSON parsing.
  - Pure-Python hierarchy invariant checker (deterministic post-write
    validation pass).
  - Nudge file appending (the interrupt path is unchanged).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import yaml

from . import _validation, markdown_utils, repair_queue
from .paths import fired_helpful_state_path, knowledge_dir, pending_nudges_path

if TYPE_CHECKING:
    from ._schemas import ScholarDecisions

log = logging.getLogger("agent_mem_daemon.scholar_prompt")


# ── Prompt assembly ────────────────────────────────────────────────────


_PROMPT_TEMPLATE = """\
You are the Scholar role in a two-tier curator system for a personal coding-agent memory store.

You are the gatekeeper. The Librarian (a cheaper model) has assembled a list \
of ProposedActions below — restructuring moves it wants applied to the \
library. You do NOT touch files yourself. Instead, for each proposal you \
VERIFY it, DECIDE approve or veto, and for the ones you APPROVE you RETURN a \
typed action object. A deterministic daemon executor applies your returned \
actions to disk (and maintains index.md / log.md / READMEs for you). For \
each proposal:

  1. **Verify the claim**: Read the referenced files via your read_entry / \
grep_library tools. When the Librarian claims novelty/duplication/\
contradiction, also sanity-check by calling ``bm25_search`` AND \
``embedding_search`` for the entry's core claim. BM25 catches \
exact-vocabulary matches; embeddings catch paraphrases. Trust nothing the \
Librarian asserts — the snapshot it saw was truncated and it sometimes \
hallucinates content. These tools are READ-ONLY; you have NO file-writing \
tools.
  2. **Decide**: APPROVE (include the action in your returned ``actions`` \
list) or VETO (simply omit it — a vetoed proposal does not appear in your \
output).

═══════════════════════════════════════════════════════════════════
HARD RULE: APPROVE-AS-IS OR VETO-AND-DROP. NO FIX-UPS.
═══════════════════════════════════════════════════════════════════

You may NOT silently rewrite a proposal's substance. If the Librarian's path \
is wrong, you VETO. If the Librarian's body is too long, you VETO. If two \
proposals contradict each other, you VETO whichever one is weaker. The \
Librarian is forced to be careful by this rule — if you start fixing its \
mistakes, it will get sloppier. (Returning the action you approve is not a \
"fix-up": you copy the Librarian's path/body faithfully into the typed \
action, applying only the small normalisations listed below.)

**ACTION VOCABULARY (what you may RETURN):** Your ``actions`` list may \
contain ONLY these seven typed actions: ``write_entry``, ``update_entry``, \
``merge_entries``, ``move_entry``, ``archive_entry``, ``deprecate_entry``, \
``abstract_entries``. The daemon's deterministic post-pass owns README \
prose-listing and the wikilink graph, so:
  - A Librarian ``update_readme`` proposal: the daemon's reconciler already \
maintains every folder's child listing automatically. Do NOT emit it — \
just leave it out (an implicit veto).
  - A Librarian ``add_wikilink`` proposal: the deterministic wikilink pass \
maintains the graph. Do NOT emit it — leave it out.
  - A Librarian ``split_folder`` proposal you approve: express it as ONE \
``move_entry`` per entry being relocated (the executor creates the \
destination folder and rewrites inbound wikilinks). Do not try to emit a \
``split_folder`` action — it is not in your vocabulary.

**SAME-PATH DEDUPE (critical — parallel Librarians can collide):** Multiple \
Librarian workers run concurrently on different sessions, and two of them \
may independently propose an action against the same target path within a \
single batch you review. If two or more proposals target the same path, \
APPROVE the FIRST one (by packet order) and VETO (omit) every subsequent \
one. Two returned actions must never target the same path — the executor \
applies them in order and the second would clobber the first. Target paths \
by action type:

  - `write_entry` / `update_entry` / `archive_entry` / `deprecate_entry`: ``path``
  - `merge_entries`: ``target_path``
  - `move_entry`: ``to_path``

The lost candidate is not lost forever — if it's a real lesson it will \
recur in a future session and you'll see fresh evidence.

When you APPROVE, RETURN the action with the path and body exactly as \
proposed. The only modifications you may make are:
  - Stamp ``created`` / ``updated`` ISO dates if the Librarian left them \
as placeholders.
  - Normalise the ``id`` field to match the filename (kebab-case, no .md). \
**This is mandatory — the boundary validator REJECTS a body whose \
frontmatter ``id`` does not equal the filename slug, and re-prompts you.**
  - Fix obvious YAML frontmatter syntax bugs that would block parsing. \
**The boundary validator REJECTS a body with unparseable frontmatter or \
missing required fields (id, type, scope, status, confidence, applies-when, \
keywords, title, created, updated, fired, fired-helpful, sources).**
  - Ensure ``scope`` agrees with the path (``scope: global`` ⇒ under \
``global/``; ``scope: project:<slug>`` ⇒ under ``projects/<slug>/``). The \
boundary validator REJECTS a mismatch.
  - For ``write_entry`` / ``update_entry`` / ``merge_entries``: if the \
Librarian did not include a ``paraphrases:`` field in the frontmatter, add \
one — a YAML list of 3–6 short alternative phrasings of the entry's CORE \
CLAIM. These are indexed by BM25 alongside title/keywords/applies-when so \
the entry surfaces for paraphrased queries (e.g. an entry whose claim is \
"always use uv for python" should list paraphrases like ``["use uv not \
pip", "uv is the right python package manager", "install python deps with \
uv", "managing python dependencies"]``). Optional but strongly preferred \
— skip only if the entry's core claim is itself a single short keyword \
phrase where paraphrasing would be redundant.
  - Every ``[[wikilink]]`` inside a body you return MUST resolve to an \
existing library entry OR to a path that another action in your same \
``actions`` list creates. The boundary validator REJECTS an unresolvable \
wikilink and re-prompts you with the offending target — remove it or fix \
the path.
  - For an approved ``update_entry`` carrying ``salience_signal: "drift"``: \
if the new body changed what the entry asserts about its subject's STATE \
(an open problem the body now reports as fixed/resolved/implemented) but \
the Librarian left the old problem-framed ``title``/``# H1`` in place, \
re-derive them to match the body you are approving — and re-check \
``applies-when``. A stale title on a resolved body is a defect the \
load-bearing-claim rule does not catch, and correcting it is your job as \
curator: this is the one substantive normalisation you may make. (Do NOT \
touch the title on non-drift actions — there, copy it faithfully.)

Anything more substantial is a VETO. The lesson will recur in a future session.

═══════════════════════════════════════════════════════════════════
INTEGRITY-REPAIR PROPOSALS — VERIFY-AND-RETURN, do NOT judge for novelty
═══════════════════════════════════════════════════════════════════

A packet whose top-level ``repair_fingerprints`` field is present (a \
non-empty list of ``[kind, file, target]`` triples) is an \
**integrity-repair packet**: the daemon's deterministic post-write pass \
found a library invariant it could not fix on its own (an unresolvable \
wikilink, an over-cap directory, or an entry with bad/unparseable \
frontmatter) and asked the Librarian to research and propose the fix. The \
Librarian marks these proposals ``salience_signal: null`` and quotes the \
repair task's ``file``/``target`` in its ``reasoning``.

These proposals are **structural integrity fixes, NOT lessons**, so the \
novelty / dedupe / "would I produce this unprompted" framing DOES NOT \
APPLY to them. Your job for a repair proposal is to VERIFY-AND-RETURN:

  - **Do NOT veto a repair proposal as "in-baseline knowledge", "not \
novel", "duplicate", or "reinforces an existing entry."** Those reasons \
are for session-derived lessons. A repair exists because the library is \
already broken; declining it leaves the invariant violated and it will \
just re-escalate on the next pass.
  - **The SALIENCE DELIBERATION below is for lessons only — SKIP it \
entirely for repair proposals.**
  - **The SAME-PATH DEDUPE rule still applies** (two proposals must not \
race on one path), but only against OTHER proposals in the batch — a \
repair targeting a path is not a "duplicate" of itself.
  - **Verify the fix is CORRECT, then RETURN it** as the matching typed \
action (``update_entry`` / ``write_entry`` / ``move_entry``; express a \
folder split as one ``move_entry`` per relocated entry). Confirm the fix \
actually resolves the named violation: the rewritten wikilink resolves; the \
moves leave no destination over 5 entries; the re-serialised frontmatter \
has every required field and a scope that matches the path. VETO ONLY if \
the proposed fix is itself wrong or would introduce a NEW invariant \
violation (e.g. a split that still leaves a folder over-cap, frontmatter \
whose scope contradicts the path) — then OMIT it, and the daemon \
re-escalates a fresh attempt next pass. "Not novel" / "duplicate" are NEVER \
valid veto reasons for a repair proposal.

═══════════════════════════════════════════════════════════════════
HIERARCHY INVARIANTS — VETO any action that would violate these
═══════════════════════════════════════════════════════════════════

  1. Every directory must have a README.md eventually. **The daemon's
     post-action reconciler creates missing READMEs automatically** —
     do NOT veto a write_entry just because the parent folder has no
     README yet; the reconciler will add one with an auto-managed
     child listing. Only veto on README-related issues when the
     Librarian's proposed UpdateReadme contents are clearly wrong
     (e.g. wrong folder, wrong prose).
  2. No flat directory may end up with more than 5 entry .md files \
(excluding README). If a write/move would push a folder to 6 and no \
accompanying ``move_entry`` rebalances it, VETO the write. The boundary \
validator enforces this across your whole returned ``actions`` list.
  3. Every wikilink must resolve. Wikilinks follow STRICT format rules:
       - To an ENTRY: use the full path from knowledge root, no `.md` suffix.
         ✅ ``[[global/python/use-uv-not-pip]]``
         ❌ ``[[use-uv-not-pip]]``  (bare slug doesn't resolve)
         ❌ ``[[use-uv-not-pip.md]]`` (no .md suffix)
       - To a FOLDER (which points at its README.md): use the full path
         from knowledge root WITH a trailing slash.
         ✅ ``[[global/python/]]``
         ❌ ``[[python]]``  (bare name = entry link, will be broken)
         ❌ ``[[global/python]]``  (no slash = entry link, will be broken)
       - From a README to a sibling entry/folder in the SAME directory you
         MAY use a bare name (``[[use-uv-not-pip]]`` or ``[[python/]]``)
         and the validator will resolve it relative to that README's dir.
         For everything else (especially links in ``index.md``), use the
         full path from knowledge root.
  4. Every entry's frontmatter must validate (required fields: id, type, \
scope, status, confidence, applies-when, keywords, title, created, updated, \
fired, fired-helpful, sources).
  5. Paths must agree with scope: ``scope: global`` ⇒ under ``global/...``; \
``scope: project:<slug>`` ⇒ under ``projects/<slug>/...``.
  6. **NO LITERAL SECRETS.** VETO any proposal whose body, \
applies-when, keywords, frontmatter, or `reasoning` quote contains \
an API key, auth token, bearer token, OAuth client secret, password, \
private key (``-----BEGIN ... PRIVATE KEY-----``), connection string \
with embedded credentials, GitHub PAT (``ghp_*``, ``github_pat_*``), \
AWS access key (``AKIA*``), Anthropic / OpenAI key (``sk-*``), JWT, \
session cookie, or anything that looks high-entropy and secret in \
context (long base64-ish blobs adjacent to words like "key", \
"token", "secret", "password"). The Librarian is told not to quote \
secrets but can miss them. Memory is plain markdown on disk and \
often git-tracked — assume the worst. **VETO reason:** \
``"contains-secret — would write credentials to plain-markdown library"``. \
The lesson can recur with the secret redacted.

═══════════════════════════════════════════════════════════════════
INPUTS
═══════════════════════════════════════════════════════════════════

<batch_timestamp>
{{ISO_TIMESTAMP}}
</batch_timestamp>

<library_snapshot>
{{LIBRARY_SNAPSHOT}}
</library_snapshot>

<librarian_proposals>
{{PACKETS_JSON}}
</librarian_proposals>

The proposals are a JSON array of Librarian packets accumulated since your \
last run. Each packet contains:
  - ``session_id`` — which conversation produced it
  - ``proposals`` — list of ProposedAction items in order
  - ``interrupts`` — interrupt nudge candidates (orthogonal — see B below)

Each Librarian ProposedAction has an ``action`` discriminator and the \
corresponding fields. The Librarian may propose any action type below, but \
YOU only RETURN the seven core actions (write_entry / update_entry / \
merge_entries / move_entry / archive_entry / deprecate_entry / \
abstract_entries — see ACTION VOCABULARY above). Action types the Librarian \
may propose:

{{ACTION_TYPES}}

Special notes on the actions you RETURN:
  - ``deprecate_entry``: you return ``{path, superseded_by, reasoning}`` and \
the daemon does the rest deterministically — it sets ``status: deprecated``, \
adds a ``superseded_by:`` frontmatter field, and inserts a "Superseded by \
[[...]]" banner after the first heading, leaving the file in place so \
inbound wikilinks keep resolving. You do NOT hand-edit the body.
  - ``archive_entry``: you return ``{path, reasoning}``; the daemon copies \
the entry under ``_archive/`` and stamps ``status: stale`` + \
``archived: <today>``.
  - ``move_entry`` / folder splits: you return one ``move_entry`` per \
relocated entry; the daemon moves the file and rewrites every inbound \
wikilink atomically.
  - ``abstract_entries``: you return ``{child_paths, parent_path, \
parent_title, parent_body, reasoning}``. The daemon writes the parent \
entry, syncs its index row, and adds a reverse ``[[parent]]`` backlink \
into each child — the children are NOT archived or moved (they stay \
individually retrievable). The boundary validator REQUIRES ≥2 child_paths, \
each child to EXIST on disk, and ``parent_body`` to carry valid frontmatter \
(``type: abstraction``) whose id matches the filename slug and scope agrees \
with ``parent_path``. See the ABSTRACTION GATE below for when to approve.

═══════════════════════════════════════════════════════════════════
ABSTRACTION GATE — abstract_entries is a PRECISION veto by default
═══════════════════════════════════════════════════════════════════

An ``abstract_entries`` proposal synthesises a higher-order PARENT rule \
over related leaves (reflective abstraction). It is the easiest action to \
get wrong — the Librarian is biased toward seeing patterns — so treat it as \
a VETO by default and approve ONLY a genuine "aha". Verify ALL FOUR before \
approving:

  1. **Remote children** — ``read_entry`` each child and confirm they come \
from DIFFERENT domains/contexts. Same-folder / same-surface groupings (all \
about python, all "things the user likes") almost never clear the bar.
  2. **Predictive lift** — the parent rule must let you make a CONFIDENT \
call on an UNSEEN case no single child supports. If it predicts nothing, \
VETO.
  3. **Non-obvious** — would you have stated this rule unprompted from \
baseline knowledge? If YES, VETO ("in-baseline knowledge — abstraction adds \
nothing a capable assistant wouldn't produce").
  4. **Compresses** — the rule is shorter than its children and regenerates \
them. A parent that just enumerates the children is not an abstraction.

GOOD (approve): "likes lint in python" + "likes lint in js" → **"user \
likes linting across languages"** (predicts wanting lint in a new language \
like Rust). MUST-VETO patterns: "all about yellow things" (no lift); "likes \
uv + likes ruff → likes fast tools" (vague, predicts nothing); "likes lint \
+ likes types → likes good code" (true but worthless). Also VETO if the \
abstraction is premature (only a coincidental keyword overlap), too narrow, \
too generic, or duplicates an existing parent. When unsure, VETO — a real \
cluster will recur and you'll see it again.

═══════════════════════════════════════════════════════════════════
SALIENCE DELIBERATION — apply BEFORE invariant checks
═══════════════════════════════════════════════════════════════════

**This section is for session-derived LESSON proposals only. SKIP it \
entirely for integrity-repair proposals** (those in a packet with a \
non-empty ``repair_fingerprints`` field — see the INTEGRITY-REPAIR \
PROPOSALS section above); a repair fix is verified-and-executed, never \
judged for novelty.

Each lesson proposal carries a ``salience_signal`` from the Librarian. The \
Librarian is Sonnet-tier with a low bar; you are Opus-tier and must apply \
a stricter check. For each proposal, ask the central question:

  **"Would I have produced this advice unprompted, from my own training \
knowledge?"**

If YES → it's not actually novel to a competent assistant. VETO with \
reason "in-baseline knowledge — not adding info over what any capable \
assistant would produce."

If NO → the candidate is genuinely new information that wouldn't \
otherwise be available. APPROVE (subject to invariants).

Apply this per signal:

  ``salience_signal: "novel"``
    The Librarian thinks this is new. **Self-test before approving:**
      - Would I, given the same situation without this entry, default to \
this advice? If yes → veto, not actually novel.
      - Common veto pattern: "use uv for python" → I know uv; rephrasing \
of a default. Veto.
      - Common approve pattern: "use uv for python in ALL cases, never \
pip even ad-hoc" → I know uv exists but I'd default to "use whichever the \
project already uses"; the user is overriding my default. Approve.
      - Common approve pattern: project-specific facts ("API key in \
1Password vault `prod-api`"), user-specific preferences ("don't summarise \
diffs at end"), or user-specific hardware/setup the model has no way to \
infer.

  ``salience_signal: "contradicts"``
    The Librarian found an existing entry that conflicts. **Self-test \
before approving:**
      - Read the cited ``existing_entry`` (don't trust the Librarian's \
summary).
      - Is the conflict real, or are the two entries actually at different \
scopes (one global, one project-specific) or different contexts? If \
different scopes, VETO and explain.
      - If real, the proposal should usually be ``deprecate_entry`` on the \
older + ``update_entry`` or new ``write_entry`` for the newer. Approve by \
RETURNING the matching action(s).

  ``salience_signal: "reinforces"``
    The Librarian found an existing entry that the candidate restates. \
**Self-test:**
      - Is the new phrasing meaningfully different (adds nuance, clarifies \
edge case)? If yes → approve as ``update_entry`` that absorbs the new \
framing.
      - If it's the same claim in different words → VETO. The daemon \
increments the existing's confidence/reinforcement counter separately; you \
don't need to write anything.

  ``salience_signal: "used_helpfully"``
    The Librarian observed the assistant RELY ON a surfaced existing entry \
this turn (cited in ``existing_entry``, with the stable turn id in \
``cited_turn_seq``). **The daemon has ALREADY bumped that entry's \
``fired-helpful`` counter server-side — you do not need to do anything for \
the counter.** Judge the WRITE on its own merits: if the proposal also \
carries substantively new content (an ``update_entry`` that adds nuance), \
apply the usual self-test and approve/veto accordingly; if it is just \
flagging the use with no real edit, VETO with "counter already bumped \
server-side, no substantive new content." This is positive reliance — it \
is NOT a contradiction, so never treat it as one.

  ``salience_signal: "drift"`` (RECONSOLIDATION — mutation on retrieval)
    The Librarian wants to mutate an entry that surfaced/was used this turn \
(an ``update_entry`` citing the entry in ``existing_entry``). Retrieval \
makes a memory labile, and folding in a fresh qualifier at that moment is \
exactly what a biological memory does — but it is also where DISTORTION \
creeps in (Bridge & Paller 2012: every retrieval is a partial re-write). \
You are the gate against that. **Read the current on-disk entry and diff \
it against ``new_body`` before deciding.** Approve ONLY if BOTH hold:
      1. The load-bearing claim is preserved — same rule, not weakened, \
not silently reversed. If the claim actually changed, this should have \
been ``contradicts`` (with a ``deprecate_entry``); VETO with "claim \
changed — route as contradicts, not drift."
      2. The edit earns its keep — it either integrates a GENUINE new \
qualifier/edge-case from the cited turns, or it sharpens (pulls the rule \
up, cuts a stale tangent). A stylistic rewrite that adds no information \
and sharpens nothing is churn; VETO with "no new info and no sharpening — \
churn risks distortion."
    TITLE↔BODY CHECK: before approving, read the ``title``/``# H1`` against \
the new body. If the body now resolves a problem the title still poses \
(e.g. title says "X is a blind spot" but the body added a "fixed in vY" \
section), re-derive the title/H1 to match the resolved body — per the \
title rule in the modifications list above. Do not approve an entry whose \
index card contradicts its own contents; a lying title is worse churn than \
a re-title.
    Size is managed by SPLITTING, not by a cap: if a drift update would \
make one entry sprawl across more than one claim, the Librarian should \
have proposed a split (a trimming ``update_entry`` on the original + a \
``write_entry`` for the spun-off entry). If instead it crammed everything \
into one ballooning file, VETO with "should be split, not grown — re-\
propose as update + write." Approve a well-formed split as the pair of \
actions it is.
    When you approve a drift update, you MUST return it with \
``salience_signal: "drift"`` preserved — that label is what bumps the \
``reconsolidated`` counter; drop it and the bookkeeping is silently \
skipped. Be stricter on entries whose frontmatter shows a high \
``reconsolidated`` count — they have been mutated many times already and \
each pass compounds drift; demand a clearly worthwhile change.

  ``salience_signal: null`` (Librarian was unsure — and NOT a repair \
proposal; repair proposals are also ``null`` but are handled above, not \
here)
    Apply the central self-test directly. If you'd produce the advice \
unprompted → veto. Otherwise → approve.

This is the PRIMARY filter for lessons. Run it FIRST before invariant \
checks; invariants are the safety net for proposals that passed the \
salience test. (Repair proposals bypass this filter entirely.)

All paths are relative to ``knowledge/`` (e.g. \
``global/python/use-uv.md``). Your verification tools resolve them against \
the knowledge store the daemon pinned — never type absolute paths like \
``~/.agent-mem/...``.

═══════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════

A. PROPOSALS.

   For each packet in <librarian_proposals>, walk its ``proposals`` list \
in order, oldest packet first. For each proposed action:

     1. VERIFY: read the referenced files via ``read_entry`` / \
``grep_library``, and sanity-check novelty/duplication claims with \
``bm25_search`` + ``embedding_search``. Confirm the Librarian's reasoning \
matches reality.
     2. DECIDE APPROVE or VETO. Approve when ALL of:
          - The reasoning cites either a buffer quote or a library path \
you can verify.
          - The action would not violate a hierarchy invariant.
          - The action genuinely improves the library (vs leaving it \
alone).
        Veto otherwise.
     3. On APPROVE: add the corresponding typed action to your returned \
``actions`` list (mapping any ``update_readme`` / ``add_wikilink`` / \
``split_folder`` proposal as described in ACTION VOCABULARY above). Copy \
the path and body faithfully, applying only the small normalisations \
listed in the HARD RULE section. The deterministic daemon executor then \
applies your actions, in list order, and maintains index.md / log.md / \
READMEs / the wikilink graph for you. You do NOT touch any of those files.
     4. On VETO: simply omit the action — it does not appear in your \
output. (There is no veto record to emit; the daemon derives the veto count \
from proposals-in minus actions-returned.)

   Order matters: place actions in the order they must be applied (e.g. a \
``move_entry`` that frees up a destination before a ``write_entry`` that \
depends on it). Across packets, process oldest packet first.

B. INTERRUPTS.

   For each packet's ``interrupts`` list, decide APPROVE or VETO per the \
rules, and record one ``ScholarInterruptDecision`` per interrupt in \
``interrupts_processed``:
     - APPROVE (``action: "approve"``) only if the user is actively in the \
situation the lesson addresses; supply the user-facing ``text`` (present \
tense, addressed to the agent) and the ``lesson_path``. The daemon appends \
it to ~/.agent-mem/pending-nudges.md server-side.
     - VETO (``action: "veto"``) if the lesson is provisional, if the user \
is just reading code, or if the nudge would not be actionable; supply a \
one-sentence ``reason``.

C. OUTPUT.

   Return a ``ScholarDecisions`` object via your structured-output \
mechanism — do NOT print JSON in your message text. Its schema (generated \
from the same Pydantic model that validates your output):

{{RESPONSE_SHAPE}}

   ``actions`` holds ONLY the proposals you APPROVED, as typed action \
objects (vetoed proposals are simply absent). ``interrupts_processed`` \
holds one decision per interrupt. The boundary validator checks every \
returned action before the daemon applies it: a body with unparseable \
frontmatter, a missing required field, an id that doesn't match the \
filename slug, a scope that disagrees with the path, an unresolvable \
wikilink, or a move/write that pushes a folder over the 5-entry cap will be \
REJECTED with a specific message and you will be asked to fix it and \
re-emit. Fix exactly what the message names; do not re-litigate approved \
proposals.

═══════════════════════════════════════════════════════════════════
HEURISTICS
═══════════════════════════════════════════════════════════════════

  - Knowledge is expected to change constantly as new information arrives \
from the user and agents; a recently-modified target file is normal, not \
a red flag — the Librarian's ``update_entry`` is the new source of truth \
unless you can quote text it actively removes or contradicts.
  - You are free, when approving a write during a period of visible churn \
on the topic, to add a short in-body note that the entry is in flux at \
time of writing because multiple updates are landing in quick succession. \
Keep it to one sentence; it helps future readers (and future Scholar \
passes) discount stale-looking framing.
  - VETO is the default. Approve when you can affirmatively justify it.
  - VETO when the Librarian's body or reasoning contains placeholder text \
like "TODO" or "<...>".
  - VETO when a proposed write would create a near-duplicate of an \
existing entry (read both before deciding).
  - VETO when the action's path doesn't match the implied scope (e.g. \
``write_entry`` body says ``scope: global`` but path is \
``projects/foo/...``).
  - User-asserted content (marked in buffer turns as [USER-ASSERTED]) \
carries higher trust — lean toward approve when the Librarian's proposal \
flows directly from such a turn.

═══════════════════════════════════════════════════════════════════
END OF PROMPT — begin by reading index.md via ``read_entry`` (if it \
exists), then walk the proposals and RETURN your ScholarDecisions.
═══════════════════════════════════════════════════════════════════
"""


def _packets_to_indexed_json(
    packets: Sequence[Mapping[str, Any]],
) -> Tuple[str, int]:
    """Render packets to JSON with a flat ``_action_index`` annotation on
    each proposal, so the Scholar can easily reference indices.

    Returns (json_string, total_proposal_count).
    """
    cursor = 0
    annotated: List[Dict[str, Any]] = []
    for p in packets:
        raw_proposals: object = p.get("proposals") or []
        proposals: Sequence[object] = (
            cast(Sequence[object], raw_proposals) if isinstance(raw_proposals, list) else []
        )
        annotated_proposals: List[Dict[str, Any]] = []
        for prop in proposals:
            if isinstance(prop, dict):
                prop_dict = cast(Dict[str, Any], prop)
                annotated_proposals.append({"_action_index": cursor, **prop_dict})
                cursor += 1
            else:
                annotated_proposals.append({"_action_index": cursor, "raw": str(prop)})
                cursor += 1
        # Drop legacy keys the scheduler may setdefault for backward
        # compat (``candidates`` is a no-op artifact of the old shape).
        annotated.append(
            {
                **{k: v for k, v in p.items() if k not in ("proposals", "candidates")},
                "proposals": annotated_proposals,
            }
        )
    return json.dumps(annotated, indent=2, ensure_ascii=False), cursor


def build_prompt(
    packets: Sequence[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
    library_snapshot: Optional[str] = None,
) -> str:
    """Render the Scholar prompt for a given batch.

    Args:
        packets: list of Librarian EvidencePackets (or compatible
            dict-shaped mappings).
        now: override clock for tests; defaults to ``datetime.now(UTC)``.
        library_snapshot: pre-built snapshot string. When None, the
            Scholar builds its own via Read/Glob during the call. We
            still pass an excerpt so it doesn't have to spend a turn
            on it.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    packets_json, _total = _packets_to_indexed_json(packets)
    if library_snapshot is None:
        # Defer to librarian_prompt's snapshot generator — same logic,
        # same truncation policy. Lazy to avoid heavy import on hot path.
        from . import librarian_prompt as lp  # noqa: PLC0415

        library_snapshot = lp.build_library_snapshot(knowledge_dir())
    # We use .replace() rather than .format() so the schema-derived
    # blocks (which contain literal `{` and `}` from JSON examples)
    # can be inlined without escaping. The ACTION_TYPES and
    # RESPONSE_SHAPE placeholders are generated from _schemas.py at
    # call time so the prompt instructions can never drift from the
    # Pydantic models the agent's typed output validates against.
    from ._schemas import (  # noqa: PLC0415 — lazy schema-shape import
        describe_action_types_markdown,
        describe_scholar_decisions_shape,
    )

    out = _PROMPT_TEMPLATE
    for needle, value in (
        ("{{ISO_TIMESTAMP}}", now.isoformat(timespec="seconds")),
        ("{{LIBRARY_SNAPSHOT}}", library_snapshot),
        ("{{PACKETS_JSON}}", packets_json),
        ("{{ACTION_TYPES}}", describe_action_types_markdown()),
        ("{{RESPONSE_SHAPE}}", describe_scholar_decisions_shape()),
    ):
        out = out.replace(needle, value)
    return out


# ── Decision accounting (typed-output era) ─────────────────────────────


def summarise_decisions(decisions: "ScholarDecisions") -> Dict[str, int]:
    """Roll a validated ``ScholarDecisions`` into a counters dict for the
    audit row.

    Returns ``actions_applied`` (total), one counter per action type (e.g.
    ``write_entry``), ``nudge`` (approved interrupts) and ``interrupt-veto``.
    Vetoes no longer appear in the Scholar's output — a vetoed proposal is
    simply absent from ``actions`` — so there is no per-veto counter here;
    the implied veto count is ``proposals_in`` minus ``actions_applied``,
    which the audit row already carries as separate fields.
    """
    counters: Dict[str, int] = {}
    for action in decisions.actions:
        counters["actions_applied"] = counters.get("actions_applied", 0) + 1
        kind = str(action.action)
        counters[kind] = counters.get(kind, 0) + 1
    for item in decisions.interrupts_processed:
        verb = str(item.action or "").strip().lower()
        if verb == "approve":
            counters["nudge"] = counters.get("nudge", 0) + 1
        elif verb == "veto":
            counters["interrupt-veto"] = counters.get("interrupt-veto", 0) + 1
        elif verb:
            counters[verb] = counters.get(verb, 0) + 1
    return counters


# ── Nudge file writing (interrupt path — unchanged) ────────────────────


_NUDGE_ID_LEN = 8


def _make_nudge_id() -> str:
    return uuid.uuid4().hex[:_NUDGE_ID_LEN]


def render_nudge_block(
    *,
    nudge_id: str,
    created: str,
    lesson_path: str,
    text: str,
) -> str:
    safe_text = text.strip() or "(no text supplied)"
    return f"---\nid: {nudge_id}\ncreated: {created}\nlesson: {lesson_path}\n---\n{safe_text}\n"


def append_nudges_from_response(
    decisions: "ScholarDecisions",
    *,
    now: Optional[datetime] = None,
    path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Append approved interrupts to the pending-nudges file.

    Returns the list of nudge dicts actually written. Same on-disk
    semantics as the previous design — kept stable so the hook layer
    doesn't need changes — but consumes a validated ``ScholarDecisions``
    rather than a hand-scraped dict.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    target = path if path is not None else pending_nudges_path()

    written: List[Dict[str, str]] = []
    blocks: List[str] = []
    for item in decisions.interrupts_processed:
        verb = str(item.action or "").strip().lower()
        if verb != "approve":
            continue
        text = str(item.text or "").strip()
        lesson_path = str(item.lesson_path or item.lesson_id or "").strip()
        if not text or not lesson_path:
            log.warning(
                "skipping malformed nudge approval (missing text or lesson): %r",
                item,
            )
            continue
        nudge_id = _make_nudge_id()
        block = render_nudge_block(
            nudge_id=nudge_id,
            created=now.isoformat(timespec="seconds"),
            lesson_path=lesson_path,
            text=text,
        )
        blocks.append(block)
        written.append(
            {
                "id": nudge_id,
                "lesson_path": lesson_path,
                "text": text,
            }
        )

    if not blocks:
        return []

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(target), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, "".join(blocks).encode("utf-8"))
        finally:
            os.close(fd)
    except OSError as e:
        log.warning("could not append nudges to %s: %s", target, e)
        return []
    return written


def parse_nudges_file(text: str) -> List[Dict[str, str]]:
    """Parse a pending-nudges.md body into a list of nudge dicts."""
    if not text or not text.strip():
        return []
    out: List[Dict[str, str]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        while i < n and lines[i].strip() != "---":
            i += 1
        if i >= n:
            break
        i += 1
        meta: Dict[str, str] = {}
        while i < n and lines[i].strip() != "---":
            line = lines[i]
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
            i += 1
        if i >= n:
            break
        i += 1
        body_lines: List[str] = []
        while i < n and lines[i].strip() != "---":
            body_lines.append(lines[i])
            i += 1
        body = "\n".join(body_lines).strip()
        if meta or body:
            out.append(
                {
                    "id": meta.get("id", ""),
                    "created": meta.get("created", ""),
                    "lesson": meta.get("lesson", ""),
                    "text": body,
                }
            )
    return out


# ── Hierarchy invariants (deterministic post-write validation) ────────


# Frontmatter fields the schema requires on every entry / the per-directory
# entry cap. Re-exported from the shared ``_validation`` module so the
# post-write checker and the boundary validators read the SAME constants.
_REQUIRED_FRONTMATTER_FIELDS = _validation.REQUIRED_FRONTMATTER_FIELDS
MAX_FLAT_DIR_ENTRIES = _validation.MAX_FLAT_DIR_ENTRIES


@dataclass(frozen=True)
class InvariantViolation:
    """One invariant violation, structured for both display and escalation.

    ``message`` is the historic one-line human-readable string (what
    :func:`check_invariants` returns and the audit/WARN log shows). The
    optional ``repair_kind`` / ``file`` / ``target`` / ``context`` fields are
    set only for violations the daemon escalates into the Librarian→Scholar
    pipeline (over-cap dirs, bad frontmatter). When ``repair_kind`` is
    ``None`` the violation is display-only — it's handled by another
    deterministic pass (READMEs by the reconciler, wikilinks by
    ``repair_broken_wikilinks``) or is not independently actionable
    (scope/path, empty body — both ride along with their entry's
    frontmatter task), so escalating it here would be redundant.
    """

    message: str
    repair_kind: Optional[str] = None
    file: str = ""
    target: str = ""
    context: str = ""

    def to_repair_task(self) -> Optional[repair_queue.RepairTask]:
        """Build the :class:`RepairTask` for this violation, or ``None`` when
        it is display-only (``repair_kind is None``)."""
        if self.repair_kind is None:
            return None
        return repair_queue.RepairTask(
            kind=self.repair_kind,
            file=self.file,
            target=self.target,
            context=self.context,
        )


def _collect_md_files(knowledge_dir_path: Path) -> tuple[List[Path], List[Path]]:
    """Walk the tree once and return ``(entry_files, all_md_files)``.
    Entries exclude README.md / index.md / log.md; ``all_md_files``
    includes them. Both lists exclude any path under ``_archive``."""
    entry_files: List[Path] = []
    all_md_files: List[Path] = []
    for md in sorted(knowledge_dir_path.rglob("*.md")):
        if "_archive" in md.parts:
            continue
        all_md_files.append(md)
        if md.name in ("README.md", "index.md", "log.md"):
            continue
        entry_files.append(md)
    return entry_files, all_md_files


def _check_readme_coverage(
    entry_files: List[Path],
    knowledge_dir_path: Path,
) -> List[InvariantViolation]:
    """Every directory that contains an entry must have a README.md.

    Display-only: the post-action reconciler creates missing READMEs
    automatically, so there is no separate repair task to escalate."""
    out: List[InvariantViolation] = []
    dirs_to_check: set[Path] = set()
    for md in entry_files:
        d = md.parent
        while d != knowledge_dir_path.parent and d.exists():
            dirs_to_check.add(d)
            if d == knowledge_dir_path:
                break
            d = d.parent
    for d in sorted(dirs_to_check):
        if "_archive" in d.parts:
            continue
        if not (d / "README.md").exists():
            rel = d.relative_to(knowledge_dir_path) if d != knowledge_dir_path else Path(".")
            out.append(InvariantViolation(message=f"missing README.md in {rel}/"))
    return out


def _check_flat_dir_caps(
    entry_files: List[Path],
    knowledge_dir_path: Path,
) -> List[InvariantViolation]:
    """Per-directory entry count must be at or below MAX_FLAT_DIR_ENTRIES.

    Over-cap dirs ESCALATE (``overcap_dir``): the fingerprint targets the
    directory, and the context lists the entries so the Librarian can
    propose a ``split_folder`` / ``move_entry`` rebalance without re-walking
    the tree."""
    out: List[InvariantViolation] = []
    by_dir: Dict[Path, List[Path]] = {}
    for md in entry_files:
        by_dir.setdefault(md.parent, []).append(md)
    for d, mds in by_dir.items():
        if len(mds) > MAX_FLAT_DIR_ENTRIES:
            rel = d.relative_to(knowledge_dir_path) if d != knowledge_dir_path else Path(".")
            rel_posix = rel.as_posix()
            entry_rels = sorted(m.relative_to(knowledge_dir_path).as_posix() for m in mds)
            out.append(
                InvariantViolation(
                    message=(
                        f"directory {rel}/ has {len(mds)} entry .md files "
                        f"(max is {MAX_FLAT_DIR_ENTRIES})"
                    ),
                    repair_kind=repair_queue.KIND_OVERCAP_DIR,
                    file=rel_posix,
                    target=rel_posix,
                    context=(
                        f"{len(mds)} entries in {rel_posix}/ (cap {MAX_FLAT_DIR_ENTRIES}): "
                        + ", ".join(entry_rels)
                    ),
                )
            )
    return out


def _wikilink_resolves(link: str, md: Path, knowledge_dir_path: Path) -> bool:
    """Apply the resolution rules (root-relative + sibling fallback) to
    one wikilink target. Returns True if it points at an existing entry.

    Thin wrapper over the shared ``_validation.wikilink_resolves`` so the
    post-write checker and the boundary ``output_validator`` use identical
    resolution rules."""
    return _validation.wikilink_resolves(link, md.parent, knowledge_dir_path)


def _check_wikilinks(
    all_md_files: List[Path],
    knowledge_dir_path: Path,
) -> List[InvariantViolation]:
    """Every wikilink resolves. We parse each file as markdown so that
    links inside code spans / fenced code blocks / YAML frontmatter are
    excluded — those were the source of historic false positives. Skip
    log.md outright (audit trail; quoted paths are not navigation).

    Display-only here: broken wikilinks escalate through the dedicated
    deterministic pass (``repair_broken_wikilinks`` →
    ``_escalate_unresolved_wikilink``), which runs BEFORE this check and
    owns the link's in-flight marker. Escalating again here would
    double-fingerprint the same issue."""
    out: List[InvariantViolation] = []
    for md in all_md_files:
        if md.name == "log.md":
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for hit in markdown_utils.extract_wikilinks(text):
            link = hit.target
            if not link:
                continue
            if _wikilink_resolves(link, md, knowledge_dir_path):
                continue
            rel = md.relative_to(knowledge_dir_path)
            out.append(InvariantViolation(message=f"broken wikilink in {rel}: [[{link}]]"))
    return out


def _check_scope_path_agreement(rel: Path, scope: str) -> str | None:
    """Per prompt invariant #5: ``scope: global`` must live under
    ``global/``; ``scope: project:<slug>`` under ``projects/<slug>/``.
    Returns the violation string or ``None`` if the file is in agreement.

    Thin wrapper over the shared ``_validation.scope_path_violation`` so the
    post-write checker and the boundary validators agree."""
    return _validation.scope_path_violation(rel, scope)


def _bad_frontmatter_violation(rel_posix: str, message: str) -> InvariantViolation:
    """Build an escalating ``bad_frontmatter`` violation for an entry whose
    frontmatter is missing, unparseable, or short the required fields. The
    fingerprint targets the entry path so a single in-flight attempt covers
    every frontmatter defect in that file at once."""
    return InvariantViolation(
        message=message,
        repair_kind=repair_queue.KIND_BAD_FRONTMATTER,
        file=rel_posix,
        target=rel_posix,
        context=message,
    )


def _check_entry_frontmatter(
    entry_files: List[Path],
    knowledge_dir_path: Path,
) -> List[InvariantViolation]:
    """Each entry must have a parseable frontmatter block with the
    required fields, a non-trivial body, and a scope that matches its
    path.

    A missing/unparseable/incomplete frontmatter block ESCALATES
    (``bad_frontmatter``) so the Librarian can re-serialise valid
    frontmatter. The scope/path and empty-body checks stay display-only —
    re-serialising frontmatter wouldn't relocate a misfiled entry or invent
    a body, and the entry's frontmatter task (if any) already carries the
    in-flight marker for that file."""
    out: List[InvariantViolation] = []
    for md in entry_files:
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = md.relative_to(knowledge_dir_path)
        rel_posix = rel.as_posix()
        fm = _parse_frontmatter(text)
        if not fm:
            out.append(
                _bad_frontmatter_violation(
                    rel_posix, f"missing or unparseable frontmatter in {rel}"
                )
            )
            continue
        missing = [f for f in _REQUIRED_FRONTMATTER_FIELDS if f not in fm]
        if missing:
            out.append(
                _bad_frontmatter_violation(
                    rel_posix,
                    f"missing frontmatter fields in {rel}: {', '.join(missing)}",
                )
            )

        body = _strip_frontmatter(text).strip()
        if len(body) < 20:
            out.append(InvariantViolation(message=f"entry body is empty or trivial in {rel}"))

        scope = str(fm.get("scope", "")).strip()
        scope_violation = _check_scope_path_agreement(rel, scope)
        if scope_violation is not None:
            out.append(InvariantViolation(message=scope_violation))
    return out


def check_invariants_detailed(knowledge_dir_path: Path) -> List[InvariantViolation]:
    """Walk the knowledge tree and return structured invariant violations.

    Each :class:`InvariantViolation` carries the human-readable ``message``
    plus — for the escalating kinds (over-cap dirs, bad frontmatter) — the
    fields needed to build a :class:`repair_queue.RepairTask`. The Scholar's
    escalation pass consumes this; :func:`check_invariants` projects it down
    to the legacy ``List[str]`` for logging/tests.

    Checks:
      1. Every directory has a README.md (display-only — reconciler fixes).
      2. No directory has >MAX_FLAT_DIR_ENTRIES entry .md files (escalates).
      3. Every wikilink resolves (display-only — repair pass escalates).
      4. Every entry's frontmatter has the required fields (escalates).
    """
    if not knowledge_dir_path.exists():
        return []

    entry_files, all_md_files = _collect_md_files(knowledge_dir_path)
    out: List[InvariantViolation] = []
    out.extend(_check_readme_coverage(entry_files, knowledge_dir_path))
    out.extend(_check_flat_dir_caps(entry_files, knowledge_dir_path))
    out.extend(_check_wikilinks(all_md_files, knowledge_dir_path))
    out.extend(_check_entry_frontmatter(entry_files, knowledge_dir_path))
    return out


def check_invariants(knowledge_dir_path: Path) -> List[str]:
    """Walk the knowledge tree and return a list of invariant violations.

    Each violation is a one-line human-readable string. Empty list means
    everything is well-formed. The Scholar calls this after executing
    approved actions and logs WARN on each violation; this is the safety
    net for "the Scholar should have caught it" cases.

    Thin projection over :func:`check_invariants_detailed` — kept as the
    stable string-list contract that the audit log and the test-suite
    assert against.
    """
    return [v.message for v in check_invariants_detailed(knowledge_dir_path)]


def _strip_frontmatter(text: str) -> str:
    """Return ``text`` with a leading YAML frontmatter block removed.
    Delegates to the shared ``_validation`` helper."""
    return _validation.strip_frontmatter(text)


# Re-exported so the reinforcement-counter bookkeeping below (which mutates
# frontmatter in place) keeps using the same regex as the shared parser.
_FRONTMATTER_HEAD_RE = _validation.FRONTMATTER_HEAD_RE


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Parse a leading YAML frontmatter block to a mapping (``{}`` when
    missing/unparseable). Delegates to the shared ``_validation`` helper."""
    return _validation.parse_frontmatter(text)


# ── Deterministic reinforcement-counter bookkeeping ──────────────────
#
# When the Librarian flags a proposal with ``salience_signal: "reinforces"``
# and a valid ``existing_entry`` path, the daemon bumps a counter on that
# entry's frontmatter — no SDK round-trip. This is an EMPIRICAL fact
# ("the user mentioned this again") not a judgment call, so it doesn't
# need the Scholar's deliberation.
#
# The Scholar still sees the proposal and decides whether to APPROVE an
# ``update_entry`` to merge in new phrasing or VETO with "counter already
# bumped server-side, no substantive new content."


def _bump_reinforced_counter(
    entry_path: Path,
    *,
    today_iso: Optional[str] = None,
) -> bool:
    """Increment ``reinforced`` in entry's frontmatter; stamp
    ``last_reinforced``. Returns True on success, False if the file
    can't be read or has no parseable frontmatter."""
    if today_iso is None:
        today_iso = datetime.now(timezone.utc).date().isoformat()
    try:
        text = entry_path.read_text(encoding="utf-8")
    except OSError:
        return False
    m = _FRONTMATTER_HEAD_RE.match(text)
    if not m:
        return False
    try:
        loaded: object = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return False
    if not isinstance(loaded, dict):
        return False
    fm = cast(Dict[str, Any], loaded)
    fm["reinforced"] = int(fm.get("reinforced") or 0) + 1
    fm["last_reinforced"] = today_iso
    try:
        new_fm = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False).rstrip()
    except yaml.YAMLError:
        return False
    rest = text[m.end() :]
    new_text = f"---\n{new_fm}\n---\n{rest}"
    try:
        tmp = entry_path.with_suffix(entry_path.suffix + ".tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, entry_path)
    except OSError:
        return False
    return True


def _resolve_reinforce_target(prop: object, root: Path) -> Path | None:
    """Return the absolute entry path a "reinforces" proposal targets,
    or ``None`` if the proposal isn't well-formed / cites a path outside
    the knowledge dir."""
    if not isinstance(prop, dict):
        return None
    prop_dict = cast(Dict[str, Any], prop)
    if prop_dict.get("salience_signal") != "reinforces":
        return None
    cited = prop_dict.get("existing_entry")
    if not isinstance(cited, str) or not cited:
        return None
    candidate = (root / cited).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        log.warning(
            "apply_reinforcement_counters: rejected path outside knowledge dir: %s",
            cited,
        )
        return None
    return candidate


def apply_reinforcement_counters(
    packets: Sequence[Mapping[str, Any]],
    knowledge_dir_path: Path,
) -> List[str]:
    """Walk all proposals across all packets; for each with
    ``salience_signal == "reinforces"`` and a valid ``existing_entry``
    path inside ``knowledge_dir_path``, bump the counter.

    Returns a list of human-readable change messages. Idempotent within
    a single batch (each unique entry bumped once even if multiple
    proposals reinforce it) — multiple separate batches will compound.
    """
    if not knowledge_dir_path.exists():
        return []
    root = knowledge_dir_path.resolve()
    today = datetime.now(timezone.utc).date().isoformat()
    changes: List[str] = []
    bumped_paths: set[Path] = set()

    for packet in packets:
        raw_proposals: object = packet.get("proposals") or []
        proposals: Sequence[object] = (
            cast(Sequence[object], raw_proposals) if isinstance(raw_proposals, list) else []
        )
        for prop in proposals:
            candidate = _resolve_reinforce_target(prop, root)
            if candidate is None:
                continue
            if candidate in bumped_paths:
                continue  # dedupe within batch
            if not candidate.exists():
                log.debug(
                    "apply_reinforcement_counters: existing_entry not found: %s",
                    candidate.relative_to(root),
                )
                continue
            if _bump_reinforced_counter(candidate, today_iso=today):
                bumped_paths.add(candidate)
                rel = candidate.relative_to(root)
                changes.append(f"reinforced {rel}")
    return changes


# ── Deterministic fired-helpful counter bookkeeping ──────────────────
#
# ``fired-helpful`` is conceptually distinct from ``reinforced``:
#   - ``reinforced``    = the conversation re-asserted the entry's CONTENT.
#   - ``fired-helpful`` = the assistant USED a surfaced entry helpfully in
#                         a turn (positive reliance).
# When the Librarian flags a proposal with ``salience_signal:
# "used_helpfully"`` it cites the relied-upon entry in ``existing_entry``
# and the STABLE turn id in ``cited_turn_seq``. The Scholar bumps the
# counter deterministically — like the reinforcement bump, this is an
# empirical fact (the assistant relied on it), not a judgment call.
#
# DOUBLE-COUNT PROBLEM + FIX:
# The Librarian's rolling buffer is never drained, so a given physical
# turn is re-scanned on every subsequent Stop until it ages out — the
# Librarian would re-emit ``used_helpfully`` for the same (turn, entry)
# many times. We attribute a bump to a (session, entry, turn) citation
# EVENT exactly once via a persisted per-(session, entry) HIGH-WATER mark
# on ``cited_turn_seq`` (the buffer's stable, monotonic per-session turn
# id — NOT the Librarian's scan-local ``turn_id``, which is recomputed per
# scan and is not stable). Within one batch we count every DISTINCT cited
# seq strictly greater than the stored mark (so coalesced Stops count the
# whole gap rather than collapsing to one), then advance the mark to the
# max counted seq. State persists across restarts via the offset-state
# idiom (atomic tmp+rename JSON at ``fired_helpful_state_path()``).
#
# Restart edge: the in-memory buffer (and its ``turn_seq`` allocator) is
# rebuilt empty on restart and the tailer resumes past old events, so the
# old turns can never be re-seen — the over-count this guards against
# cannot occur across a restart. A session that *continues* across a
# restart restarts its ``turn_seq`` at 1, which could under-count at most
# one bump per entry; benign and far rarer than the over-count we fix.

# Shape: ``{session_id: {entry_rel_path: last_counted_turn_seq}}``.
FiredHelpfulState = Dict[str, Dict[str, int]]


def _load_fired_helpful_state(state_path: Path) -> FiredHelpfulState:
    """Read the persisted high-water state. Returns ``{}`` on missing or
    corrupt state (we re-validate every value before use, so a malformed
    file just means we start from scratch — at worst one re-count)."""
    try:
        loaded: object = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: FiredHelpfulState = {}
    for sid, entries in cast(Dict[Any, Any], loaded).items():
        if not isinstance(sid, str) or not isinstance(entries, dict):
            continue
        per_entry: Dict[str, int] = {}
        for path, seq in cast(Dict[Any, Any], entries).items():
            if isinstance(path, str) and isinstance(seq, int) and not isinstance(seq, bool):
                per_entry[path] = seq
        if per_entry:
            out[sid] = per_entry
    return out


def _save_fired_helpful_state(state_path: Path, state: FiredHelpfulState) -> None:
    """Persist the high-water state atomically (tmp+rename). Swallows OS
    errors — the counter is bookkeeping; a failed persist just risks one
    future re-count, it must never break the review pipeline."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, state_path)
    except OSError as e:
        log.debug("could not persist fired-helpful state to %s: %s", state_path, e)


def _bump_fired_helpful_counter(
    entry_path: Path,
    *,
    by: int,
    today_iso: Optional[str] = None,
) -> bool:
    """Increment ``fired-helpful`` by ``by`` in the entry's frontmatter;
    stamp ``last_fired_helpful``. Returns True on success, False if the
    file can't be read or has no parseable frontmatter. Mirrors
    :func:`_bump_reinforced_counter`."""
    if by <= 0:
        return False
    if today_iso is None:
        today_iso = datetime.now(timezone.utc).date().isoformat()
    try:
        text = entry_path.read_text(encoding="utf-8")
    except OSError:
        return False
    m = _FRONTMATTER_HEAD_RE.match(text)
    if not m:
        return False
    try:
        loaded: object = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return False
    if not isinstance(loaded, dict):
        return False
    fm = cast(Dict[str, Any], loaded)
    fm["fired-helpful"] = int(fm.get("fired-helpful") or 0) + by
    fm["last_fired_helpful"] = today_iso
    try:
        new_fm = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False).rstrip()
    except yaml.YAMLError:
        return False
    rest = text[m.end() :]
    new_text = f"---\n{new_fm}\n---\n{rest}"
    try:
        tmp = entry_path.with_suffix(entry_path.suffix + ".tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, entry_path)
    except OSError:
        return False
    return True


def _resolve_used_helpfully(prop: object, root: Path) -> Optional[Tuple[Path, str, int]]:
    """Return ``(abs_entry_path, rel_entry_path, cited_turn_seq)`` for a
    well-formed ``used_helpfully`` proposal, or ``None`` if it is not one /
    is malformed / cites a path outside the knowledge dir / omits a valid
    ``cited_turn_seq``.

    A missing or non-positive ``cited_turn_seq`` is rejected: without a
    stable turn id we cannot dedup, and counting blindly would re-introduce
    the over-count this whole mechanism exists to prevent. Better to drop a
    bump than to inflate the counter."""
    if not isinstance(prop, dict):
        return None
    prop_dict = cast(Dict[str, Any], prop)
    if prop_dict.get("salience_signal") != "used_helpfully":
        return None
    cited = prop_dict.get("existing_entry")
    if not isinstance(cited, str) or not cited:
        return None
    raw_seq = prop_dict.get("cited_turn_seq")
    if not isinstance(raw_seq, int) or isinstance(raw_seq, bool) or raw_seq <= 0:
        log.debug(
            "apply_fired_helpful_counters: used_helpfully for %s missing a valid "
            "cited_turn_seq (%r); skipping to avoid an un-dedupable bump",
            cited,
            raw_seq,
        )
        return None
    candidate = (root / cited).resolve()
    try:
        rel = candidate.relative_to(root)
    except ValueError:
        log.warning(
            "apply_fired_helpful_counters: rejected path outside knowledge dir: %s",
            cited,
        )
        return None
    return candidate, str(rel), raw_seq


def apply_fired_helpful_counters(
    packets: Sequence[Mapping[str, Any]],
    knowledge_dir_path: Path,
    *,
    state_path: Optional[Path] = None,
) -> List[str]:
    """Walk all proposals across all packets; for each with
    ``salience_signal == "used_helpfully"``, a valid ``existing_entry``
    inside ``knowledge_dir_path``, and a valid ``cited_turn_seq``, bump the
    entry's ``fired-helpful`` counter ONCE per distinct (session, entry,
    turn) citation event and stamp ``last_fired_helpful``.

    Double-counting is prevented by a persisted per-(session, entry)
    high-water mark on ``cited_turn_seq``: only seqs strictly greater than
    the stored mark are counted, then the mark advances to the max counted
    seq. Distinct new seqs within a single batch each count (coalesced
    Stops count the whole gap). The session id comes from each PACKET (a
    batch may mix sessions), defaulting to ``"?"`` when absent so a
    session-less packet still dedups against itself.

    Returns a list of human-readable change messages. ``state_path``
    defaults to :func:`fired_helpful_state_path` (overridable for tests).
    """
    if not knowledge_dir_path.exists():
        return []
    root = knowledge_dir_path.resolve()
    today = datetime.now(timezone.utc).date().isoformat()
    sp = state_path if state_path is not None else fired_helpful_state_path()
    state = _load_fired_helpful_state(sp)

    # Pass 1: gather distinct new (session, entry) -> set of seqs above the
    # stored high-water, plus the resolved path for each entry. We resolve
    # paths once and aggregate before touching any file so a single entry
    # cited across multiple turns/packets is bumped by the right total.
    pending: Dict[Tuple[str, str], set[int]] = {}
    abs_paths: Dict[Tuple[str, str], Path] = {}
    for packet in packets:
        session_id = str(packet.get("session_id") or "?")
        raw_proposals: object = packet.get("proposals") or []
        proposals: Sequence[object] = (
            cast(Sequence[object], raw_proposals) if isinstance(raw_proposals, list) else []
        )
        for prop in proposals:
            resolved = _resolve_used_helpfully(prop, root)
            if resolved is None:
                continue
            candidate, rel, seq = resolved
            mark = state.get(session_id, {}).get(rel, 0)
            if seq <= mark:
                continue  # already counted (re-seen turn) or stale
            key = (session_id, rel)
            pending.setdefault(key, set()).add(seq)
            abs_paths[key] = candidate

    # Pass 2: apply. One file write per (session, entry) carrying the full
    # count of newly-seen turns; advance the high-water to the max seq only
    # if the on-disk bump actually succeeded (so a transient write failure
    # re-counts next time rather than silently dropping the use).
    changes: List[str] = []
    dirty = False
    for (session_id, rel), seqs in pending.items():
        candidate = abs_paths[(session_id, rel)]
        if not candidate.exists():
            log.debug("apply_fired_helpful_counters: existing_entry not found: %s", rel)
            continue
        n = len(seqs)
        if _bump_fired_helpful_counter(candidate, by=n, today_iso=today):
            state.setdefault(session_id, {})[rel] = max(seqs)
            dirty = True
            changes.append(f"fired-helpful +{n} {rel}")

    if dirty:
        _save_fired_helpful_state(sp, state)
    return changes


# ── README reconciliation (deterministic bookkeeping) ──────────────────
#
# After the Scholar finishes its action batch, walk every folder in the
# knowledge tree and make sure each README.md exists and its
# auto-managed "children list" section matches the folder's actual
# contents. The LLM owns prose; this owns the bullet list.
#
# Markers delimit the auto-managed block. Anything outside is preserved
# verbatim, so the Librarian can write a description above and trust the
# child-listing below to stay in sync.

_AUTO_CHILDREN_BEGIN = "<!-- ULTAN:children (auto) -->"
_AUTO_CHILDREN_END = "<!-- /ULTAN:children -->"


def _entry_title(entry: Path) -> str:
    """Return the entry's frontmatter title, falling back to its stem."""
    try:
        text = entry.read_text(encoding="utf-8")
    except OSError:
        return entry.stem
    fm = _parse_frontmatter(text)
    title = fm.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip().strip('"').strip("'")
    return entry.stem


def _folder_title(folder: Path) -> str:
    """Return the folder's README h1 if present, else humanise the name."""
    readme = folder / "README.md"
    if readme.exists():
        try:
            for line in readme.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
        except OSError:
            pass
    return folder.name.replace("-", " ").replace("_", " ").title()


def _format_children_section(folder: Path, knowledge_dir: Path) -> str:
    """Build the auto-managed children list for ``folder`` as text.

    Subfolders first (linked with trailing slash → README), then entries
    (linked with full path from knowledge root, no .md). Excludes
    ``_archive`` subtrees and the folder's own README.
    """
    lines = [_AUTO_CHILDREN_BEGIN]
    subfolders = sorted(p for p in folder.iterdir() if p.is_dir() and p.name != "_archive")
    for sub in subfolders:
        rel = sub.relative_to(knowledge_dir)
        title = _folder_title(sub)
        lines.append(f"- [[{rel}/]] — {title}")
    entries = sorted(
        p for p in folder.glob("*.md") if p.name not in ("README.md", "index.md", "log.md")
    )
    for entry in entries:
        rel = entry.relative_to(knowledge_dir)
        link = str(rel.with_suffix(""))
        title = _entry_title(entry)
        lines.append(f"- [[{link}]] — {title}")
    if not subfolders and not entries:
        lines.append("_(empty)_")
    lines.append(_AUTO_CHILDREN_END)
    return "\n".join(lines)


def _splice_children_section(existing: str, section: str) -> str:
    """Replace the marker-delimited block in ``existing`` with ``section``.

    If the markers aren't present, append the section to the end with a
    blank-line separator. If only the BEGIN marker is present (corrupted
    file), append a fresh section and leave the corrupted one in place
    for the user to clean up — never silently destroy content.
    """
    begin = existing.find(_AUTO_CHILDREN_BEGIN)
    if begin == -1:
        sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        return existing + sep + section + "\n"
    end = existing.find(_AUTO_CHILDREN_END, begin)
    if end == -1:
        sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        return existing + sep + section + "\n"
    end += len(_AUTO_CHILDREN_END)
    return existing[:begin] + section + existing[end:]


def reconcile_readmes(knowledge_dir: Path) -> List[str]:
    """For every directory under ``knowledge_dir`` (excluding ``_archive``),
    ensure a README.md exists and its auto-managed children list matches
    the directory's actual contents.

    Returns a list of human-readable change messages. Empty list means
    everything was already in sync. Safe to run repeatedly; idempotent.

    This is the deterministic safety net for the rule "every folder has
    a README that reflects its contents." The Librarian/Scholar can
    still propose their own README updates (for prose); reconcile owns
    the bullet list of children below the auto markers.
    """
    if not knowledge_dir.exists():
        return []
    changes: List[str] = []

    # Collect every dir except _archive subtrees. Include the knowledge
    # root itself.
    all_dirs: List[Path] = [knowledge_dir]
    for d in knowledge_dir.rglob("*"):
        if d.is_dir() and "_archive" not in d.parts:
            all_dirs.append(d)

    for folder in sorted(set(all_dirs)):
        readme = folder / "README.md"
        section = _format_children_section(folder, knowledge_dir)
        if not readme.exists():
            display = (
                "Knowledge"
                if folder == knowledge_dir
                else folder.name.replace("-", " ").replace("_", " ").title()
            )
            body = f"# {display}\n\n{section}\n"
            try:
                readme.write_text(body, encoding="utf-8")
                rel = folder.relative_to(knowledge_dir) if folder != knowledge_dir else "."
                changes.append(f"created README in {rel}/")
            except OSError as e:
                log.warning("could not create %s: %s", readme, e)
            continue
        try:
            current = readme.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("could not read %s: %s", readme, e)
            continue
        updated = _splice_children_section(current, section)
        if updated != current:
            try:
                readme.write_text(updated, encoding="utf-8")
                rel = folder.relative_to(knowledge_dir) if folder != knowledge_dir else Path(".")
                changes.append(f"reconciled children in {rel}/README.md")
            except OSError as e:
                log.warning("could not write %s: %s", readme, e)

    return changes


# ── Deterministic broken-wikilink repair (post-write safety net) ──────
#
# ``check_invariants`` only WARNS about broken wikilinks. A hallucinated
# phantom index.md row (an entry the Scholar referenced but never wrote)
# would re-trip that warning on every single run forever — observed 47×
# for one phantom ``projects/some-fake-project/...`` row.
#
# The Scholar agent's ``output_validator`` only inspects the actions the
# model is about to apply this pass; it never touches links already sitting
# on disk that no returned action rewrites — which is exactly the
# phantom-row case. This pass closes that gap: after the executor finishes,
# walk the tree and repair broken links in place. Integrity-first — we'd
# rather remove a known-bogus phantom than leave the graph broken.


def _resolve_broken_link_leaf(link: str, knowledge_dir_path: Path) -> str | None:
    """If ``link``'s final path segment matches exactly one existing entry
    in the tree, return that entry's canonical knowledge-root-relative
    wikilink target (no ``.md``). Returns ``None`` when there are zero or
    multiple matches (ambiguous — don't guess) or the leaf already equals
    the link (no move to make)."""
    leaf = link.rsplit("/", 1)[-1]
    if leaf.endswith(".md"):
        leaf = leaf[:-3]
    if not leaf:
        return None
    matches = [m for m in knowledge_dir_path.rglob(f"{leaf}.md") if "_archive" not in m.parts]
    if len(matches) != 1:
        return None
    canonical = matches[0].relative_to(knowledge_dir_path).with_suffix("").as_posix()
    return canonical if canonical != link else None


def _neutralise_wikilink(text: str, raw: str, display: str) -> str:
    """Replace the literal ``raw`` ``[[…]]`` token with ``display`` plain
    text wherever it appears in ``text``. Surrounding content is left
    untouched — we strip only the broken navigation edge, never the
    sentence around it."""
    return text.replace(raw, display)


def _repair_index_rows(text: str, broken_targets: set[str]) -> tuple[str, int]:
    """Drop every line of an index.md whose only role is to catalog a
    now-broken entry. A catalog row is a markdown table row (starts with
    ``|``) that contains a broken wikilink. Returns ``(new_text, removed)``.

    Non-row occurrences (e.g. prose in the index header) are left for the
    body-link path so we never delete narrative lines wholesale."""
    if not broken_targets:
        return text, 0
    kept: List[str] = []
    removed = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        # Match the FULL wikilink token, not a prefix: a broken target
        # ``global/foo`` must not delete a row whose only link is the
        # valid ``[[global/foobar]]`` (prefix ``[[global/foo`` would match
        # inside it). A table-cell wikilink is terminated by ``]]`` or by
        # ``|`` (the alias separator), so require one of those right after
        # the target.
        if stripped.startswith("|") and any(
            f"[[{t}]]" in line or f"[[{t}|" in line for t in broken_targets
        ):
            removed += 1
            continue
        kept.append(line)
    return ("".join(kept), removed) if removed else (text, 0)


# Callback the escalation layer injects: given ``(rel_file, target,
# context)`` for a wikilink the deterministic pass could not resolve, it
# escalates the issue into the Librarian→Scholar pipeline and returns
# ``True`` if the issue is now owned by that escalation path. When an
# escalator owns the issue we deliberately LEAVE the link broken on disk
# (do not neutralise) so the next deterministic pass can re-detect it and
# re-escalate until the Scholar actually fixes it — neutralising would
# erase the only on-disk signal. With no escalator wired (callback is
# ``None``) the historical neutralise-as-stopgap behaviour is preserved.
OnUnresolved = Callable[[str, str, str], bool]


def _link_context(text: str, raw: str, *, window: int = 60) -> str:
    """Return a short snippet of ``text`` around the first occurrence of the
    raw ``[[…]]`` token — enough for the Librarian to see where/how the
    broken link appears without re-reading the whole file."""
    idx = text.find(raw)
    if idx == -1:
        return raw
    start = max(0, idx - window)
    end = min(len(text), idx + len(raw) + window)
    snippet = text[start:end].replace("\n", " ").strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _broken_links_in(
    text: str, md: Path, knowledge_dir_path: Path
) -> List[markdown_utils.WikilinkHit]:
    """Return the prose wikilinks in ``text`` that fail to resolve."""
    out: List[markdown_utils.WikilinkHit] = []
    for hit in markdown_utils.extract_wikilinks(text):
        if hit.target and not _wikilink_resolves(hit.target, md, knowledge_dir_path):
            out.append(hit)
    return out


def _repair_body_links(
    text: str,
    broken: Sequence["markdown_utils.WikilinkHit"],
    knowledge_dir_path: Path,
    *,
    rel: str,
    on_unresolved: Optional[OnUnresolved] = None,
) -> tuple[str, List[str]]:
    """Best-effort repair of broken wikilinks in an entry/README body.

    Each broken link is either resolved to a unique existing entry (the
    link is rewritten, alias preserved) or — when unresolvable — handled
    one of two ways:

      - If an ``on_unresolved`` escalator is wired and it takes ownership
        of the issue, the link is LEFT BROKEN on disk so a future pass can
        re-detect and re-escalate it (neutralising would destroy that
        signal). We record a note but do not mutate the link.
      - Otherwise the link is neutralised to plain text so the broken edge
        is removed without destroying the surrounding prose.

    Returns ``(new_text, notes)``."""
    notes: List[str] = []
    for hit in broken:
        resolved = _resolve_broken_link_leaf(hit.target, knowledge_dir_path)
        if resolved is not None:
            alias = f"|{hit.alias}" if hit.alias else ""
            text = text.replace(hit.raw, f"[[{resolved}{alias}]]")
            notes.append(f"rewrote [[{hit.target}]] → [[{resolved}]]")
            continue
        if on_unresolved is not None and on_unresolved(
            rel, hit.target, _link_context(text, hit.raw)
        ):
            # Escalated — keep the broken link so re-detection works.
            notes.append(f"escalated unresolvable [[{hit.target}]] to the Librarian")
            continue
        display = hit.alias if hit.alias else hit.target.rsplit("/", 1)[-1]
        text = _neutralise_wikilink(text, hit.raw, display)
        notes.append(f"neutralised broken [[{hit.target}]] → {display!r}")
    return text, notes


def _repair_one_file(
    md: Path,
    text: str,
    knowledge_dir_path: Path,
    *,
    on_unresolved: Optional[OnUnresolved] = None,
) -> tuple[str, List[str]]:
    """Repair broken wikilinks in a single file's ``text``. Returns
    ``(new_text, change_notes)``; ``new_text == text`` when nothing was
    broken. Splits index-row removal from body-link repair."""
    broken = _broken_links_in(text, md, knowledge_dir_path)
    if not broken:
        return text, []
    rel = md.relative_to(knowledge_dir_path)
    changes: List[str] = []
    new_text = text
    if md.name == "index.md":
        new_text, removed = _repair_index_rows(new_text, {h.target for h in broken})
        if removed:
            changes.append(f"removed {removed} phantom row(s) from {rel}")
        # Re-scan: links that weren't catalog rows still need body repair.
        broken = _broken_links_in(new_text, md, knowledge_dir_path)
    if broken:
        new_text, notes = _repair_body_links(
            new_text,
            broken,
            knowledge_dir_path,
            rel=rel.as_posix(),
            on_unresolved=on_unresolved,
        )
        changes.extend(f"{rel}: {n}" for n in notes)
    return new_text, changes


def repair_broken_wikilinks(
    knowledge_dir_path: Path,
    *,
    on_unresolved: Optional[OnUnresolved] = None,
) -> List[str]:
    """Best-effort, in-place repair of broken wikilinks across the tree.

    Cases, integrity-first:
      - An ``index.md`` catalog row pointing at a non-existent entry → the
        whole phantom row is removed (the entry never existed; the row is
        pure noise that re-trips the broken-wikilink invariant forever).
      - A broken wikilink in any other body → resolve to a unique existing
        entry by leaf-name match and rewrite (alias preserved). If
        unresolvable AND an ``on_unresolved`` escalator takes ownership,
        the link is left broken (so the issue stays detectable until the
        Scholar fixes it via the Librarian pipeline); otherwise it is
        neutralised to plain text without touching surrounding content.

    Skips ``log.md`` (audit trail; quoted paths are not navigation) and
    ``_archive`` subtrees. Returns a list of human-readable change
    messages; empty when the graph was already intact. Idempotent — note
    that an escalated (left-broken) link will surface in the change list
    on every pass until fixed, which is the intended "keep escalating"
    behaviour; the in-flight guard in ``repair_queue`` prevents duplicate
    concurrent escalations.
    """
    if not knowledge_dir_path.exists():
        return []
    changes: List[str] = []
    _, all_md_files = _collect_md_files(knowledge_dir_path)
    for md in all_md_files:
        if md.name == "log.md":
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_text, file_changes = _repair_one_file(
            md, text, knowledge_dir_path, on_unresolved=on_unresolved
        )
        # Even when the file body is unchanged (e.g. the only broken link
        # was escalated and left in place), surface the escalation notes so
        # the audit trail shows we acted.
        if new_text != text:
            try:
                md.write_text(new_text, encoding="utf-8")
            except OSError as e:
                log.warning("could not write repaired %s: %s", md, e)
                continue
        changes.extend(file_changes)
    return changes
