---
name: epiphany
description: Roam the Ultan memory library and surface ONE non-obvious, useful connection between distant entries — an on-demand "epiphany". Use when the user asks for a spark/insight/epiphany from their memory, wants a hidden cross-cutting pattern surfaced, asks "show me something interesting in my notes", or runs /epiphany. Can be scoped to a single project (e.g. /epiphany vol-predictor) to surface a cross-subsystem insight about that project. Read-only — it never writes to the library. Works by fanning out parallel scouts (Opus), converging over rounds with iterative grounding against the actual source artifacts the entries point at, then filtering to the single best connection.
---

# Epiphany

A horizontal connection-finder for the Ultan library. The daemon already does
*vertical* abstraction (rolling children up into a parent rule); this does the
thing nothing else does — spots that two **distant, unrelated** entries share a
latent structure, and surfaces the single best one.

The design rationale, and the one rule that makes this worth running: an
epiphany is a *remote association* — a far pair with shared deep structure.
Near-neighbours (same folder, same topic, already cross-linked) are obvious and
worthless here. **The value lives in pairs that are far apart in the graph but
structurally rhyme.** Everything below is built to find those and reject
everything else.

## Usage

```
/epiphany                  # free-roam the whole library (cross-domain)
/epiphany <seed/topic>     # free-roam, biased toward a region or theme
/epiphany <project>        # SCOPED: search one project, bridge across its subsystems
```

**Free-roam** finds the most surprising pair anywhere in the library. **Scoped**
(e.g. `/epiphany vol-predictor`) constrains the search to one project and hunts a
non-obvious connection *across its subsystems* (`model/entry-signal` ↔
`infrastructure/data-pipeline`, `evaluation` ↔ `policy`, …), optionally bridging up
to a `global/` principle. "Distance" rescales from cross-domain to cross-subsystem —
still genuinely far, and actionable for the project you're working in. Scoped is the
better daily mode.

## Method

The agent invoking this skill is the **orchestrator**. The scouts are the
generators — and they do not fire once and stop. They **converge over rounds**,
exchanging findings through a shared pool the orchestrator relays between them.
(True peer-to-peer agent chatter isn't available; an orchestrator-mediated
*blackboard* is the realistic — and more faithful — form: it mirrors a global
workspace that competing processes read from and write to.) Run every step.

**0 — Map the territory.**

```bash
# free-roam: the whole library
python3 "<this-skill-dir>/inventory.py" --regions
# scoped to a project: partition it into subsystems (those become the scout regions)
python3 "<this-skill-dir>/inventory.py" --region projects/<PROJECT> --depth 3 --regions
```

**1 — Round 1: independent generation (blind).** Spawn **5–8 scouts in parallel —
one message, multiple `Agent` calls — using Opus** (`model: "opus"`; connection-
finding is reasoning-heavy, not just search). Keep them **blind to each other this
round** — independent diversity is the asset you're protecting. Assign each a
distinct **home region** (free-roam: a project or global subtree; scoped: one
subsystem from step 0). Brief each:

> You are hunting for an *epiphany* — a non-obvious connection between two distant
> entries in a personal knowledge library. Your home region is `<REGION>`. Run
> `python3 <skill-dir>/inventory.py <SCOPE-FLAGS>` to see the territory. Skim your
> home region, pick 2–4 entries that carry a real principle, read them fully. Then
> deliberately bridge **OUT of your home region** — `<BRIDGE TARGET: another
> subsystem of this project / another domain / up to a global principle>` — and find
> entries whose *underlying structure* rhymes with yours (same failure mode,
> trade-off, or mechanism) though the surface topic differs. Read those fully too.
> Return **2–3 candidates**, each as: `a`,`b` (wikilinks) · `pattern` (shared
> structure, one sentence) · `evidence` (a concrete quote/fact from EACH entry — no
> paraphrase-only claims) · `nonobvious` (why it isn't already-linked/trivial) ·
> `sowhat` (the payoff). **Never pair two entries from the same region/subsystem** —
> that's a cross-reference, not an epiphany. Fewer sharp candidates beat more vague ones.

**2 — Rounds 2..N: converge (the agents talk).** Pool every candidate and the
entries scouts surfaced into a shared **blackboard**. Continue the SAME scouts with
`SendMessage` (their context persists — they keep their own reasoning) and hand each
the blackboard:

> Here is the full pool of candidates and entries the other scouts found. Do four
> things: **(1) build** — find a stronger connection bridging ACROSS two *different*
> scouts' material that none of them saw alone; **(2) challenge** — name the weakest
> candidate in the pool and why it's coincidental, already-linked, or hollow;
> **(3) ground** — for any candidate resting on a technical claim, do NOT trust the
> entry's wording: follow its pointers (`sources:`, file paths, repo/tool names in the
> body) to the **actual artifact** — locate it on disk (`find`/`grep`), read it, quote
> the real thing — and refine or kill the bridge by what the source actually says;
> **(4) vote** — your single best with one line of why. Do NOT just agree: if the pool
> is converging on something shallow, or on a claim no one has checked against the
> source, say so and pull it apart.

Assign **one scout a standing skeptic role** — its job is to refute, never to ratify,
and in particular to **demand the source**: "which line of which file shows that?" This
is the *lateral inhibition* that stops the group converging on a confident-but-wrong
answer (see the notes below). Grounding is **iterative, not one-shot**: a technical claim
gets checked against the artifact, the verbatim finding goes back on the blackboard, and
the next round refines, narrows, or kills the bridge in response — so the epiphany is
*shaped by the source*, not merely approved by it. Repeat until **stable**: the same 1–2
candidates win across a round, no new far pair appears, **and the survivor's technical
claims have each been confirmed against the source artifact, not the summary.** Cap at
~3–4 rounds. Convergence is by **evidence survival**, not agreement.

**3 — Judge ratifies.** The orchestrator does NOT just rubber-stamp the vote winner.
Cut hard:

- **Already known / abstraction-of-instance** → drop it if the two entries already
  wikilink each other, if `global/connections/` already covers the pairing, or if
  one entry's `related:`/`sources:` frontmatter cites the other. A global principle
  whose `sources`/`related` point at the project instance is an *abstraction* of
  that instance, not a novel connection — the tell is near-identical numbers or
  wording across the pair. Read the frontmatter before trusting "novel"; this is the
  most seductive false positive. **Cluster-level version:** when many "blind" scouts
  converge on the same neighborhood, check whether those entries already cross-link
  each other in frontmatter before treating the convergence as signal — convergence
  onto a pre-wired cluster is the link graph reflected back, not independent discovery.
- **Not a far pair** → drop same-subsystem / same-topic pairings.
- **Confabulated, or summary-only** → grounding should already have happened in the
  rounds; the judge confirms it held. For every technical claim the survivor rests on,
  the check is against the **source artifact** the entry points at (the code/file/data),
  not the entry's prose. A bridge that holds only at the summary level is the most
  dangerous failure — it reads as profound and cites real entries. If a claim was never
  source-checked, send it back for grounding rather than ratifying.
- **No "so what"** → if it doesn't change a decision or explain something, drop it.

Run a final refute pass, then surface the single best (or a short ranked few if the
user asked for more than one).

### Why convergence needs a skeptic (don't skip this)

Letting agents talk and agree is how you get **groupthink** — an information cascade
where they anchor on the first confident-sounding pairing and reinforce it, throwing
away the independent diversity that made the fan-out worth running. Real neural
convergence isn't agreement; it's *competition* — coalitions that inhibit each other
until one survives. So convergence here must stay adversarial: independent round
first, a standing skeptic throughout, and a final judge that ratifies on grounded
evidence, not on vote count. Convergence is the mechanism; evidence is the criterion.

And discount convergence that lands on an **already-cross-linked cluster** — that is the
corpus's existing link structure reflected back, not N independent discoveries. The
richest epiphany is often a *tension* (two entries that quietly contradict), not a
restatement the scouts agree on — but a tension only counts **once it survives the source**.

### Ground the epiphany in the source, iteratively (the load-bearing fix)

Memory entries are **leads, not ground truth** — a gist, often vague, sometimes just a
pointer at a repo or file. A connection that rhymes at the *summary* level can be an
artifact of lossy wording, and the convergence round will happily agree on it. So
grounding is a **dimension of the discussion, not a final gate**: whenever a candidate
rests on a technical claim, an agent follows the entry's pointers to the **actual
artifact** (locate the repo/file on disk — `find`/`grep` for the path the `sources:` or
body names — and read it), quotes the real thing, and the bridge is refined or killed by
what the source says. This repeats across rounds until the survivor has met the code.

Why this is load-bearing: a live `research` run produced a confident epiphany — *"your
prime-Kt measurement contradicts your v2-lean result"* — that the converge round **and**
the skeptic both endorsed, because every entry said "isolated cost." It was **wrong**.
Opening the actual Lean (`TimeHierarchy.lean:128` → `Lnat n ≤ 1`) showed the v2 result is
about a single **bit**, not a multi-digit prime; the "contradiction" dissolved on contact
with the code. The summaries agreed; the source refuted. The skill is a **hypothesis
generator** — it proposes the bridge; the source decides.

Neurologically this is the **active-inference loop**: a candidate bridge is a *prediction*,
reading the source is *sampling the world*, and iterating until the prediction survives the
evidence is prediction-error minimisation. Settling on the prior (the remembered gist)
without sampling the world is exactly the failure mode to design against.

## The quality bar (this is the whole point)

Generating a plausible connection between any two notes is trivial — any model
will do it endlessly. The entire worth of this skill is the **filter**. Hold the
output to all four:

- **Novel** — not already linked, not already a `connection` entry.
- **Non-obvious** — a genuine far pair, not two siblings.
- **Useful** — changes a decision or explains something you didn't see.
- **Grounded in the source, not the summary** — every technical claim traces to the
  **artifact the entry points at** (the code/file/data it references), not the entry's
  own prose. Entries are *leads*, often vague; a bridge that only rhymes at the summary
  level dies on contact with the source. No confabulation.

If nothing clears the bar, **say so** and report the closest near-miss rather
than inventing a flashy-but-hollow link. A false epiphany is worse than none.

## Output format

```
💡 <the epiphany as one sharp sentence>

Bridges:
- [[region/entry-a]] — <what it actually says>
- [[region/entry-b]] — <what it actually says>

Shared structure: <the latent pattern both instances share>

Why it's non-obvious: <they live in unrelated domains / aren't linked>

So what: <the decision it changes or the thing it explains>
```

End by offering — not doing — persistence: *"Want this saved as a `connection`
entry? Say the word and I'll draft one."* This skill is **read-only**; never
write to the library unprompted.

## When NOT to use

- The user wants an answer to a *specific* question → `/ultan-advisor <question>`.
- The user wants to *fetch* a known entry → `ultan-search`.
- The user wants to *write* a memory → `/ultan <text>`.

## Dependencies

`inventory.py` is a stdlib-only filesystem walk — no daemon needed, so it works
even while the daemon is warming. Scouts use the `ultan-search` skill to read
full entries (that one does need the daemon; if it's still warming, scouts can
fall back to reading the `.md` files directly under `~/.agent-mem/knowledge/`).

For the **grounding** step, scouts follow an entry's pointers out to the referenced
**source artifacts** — the code/data repo an entry's `sources:` or body names (e.g. a
Lean or code project) — by locating it on disk (`find`/`grep` from `~`) and reading the
real files. The entry is the lead; the artifact is the evidence.
