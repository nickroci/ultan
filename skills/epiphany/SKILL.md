---
name: epiphany
description: Roam the Ultan memory library and surface ONE non-obvious, useful connection between distant entries — an on-demand "epiphany". Use when the user asks for a spark/insight/epiphany from their memory, wants a hidden cross-cutting pattern surfaced, asks "show me something interesting in my notes", or runs /epiphany. Can be scoped to a single project (e.g. /epiphany vol-predictor) to surface a cross-subsystem insight about that project. Read-only — it never writes to the library. Works by fanning out parallel scouts (Opus) across the knowledge base, then adversarially filtering to the single best connection.
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

> Here is the full pool of candidates and entries the other scouts found. Do three
> things: **(1) build** — find a stronger connection bridging ACROSS two *different*
> scouts' material that none of them saw alone; **(2) challenge** — name the weakest
> candidate in the pool and why it's coincidental, already-linked, or hollow;
> **(3) vote** — your single best with one line of why. Do NOT just agree: if the
> pool is converging on something shallow, say so and pull it apart.

Assign **one scout a standing skeptic role** — its job is to refute, never to
ratify. This is the *lateral inhibition* that stops the group converging on a
confident-but-wrong answer (see the convergence note below). Repeat until **stable**:
the same 1–2 candidates win the vote across a round with no new far pair appearing.
Cap at ~3 convergence rounds. The goal is convergence by **evidence survival**, not
by agreement.

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
- **Confabulated** → spot-check the `evidence` against the real entries. If the entry
  doesn't actually say it, kill it — a fluent link the text doesn't support is the
  most common failure.
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

And discount convergence that lands on an **already-cross-linked cluster** — that is
the corpus's existing link structure reflected back, not N independent discoveries.
The richest epiphany is often a *tension* (two of the user's own entries that quietly
contradict), not a restatement the scouts agree on; a live `research` run found exactly
this — the agreed-upon cluster was pre-wired, while the real find was an unlinked
contradiction the skeptic had to dig out.

## The quality bar (this is the whole point)

Generating a plausible connection between any two notes is trivial — any model
will do it endlessly. The entire worth of this skill is the **filter**. Hold the
output to all four:

- **Novel** — not already linked, not already a `connection` entry.
- **Non-obvious** — a genuine far pair, not two siblings.
- **Useful** — changes a decision or explains something you didn't see.
- **Grounded** — every claim traces to actual entry text. No confabulation.

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
