# Curator agent evals

Behavioural evals for the curator agents (Librarian / Scholar). They are
**pytest tests**, but not unit tests: each case seeds a throwaway knowledge
library (a copy of the corpus below), drives a *real* agent through the Claude
Code **subscription** (`claude_agent_sdk` — never the metered API), and asserts
the agent does the obviously-right thing.

They live under `daemon/evals/`, **outside** `testpaths = ["tests"]`, so a
normal `pytest` run never collects them (every case spends subscription quota
and takes ~1 minute).

## Running

```sh
cd daemon
uv run --frozen pytest evals/ --no-cov              # run all
uv run --frozen pytest evals/ --no-cov -k dedupe    # one case
uv run --frozen pytest evals/ --co -q               # list cases, no agent calls
```

`--no-cov` is required: the daemon's 90%-coverage gate (in `addopts`) would
otherwise fail an eval-only run.

- A **wrong answer** is a failed assertion → non-zero exit (blocks a push).
- An **infra hiccup** (timeout / transport / not authenticated) is a
  `pytest.skip` → does not block. Cost + latency are shown in the failure
  message when an assertion fails.
- `ULTAN_SKIP_EVALS=1` skips the whole module; if `claude_agent_sdk` isn't
  importable the module skips itself.

## Why local + subscription only, never CI

CI would need the metered Anthropic API (a real bill); the subscription only
exists on a developer machine. So these run **locally**, as a **pre-push** hook —
never in GitHub Actions. (Cost shown is the subscription-equivalent the SDK
reports, not a metered charge.)

## The corpus

`corpus/knowledge/` is a synthetic **cooking** library — fictional toy data, so
a fixture can never overlap with (or be mistaken for) a real knowledge base. The
harness copies it into a temp dir per run; edit the markdown to change what the
agent sees. The agent is hard-scoped to that temp copy: its research tools are
rooted there and refuse to read outside it, and `run_typed` gives the curator
`setting_sources=[]` (no Read/Bash/filesystem tools), so it can never reach the
user's real `~/.agent-mem`.

## The cases

Deliberately the most clear-cut situations, so a non-deterministic model doesn't
flake the gate. Each asserts the single behaviour that matters and tolerates
incidental variation:

| Case | Situation | Right answer |
|---|---|---|
| `dedupe_pasta_salt` | User restates a rule already in the corpus (salt pasta water) | **No** new entry — recognise the duplicate (negative assertion: robust) |
| `novel_cpp_for_ai` | User states a novel **in-domain** fact (a new C++ for AI), conversationally | Propose a `write_entry` — a genuine surprise/salience test, not a forced write |

They're mutually constraining: the first fails an agent that over-writes, the
second fails one that never writes.

> `novel_cpp_for_ai` is *conversational* (not a forced `/ultan` write) and still
> reliable because the fact is clearly **in-domain** — it fired 5/5 in
> measurement. A single conversational mention of *out-of-domain* trivia is a
> poorer gate (it missed 1/6), so judgment-at-the-margin cases belong in
> repeated-sample experiments (e.g. the `pytest-repeat` plugin's `--count`), not
> gated checks.

Adding a case: append an `EvalCase` to `CASES` in `cases.py` (exchanges + a
tolerant `check` predicate); add any new seed entries as markdown under
`corpus/knowledge/`. The harness handles seeding, the agent call, and `pytest`
handles collection/reporting.

## Pre-push hook

Wired in the repo-root `.pre-commit-config.yaml` as `ultan-agent-evals`,
`stages: [pre-push]`, running `pytest evals/ --no-cov`, gated to the files that
define curator behaviour (the prompts, `_schemas.py`, the agent runners, and
`evals/` itself) — so an unrelated push never triggers them.

- Skip once: `SKIP=ultan-agent-evals git push`
- Skip always: `ULTAN_SKIP_EVALS=1 git push`
