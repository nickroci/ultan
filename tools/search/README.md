# agent-mem search

Three-mode search over the `agent-mem` knowledge store. Implements PLAN section 3:
hierarchy traversal, index-led LLM retrieval, and BM25 over article bodies — plus a
merged default that runs BM25 and index-led together and deduplicates hits by file path.

## Install

This package is uv-managed. From `tools/search/`:

```bash
uv venv
uv sync   # installs rank-bm25, claude-agent-sdk, pyyaml
```

Or run anything ad-hoc with `uv run` — it will resolve deps on demand:

```bash
uv run python cli.py search --bm25 "factory pattern"
```

## Knowledge dir resolution

In order of precedence:

1. `--knowledge-dir <path>` on the CLI.
2. `$AGENT_MEM_KNOWLEDGE` environment variable.
3. `~/.agent-mem/knowledge/` (the PLAN default).

If `~/.agent-mem/knowledge/` doesn't exist yet, point the CLI at the bundled fixtures:

```bash
uv run python cli.py --knowledge-dir fixtures/knowledge search --bm25 "factory"
```

## Modes

### Hierarchy (`--hierarchy [SUBPATH]`)

```bash
uv run python cli.py search --hierarchy
uv run python cli.py search --hierarchy global/concepts
```

Walks `<knowledge_dir>/<SUBPATH>` and prints every `.md` path. No LLM, no scoring.
`_archive/` is skipped. The path is refused if it escapes the knowledge dir.

### Index-led LLM retrieval (`--index <query>`)

```bash
uv run python cli.py search --index "how should I structure new APIs?"
```

Loads `<knowledge_dir>/index.md`, hands it + the query to the Claude Agent SDK with
`allowed_tools=["Read"]` and `system_prompt={"type":"preset","preset":"claude_code"}`,
and asks the model which entries are relevant. The model reads them and synthesizes an
answer; the CLI parses lines prefixed `SOURCE:` to surface cited paths.

Requires `claude-agent-sdk` and a working Claude Code auth (same auth as the rest of
the Claude Agent SDK uses).

### BM25 (`--bm25 <query>`)

```bash
uv run python cli.py search --bm25 "postgres database tests"
uv run python cli.py search --bm25 "paradigms"
uv run python cli.py search --bm25 "factory" --rebuild
```

Stock `rank_bm25.BM25Okapi` over tokenized article bodies. The index is persisted
to `<knowledge_dir>/../.bm25.idx` (i.e. `~/.agent-mem/.bm25.idx`) and rebuilt on
demand when any source `.md` has changed, been added, or been removed.

Output: top-k entries with score and a one-line snippet centered on the first
query-token hit in the body.

### Merged default (no flag)

```bash
uv run python cli.py search "what paradigms do we use for APIs?"
```

Runs BM25 + index-led retrieval in parallel and merges results, deduplicated by
absolute file path. One row per file, tagged with the mode(s) that surfaced it
(`bm25`, `index`, or `bm25+index`). Entries surfaced by more modes rank higher;
ties broken by BM25 score. The LLM synthesis is printed underneath.

Hierarchy mode is intentionally not part of the merged default — it's a browsing
tool, not a query tool.

If BM25 returns nothing AND the index retrieval returns nothing, the CLI says so
explicitly; it does not fabricate.

## Tokenization choices (v1)

These choices are pinned here so the PLAN can adopt or override them:

- **YAML frontmatter** is stripped before tokenization, except:
  - `keywords:` values are appended to the searchable text (list or scalar).
  - `applies-when:` values are appended (block scalar or list).
  - Everything else in the frontmatter (`id`, `status`, `confidence`, `sources`, etc.)
    is dropped — it's metadata, not content.
- **Code fences are kept verbatim.** Identifiers like `PaymentsClientFactory.create`
  are valuable search terms. We do not strip ```` ``` ```` markers.
- **Lowercase everything.**
- **Split on `[^a-z0-9]+`.** No language-specific stemming. No stopword removal.
  At this scale, the index is small enough that stopwords don't hurt; stemming is
  worth less than the risk of dropping a legitimate keyword.
- **Drop tokens shorter than 2 characters.** Single-letter tokens are noise.
- **`_archive/`** subdirectories are excluded from the index entirely.
- **Top-level `index.md` and `log.md` are excluded from BM25.** They're catalogs
  that mention nearly every entry; including them collapses IDF for any term
  that appears as an entry name (BM25Okapi clamps IDF to zero when a term is in
  >= half the documents). The catalog is what `--index` mode consumes directly,
  so BM25 doesn't need to see it.

If the PLAN wants stemming or stopwords later, they go here in `tokenize()`.

## Semantic search (`embeddings.py`)

BM25 finds keyword overlap. It misses paraphrased matches — *"I'm about to deploy"*
doesn't surface *"never push to prod without approval"* because they share no
tokens beyond maybe "deploy". `embeddings.py` is a parallel index, mirroring
`bm25.BM25Index`'s shape, that does dense retrieval over the same corpus.

It is **infrastructure only** for now — not wired into the CLI's `search`
subcommands. Integration (BM25 + embeddings fusion / RRF / etc.) is a separate
step. Until then the CLI remains pure BM25.

### Install

`uv sync` from `tools/search/` picks up `sentence-transformers` and `numpy`
automatically. First use of the module downloads
`sentence-transformers/all-MiniLM-L6-v2` (~80 MB) from HuggingFace into
`~/.cache/huggingface/`; subsequent runs load from disk in ~1 second.

### Basic usage

```python
from pathlib import Path
from embeddings import load_or_build

index = load_or_build(Path("~/.agent-mem/knowledge").expanduser())
for hit in index.search("managing python dependencies", k=5):
    print(f"{hit.score:.3f}  {hit.path.name}")
    print(f"        {hit.snippet}")
```

The first call builds the index and persists it to
`<knowledge_dir>/../.embeddings.idx`; later calls reload from disk if every
tracked file is unchanged. Force a rebuild with `load_or_build(..., force_rebuild=True)`.

### How it differs from BM25

| | BM25 | embeddings |
|---|------|-----------|
| Match basis | exact token overlap | semantic similarity (cosine over 384-dim vectors) |
| Strength | precise — exact phrase / identifier hits | recall — finds paraphrases, related concepts |
| Weakness | misses paraphrased queries | weaker on novel proper nouns, identifiers |
| Cost (warm) | sub-ms | ~4 ms per query, single thread, CPU |
| Persistence | `.bm25.idx` | `.embeddings.idx` (pickled numpy array) |
| Cold start | none | ~5 s to load the model the first time per process |

Same file-selection rules: `_archive/` and the top-level `index.md` / `log.md`
catalogs are excluded. Frontmatter is stripped except `keywords:` and
`applies-when:`, which are prepended to the embedded text (matches
`bm25.tokenize`).

### Notes

- **Pure CPU.** The module does not probe for GPUs; `all-MiniLM-L6-v2` is fast
  enough on CPU for personal-corpus scale.
- **Pickle is not portable across Python versions.** Delete `.embeddings.idx`
  after an interpreter upgrade. `load_or_build` swallows unpickle errors and
  rebuilds, so this is self-healing.
- **Model cache is module-level**, keyed on model name. Multiple
  `EmbeddingIndex` instances in the same process share one loaded model.

## Files

```
tools/search/
  pyproject.toml          # uv-managed; depends on rank-bm25, claude-agent-sdk, pyyaml,
                          #   sentence-transformers, numpy
  bm25.py                 # BM25 indexer + searcher
  embeddings.py           # sentence-transformer indexer + searcher (parallel module)
  cli.py                  # `agent-mem search` entry point, three modes + merged default
  test_bm25.py            # BM25 sanity tests
  test_embeddings.py      # embeddings sanity tests (first run downloads the model)
  fixtures/knowledge/     # 5 dummy entries + index.md, schema-conformant per PLAN section 2
  README.md
```

## Running tests

```bash
cd tools/search
uv run python test_bm25.py        # plain runner, exits non-zero on failure
# or
uv run python -m pytest test_bm25.py -v
```

The tests don't touch the Claude Agent SDK — they only exercise BM25 + tokenization.

## Notes / limitations

- Pickle persistence is fine for v1 but is not portable across Python versions.
  Delete `~/.agent-mem/.bm25.idx` after an interpreter upgrade. `load_or_build`
  swallows unpickle errors and rebuilds, so this is self-healing.
- The merged default invokes the Claude Agent SDK every time. If you just want
  keyword hits with no LLM cost, use `--bm25` explicitly.
- The `agent-mem` console script in `pyproject.toml` only exposes the `search`
  subcommand for now. `review`, `promote`, `doctor` etc. (PLAN section 1) are
  out of scope for this package.
