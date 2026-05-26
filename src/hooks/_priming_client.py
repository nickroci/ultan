"""Thin priming client for the UserPromptSubmit hook.

The hook runs as a fresh Python process every turn. Loading the
sentence-transformer model in-process is ~5s; loading rank-bm25 + a
fresh BM25 index is several hundred ms. Neither fits the hook's
sub-200 ms wall-clock budget.

This module gives the hook two paths, in order of preference:

  1. **Daemon socket (target <100ms).** Connect to the daemon's
     ``priming.sock`` and ship the user's prompt over. The daemon
     reuses ``priming._hybrid_search`` (BM25 + embeddings via RRF)
     and ``priming._assemble_output`` — same rendered markdown shape
     the agent used to read from ``hot-context.md``. Total budget
     200 ms (connect + send + recv).

  2. **In-hook lexical fallback (target <100ms).** When the socket is
     missing or unreachable, do a pure-stdlib token-overlap scan over
     the knowledge tree. Crude — no IDF, no embeddings — but it
     surfaces *something* keyed on the user's prompt rather than
     leaving the agent flying blind. Skips ``_archive`` and the
     ``index.md`` / ``log.md`` catalogs the daemon's BM25 also skips.

Both paths return either rendered markdown or the empty string. The
function never raises. The hook treats empty as "no priming to inject"
and moves on to the nudge pipeline.

We deliberately don't import the daemon's ``priming`` module — the
hook lives in a separate uv project root (``src/``) with no path dep
on the daemon. Mirroring the renderer here keeps the boundary clean.
"""

from __future__ import annotations

import json
import os
import re
import socket
import struct
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict, Union, cast

# Frontmatter values can be scalar (str), list[str], or absent.
FrontmatterValue = Union[str, List[str]]
Frontmatter = dict[str, FrontmatterValue]
ScoredEntry = Tuple[Path, float, int, Frontmatter]


class PrimingRequest(TypedDict):
    """Payload sent over the Unix socket to the daemon's priming RPC."""

    op: str
    prompt: str
    project_slug: Optional[str]
    k: int
    char_budget: int


class PrimingResponse(TypedDict, total=False):
    """Daemon RPC reply shape.

    Mirrors the on-the-wire contract documented in
    ``daemon/agent_mem_daemon/priming_rpc.py``: success carries
    ``priming_md`` + ``took_ms`` + ``lane``; failure carries ``error``.
    """

    ok: bool
    priming_md: str
    took_ms: int
    lane: str
    error: str


# ── Tunables ─────────────────────────────────────────────────────────


# Total hook-side budget (connect + send + recv). The daemon's
# SERVER_REQUEST_TIMEOUT_S sits below this. 2s accommodates the
# cross-encoder rerank stage on slower machines — warm steady-state is
# ~300ms on Apple Silicon, but CPU-only or older hardware can push that
# higher and we'd rather wait than skip the precision lift.
_TOTAL_BUDGET_MS = 2000

# Length-prefix header size; must match daemon's priming_rpc.
_LEN_HEADER = 4
_MAX_BODY_BYTES = 1 << 20  # 1 MiB sanity cap, mirrors server side


# ── Path resolution (kept local; can't import daemon) ────────────────


def _agent_mem_home() -> Path:
    override = os.environ.get("AGENT_MEM_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".agent-mem"


def _priming_socket_path() -> Path:
    return _agent_mem_home() / "priming.sock"


def _knowledge_dir() -> Path:
    return _agent_mem_home() / "knowledge"


# ── Wire helpers ─────────────────────────────────────────────────────


def _parse_response(buf: bytes) -> Optional[PrimingResponse]:
    """Decode the response body. Returns ``None`` on any decode error
    or if the payload isn't a JSON object.
    """
    try:
        parsed: Any = json.loads(buf.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return cast("PrimingResponse", parsed)


def _send_request(
    socket_path: Path,
    request: PrimingRequest,
    *,
    total_budget_ms: int,
) -> Optional[PrimingResponse]:
    """One-shot length-prefixed JSON exchange. Returns parsed response
    or ``None`` on any failure (timeout, refused, closed, bad JSON).

    The total budget is split between connect, send and recv —
    ``settimeout`` on the socket bounds each blocking syscall;
    ``deadline`` bounds the cumulative wall time so a slow handshake
    can't burn through the whole budget before we even send the body.
    """
    body = json.dumps(request).encode("utf-8")
    deadline = time.monotonic() + total_budget_ms / 1000.0

    def _remaining() -> float:
        return max(0.001, deadline - time.monotonic())

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(_remaining())
        sock.connect(str(socket_path))

        sock.settimeout(_remaining())
        sock.sendall(struct.pack(">I", len(body)) + body)

        # Header
        header = b""
        while len(header) < _LEN_HEADER:
            sock.settimeout(_remaining())
            chunk = sock.recv(_LEN_HEADER - len(header))
            if not chunk:
                return None
            header += chunk
        (length,) = struct.unpack(">I", header)
        if length == 0 or length > _MAX_BODY_BYTES:
            return None

        # Body
        buf = b""
        while len(buf) < length:
            sock.settimeout(_remaining())
            chunk = sock.recv(length - len(buf))
            if not chunk:
                return None
            buf += chunk
        return _parse_response(buf)
    except (
        FileNotFoundError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
        socket.timeout,
        OSError,
    ):
        return None
    finally:
        try:
            sock.close()
        except OSError:
            pass


# ── In-hook lexical fallback (stdlib-only) ───────────────────────────


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


_HEADER = "## Ultan — your library says (cite or follow when applicable)"
_FOOTER = (
    "*Wikilinks resolve to real entries. Use the `ultan-search` skill to read one "
    "(returns content + sibling entries + subfolders + parent README so you can traverse), "
    "or `/ultan-advisor <question>` to have Sonnet + Opus intelligently synthesise "
    "across multiple entries.*"
)


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(text.lower()) if len(t) >= 2]


def _parse_yaml_lite(fm_block: str) -> Frontmatter:
    """Just enough YAML to pull ``reinforced``, ``title``, ``applies-when``,
    ``keywords``. Pure stdlib — no PyYAML dep.

    Handles:
      - Simple ``key: value`` lines
      - Inline arrays: ``keywords: [a, b, c]``
      - Block-scalar ``applies-when: |`` followed by indented lines
    """
    out: Frontmatter = {}
    lines = fm_block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, rest = m.group(1), m.group(2)
        rest = rest.rstrip()
        if rest == "|" or rest == ">":
            # Block scalar: collect subsequent indented lines.
            i += 1
            block: List[str] = []
            while i < len(lines) and (lines[i].startswith("  ") or lines[i].strip() == ""):
                block.append(lines[i][2:] if lines[i].startswith("  ") else "")
                i += 1
            out[key] = "\n".join(block).rstrip()
            continue
        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1]
            out[key] = [s.strip().strip('"').strip("'") for s in inner.split(",") if s.strip()]
        else:
            out[key] = rest.strip().strip('"').strip("'")
        i += 1
    return out


def _iter_markdown(knowledge_dir: Path) -> List[Path]:
    """Same selection rules as ``bm25._iter_markdown``: skip ``_archive``
    subtrees and the top-level ``index.md`` / ``log.md`` catalogs."""
    files: List[Path] = []
    catalog_names = {"index.md", "log.md", "README.md"}
    for p in sorted(knowledge_dir.rglob("*.md")):
        if "_archive" in p.parts:
            continue
        if p.parent == knowledge_dir and p.name in catalog_names:
            continue
        if p.name == "README.md":
            # Folder READMEs are catalog-like; skip to avoid IDF dilution.
            continue
        files.append(p)
    return files


def _score_doc(doc_tokens: set[str], query_tokens: List[str]) -> float:
    """Tiny lexical score: count of query-token hits, weighted by token
    rarity in the query (de-duplicate first). No IDF — the corpus is
    small and IDF would require a corpus scan we can't afford."""
    if not doc_tokens or not query_tokens:
        return 0.0
    # De-duplicate query tokens to avoid double-counting repetition.
    qset = set(query_tokens)
    hits = qset & doc_tokens
    if not hits:
        return 0.0
    return float(len(hits))


def _wikilink(entry: Path, knowledge_dir: Path) -> str:
    try:
        rel = entry.resolve().relative_to(knowledge_dir.resolve())
    except ValueError:
        rel = entry
    s = str(rel).replace(os.sep, "/")
    if s.endswith(".md"):
        s = s[:-3]
    return s


def _shorten(text: str, max_chars: int = 80) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _summary(fm: Frontmatter, path: Path) -> str:
    aw: Optional[FrontmatterValue] = fm.get("applies-when") or fm.get("applies_when")
    if isinstance(aw, list):
        for item in aw:
            s = item.strip()
            if s:
                return _shorten(s)
    elif isinstance(aw, str):
        for line in aw.splitlines():
            s = line.strip()
            if s:
                return _shorten(s)
    title: Optional[FrontmatterValue] = fm.get("title")
    if isinstance(title, str) and title.strip():
        return _shorten(title.strip())
    return path.stem.replace("-", " ")


def _split_frontmatter_body(text: str) -> tuple[Frontmatter, str]:
    """Return ``(frontmatter_dict, body_text)``; empty fm if none present."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    return _parse_yaml_lite(m.group(1)), text[m.end() :]


def _doc_token_bag(body: str, fm: Frontmatter) -> set[str]:
    """Tokenize body + search-relevant frontmatter fields into one bag.

    Mirrors bm25's frontmatter extraction so the lexical fallback ranks
    entries the same way the daemon would.
    """
    tokens: set[str] = set(_tokenize(body))
    kw: Optional[FrontmatterValue] = fm.get("keywords")
    if isinstance(kw, list):
        for w in kw:
            tokens.update(_tokenize(w))
    aw: Optional[FrontmatterValue] = fm.get("applies-when") or fm.get("applies_when")
    if isinstance(aw, str):
        tokens.update(_tokenize(aw))
    elif isinstance(aw, list):
        for w in aw:
            tokens.update(_tokenize(w))
    return tokens


def _reinforced_count(fm: Frontmatter) -> int:
    """Read the ``reinforced`` frontmatter int; clamp at 0 on bad input."""
    raw: Optional[FrontmatterValue] = fm.get("reinforced")
    if not isinstance(raw, str):
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def _rank_entries(
    files: List[Path],
    q_tokens: List[str],
) -> List[ScoredEntry]:
    """Score and sort the knowledge files against the query tokens.

    Returns ``(path, boosted_score, reinforced, frontmatter)`` tuples
    sorted by score desc, ties broken by path for stability.
    """
    scored: List[ScoredEntry] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        fm, body = _split_frontmatter_body(text)
        score = _score_doc(_doc_token_bag(body, fm), q_tokens)
        if score <= 0:
            continue
        reinforced = _reinforced_count(fm)
        # Boost matches the daemon's _boost_with_reinforcement (×0.5).
        scored.append((p, score + reinforced * 0.5, reinforced, fm))

    scored.sort(key=lambda t: (-t[1], str(t[0])))
    return scored


def _render_bullets(
    top: List[ScoredEntry],
    kdir: Path,
) -> List[str]:
    """Format the top-ranked entries as markdown bullets."""
    bullets: List[str] = []
    for path, _score, reinforced, fm in top:
        link = _wikilink(path, kdir)
        cnt = f" (×{reinforced})" if reinforced > 0 else ""
        summary = _summary(fm, path)
        if summary:
            bullets.append(f"- [[{link}]]{cnt} — {summary}")
        else:
            bullets.append(f"- [[{link}]]{cnt}")
    return bullets


def _assemble(bullets: List[str]) -> str:
    """Wrap bullets in the standard header / footer block."""
    return f"{_HEADER}\n\n" + "\n".join(bullets) + f"\n\n{_FOOTER}\n"


def _fit_to_budget(bullets: List[str], char_budget: int) -> str:
    """Assemble; drop trailing bullets until the rendered string fits."""
    full = _assemble(bullets)
    if len(full) <= char_budget:
        return full
    # Drop trailing bullets until we fit. Same strategy as the daemon's
    # ``priming._assemble_output`` — keep the framing, trim content.
    while len(bullets) > 1:
        bullets.pop()
        candidate = _assemble(bullets)
        if len(candidate) <= char_budget:
            return candidate
    return _assemble(bullets)


def _local_priming(
    prompt: str,
    *,
    k: int,
    char_budget: int,
    knowledge_dir: Optional[Path] = None,
) -> str:
    """Pure-stdlib token-overlap rank. Returns rendered markdown or ""."""
    kdir = knowledge_dir or _knowledge_dir()
    if not kdir.exists():
        return ""

    q_tokens = _tokenize(prompt)
    if not q_tokens:
        return ""

    files = _iter_markdown(kdir)
    if not files:
        return ""

    scored = _rank_entries(files, q_tokens)
    if not scored:
        return ""

    bullets = _render_bullets(scored[:k], kdir)
    return _fit_to_budget(bullets, char_budget)


# ── Public API ───────────────────────────────────────────────────────


def get_priming(
    prompt: str,
    *,
    project_slug: Optional[str] = None,
    k: int = 5,
    char_budget: int = 1500,
    total_budget_ms: int = _TOTAL_BUDGET_MS,
) -> str:
    """Return rendered priming markdown, or empty string on any failure.

    Order of operations:
      1. Try the daemon socket (target <100 ms; hard cap
         ``total_budget_ms``).
      2. On any failure (socket missing, connect refused, timeout, bad
         JSON, ``ok: false`` response): fall back to inline lexical
         search.
      3. On both failures: return empty string. Hook still emits the
         nudge half normally; this function never raises.

    The non-priming happy path (no library on disk, empty prompt) should
    return in well under 50 ms so it doesn't push the hook past its
    overall budget.
    """
    # Defensive: even though the signature pins ``str``, callers in the
    # wild (and our own tests via ``# type: ignore``) can pass ``None``
    # or numbers. Coerce to "" rather than crash the hook.
    runtime_prompt: Any = prompt
    if not isinstance(runtime_prompt, str) or not runtime_prompt.strip():
        return ""

    socket_path = _priming_socket_path()
    if socket_path.exists():
        request: PrimingRequest = {
            "op": "priming",
            "prompt": prompt,
            "project_slug": project_slug,
            "k": k,
            "char_budget": char_budget,
        }
        resp = _send_request(socket_path, request, total_budget_ms=total_budget_ms)
        if resp is not None and resp.get("ok"):
            priming_md = resp.get("priming_md")
            if isinstance(priming_md, str):
                return priming_md
        # Daemon answered but with ok=false, OR transport failed. Fall through
        # to the local path so the agent gets something rather than nothing.

    return _local_priming(prompt, k=k, char_budget=char_budget)
