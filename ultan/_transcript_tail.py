"""Transcript-tail backfill for the priming query.

Hook-hot-path safe: stdlib plus ``stopwordsiso`` (a data-only stopword
list, no transitive deps, ~27ms import+load). No ML imports.

The priming retrieval query used to be the user's prompt, verbatim. That
fails exactly when priming would matter most: confirmation turns ("ok",
"go ahead", "continue") carry no topical tokens — the signal lives in
the ASSISTANT's preceding message ("I'll pip install into the system
Python…"). BM25 then matches the bare token against arbitrary entry
text, which is where the feedback log's junk firings came from.

Fix: when the prompt is LOW-SIGNAL, append the last
``TAIL_MAX_CHARS`` characters of recent conversation text to the query.
Measured on the live 919-entry library (2026-07-03 sweep): backfilled
low-signal turns went from 3 scope-leaked junk firings / 1-of-6 target
hits to 0 junk / 3-of-6 hits, with 500 chars of tail already sufficient.

The gate is load-bearing. The same sweep showed that backfilling a RICH
prompt is actively harmful: appending unrelated chatter to "check why we
see no collateral timings for ZOREP" silenced a previously perfect
firing (the cross-encoder co-attends over the whole query, and the
extra topic drowned the real one). Rich prompts keep the verbatim query.

Cognitive analog: encoding specificity (Tulving & Thomson 1973) — recall
succeeds when the retrieval cue matches what was encoded. A bare "ok"
is a cue for nothing; the conversation context IS the cue.

Extraction rules (shapes verified against real Claude Code transcripts):
  - ``type == "user"`` entries with STRING content: the user's actual
    prompts. List-content user entries are tool results — skipped.
  - ``type == "assistant"`` entries: ``text`` blocks only. ``thinking``
    and ``tool_use`` blocks are model-internal — skipped.
  - Sidechain (subagent) entries and hook-injected attachment/system
    entries are skipped; ``<system-reminder>`` spans are stripped as
    defence-in-depth so prior priming output can never feed back into
    the retrieval query (a self-reinforcing echo).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional, cast

import stopwordsiso

# Tail budget appended to a low-signal prompt. The 2026-07-03 sweep showed
# 500 chars already flips every tested low-signal miss; 1500 gave identical
# results and covers longer assistant turns. Beyond that only dilutes BM25.
TAIL_MAX_CHARS = 1500

# Read at most this much of the transcript file. Transcripts grow to many
# MB; the tail we need lives in the last few entries. 256 KiB comfortably
# covers TAIL_MAX_CHARS of extracted prose even with heavy tool traffic
# between text blocks, at negligible read cost on the hook hot path.
_READ_BACK_BYTES = 262_144

# A prompt with fewer content tokens than this is "low-signal" — too little
# for retrieval to key on, so the transcript tail is appended. Calibrated on
# the sweep's cases against the stopwords-iso list: "ok" (0), "go ahead" (0),
# "continue" (0), "this is meant to be a one pager" (2) all gate in; "check
# why we see no collateral timings for ZOREP" (4) and real work prompts stay
# verbatim. NOTE: this threshold and the stopword list are calibrated as a
# PAIR — the iso list is aggressive (1298 words; it stops "one", "why",
# "fix"), which is exactly why the threshold sits at 3 rather than 4.
_LOW_SIGNAL_CONTENT_TOKENS = 3

# General-English function words come from the maintained stopwords-iso
# dataset (data-only package, no transitive deps, ~27ms import+load —
# hook-budget safe). IDF over the 919-entry library CANNOT do this job:
# conversational filler is rare in terse technical entries, so BM25 scores
# "ok" at 5.6 and "go ahead" above "did ultan work?" (measured 2026-07-03).
# Dialogue-filler-ness is a fact about conversational English, not about
# the corpus — hence an external list.
_STOPWORDS: frozenset[str] = frozenset(stopwordsiso.stopwords("en"))

# Residual confirmation vocabulary the general-English list lacks: chat-isms
# that dominate the low-signal turns priming used to miss. Deliberately
# ONLY dialogue tokens — anything general-English belongs upstream in
# stopwords-iso (contribute there rather than growing this set).
_FILLER_TOKENS = frozenset(
    {
        "continue",
        "proceed",
        "sounds",
        "yeah",
        "yep",
        "yup",
        "pls",
        "ty",
        "thx",
        "kk",
        "lgtm",
    }
)

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def is_low_signal(prompt: str) -> bool:
    """True when the prompt carries too few content tokens to retrieve on."""
    tokens = [t for t in _TOKEN_SPLIT_RE.split(prompt.lower()) if len(t) >= 2]
    content = [t for t in tokens if t not in _STOPWORDS and t not in _FILLER_TOKENS]
    return len(content) < _LOW_SIGNAL_CONTENT_TOKENS


def _entry_text(entry: dict[str, Any]) -> str:
    """Extract conversational prose from one transcript entry, or ""."""
    if entry.get("isSidechain"):
        return ""
    etype = entry.get("type")
    if etype not in ("user", "assistant"):
        return ""
    message_obj: Any = entry.get("message")
    if not isinstance(message_obj, dict):
        return ""
    message = cast("dict[str, Any]", message_obj)
    content: Any = message.get("content")
    if etype == "user":
        # String content = the user's actual prompt. List content on a
        # user-type entry is tool_result traffic — no prose, skip.
        return content if isinstance(content, str) else ""
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block_obj in cast("list[object]", content):
        if not isinstance(block_obj, dict):
            continue
        block = cast("dict[str, Any]", block_obj)
        if block.get("type") == "text":
            text: Any = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
    return "\n".join(parts)


def _prose_chunks(lines: List[str]) -> List[str]:
    """Conversational prose per transcript line; skips everything else."""
    chunks: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        text = _entry_text(cast("dict[str, Any]", parsed))
        if not text:
            continue
        text = _SYSTEM_REMINDER_RE.sub(" ", text)
        text = " ".join(text.split())
        if text:
            chunks.append(text)
    return chunks


def transcript_tail(transcript_path: Path, *, max_chars: int = TAIL_MAX_CHARS) -> str:
    """Last ``max_chars`` of conversational text from the transcript.

    Never raises; returns "" on any failure (missing file, bad JSON,
    unexpected shapes) — the caller degrades to the verbatim prompt.
    """
    if max_chars <= 0:
        return ""
    try:
        with transcript_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - _READ_BACK_BYTES))
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""

    lines = raw.splitlines()
    if len(raw) == _READ_BACK_BYTES:
        # Started mid-file: the first line is almost certainly truncated.
        lines = lines[1:]

    chunks = _prose_chunks(lines)
    if not chunks:
        return ""
    joined = "\n".join(chunks)
    return joined[-max_chars:]


def build_priming_query(prompt: str, transcript_path: Optional[str]) -> str:
    """The retrieval query for this turn: verbatim prompt, or prompt plus
    transcript tail when the prompt alone is too thin to retrieve on.

    The prompt goes LAST so recency reads naturally and the fallback
    lexical lane (pure token overlap) still sees every prompt token.
    """
    if not is_low_signal(prompt):
        return prompt
    if not transcript_path:
        return prompt
    tail = transcript_tail(Path(transcript_path))
    if not tail:
        return prompt
    return f"{tail}\n{prompt}"
