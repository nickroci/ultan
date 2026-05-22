"""Run logging — JSONL audit trail + full per-invocation transcripts + cost tracking.

Every Librarian / Scholar invocation produces three artefacts:

1. **One line in `~/.agent-mem/runs/<YYYY-MM-DD>.jsonl`** — structured record
   suitable for `jq` queries. Always written, even on failure.
2. **`~/.agent-mem/runs/<ts>-<role>-<session_id>.md`** — the full prompt that
   was sent and the full text response that came back. Verbose, no truncation,
   for post-mortem debugging.
3. **An update to `~/.agent-mem/cost.json`** — running today / lifetime cost
   totals (informational only — there is no enforcement cap; see the
   removed ``daily_cap_usd``/``over_daily_cap`` for history).

Rotation: per-invocation `.md` transcripts older than 7 days are deleted on
each call (best-effort sweep). JSONL files are never rotated — they're one
per day, small, useful as an audit trail.

The daemon also writes high-level activity to `~/.agent-mem/daemon.log` via
the existing logging_setup pipeline. That's the "live tail" surface; this
module is the audit / replay surface.
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

from .paths import home

log = logging.getLogger("agent_mem_daemon.runs")


# How long per-invocation transcripts live before sweep deletes them.
TRANSCRIPT_RETENTION_DAYS = 7

# Cap on `output_raw` in the JSONL line (full text still lives in the .md).
JSONL_OUTPUT_RAW_MAX_BYTES = 8 * 1024  # 8 KiB


def runs_dir() -> Path:
    return home() / "runs"


def cost_file() -> Path:
    return home() / "cost.json"


def ensure_runs_dir() -> Path:
    d = runs_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Cost tracking ──────────────────────────────────────────────────────


def _today_iso() -> str:
    return date.today().isoformat()


def load_cost_state() -> Dict[str, Any]:
    """Read ~/.agent-mem/cost.json. Returns a defaulted dict on any error.

    Schema:
        {"today": "YYYY-MM-DD",
         "today_usd": float,
         "lifetime_usd": float}
    """
    path = cost_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"today": _today_iso(), "today_usd": 0.0, "lifetime_usd": 0.0}
    # json.loads returns Any; if the file isn't a JSON object treat it as
    # corrupt and reset rather than propagating a stranger shape downstream.
    if not isinstance(raw, dict):
        return {"today": _today_iso(), "today_usd": 0.0, "lifetime_usd": 0.0}
    data: Dict[str, Any] = cast("Dict[str, Any]", raw)
    today = _today_iso()
    if data.get("today") != today:
        # Day rolled over — reset today_usd; keep lifetime.
        fresh: Dict[str, Any] = {
            "today": today,
            "today_usd": 0.0,
            "lifetime_usd": float(data.get("lifetime_usd", 0.0) or 0.0),
        }
        data = fresh
    # Coerce types defensively.
    data.setdefault("today", today)
    data["today_usd"] = float(data.get("today_usd", 0.0) or 0.0)
    data["lifetime_usd"] = float(data.get("lifetime_usd", 0.0) or 0.0)
    return data


def save_cost_state(state: Dict[str, Any]) -> None:
    path = cost_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.warning("could not write cost file %s: %s", path, e)


def add_cost(usd: float) -> Dict[str, Any]:
    """Append `usd` to today_usd and lifetime_usd. Returns the new state."""
    state = load_cost_state()
    state["today_usd"] = float(state["today_usd"]) + float(usd or 0.0)
    state["lifetime_usd"] = float(state["lifetime_usd"]) + float(usd or 0.0)
    save_cost_state(state)
    return state


# Note: the daily cost cap (``daily_cap_usd`` / ``over_daily_cap`` /
# ``DEFAULT_DAILY_USD_CAP`` / ``AGENT_MEM_DAILY_USD_CAP``) was removed in
# 2026-05; tracking stays, enforcement is gone. The Scholar always runs.


# ── Sanitisation for filenames ─────────────────────────────────────────


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(token: str, max_len: int = 64) -> str:
    safe = _FILENAME_SAFE.sub("-", token or "unknown").strip("-")
    if not safe:
        safe = "unknown"
    return safe[:max_len]


# ── Per-invocation record ──────────────────────────────────────────────


@dataclass
class InvocationRecord:
    """All telemetry for one Librarian/Scholar call.

    Constructed at the call site, mutated during the call, persisted by
    ``finalise()``.
    """

    role: str  # "librarian" | "scholar"
    session_id: str
    started_at: float = field(default_factory=time.time)
    input_prompt: str = ""
    input_buffer_turns: int = 0
    output_raw: str = ""
    parsed_ok: bool = False
    decisions: Dict[str, int] = field(default_factory=dict[str, int])
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: Optional[str] = None
    transcript_path: Optional[Path] = None

    def mark_error(self, exc: BaseException) -> None:
        self.error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    @property
    def ts_iso(self) -> str:
        return datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )

    def _jsonl_path(self) -> Path:
        d = ensure_runs_dir()
        day = datetime.fromtimestamp(self.started_at, tz=timezone.utc).date().isoformat()
        return d / f"{day}.jsonl"

    def _transcript_path(self) -> Path:
        d = ensure_runs_dir()
        ts = datetime.fromtimestamp(self.started_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sid = _sanitize(self.session_id)
        role = _sanitize(self.role)
        return d / f"{ts}-{role}-{sid}.md"

    def write_transcript(self) -> Path:
        path = self._transcript_path()
        body = (
            f"# {self.role} invocation\n\n"
            f"- timestamp: {self.ts_iso}\n"
            f"- session_id: {self.session_id}\n"
            f"- input_buffer_turns: {self.input_buffer_turns}\n"
            f"- duration_ms: {self.duration_ms}\n"
            f"- cost_usd: {self.cost_usd:.6f}\n"
            f"- parsed_ok: {self.parsed_ok}\n"
            f"- decisions: {json.dumps(self.decisions, sort_keys=True)}\n"
            f"- error: {bool(self.error)}\n\n"
            f"## Prompt sent\n\n"
            f"```\n{self.input_prompt}\n```\n\n"
            f"## Response received\n\n"
            f"```\n{self.output_raw}\n```\n"
        )
        if self.error:
            body += f"\n## Error\n\n```\n{self.error}\n```\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            self.transcript_path = path
        except OSError as e:
            log.warning("could not write transcript %s: %s", path, e)
        return path

    def append_jsonl(self) -> None:
        path = self._jsonl_path()
        raw_trunc = self.output_raw
        if len(raw_trunc.encode("utf-8", errors="replace")) > JSONL_OUTPUT_RAW_MAX_BYTES:
            # Truncate by character count first, then by bytes to be safe.
            raw_trunc = raw_trunc[:JSONL_OUTPUT_RAW_MAX_BYTES] + "…[truncated]"
        row: Dict[str, Any] = {
            "ts": self.ts_iso,
            "role": self.role,
            "session_id": self.session_id,
            "input_prompt_chars": len(self.input_prompt),
            "input_buffer_turns": self.input_buffer_turns,
            "output_raw": raw_trunc,
            "parsed_ok": self.parsed_ok,
            "decisions": self.decisions,
            "cost_usd": round(float(self.cost_usd), 6),
            "duration_ms": int(self.duration_ms),
            "error": self.error,
            "transcript_path": str(self.transcript_path) if self.transcript_path else None,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            log.warning("could not append jsonl %s: %s", path, e)

    def finalise(self) -> None:
        """End-of-call hook: stamp duration, write transcript + JSONL,
        bump cost totals, sweep old transcripts. Never raises."""
        self.duration_ms = int((time.time() - self.started_at) * 1000)
        try:
            self.write_transcript()
        except Exception:
            log.exception("write_transcript failed")
        try:
            self.append_jsonl()
        except Exception:
            log.exception("append_jsonl failed")
        try:
            if self.cost_usd:
                add_cost(self.cost_usd)
        except Exception:
            log.exception("add_cost failed")
        try:
            sweep_old_transcripts()
        except Exception:
            log.exception("sweep_old_transcripts failed")


# ── Transcript sweep ───────────────────────────────────────────────────


def sweep_old_transcripts(
    *,
    retention_days: int = TRANSCRIPT_RETENTION_DAYS,
    now: Optional[float] = None,
) -> int:
    """Delete per-invocation .md transcripts older than ``retention_days``.

    JSONL day-rolls are kept indefinitely (they're tiny). Returns the
    number of files deleted. Best-effort; never raises.
    """
    if now is None:
        now = time.time()
    cutoff = now - retention_days * 24 * 3600
    d = runs_dir()
    if not d.exists():
        return 0
    deleted = 0
    for p in d.glob("*.md"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                deleted += 1
        except OSError:
            continue
    return deleted
