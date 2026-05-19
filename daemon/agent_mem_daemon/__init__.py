"""agent-mem daemon — event ingestion, rolling buffer, scheduler.

This package is the *skeleton* phase: it does the plumbing (tail a JSONL
event log, aggregate turns per session, fire stub Librarian/Scholar
callables) but it does not call any LLM. The real Librarian/Scholar
implementations are a separate deliverable.

See ``PLAN.md`` §1 and §4 for the architecture this implements.
"""

__version__ = "0.1.0"
