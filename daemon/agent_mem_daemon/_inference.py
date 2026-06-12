"""Single-thread executor for the daemon's PyTorch/MPS model inference.

The sentence-transformer embedder and the cross-encoder reranker run on MPS
(Apple-Silicon GPU). PyTorch MPS is NOT safe to drive from multiple threads:
concurrent forward passes from different threads share the model's internal
buffers, which both

  1. **deadlocks** after the first cross-thread call (the priming RPC served
     exactly one request, then every later request hung), and
  2. **corrupts** intermediate tensors — observed as
     ``RuntimeError: size of tensor a (45) must match tensor b (35) at
     non-singleton dimension 1`` (one query's sequence length colliding with
     another's attention mask).

Several daemon threads want to run inference: the priming RPC handler pool, the
Scholar/Librarian agent-research workers (embedding search via
``library_tools``), and the startup warmup. Funnel EVERY model forward pass
through the one dedicated thread here, so MPS is only ever driven from a single
thread. Connection I/O and BM25 (CPU) stay concurrent; only the model calls
serialise.

Wrap LEAF model calls only (``index.search`` / ``rerank`` / ``encode``), never a
whole pipeline that itself calls :func:`run` — submitting to this 1-worker pool
from its own worker thread would deadlock. The startup warmup must also go
through :func:`run` so the MPS kernels JIT-compile on the serving thread.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Lazily-created single-worker executor. Re-creatable: :func:`shutdown` resets it
# so a fresh one is built on the next :func:`run` — without this, a shutdown
# (daemon exit, or a test exercising it) would poison every later call with
# "cannot schedule new futures after shutdown".
_lock = threading.Lock()
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mps-inference")
        return _executor


def run(fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Run a leaf model-inference callable on the single inference thread and
    return its result (blocking the caller until it completes).

    MUST wrap only leaf calls (``encode`` / ``predict`` / ``index.search`` /
    ``rerank``); never a pipeline that re-enters :func:`run`, which would submit
    to this 1-worker pool from its own worker thread and deadlock.
    """
    return _get_executor().submit(fn, *args, **kwargs).result()


def shutdown() -> None:
    """Stop the inference thread (daemon shutdown) so the executor's at-exit
    join doesn't delay process exit. Idempotent; a later :func:`run` lazily
    recreates the executor."""
    global _executor
    with _lock:
        ex, _executor = _executor, None
    if ex is not None:
        ex.shutdown(wait=False, cancel_futures=True)
