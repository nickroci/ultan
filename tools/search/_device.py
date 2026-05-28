"""Torch device selection, shared by the embedding and reranker models.

Both models pick the same device the same way; keeping the logic here
avoids drift between the two copies.
"""

from __future__ import annotations


def select_device() -> str:
    """Return the best available torch device.

    Apple Silicon's MPS is preferred when present; falls back to CUDA on
    Linux/NVIDIA hosts; CPU otherwise. Detection is one-shot at load time;
    callers don't pay it per request.
    """
    try:
        import torch  # noqa: PLC0415 — lazy import keeps cold start lean.

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
