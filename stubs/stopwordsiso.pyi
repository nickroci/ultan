"""Signature stub for the data-only ``stopwordsiso`` package (untyped upstream).

Only the single function the hook hot path uses is declared.
"""

from collections.abc import Iterable

def stopwords(langs: str | Iterable[str]) -> set[str]: ...
