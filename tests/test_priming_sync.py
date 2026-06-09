"""Guard: ultan/_priming.py is a verbatim copy of src/hooks/_priming_client.py.

The two deployment modes (plugin wheel vs from-source legacy hooks) each need
the lexical-fallback module without cross-package imports, so the file is
duplicated by design. This test pins that duplication so the copies can't
drift apart silently: edit one, then `cp` it over the other.
"""

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_PLUGIN_COPY = _REPO / "ultan" / "_priming.py"
_LEGACY_COPY = _REPO / "src" / "hooks" / "_priming_client.py"


@pytest.mark.skipif(not _LEGACY_COPY.exists(), reason="src/ not present (installed mode)")
def test_priming_copies_are_byte_identical() -> None:
    assert _PLUGIN_COPY.read_bytes() == _LEGACY_COPY.read_bytes(), (
        "ultan/_priming.py and src/hooks/_priming_client.py have drifted — "
        "they are deliberate verbatim copies; apply the same edit to both "
        "(edit one, `cp` it over the other)."
    )
