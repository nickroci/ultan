# Type stub for the `advisor` flat module (agent-mem-tools, ships via the
# [retrieval] extra). Lets pyright type the lazy call sites in
# ultan/__main__.py identically whether or not the extra is installed
# (thin CI env vs full dev venv). Keep in sync with tools/ultan/advisor.py.
def run(question: str) -> int: ...
