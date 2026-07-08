"""The one sNrN run-id parser, shared by every gaptrace package."""

import re

_TARGET_RE = re.compile(r"^s(\d+)r(\d+)$", re.IGNORECASE)


def parse_target_id(target: str) -> tuple[int, int]:
    """Parse an sNrN run identifier into (session_id, run_seq).

    Pure — no store access.

    Inputs: ``target``, a run id like ``"s4r3"`` (case-insensitive).
    Returns: ``(session_id, run_seq)`` as ints, e.g. ``(4, 3)``.
    Errors: raises ``TypeError`` if ``target`` is not a string; raises
    ``ValueError`` if it is a string but not in sNrN format.
    """
    if not isinstance(target, str):
        raise TypeError(
            f"Run id must be a string in sNrN format (e.g. 's4r3'), "
            f"got {type(target).__name__}: {target!r}"
        )
    m = _TARGET_RE.match(target)
    if not m:
        raise ValueError(f"Run id must be in sNrN format (e.g. 's4r3'), got: {target!r}")
    return int(m.group(1)), int(m.group(2))
