"""Spotting a reply that has got stuck repeating itself.

With greedy decoding a model can fall into a loop it never leaves: on ruled
forms Gemma 4 kept writing empty table cells ("| | | |" across, or "|" on
line after line) until it hit the token limit, which wastes minutes per page.
`find_loop` looks at the end of the text so far and says where a loop began,
so the reader can stop early and try again.

The thresholds are deliberately high. Pages legitimately repeat short
things (dotted leaders, rows of underscores, a column of "$0.00"), so a short
unit has to repeat for 300 characters, and a longer one ten times over 500.
Whitespace is collapsed before comparing, so a pipe on each line counts the
same as pipes across one line.
"""

from __future__ import annotations

import re

TAIL = 4000            # only the end of the text is examined
SHORT_UNIT = 4         # units up to this many characters...
SHORT_SPAN = 300       # ...must repeat for this many characters
LONG_UNIT = 80         # longer units, up to this length...
LONG_REPEATS = 10      # ...must repeat this many times
LONG_SPAN = 500        # ...and cover this many characters
BLANK_RUN = 300        # this much whitespace in a row is a loop on its own

_WS = re.compile(r"\s+")
_BLANK_TAIL = re.compile(r"\s+\Z")


def find_loop(text: str) -> int | None:
    """Index in `text` where a loop at its end starts, or None if there isn't one."""
    blank = _BLANK_TAIL.search(text)
    if blank and len(blank.group()) >= BLANK_RUN:
        return blank.start()

    offset = max(0, len(text) - TAIL)
    tail = text[offset:]
    # Collapse whitespace runs to one space, remembering where each kept
    # character came from so the answer can be given in `text` positions.
    compact: list[str] = []
    origin: list[int] = []
    pos = 0
    for m in _WS.finditer(tail):
        for i in range(pos, m.start()):
            compact.append(tail[i])
            origin.append(i)
        compact.append(" ")
        origin.append(m.start())
        pos = m.end()
    for i in range(pos, len(tail)):
        compact.append(tail[i])
        origin.append(i)
    c = "".join(compact).rstrip()
    n = len(c)

    for unit_len in range(1, LONG_UNIT + 1):
        if 2 * unit_len > n:
            break
        unit = c[n - unit_len:]
        if not unit.strip():
            continue
        start = n - unit_len
        while start >= unit_len and c[start - unit_len:start] == unit:
            start -= unit_len
        span = n - start
        repeats = span // unit_len
        if (unit_len <= SHORT_UNIT and span >= SHORT_SPAN) or \
           (repeats >= LONG_REPEATS and span >= LONG_SPAN):
            return offset + origin[start]
    return None
