"""The page prompt, and turning the model's reply into database columns.

The model answers in two tagged sections:

    <meta>{ ...JSON with ordering clues and answers... }</meta>
    <text>
    ...verbatim transcription...
    </text>

Metadata comes first because it's short and it's what the sorting pass needs
most: if a long page runs out of tokens, we lose the end of the transcription
rather than the clues. The transcription is plain text rather than a JSON
string, because models often break JSON escaping on long text.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Bump when the prompt changes in a way that affects results. Stored per page,
# so later notebooks can tell which pages were read with which prompt.
PROMPT_VERSION = 4   # 2: tables as one line per filled row; 3: [?] for unclear marks, strips; 4: heading band on strips

META_FIELDS = {
    "doc_title": "The document's title or subject line as printed, or a short description if none is printed.",
    "doc_type": "One of: letter, memo, email, invoice, receipt, form, report, article, contract, "
                "statement, notes, table, figure, other.",
    "page_number_printed": "The page number exactly as printed on the page (e.g. \"3\", \"iv\", \"Page 2 of 5\"), or null.",
    "dates": "Every date that appears on the page, as YYYY-MM-DD where possible (YYYY-MM or YYYY if partial).",
    "primary_date": "The date this document was written or sent, as YYYY-MM-DD where possible, or null.",
    "running_header": "Text repeated at the top of the page (letterhead, header line), or null.",
    "running_footer": "Text repeated at the bottom of the page (footer, reference number), or null.",
    "starts_mid_sentence": "true if the first line continues a sentence from a previous page.",
    "ends_mid_sentence": "true if the last line breaks off mid-sentence and continues on another page.",
    "language": "ISO 639-1 code of the main language, e.g. \"en\".",
}


# Tables and forms. Asking for Markdown tables made the model loop on ruled
# forms with mostly empty cells (it kept writing "| | |" until it ran out of
# tokens), so grids are written as one line per filled row instead. A ticked
# box becomes its column's heading, which is the information the mark carries.
TABLE_RULES = (
    "Tables, forms and grids: never draw borders, grid lines or empty cells. Write instructions, "
    "titles and other text around a table as ordinary lines, not side by side with the table's "
    "columns. Write the column headings once, on one line, separated by \" | \". Then write each row "
    "that has something in it as one line, with its filled cells in order separated by \" | \". "
    "Where a row marks a box with an X, tick or check, look straight up from the mark to the heading "
    "of the column it sits in, and write that heading instead of the mark. A row usually has at most "
    "one mark in each group of related columns. If you cannot tell which column a mark is in, write "
    "[?] instead of guessing. Leave out empty cells and empty rows. Start a printed example row with "
    "\"Example:\". For form fields, write \"Label: value\", or just the label if it was left blank."
)

TEXT_RULES = (
    "Transcribe every word exactly as written, top to bottom, keeping line breaks. Keep spelling, "
    "numbers, names and punctuation exactly as printed, even if they look wrong. Mark words you "
    "cannot read as [illegible]. Do not summarize or skip any text."
)


_INTRO = ("You are transcribing one scanned page from a pile of documents. "
          "Pages may be out of order and from many different documents.\n\n")


def _meta_block(questions: dict[str, str]) -> str:
    fields = "\n".join(f'  "{k}": {v}' for k, v in META_FIELDS.items())
    if questions:
        answers = "\n".join(f'    "{k}": {v}' for k, v in questions.items())
        answers_block = f'  "answers": an object with these keys:\n{answers}\n'
    else:
        answers_block = '  "answers": {}\n'
    return ("a JSON object inside <meta></meta> tags with these keys:\n"
            f"{fields}\n{answers_block}"
            "Use null for anything not shown on the page. Do not guess.")


def build_prompt(questions: dict[str, str]) -> str:
    """The prompt for reading a whole page in one go: metadata, then text."""
    return (
        _INTRO
        + "Reply with exactly two sections and nothing else.\n\n"
        + "First, " + _meta_block(questions) + "\n\n"
        + "Second, inside <text></text> tags: " + TEXT_RULES + "\n\n"
        + TABLE_RULES
    )


def build_meta_prompt(questions: dict[str, str]) -> str:
    """When a page is read in strips: the whole page, for its metadata only."""
    return (
        _INTRO
        + "Reply with only " + _meta_block(questions) + "\n\n"
        + "Do not transcribe the page; that is done separately."
    )


def build_strip_prompt(part: int, parts: int, heading: bool = False) -> str:
    """When a page is read in strips: the text of one horizontal strip.

    Sent with two images, the whole page and then the strip, so the model can
    see column headings that are outside the strip. With `heading`, the
    table's heading band has been pasted above the strip, under a grey bar.
    """
    head = ("" if not heading else
            "Above the grey bar at the top of the close-up is a copy of the table's column headings "
            "from higher up the page, so you can see which column is which. Do not transcribe that "
            "copy; transcribe only what is below the grey bar, and use the headings above it to name "
            "the column each mark sits in.\n\n")
    return (
        "You are transcribing one scanned page in parts. The first image is the whole page. The "
        f"second image is a close-up of part {part} of {parts}, a horizontal strip of the same page "
        "(part 1 is the top). Transcribe only what is in the close-up, using the whole page to see "
        "column headings and context that are outside it.\n\n"
        + head
        + "Reply with only the transcription inside <text></text> tags. " + TEXT_RULES + "\n\n"
        + TABLE_RULES
    )


def text_section(raw: str) -> tuple[str, bool]:
    """The text inside <text></text> in one strip's reply, and whether it was closed."""
    cleaned = clean_response(raw)
    m = _TEXT_RE.search(cleaned)
    if not m:
        return _META_RE.sub("", cleaned).strip(), True
    return m.group(1).strip(), "</text>" in cleaned.lower()


def meta_section(raw: str) -> str:
    """Just the <meta>...</meta> block of a reply, closed even if the reply was cut off."""
    m = _META_RE.search(clean_response(raw))
    return f"<meta>{m.group(1).strip() if m else '{}'}</meta>"


_SQUASH = re.compile(r"[\s|]+")


def _squash(line: str) -> str:
    return _SQUASH.sub("", line).lower()


def join_strips(texts: list[str], window: int = 10) -> tuple[str, list[list[str]]]:
    """Join the strips' transcriptions, dropping lines a strip repeats from the one above.

    Where two strips meet, the model sometimes writes the same line in both
    (a row near the cut, or one it read off the whole page). A line among the
    first `window` lines of a strip is dropped when it is the same as, or is
    contained in or contains, one of the last `window` lines written so far,
    ignoring spacing and "|". Nearly-the-same lines are kept: two table rows
    can differ only in their row number. Short lines (under 8 characters) are
    never dropped, so row numbers and the like stay. Returns the text and the lines dropped from
    each strip.
    """
    out: list[str] = []
    dropped: list[list[str]] = []
    for k, text in enumerate(texts):
        lines = [ln for ln in text.split("\n")] if text else []
        gone: list[str] = []
        if k and out:
            tail = [_squash(ln) for ln in out[-window:] if len(_squash(ln)) >= 8]
            kept = []
            for n, line in enumerate(lines):
                sq = _squash(line)
                if n < window and len(sq) >= 8 and any(sq in t or t in sq for t in tail):
                    gone.append(line)
                else:
                    kept.append(line)
            lines = kept
        dropped.append(gone)
        out.extend(lines)
    return "\n".join(out).strip(), dropped


# --- tidying tables ---------------------------------------------------------------

_ROW_NUMBER = re.compile(r"^\d{1,3}\.?$")


def tidy_text(text: str | None) -> str | None:
    """Drop the grid padding the model sometimes still writes.

    On lines with " | " cells, empty cells are removed, and a line left with
    nothing, or with only a row number, is dropped. Other lines are untouched.
    The model's reply is kept as it was in `raw_response`.
    """
    if text is None:
        return None
    out = []
    for line in text.split("\n"):
        if "|" not in line:
            out.append(line)
            continue
        cells = [c.strip() for c in line.split("|")]
        kept = [c for c in cells if c]
        if not kept or (len(kept) == 1 and _ROW_NUMBER.match(kept[0]) and len(cells) > 1):
            continue
        out.append(" | ".join(kept))
    return "\n".join(out).strip()


# --- parsing -----------------------------------------------------------------

_THOUGHT_RE = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
# Gemma 4's control tokens contain a pipe (e.g. "<|turn>", "<turn|>"); older
# Gemma tokens are listed explicitly. Ordinary angle-bracket text on the page is
# left alone.
_SPECIAL_RE = re.compile(r"<\|[^<>\s]*>|<[^<>\s]*\|>|<(?:eos|bos|pad|start_of_turn|end_of_turn)>")
_META_RE = re.compile(r"<meta>(.*?)(?:</meta>|$)", re.DOTALL | re.IGNORECASE)
_TEXT_RE = re.compile(r"<text>\n?(.*?)(?:\n?</text>|$)", re.DOTALL | re.IGNORECASE)


def clean_response(raw: str) -> str:
    """Strip Gemma 4's (empty) thinking block and any leftover special tokens.

    With thinking off, the larger Gemma 4 models still emit an empty
    `<|channel>thought ...<channel|>` block before the answer.
    """
    text = _THOUGHT_RE.sub("", raw)
    text = _SPECIAL_RE.sub("", text)
    return text.strip()


def _loads_lenient(s: str) -> dict[str, Any] | None:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = s[start:end + 1]
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            obj = json.loads(attempt)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(c not in _ROMAN for c in s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b != " " and _ROMAN.get(b, 0) > v else v
    return total


def parse_page_number(printed: str | None) -> tuple[int | None, int | None]:
    """'Page 3 of 7' -> (3, 7); '12' -> (12, None); 'iv' -> (4, None)."""
    if not printed:
        return None, None
    s = str(printed).strip()
    m = re.search(r"(\d+)\s*(?:of|/)\s*(\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0)), None
    word = re.sub(r"(?i)^(page|p\.?|pg\.?)\s*", "", s).strip(" -–—.")
    return _roman_to_int(word), None


def _bool_or_none(v: Any) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return int(v.strip().lower() == "true")
    return None


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("null", "none", "n/a") else None


def first_last_lines(text: str | None) -> tuple[str | None, str | None]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None, None
    return lines[0], lines[-1]


def parse_response(raw: str) -> dict[str, Any]:
    """Turn a model reply into values for `db.save_page`.

    Never raises on malformed output: what can be recovered is kept, and the
    raw reply is always stored so a page can be re-parsed later.
    """
    cleaned = clean_response(raw)
    meta_match = _META_RE.search(cleaned)
    meta = _loads_lenient(meta_match.group(1)) if meta_match else None
    text_match = _TEXT_RE.search(cleaned)
    if text_match:
        text = tidy_text(text_match.group(1).strip())
        truncated = "</text>" not in cleaned.lower()
    else:
        # The model ignored the format. Keep whatever isn't the meta block.
        text = _META_RE.sub("", cleaned).strip() or None
        truncated = False

    meta = meta or {}
    printed = _str_or_none(meta.get("page_number_printed"))
    page_int, page_total = parse_page_number(printed)
    dates = meta.get("dates")
    if not isinstance(dates, list):
        dates = [dates] if _str_or_none(dates) else []
    dates = [str(d).strip() for d in dates if _str_or_none(d)]
    answers = meta.get("answers") if isinstance(meta.get("answers"), dict) else {}
    first, last = first_last_lines(text)

    return {
        "ocr_text": text,
        "truncated": int(truncated),
        "doc_title": _str_or_none(meta.get("doc_title")),
        "doc_type": _str_or_none(meta.get("doc_type")),
        "page_number_printed": printed,
        "page_number_int": page_int,
        "page_total_int": page_total,
        "dates_json": json.dumps(dates, ensure_ascii=False),
        "primary_date": _str_or_none(meta.get("primary_date")),
        # Computed from the transcription rather than asked of the model: exact
        # and free, and the sorting pass leans on them for continuity.
        "first_line": first,
        "last_line": last,
        "running_header": _str_or_none(meta.get("running_header")),
        "running_footer": _str_or_none(meta.get("running_footer")),
        "starts_mid_sentence": _bool_or_none(meta.get("starts_mid_sentence")),
        "ends_mid_sentence": _bool_or_none(meta.get("ends_mid_sentence")),
        "language": _str_or_none(meta.get("language")),
        "answers_json": json.dumps(answers, ensure_ascii=False),
        "meta_json": json.dumps(meta, ensure_ascii=False) if meta else None,
        "raw_response": raw,
    }
