"""Compare two scan databases page by page, e.g. a 4-bit run against an 8-bit run.

Pages are matched on (file_sha256, page_index), so both runs must have read the
same files. Use the same `sample_limit` for both and they will.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from pathlib import Path

_NUM_RE = re.compile(r"\d[\d,./:-]*\d|\d")


def _numbers(text: str | None) -> list[str]:
    return _NUM_RE.findall(text or "")


def _load(path: str | Path) -> dict[tuple[str, int], sqlite3.Row]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT p.*, f.source_path FROM pages p JOIN files f USING (file_sha256)
               WHERE p.status='done'"""
        ).fetchall()
    finally:
        conn.close()
    return {(r["file_sha256"], r["page_index"]): r for r in rows}


def compare(db_a: str | Path, db_b: str | Path) -> list[dict]:
    """One dict per page present in both databases.

    * `text_similarity`: 0-1 ratio of matching characters between transcriptions.
    * `numbers_match`: the sequence of numbers (dates, amounts, IDs) is identical.
      Numbers are where lower-precision models tend to slip first.
    * `numbers_only_in_a` / `_b`: the numbers that differ, to eyeball which is right.
    """
    a, b = _load(db_a), _load(db_b)
    out = []
    for key in sorted(a.keys() & b.keys(), key=lambda k: (a[k]["source_path"], k[1])):
        ra, rb = a[key], b[key]
        ta, tb = ra["ocr_text"] or "", rb["ocr_text"] or ""
        na, nb = _numbers(ta), _numbers(tb)
        out.append({
            "page": f"{ra['source_path']} p{key[1] + 1}",
            "text_similarity": round(difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio(), 4),
            "numbers_match": na == nb,
            "numbers_only_in_a": sorted(set(na) - set(nb)),
            "numbers_only_in_b": sorted(set(nb) - set(na)),
            "title_a": ra["doc_title"], "title_b": rb["doc_title"],
            "page_no_a": ra["page_number_printed"], "page_no_b": rb["page_number_printed"],
            "seconds_a": ra["elapsed_s"], "seconds_b": rb["elapsed_s"],
        })
    return out


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "No pages in common."
    n = len(rows)
    sim = sum(r["text_similarity"] for r in rows) / n
    nums = sum(r["numbers_match"] for r in rows)
    sa = sum(r["seconds_a"] or 0 for r in rows) / n
    sb = sum(r["seconds_b"] or 0 for r in rows) / n
    return (f"{n} pages compared. Mean text similarity {sim:.3f}. "
            f"Numbers identical on {nums}/{n} pages. "
            f"Mean seconds per page: A {sa:.1f}, B {sb:.1f}.")
