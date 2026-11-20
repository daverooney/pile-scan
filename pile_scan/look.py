"""Looking at how a page was read in strips, for the notebook.

`page_strips` finds a page, cuts it the way it was cut when it was scanned
(from the database, when that page was read in strips) or the way it would be
cut now, and returns each strip's picture alongside the model's reply.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import files, strips
from .config import ScanConfig


@dataclass
class StripView:
    part: int
    top: int
    bottom: int
    heading: tuple[int, int] | None
    image: Image.Image = field(repr=False)
    reply: str | None = None             # the model's reply for this strip, if it was scanned
    dropped_lines: list[str] = field(default_factory=list)


@dataclass
class PageView:
    rel_path: str
    page_number: int                    # 1-based
    image: Image.Image = field(repr=False)
    strips: list[StripView]
    from_database: bool                 # True: the cuts the scan used; False: what STRIPS would do now
    ocr_text: str | None = None
    meta_reply: str | None = None


def _database(cfg: ScanConfig) -> Path | None:
    for path in (cfg.local_db_path, cfg.drive_db_path):
        if Path(path).exists():
            return Path(path)
    return None


def page_strips(cfg: ScanConfig, name: str, page_number: int, parts: int | None = None) -> PageView:
    """A page's strips. `name` is part of the file's path under the input folder."""
    found = [f for f in files.discover(cfg.input_dir, recursive=cfg.recursive)
             if f.supported and name.lower() in f.rel_path.lower()]
    if not found:
        raise FileNotFoundError(f"No file under {cfg.input_dir} has {name!r} in its name.")
    f = found[0]
    n = files.page_count(f.path)
    if not 1 <= page_number <= n:
        raise ValueError(f"{f.rel_path} has {n} pages.")
    idx = page_number - 1
    page = next(files.iter_pages(f.path, dpi=cfg.pdf_dpi, max_side=cfg.max_image_side,
                                 skip=set(range(n)) - {idx}))
    image = page.image

    stored, ocr_text = None, None
    db_path = _database(cfg)
    if db_path is not None:
        sha = files.sha256_file(f.path)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT strips_json, ocr_text FROM pages WHERE file_sha256=? AND page_index=?",
                               (sha, idx)).fetchone()
        except sqlite3.OperationalError:     # a database from before strips were saved
            row = None
        finally:
            conn.close()
        if row:
            stored = json.loads(row[0]) if row[0] else None
            ocr_text = row[1]

    if stored:
        views = [StripView(r["part"], r["top"], r["bottom"],
                           tuple(r["heading"]) if r.get("heading") else None,
                           strips.compose(image, r["top"], r["bottom"], r.get("heading")),
                           r.get("reply"), r.get("dropped_lines") or [])
                 for r in stored["strips"]]
        return PageView(f.rel_path, page_number, image, views, True, ocr_text, stored.get("meta_reply"))
    cut = strips.make_strips(image, parts or cfg.strips)
    views = [StripView(s.part, s.top, s.bottom, s.heading, s.image) for s in cut]
    return PageView(f.rel_path, page_number, image, views, False, ocr_text)
