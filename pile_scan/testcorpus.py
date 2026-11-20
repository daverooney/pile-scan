"""Make a test pile from a born-digital PDF, with an answer key the scanner never sees.

The PDF's pages are rendered to images (so the text layer is gone, as in a
scanned dump), optionally roughed up to look like photocopies, some are turned
by 90, 180 or 270 degrees, and the order is partly shuffled. Everything the
scanner and the sort pass should rebuild goes to `answer_key/` instead:

    CORPUS_DIR/
      pages/                  the pile: point notebook 1's INPUT_DIR here
      answer_key/
        pages.csv             one row per page in pile order (no text)
        pages.jsonl           the same rows plus each page's text from the PDF's text layer
        true_order.txt        file names in true reading order
        held_out/             sections taken out of the pile (index, table of cases, ...)
        manifest.json         settings, source, section ranges, counts
        README.md

The section detection is tuned to the United States Reports (running heads
like "ORDERS 803", "TABLE OF CASES REPORTED xvi", "1262 INDEX"). Any other PDF
still works, but its section labels won't mean much (numbered pages come out as
"opinions", the rest as "other") and nothing gets held out.

Everything random is seeded. Each page's own treatment (rotation, roughing up,
random file name) is seeded by the page, so changing the page range or the
share of rotated pages doesn't change what happens to the other pages.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

US_REPORTS_586_URL = "https://www.govinfo.gov/content/pkg/USREPORTS-586/pdf/USREPORTS-586.pdf"

SECTIONS = ("cover", "front_matter", "table_of_cases", "opinions", "orders", "index", "other")
DEFAULT_HOLD_OUT = ("cover", "table_of_cases", "index")

# A page turned "portrait saved as landscape" is the usual mistake, so quarter
# turns are more common than upside-down pages.
ROTATION_WEIGHTS = {90: 0.4, 270: 0.4, 180: 0.2}


# --------------------------------------------------------------------------
# Settings


@dataclass
class CorpusConfig:
    corpus_dir: str                      # gets pages/ and answer_key/
    source_pdf: str                      # local path; downloaded from source_url when missing
    source_url: str | None = US_REPORTS_586_URL
    pages: str = "1-145"                 # printed page range, "all", or "pdf:201-345" (1-based PDF pages)
    hold_out: tuple[str, ...] = DEFAULT_HOLD_OUT
    hold_out_images: bool = True         # also save held-out pages as images, not only their text
    seed: int = 586
    dpi: int = 200
    rotate_share: float = 0.10           # share of pages turned 90, 180 or 270 degrees
    degrade_share: float = 0.0           # share of pages roughed up to look photocopied
    shuffle: str = "partial"             # "partial", "none" or "full"
    filenames: str = "pile"              # "pile": scan_00001.png in pile order; "random": random hex names
    jitter: float = 0.7                  # partial shuffle: how far pages wander locally (0.7: ~1 in 25 neighbours swap)
    block_share: float = 0.08            # partial shuffle: share of pages moved in contiguous groups
    stray_share: float = 0.03            # partial shuffle: share of single pages that land far away
    overwrite: bool = False

    def validate(self) -> None:
        if not 36 <= self.dpi <= 600:
            raise ValueError("dpi should be between 36 and 600")
        for name in ("rotate_share", "degrade_share", "block_share", "stray_share"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {v}")
        if self.block_share + self.stray_share > 0.9:
            raise ValueError("block_share + stray_share leaves too few pages in place")
        if self.shuffle not in ("partial", "none", "full"):
            raise ValueError("shuffle must be 'partial', 'none' or 'full'")
        if self.filenames not in ("pile", "random"):
            raise ValueError("filenames must be 'pile' or 'random'")
        bad = [s for s in self.hold_out if s not in SECTIONS]
        if bad:
            raise ValueError(f"Unknown section(s) to hold out: {bad}. Known: {', '.join(SECTIONS)}")
        parse_page_range(self.pages)


def parse_hold_out(text: str) -> tuple[str, ...]:
    """'cover, table_of_cases index' -> ('cover', 'table_of_cases', 'index')."""
    return tuple(s for s in re.split(r"[\s,]+", text.strip()) if s)


def parse_page_range(spec: str) -> tuple[str, int | None, int | None]:
    """'all' -> ('all', None, None); '1-145' -> ('printed', 1, 145); 'pdf:20-40' -> ('pdf', 20, 40)."""
    s = spec.strip().lower().replace(" ", "")
    if s in ("", "all"):
        return ("all", None, None)
    kind = "printed"
    if s.startswith("pdf:"):
        kind, s = "pdf", s[4:]
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", s)
    if not m:
        raise ValueError(f"Page range should look like '1-145', 'all' or 'pdf:201-345', got {spec!r}")
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    if b < a:
        raise ValueError(f"Page range runs backwards: {spec!r}")
    return (kind, a, b)


# --------------------------------------------------------------------------
# Reading the PDF's text layer: printed page numbers and sections


_ROMAN = [(100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]


def int_to_roman(n: int) -> str:
    out = []
    for value, sym in _ROMAN:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def roman_to_int(s: str) -> int | None:
    """Lower-case roman numerals up to 399 (front matter), else None.

    Round-tripping rejects words that merely use roman letters ("mix", "civil").
    """
    if not s or not re.fullmatch(r"c{0,3}(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})", s):
        return None
    total, rest = 0, s
    for value, sym in _ROMAN:
        while rest.startswith(sym):
            total += value
            rest = rest[len(sym):]
    return total if total and int_to_roman(total) == s else None


_NUM = r"(\d{1,4}|[ivxlc]+)"


def _page_token(tok: str) -> str | None:
    if tok.isdigit():
        return tok if 0 < int(tok) < 5000 else None
    return tok if roman_to_int(tok) else None


def text_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def printed_page(lines: list[str]) -> tuple[str | None, str | None]:
    """The page number printed on a page, from its running head or a bare number at the foot.

    Returns (number as printed, "head" | "foot") or (None, None).
    Heads look like "2 MOUNT LEMMON FIRE DIST. v. GUIDO", "Cite as: 586 U. S. 1 (2018) 3",
    "ORDERS 803", "liv TABLE OF CASES REPORTED". First pages carry the number at the foot.
    """
    if not lines:
        return None, None
    first = lines[0]
    toks = first.split()
    if len(toks) == 1:
        tok = _page_token(toks[0])
        if tok:
            return tok, "head"
    elif len(toks) > 1:
        # Leading number, but not "586 U. S. October 1, 2018".
        if toks[1] not in ("U.", "U.S.") and (tok := _page_token(toks[0])):
            return tok, "head"
        # Trailing number, but not the year in "OCTOBER TERM, 2018".
        if not toks[-2].endswith(",") and (tok := _page_token(toks[-1])):
            return tok, "head"
    last = lines[-1]
    if len(lines) > 1 and re.fullmatch(_NUM, last) and (tok := _page_token(last)):
        return tok, "foot"
    return None, None


def is_roman(tok: str | None) -> bool:
    return bool(tok) and not tok.isdigit()


_TERM_HEAD = re.compile(r"^(?:\d+ )?(?:OCTOBER|OCT\.) TERM, \d{4}(?: \d+)?$")
_ORDERS_HEAD = re.compile(r"^(?:ORDERS \d+|\d+ ORDERS)$")
_INDEX_HEAD = re.compile(r"^(?:\d+ )?INDEX(?: \d+)?$")
_DATE = r"([A-Z][a-z]+ \d{1,2}, \d{4})"
# "586 U. S. October 1, 2018" under an orders page's running head (or the other way round).
_ORDERS_DATE = re.compile(rf"^(?:\d+ U\. ?S\. {_DATE}|{_DATE} \d+ U\. ?S\.)$")
_TITLE_END = re.compile(r"^(?:certiorari|on (?:petition|writ|application|appeal|motion|bill)|appeal from"
                        r"|No\. |Nos\. |Argued|Decided)", re.IGNORECASE)


@dataclass
class PdfPage:
    pdf_index: int                       # 0-based
    text: str
    printed: str | None = None           # page number as printed ("145", "xvi")
    printed_from: str | None = None      # "head" or "foot"
    section: str = "other"
    document: str | None = None          # what the page belongs to: "opinion at p. 27", "orders March 4, 2019", ...
    title: str | None = None             # an opinion's case name, on all of its pages

    @property
    def pdf_page(self) -> int:
        return self.pdf_index + 1


def _case_title(lines: list[str]) -> str | None:
    """The case name on an opinion's first page: the lines after "OCTOBER TERM, 2018"
    (and "Syllabus"), up to "certiorari to ...", "on petition ..." or "No. 17-587"."""
    heads = [k for k, ln in enumerate(lines[:8]) if _TERM_HEAD.match(ln)]
    if not heads:
        return None
    parts: list[str] = []
    for ln in lines[heads[-1] + 1:heads[-1] + 8]:
        if ln == "Syllabus" and not parts:
            continue
        if _TITLE_END.match(ln) or len(parts) == 4:
            break
        parts.append(ln)
    return " ".join(parts) or None


def classify(texts: list[str]) -> list[PdfPage]:
    """Number and label every page, walking the volume in order.

    Sections follow each other in a book, so a page with no clue of its own
    (a blank verso) takes the section of the page before it.
    """
    pages: list[PdfPage] = []
    state: str | None = None
    doc: str | None = None
    title: str | None = None
    for i, text in enumerate(texts):
        lines = text_lines(text)
        printed, where = printed_page(lines)
        edge = lines[:3] + lines[-3:]
        squashed = [ln.replace(" ", "").upper() for ln in lines[:4]]

        if "camera copy" in text.lower():
            section = "cover"
        elif "INDEX" in squashed or any(_INDEX_HEAD.match(ln) for ln in lines[:2]):
            section = "index"
        elif any("TABLE OF CASES REPORTED" in ln for ln in edge):
            section = "table_of_cases"
        elif (any(_ORDERS_HEAD.match(ln) for ln in lines[:2])
              or any(ln.startswith("ORDERS FOR ") for ln in lines[:8])
              or any(_ORDERS_DATE.match(ln) for ln in lines[1:3])):
            section = "orders"
        elif state in ("orders", "index") and lines:
            section = state
        elif printed and not is_roman(printed):
            # The cases begin with "CASES ADJUDGED ... OCTOBER TERM, 2018". Until then a
            # number read off a table-of-cases page is a case's page, not this page's.
            starts = any(_TERM_HEAD.match(ln) for ln in lines[:6]) or "CASES ADJUDGED" in lines[:3]
            section = "table_of_cases" if state == "table_of_cases" and not starts else "opinions"
        elif printed and is_roman(printed):
            section = "front_matter"
        elif not lines and state:
            section = state                      # blank page: stays in its section
        elif state == "cover":
            section = "front_matter"             # the unnumbered title page
        elif state in ("front_matter", "table_of_cases", "opinions"):
            section = state
        else:
            section = "other"

        if section == "opinions":
            if any(_TERM_HEAD.match(ln) for ln in lines[:6]) or doc is None or not doc.startswith("opinion"):
                if lines:
                    doc = f"opinion at p. {printed}" if printed else f"opinion at PDF page {i + 1}"
                    title = _case_title(lines)
        elif section == "orders":
            m = next((_ORDERS_DATE.match(ln) for ln in lines[:3] if _ORDERS_DATE.match(ln)), None)
            bare = next((ln for ln in lines[:6] if re.fullmatch(_DATE, ln)), None)
            if m:
                doc = f"orders {m.group(1) or m.group(2)}"
            elif bare:
                doc = f"orders {bare}"           # the first page of the orders: "October 1, 2018"
            elif doc is None or not doc.startswith("orders"):
                doc = "orders"
            title = None
        else:
            doc, title = section, None

        pages.append(PdfPage(i, text, printed, where, section, doc, title))
        if lines or state is None:
            state = section
    return pages


def section_ranges(pages: list[PdfPage]) -> list[dict]:
    """Contiguous runs of each section, for the manifest and a quick sanity check."""
    runs: list[dict] = []
    for p in pages:
        if runs and runs[-1]["section"] == p.section:
            run = runs[-1]
            run["pdf_last"] = p.pdf_page
            run["pages"] += 1
            if p.printed:
                run["printed_last"] = p.printed
                run.setdefault("printed_first", p.printed)
        else:
            runs.append({"section": p.section, "pdf_first": p.pdf_page, "pdf_last": p.pdf_page, "pages": 1,
                         **({"printed_first": p.printed, "printed_last": p.printed} if p.printed else {})})
    return runs


def select_pages(pages: list[PdfPage], spec: str) -> list[PdfPage]:
    """Pages in the range, in true order. Printed ranges count arabic page numbers only."""
    kind, a, b = parse_page_range(spec)
    if kind == "all":
        return list(pages)
    if kind == "pdf":
        return [p for p in pages if a <= p.pdf_page <= b]
    chosen: list[PdfPage] = []
    inside = False
    for p in pages:
        if p.printed and not is_roman(p.printed):
            n = int(p.printed)
            if a <= n <= b:
                inside = True
            elif inside and n > b:
                break
            elif n < a:
                inside = False
        if inside and p.section not in ("cover", "front_matter", "table_of_cases"):
            chosen.append(p)          # includes blank pages between numbered ones
    # Trailing blank pages after the last numbered page don't belong to the range.
    while chosen and not chosen[-1].printed and not text_lines(chosen[-1].text):
        chosen.pop()
    return chosen


# --------------------------------------------------------------------------
# Shuffling


def partial_shuffle(n: int, rng: random.Random, *, jitter: float = 0.7, block_share: float = 0.08,
                    stray_share: float = 0.03, block_len: tuple[int, int] = (3, 12)
                    ) -> tuple[list[int], dict[int, str]]:
    """A pile order for n pages that looks like a real dump, not a deck of cards.

    1. Every page wanders a little (`jitter`): neighbours sometimes swap.
    2. A few contiguous groups (`block_share` of the pages, 3 to 12 at a time)
       are lifted out and put back somewhere well away from where they were.
    3. A few single pages (`stray_share`) land far from their place.

    Returns the pile order (true positions, 0-based) and a tag for each moved
    page: "block 1", "block 2", ... or "stray".
    """
    keys = [i + rng.uniform(-jitter, jitter) for i in range(n)]
    order = sorted(range(n), key=lambda i: keys[i])
    tags: dict[int, str] = {}

    lo, hi = block_len
    target = round(n * block_share)
    moved = blocks = attempts = 0
    while moved < target and attempts < 200 and n >= 2 * lo:
        attempts += 1
        length = min(rng.randint(lo, hi), max(lo, target - moved), len(order) // 2)
        start = rng.randrange(0, len(order) - length + 1)
        block = order[start:start + length]
        if any(t in tags for t in block):
            continue
        del order[start:start + length]
        far = max(length, n // 10)
        spots = [p for p in range(len(order) + 1) if abs(p - start) >= far] or list(range(len(order) + 1))
        at = rng.choice(spots)
        order[at:at] = block
        blocks += 1
        for t in block:
            tags[t] = f"block {blocks}"
        moved += length

    for _ in range(round(n * stray_share)):
        free = [k for k, t in enumerate(order) if t not in tags]
        if not free or n < 4:
            break
        k = rng.choice(free)
        page = order.pop(k)
        far = max(2, n // 4)
        spots = [p for p in range(len(order) + 1) if abs(p - k) >= far] or list(range(len(order) + 1))
        order.insert(rng.choice(spots), page)
        tags[page] = "stray"
    return order, tags


def pile_order(n: int, cfg: CorpusConfig) -> tuple[list[int], dict[int, str]]:
    rng = random.Random(f"{cfg.seed}:shuffle")
    if cfg.shuffle == "none":
        return list(range(n)), {}
    if cfg.shuffle == "full":
        order = list(range(n))
        rng.shuffle(order)
        return order, {}
    return partial_shuffle(n, rng, jitter=cfg.jitter, block_share=cfg.block_share,
                           stray_share=cfg.stray_share)


# --------------------------------------------------------------------------
# Making the images


def photocopy(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    """Rough up a clean grayscale page: grey paper, a slight skew, speckle,
    sometimes a shadow along one edge, a little blur and JPEG artefacts."""
    w, h = img.size
    params: dict = {}
    paper = rng.randint(225, 248)
    ink = rng.randint(0, 45)
    img = img.point(lambda v: round(ink + (paper - ink) * v / 255))
    params.update(paper=paper, ink=ink)

    skew = round(rng.uniform(0.3, 1.6) * rng.choice((-1, 1)), 2)
    img = img.rotate(skew, resample=Image.Resampling.BICUBIC, fillcolor=paper)
    params["skew_deg"] = skew   # counter-clockwise

    draw = ImageDraw.Draw(img)
    speckles = int(rng.uniform(150, 1200) * (w * h) / (1200 * 1800))
    for _ in range(speckles):
        x, y = rng.randrange(w), rng.randrange(h)
        r = rng.choice((0, 0, 0, 1, 1, 2))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rng.randint(0, 120))
    params["speckles"] = speckles

    params["edge_shadow"] = None
    if rng.random() < 0.3:
        side = rng.choice(("left", "right", "top", "bottom"))
        width = rng.randint(max(4, w // 80), max(8, w // 25))
        for k in range(width):
            shade = round(40 + (paper - 40) * k / width)
            if side == "left":
                draw.line((k, 0, k, h), fill=shade)
            elif side == "right":
                draw.line((w - 1 - k, 0, w - 1 - k, h), fill=shade)
            elif side == "top":
                draw.line((0, k, w, k), fill=shade)
            else:
                draw.line((0, h - 1 - k, w, h - 1 - k), fill=shade)
        params["edge_shadow"] = side

    blur = round(rng.uniform(0.3, 0.9), 2)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    params["blur_radius"] = blur

    quality = rng.randint(30, 70)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert("L")
    params["jpeg_quality"] = quality
    return img, params


def rotate(img: Image.Image, cw: int) -> Image.Image:
    """Turn a page clockwise by 0, 90, 180 or 270 degrees, losslessly."""
    return {0: lambda: img,
            90: lambda: img.transpose(Image.Transpose.ROTATE_270),
            180: lambda: img.transpose(Image.Transpose.ROTATE_180),
            270: lambda: img.transpose(Image.Transpose.ROTATE_90)}[cw]()


def _page_rng(cfg: CorpusConfig, pdf_index: int, purpose: str) -> random.Random:
    return random.Random(f"{cfg.seed}:{purpose}:{pdf_index}")


def _choose_rotation(cfg: CorpusConfig, pdf_index: int) -> int:
    rng = _page_rng(cfg, pdf_index, "rotate")
    if rng.random() >= cfg.rotate_share:
        return 0
    angles, weights = zip(*ROTATION_WEIGHTS.items())
    return rng.choices(angles, weights)[0]


def render(pdf, pdf_index: int, dpi: int) -> Image.Image:
    page = pdf[pdf_index]
    try:
        return page.render(scale=dpi / 72.0).to_pil().convert("L")
    finally:
        page.close()


# --------------------------------------------------------------------------
# The whole job


def read_text_layer(pdf) -> list[str]:
    texts = []
    for i in range(len(pdf)):
        page = pdf[i]
        try:
            tp = page.get_textpage()
            try:
                raw = tp.get_text_range()
            finally:
                tp.close()
        finally:
            page.close()
        texts.append(raw.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", ""))
    return texts


def ensure_source(cfg: CorpusConfig, *, download=None, log=print) -> Path:
    path = Path(cfg.source_pdf)
    if path.exists() and path.stat().st_size > 0:
        return path
    if not cfg.source_url:
        raise FileNotFoundError(f"No PDF at {path} and no URL to download it from.")
    if download is None:
        from .llama import _download as download
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {cfg.source_url}")
    download(cfg.source_url, path, log=log)
    with open(path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            bad = path.with_name(path.name + ".not-a-pdf")
            path.replace(bad)
            raise RuntimeError(f"The download isn't a PDF (kept as {bad.name}). "
                               "Download it in a browser and put it at SOURCE_PDF instead.")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CorpusSummary:
    corpus_dir: Path
    pages_dir: Path
    answer_dir: Path
    pdf_pages: int
    pile_pages: int
    held_out: dict[str, int]
    rotated: int
    degraded: int
    moved: dict[str, int]
    sections: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        held = ", ".join(f"{k} {v}" for k, v in self.held_out.items()) or "nothing"
        return (f"{self.pile_pages} pages in the pile (from a {self.pdf_pages}-page PDF); held out: {held}.\n"
                f"Rotated {self.rotated}, roughed up {self.degraded}; "
                f"moved in groups {self.moved.get('block', 0)}, strays {self.moved.get('stray', 0)}; "
                f"{self.moved.get('next_page_follows', 0)} of {max(self.pile_pages - 1, 0)} pages "
                f"are still followed by their true next page.\n"
                f"Scan this folder: {self.pages_dir}")


ANSWER_README = """# Answer key

Made by `pile_scan.testcorpus`. The scanner should read only `../pages/`; nothing
in this folder is meant to be seen by it.

- `pages.csv`: one row per image in `../pages/`, in pile order (the order the
  file names sort in). Columns:
  - `file`: image name. `pile_position`: 1-based place in the pile.
    `true_position`: 1-based place in true reading order. `displacement` is the difference;
    a group moved elsewhere shifts every page in between, so judge order by neighbours instead
    (how often a page's true next page follows it in the pile).
  - `pdf_page`: 1-based page of the source PDF. `printed_page`: the number printed on it, if any.
  - `section`: cover, front_matter, table_of_cases, opinions, orders, index or other.
  - `document`: what the page belongs to ("opinion at p. 27", "orders January 7, 2019").
    `title`: an opinion's case name. Use these to score grouping.
  - `rotation_cw`: degrees the image was turned clockwise. Turn it back counter-clockwise by this much.
  - `degraded` and the photocopy settings (`skew_deg` counter-clockwise, `speckles`, `blur_radius`,
    `jpeg_quality`, `edge_shadow`).
  - `moved`: "block N" if the page was moved in a group, "stray" if it was moved alone,
    empty if it only wandered a page or two (or not at all).
- `pages.jsonl`: the same rows plus `text`, the page's text from the PDF's text layer.
  That's the ground truth for transcription, with one caveat: govinfo's text of this volume
  drops the "fi"/"fl"/"ffi" ligatures ("defnition", "offce", "fled" for "filed"), and the PDF's
  text layer most likely does too, so a correct reading of those words can look like a mismatch.
- `true_order.txt`: the image names in true reading order.
- `held_out/`: sections taken out of the pile, each as `<section>.txt` (every page, with
  `=== page ... ===` markers), `<section>.jsonl`, and page images when that was switched on.
  These are what notebook 2 should be able to rebuild (or at least check itself against).
- `manifest.json`: settings, the source PDF and its SHA-256, where each section starts and ends, counts.
"""


def make_corpus(cfg: CorpusConfig, *, log=print, download=None) -> CorpusSummary:
    cfg.validate()
    import pypdfium2 as pdfium

    root = Path(cfg.corpus_dir)
    pages_dir, answer_dir = root / "pages", root / "answer_key"
    if (pages_dir.exists() and any(pages_dir.iterdir())) or answer_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{root} already has a corpus. Pick another folder or set overwrite.")
        shutil.rmtree(pages_dir, ignore_errors=True)
        shutil.rmtree(answer_dir, ignore_errors=True)

    source = ensure_source(cfg, download=download, log=log)
    pdf = pdfium.PdfDocument(str(source))
    try:
        log(f"Reading the text layer of {len(pdf)} pages")
        all_pages = classify(read_text_layer(pdf))
        sections = section_ranges(all_pages)
        for run in sections:
            printed = (f", printed {run['printed_first']}-{run['printed_last']}"
                       if run.get("printed_first") else "")
            log(f"  {run['section']:<15} PDF pages {run['pdf_first']}-{run['pdf_last']} "
                f"({run['pages']}){printed}")

        chosen = [p for p in select_pages(all_pages, cfg.pages) if p.section not in cfg.hold_out]
        if not chosen:
            raise ValueError(f"No pages in range {cfg.pages!r} once {', '.join(cfg.hold_out) or 'nothing'} "
                             "is held out. The list above shows the sections and page numbers found in the "
                             "PDF; a range of PDF pages such as 'pdf:201-345' doesn't need page numbers.")
        held = [p for p in all_pages if p.section in cfg.hold_out]

        pages_dir.mkdir(parents=True, exist_ok=True)
        answer_dir.mkdir(parents=True, exist_ok=True)

        order, tags = pile_order(len(chosen), cfg)
        names: dict[int, str] = {}
        for pos, true_pos in enumerate(order):
            p = chosen[true_pos]
            if cfg.filenames == "pile":
                names[true_pos] = f"scan_{pos + 1:05d}.png"
            else:
                names[true_pos] = f"{_page_rng(cfg, p.pdf_index, 'name').getrandbits(48):012x}.png"

        rows: list[dict] = []
        log(f"Writing {len(chosen)} page images to {pages_dir}")
        for pos, true_pos in enumerate(order):
            p = chosen[true_pos]
            img = render(pdf, p.pdf_index, cfg.dpi)
            deg: dict = {}
            if _page_rng(cfg, p.pdf_index, "degrade").random() < cfg.degrade_share:
                img, deg = photocopy(img, _page_rng(cfg, p.pdf_index, "photocopy"))
            cw = _choose_rotation(cfg, p.pdf_index)
            img = rotate(img, cw)
            img.save(pages_dir / names[true_pos], optimize=False)
            tag = tags.get(true_pos, "")
            rows.append({
                "file": names[true_pos], "pile_position": pos + 1, "true_position": true_pos + 1,
                "displacement": pos - true_pos, "pdf_page": p.pdf_page, "printed_page": p.printed or "",
                "section": p.section, "document": p.document or "", "title": p.title or "",
                "rotation_cw": cw, "degraded": bool(deg),
                "skew_deg": deg.get("skew_deg", ""), "speckles": deg.get("speckles", ""),
                "blur_radius": deg.get("blur_radius", ""), "jpeg_quality": deg.get("jpeg_quality", ""),
                "edge_shadow": deg.get("edge_shadow") or "", "moved": tag,
                "width": img.width, "height": img.height, "text_chars": len(p.text.strip()),
                "text": p.text,
            })
            if (pos + 1) % 50 == 0:
                log(f"  {pos + 1} / {len(chosen)}")

        fields = [k for k in rows[0] if k != "text"]
        with open(answer_dir / "pages.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        with open(answer_dir / "pages.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (answer_dir / "true_order.txt").write_text(
            "".join(names[i] + "\n" for i in range(len(chosen))), encoding="utf-8")

        held_counts: dict[str, int] = {}
        for section in cfg.hold_out:
            these = [p for p in held if p.section == section]
            if not these:
                continue
            held_counts[section] = len(these)
            hdir = answer_dir / "held_out"
            hdir.mkdir(exist_ok=True)
            log(f"Holding out {len(these)} {section} pages")
            with open(hdir / f"{section}.txt", "w", encoding="utf-8") as txt, \
                    open(hdir / f"{section}.jsonl", "w", encoding="utf-8") as js:
                for p in these:
                    label = f"page {p.printed}" if p.printed else "unnumbered page"
                    txt.write(f"=== {label} (PDF page {p.pdf_page}) ===\n{p.text.strip()}\n\n")
                    js.write(json.dumps({"pdf_page": p.pdf_page, "printed_page": p.printed,
                                         "section": section, "text": p.text}, ensure_ascii=False) + "\n")
                    if cfg.hold_out_images:
                        (hdir / section).mkdir(exist_ok=True)
                        stem = f"pdf{p.pdf_page:04d}" + (f"_p{p.printed}" if p.printed else "")
                        render(pdf, p.pdf_index, cfg.dpi).save(hdir / section / f"{stem}.png")

        moved = {"block": sum(1 for t in tags.values() if t.startswith("block")),
                 "stray": sum(1 for t in tags.values() if t == "stray"),
                 "next_page_follows": sum(1 for a, b in zip(order, order[1:]) if b == a + 1)}
        summary = CorpusSummary(
            corpus_dir=root, pages_dir=pages_dir, answer_dir=answer_dir, pdf_pages=len(all_pages),
            pile_pages=len(rows), held_out=held_counts,
            rotated=sum(1 for r in rows if r["rotation_cw"]),
            degraded=sum(1 for r in rows if r["degraded"]), moved=moved, sections=sections)

        settings = asdict(cfg)
        settings["hold_out"] = list(cfg.hold_out)
        manifest = {
            "made_by": "pile_scan.testcorpus", "made_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "settings": settings,
            "source": {"path": str(source), "url": cfg.source_url, "sha256": _sha256(source),
                       "pages": len(all_pages)},
            "sections": sections,
            "pile": {"pages": len(rows), "rotated": summary.rotated, "degraded": summary.degraded,
                     "moved": moved, "first_pdf_page": chosen[0].pdf_page, "last_pdf_page": chosen[-1].pdf_page,
                     "first_printed": chosen[0].printed, "last_printed": chosen[-1].printed},
            "held_out": held_counts,
        }
        (answer_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (answer_dir / "README.md").write_text(ANSWER_README, encoding="utf-8")
    finally:
        pdf.close()
    log(str(summary))
    return summary
