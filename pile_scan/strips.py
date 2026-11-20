"""Cutting a page into horizontal strips for a closer look.

The model fits each image into a fixed budget of visual tokens (1120 at most)
while keeping its shape, so a strip that is a third of the page's height is
seen in about 1.7 times more detail across and down. That matters for small
marks in narrow columns, like the X's on a checkbox form.

Cuts are placed near the even split points, but moved to a nearby row of
pixels that crosses as little ink as possible: blank paper, a ruled line, or a
gap between text lines (where only a table's vertical rules are crossed). So a
line of text is not cut in half where there's a way around it.

A strip that sits inside a ruled table whose headings are higher up the page
gets a copy of the table's heading band pasted above it, under a grey bar, so
the model can line marks up with the right column. Pages without a ruled grid
(prose, letters) get nothing added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from PIL import Image

WINDOW = 0.12      # look this far (as a fraction of the height) either side of each even cut
INK = 160          # a grey level darker than this counts as ink
WIDTH = 1000       # rows are measured on a copy at most this wide
MARGIN = 2         # a cut needs this many rows either side that are just as clear
RULE = 0.5         # a ruled line runs unbroken across at least this share of the width
GRID_GAP = 0.35    # ruled lines further apart than this share of the height aren't one grid
TALL = 1.3         # a heading band is at least this much taller than the grid's typical row
MAX_HEADING = 1.0  # skip a heading band taller than this share of the strip it would sit on
BAR = 12           # height of the grey bar between the heading band and the strip
BAR_GREY = 150


@dataclass
class Rule:
    top: int
    bottom: int

    @property
    def middle(self) -> float:
        return (self.top + self.bottom) / 2


@dataclass
class Strip:
    part: int                                 # 1-based
    top: int
    bottom: int
    heading: tuple[int, int] | None = None    # rows of the heading band pasted above it
    image: Image.Image | None = field(default=None, repr=False)  # what the model is shown

    def record(self) -> dict:
        return {"part": self.part, "top": self.top, "bottom": self.bottom,
                "heading": list(self.heading) if self.heading else None}


def _row_stats(image: Image.Image) -> tuple[list[int], list[float]]:
    """For each pixel row: how many stretches of ink it crosses, and the longest
    one as a share of the width."""
    gray = image.convert("L")
    if gray.width > WIDTH:
        gray = gray.resize((WIDTH, gray.height), Image.Resampling.BOX)
    w = gray.width
    data = gray.tobytes()
    runs, longest = [], []
    for y in range(gray.height):
        row = data[y * w:(y + 1) * w]
        count = best = length = 0
        for v in row:
            if v < INK:
                if length == 0:
                    count += 1
                length += 1
                if length > best:
                    best = length
            else:
                length = 0
        runs.append(count)
        longest.append(best / w)
    return runs, longest


def ink_runs(image: Image.Image) -> list[int]:
    """For each pixel row, how many separate stretches of ink it crosses."""
    return _row_stats(image)[0]


def ruled_lines(image: Image.Image, longest: list[float] | None = None) -> list[Rule]:
    """Horizontal lines that run across at least half the page, top to bottom."""
    if longest is None:
        longest = _row_stats(image)[1]
    rules: list[Rule] = []
    for y, share in enumerate(longest):
        if share >= RULE:
            if rules and y - rules[-1].bottom <= 3:
                rules[-1].bottom = y
            else:
                rules.append(Rule(y, y))
    return rules


def _cuts(runs: list[int], h: int, parts: int) -> list[int]:
    if parts <= 1 or h < parts * 20:
        return []
    # A row's score is the worst of its neighbours, so cuts land in a gap, not on its edge.
    score = [max(runs[max(0, r - MARGIN):r + MARGIN + 1]) for r in range(h)]
    window = max(4, int(h * WINDOW))
    cuts: list[int] = []
    for k in range(1, parts):
        target = round(h * k / parts)
        lo = max((cuts[-1] if cuts else 0) + 10, target - window)
        hi = min(h - 10, target + window)
        best = min(range(lo, hi), key=lambda r: (score[r], abs(r - target)))
        # Middle of the clear stretch around it.
        top = bottom = best
        while top - 1 >= lo and score[top - 1] == score[best]:
            top -= 1
        while bottom + 1 < hi and score[bottom + 1] == score[best]:
            bottom += 1
        cuts.append((top + bottom) // 2)
    return cuts


def cut_rows(image: Image.Image, parts: int) -> list[int]:
    """Pixel rows to cut at, top to bottom (parts - 1 of them)."""
    return _cuts(ink_runs(image), image.height, parts)


def heading_band(rules: list[Rule], top: int, bottom: int, height: int) -> tuple[int, int] | None:
    """The heading band of the ruled grid that a strip from `top` to `bottom` sits in.

    The grid is the run of ruled lines, each no further than GRID_GAP of the
    page from the next, that reaches from above the strip into it (and on
    below it). Its heading
    band is its tallest band, if that is clearly taller than a typical row and
    lies wholly above the strip. None otherwise: no grid, a grid of even rows
    (no heading to find), or a strip that already shows its own headings.
    """
    max_gap = height * GRID_GAP
    inside = [r for r in rules if top + 3 < r.middle < bottom]
    above = [r for r in rules if r.middle <= top + 3]
    if not inside or not above or inside[0].middle - above[-1].middle > max_gap:
        return None
    grid = [above[-1]]
    for r in reversed(above[:-1]):
        if grid[0].middle - r.middle > max_gap:
            break
        grid.insert(0, r)
    whole = grid + inside
    for r in rules:            # the rest of the grid below the strip
        if r.middle >= bottom and r.middle - whole[-1].middle <= max_gap:
            whole.append(r)
    bands = list(zip(whole, whole[1:]))
    size = [b.middle - a.middle for a, b in bands]
    k = max(range(len(bands)), key=lambda i: size[i])
    a, b = bands[k]
    if size[k] < TALL * median(size) or b.middle > top + 3:
        return None
    return a.top, b.bottom + 1


def make_strips(image: Image.Image, parts: int) -> list[Strip]:
    """The page as `parts` strips, each with the image the model is shown."""
    runs, longest = _row_stats(image)
    h = image.height
    edges = [0, *_cuts(runs, h, parts), h]
    if len(edges) == 2:
        return [Strip(1, 0, h, None, image)]
    rules = ruled_lines(image, longest)
    out = []
    for k, (top, bottom) in enumerate(zip(edges, edges[1:]), start=1):
        band = heading_band(rules, top, bottom, h) if k > 1 else None
        if band and band[1] - band[0] > MAX_HEADING * (bottom - top):
            band = None
        out.append(Strip(k, top, bottom, band, compose(image, top, bottom, band)))
    return out


def compose(image: Image.Image, top: int, bottom: int,
            heading: tuple[int, int] | list[int] | None = None) -> Image.Image:
    """The image the model is shown for a strip: the heading band (if any), a grey bar, the strip."""
    piece = image.crop((0, top, image.width, bottom))
    if not heading:
        return piece
    head = image.crop((0, heading[0], image.width, heading[1]))
    grey = (BAR_GREY,) * len(image.getbands()) if len(image.getbands()) > 1 else BAR_GREY
    combined = Image.new(image.mode, (image.width, head.height + BAR + piece.height), "white")
    combined.paste(head, (0, 0))
    combined.paste(Image.new(image.mode, (image.width, BAR), grey), (0, head.height))
    combined.paste(piece, (0, head.height + BAR))
    return combined


def split(image: Image.Image, parts: int) -> list[Image.Image]:
    """The page as `parts` horizontal strips, without heading bands."""
    edges = [0, *cut_rows(image, parts), image.height]
    if len(edges) == 2:
        return [image]
    return [image.crop((0, top, image.width, bottom)) for top, bottom in zip(edges, edges[1:])]
