"""Cutting a page into horizontal strips for a closer look.

The model fits each image into a fixed budget of visual tokens (1120 at most)
while keeping its shape, so a strip that is a third of the page's height is
seen in about 1.7 times more detail across and down. That matters for small
marks in narrow columns, like the X's on a checkbox form.

Cuts are placed near the even split points, but moved to a nearby row of
pixels that crosses as little ink as possible: blank paper, a ruled line, or a
gap between text lines (where only a table's vertical rules are crossed). So a
line of text is not cut in half where there's a way around it.
"""

from __future__ import annotations

from PIL import Image

WINDOW = 0.12      # look this far (as a fraction of the height) either side of each even cut
INK = 160          # a grey level darker than this counts as ink
WIDTH = 1000       # rows are measured on a copy at most this wide
MARGIN = 2         # a cut needs this many rows either side that are just as clear


def ink_runs(image: Image.Image) -> list[int]:
    """For each pixel row, how many separate stretches of ink it crosses."""
    gray = image.convert("L")
    if gray.width > WIDTH:
        gray = gray.resize((WIDTH, gray.height), Image.Resampling.BOX)
    w = gray.width
    data = gray.tobytes()
    runs = []
    for y in range(gray.height):
        row = data[y * w:(y + 1) * w]
        count, inside = 0, False
        for v in row:
            dark = v < INK
            if dark and not inside:
                count += 1
            inside = dark
        runs.append(count)
    return runs


def cut_rows(image: Image.Image, parts: int) -> list[int]:
    """Pixel rows to cut at, top to bottom (parts - 1 of them)."""
    h = image.height
    if parts <= 1 or h < parts * 20:
        return []
    runs = ink_runs(image)
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


def split(image: Image.Image, parts: int) -> list[Image.Image]:
    """The page as `parts` horizontal strips (just the page itself if parts is 1)."""
    edges = [0, *cut_rows(image, parts), image.height]
    if len(edges) == 2:
        return [image]
    return [image.crop((0, top, image.width, bottom)) for top, bottom in zip(edges, edges[1:])]
