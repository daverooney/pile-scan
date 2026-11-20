from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pile_scan.config import ScanConfig  # noqa: E402
from pile_scan.reader import ReadResult  # noqa: E402


def make_page(text: str, size=(850, 1100)) -> Image.Image:
    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).text((60, 60), text, fill="black")
    return im


@pytest.fixture
def pile(tmp_path: Path) -> Path:
    """A small mixed pile: two PNGs, a 3-page PDF, a 2-frame TIFF, a duplicate, junk."""
    inbox = tmp_path / "drive" / "inbox"
    (inbox / "sub").mkdir(parents=True)
    make_page("Letter page one").save(inbox / "a_letter.png")
    make_page("Receipt").save(inbox / "sub" / "b_receipt.jpg", quality=90)
    pages = [make_page(f"Report page {i}") for i in range(1, 4)]
    pages[0].save(inbox / "c_report.pdf", save_all=True, append_images=pages[1:], resolution=100)
    frames = [make_page(f"Fax {i}") for i in (1, 2)]
    frames[0].save(inbox / "d_fax.tiff", save_all=True, append_images=frames[1:])
    # Same bytes as a_letter.png under another name: should be scanned once.
    (inbox / "sub" / "a_letter_copy.png").write_bytes((inbox / "a_letter.png").read_bytes())
    (inbox / "notes.txt").write_text("not a scan")
    (inbox / ".hidden.png").write_bytes((inbox / "a_letter.png").read_bytes())
    return inbox


@pytest.fixture
def cfg(pile: Path, tmp_path: Path) -> ScanConfig:
    return ScanConfig(
        input_dir=str(pile),
        output_dir=str(tmp_path / "drive" / "output"),
        local_dir=str(tmp_path / "local"),
        checkpoint_every=2,
        pdf_dpi=72,
    )


class FakeReader:
    """Stands in for Gemma: returns a well-formed reply, optionally dies after N pages."""

    def __init__(self, die_after: int | None = None, fail_on: set[int] | None = None):
        self.calls = 0
        self.die_after = die_after
        self.fail_on = fail_on or set()

    def read(self, image, prompt: str) -> ReadResult:
        if self.die_after is not None and self.calls >= self.die_after:
            raise SystemExit("simulated Colab disconnect")
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("simulated model failure")
        meta = {
            "doc_title": f"Doc {self.calls}", "doc_type": "letter",
            "page_number_printed": f"Page {self.calls} of 9", "dates": ["2019-03-04"],
            "primary_date": "2019-03-04", "running_header": None, "running_footer": None,
            "starts_mid_sentence": False, "ends_mid_sentence": "true", "language": "en",
            "answers": {"summary": "A test page.", "author": None},
        }
        raw = (f"<|channel>thought\n<channel|><meta>{json.dumps(meta)}</meta>\n"
               f"<text>\nFirst line {self.calls}\n\nmiddle\nLast line\n</text><turn|>")
        return ReadResult(raw=raw, output_tokens=50, image_soft_tokens=1080, elapsed_s=0.01)
