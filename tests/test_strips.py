"""Reading pages in strips, prompt v3, and tidying table rows."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from pile_scan import extract, llama, strips
from pile_scan.config import ScanConfig

sys.path.insert(0, str(Path(__file__).parent))
from test_llama import sent, start_fake  # noqa: E402


def form_page():
    """A white page with a ruled grid in the middle, like a disclosure form."""
    img = Image.new("RGB", (600, 900), "white")
    d = ImageDraw.Draw(img)
    for y in range(200, 801, 40):
        d.line([(20, y), (580, y)], fill="black", width=2)
    for y in range(40, 160, 20):
        d.text((30, y), "Some heading text above the table", fill="black")
    return img


def test_cuts_land_near_even_splits_and_cover_the_page():
    img = form_page()
    for parts in (2, 3):
        pieces = strips.split(img, parts)
        assert len(pieces) == parts
        assert sum(p.height for p in pieces) == img.height and all(p.width == img.width for p in pieces)
        for k, cut in enumerate(strips.cut_rows(img, parts), start=1):
            assert abs(cut - img.height * k / parts) <= img.height * strips.WINDOW + 1


def test_cut_avoids_text_lines():
    img = form_page()
    runs = strips.ink_runs(img)
    for cut in strips.cut_rows(img, 3):
        assert runs[cut] <= 1     # blank paper or a ruled line, never through text
    # A block of text lines with a gap between them: the cut goes in a gap.
    text = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(text)
    for y in range(5, 290, 14):
        d.text((10, y), "The quick brown fox jumps over the lazy dog", fill="black")
    cut = strips.cut_rows(text, 2)[0]
    assert strips.ink_runs(text)[cut] == 0


def test_one_strip_means_no_cut():
    assert strips.cut_rows(form_page(), 1) == [] and len(strips.split(form_page(), 1)) == 1


def test_prompt_v3_rules():
    p = extract.build_prompt({})
    assert extract.PROMPT_VERSION == 3
    assert "[?]" in p and "never draw borders" in p and "ordinary lines" in p
    strip = extract.build_strip_prompt(2, 3)
    assert "part 2 of 3" in strip and "first image is the whole page" in strip and "[?]" in strip
    meta = extract.build_meta_prompt({"who": "who filed it"})
    assert "<meta>" in meta and '"who"' in meta and "<text>" not in meta


def test_tidy_drops_empty_cells_and_number_only_rows():
    text = ("Owner | Asset | Type\n"
            "| | | |\n"
            "2. | | |\n"
            "| 3 |\n"
            "MH | Built to Last LLC | | Sale | 12/16/25\n"
            "Plain line with no table\n"
            "Page 3")
    assert extract.tidy_text(text) == ("Owner | Asset | Type\n"
                                       "MH | Built to Last LLC | Sale | 12/16/25\n"
                                       "Plain line with no table\n"
                                       "Page 3")
    parsed = extract.parse_response("<meta>{}</meta>\n<text>\nA | | B\n| |\n</text>")
    assert parsed["ocr_text"] == "A | B"


def test_reader_in_strips_sends_meta_then_each_strip_with_the_whole_page(tmp_path):
    server, port, rec = start_fake(tmp_path)
    try:
        cfg = ScanConfig(server_port=port, strips=3)
        reader = llama.LlamaPageReader(server, cfg, log=lambda *_: None)
        res = reader.read(form_page(), extract.build_prompt(cfg.questions))
    finally:
        server.stop()
    reqs = sent(rec)
    assert len(reqs) == 4      # metadata from the whole page, then three strips
    images = [[c for c in r["messages"][-1]["content"] if c["type"] == "image_url"] for r in reqs]
    assert [len(i) for i in images] == [1, 2, 2, 2]
    whole = images[0][0]["image_url"]["url"]
    assert all(i[0]["image_url"]["url"] == whole for i in images[1:])           # whole page first
    assert len({i[1]["image_url"]["url"] for i in images[1:]}) == 3              # then a different strip
    texts = [next(c["text"] for c in r["messages"][-1]["content"] if c["type"] == "text") for r in reqs]
    assert "Do not transcribe" in texts[0] and "part 3 of 3" in texts[3]

    assert not res.truncated and res.image_soft_tokens == 1080
    assert res.raw.count("<meta>") == 1 and res.raw.count("<text>") == 1
    parsed = extract.parse_response(res.raw)
    assert parsed["ocr_text"] == "Hello\nWorld\nHello\nWorld\nHello\nWorld"
    assert parsed["doc_title"] == "Fake 1"


def test_strip_that_loops_is_retried_on_its_own(tmp_path):
    server, port, rec = start_fake(tmp_path, "--loop", "1")
    try:
        cfg = ScanConfig(server_port=port, strips=2, max_new_tokens=400)
        reader = llama.LlamaPageReader(server, cfg, log=lambda *_: None)
        res = reader.read(form_page(), "unused")
    finally:
        server.stop()
    # The meta read loops first (the fake loops on the first image request) and is retried with DRY.
    assert len(sent(rec)) == 4 and res.loop_detected and res.retry_mode == "dry" and not res.truncated
