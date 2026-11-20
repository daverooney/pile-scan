"""Reading pages in strips, prompt v3, and tidying table rows."""

import json
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
    assert extract.PROMPT_VERSION >= 3
    assert "[?]" in p and "never draw borders" in p and "ordinary lines" in p
    strip = extract.build_strip_prompt(2, 3)
    assert "part 2 of 3" in strip and "first image is the whole page" in strip and "[?]" in strip
    assert "grey bar" not in strip and "grey bar" in extract.build_strip_prompt(3, 3, heading=True)
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


def ruled_form():
    """A landscape form: title text, a tall heading band, then 12 even rows with an X in each."""
    img = Image.new("RGB", (1200, 900), "white")
    d = ImageDraw.Draw(img)
    for y in range(20, 120, 16):
        d.text((30, y), "Report any purchase, sale, or exchange of securities", fill="black")
    rules = [140, 300] + list(range(345, 880, 45))
    for y in rules:
        d.line([(20, y), (1180, y)], fill="black", width=2)
    for x in range(400, 1181, 60):
        d.line([(x, 140), (x, rules[-1])], fill="black", width=1)
    for x in range(410, 1181, 60):
        d.text((x, 200), "$1-$2", fill="black")
    for y in rules[2:-1]:
        d.text((30, y + 15), "Some asset LLC", fill="black")
        d.text((470, y + 15), "X", fill="black")
    return img, rules


def test_heading_band_is_pasted_on_strips_inside_the_table():
    img, rules = ruled_form()
    pieces = strips.make_strips(img, 3)
    assert pieces[0].heading is None
    for piece in pieces[1:]:
        if piece.top > 300:
            top, bottom = piece.heading
            assert abs(top - 140) <= 3 and abs(bottom - 302) <= 3
            assert piece.image.height == (bottom - top) + strips.BAR + (piece.bottom - piece.top)
            # The grey bar sits between the copied headings and the strip.
            assert piece.image.getpixel((5, bottom - top + strips.BAR // 2))[0] == strips.BAR_GREY
    assert any(p.heading for p in pieces)


def test_no_heading_band_on_prose_or_an_even_grid():
    text = Image.new("RGB", (800, 900), "white")
    d = ImageDraw.Draw(text)
    for y in range(10, 890, 14):
        d.text((10, y), "The quick brown fox jumps over the lazy dog", fill="black")
    assert all(p.heading is None for p in strips.make_strips(text, 3))
    grid = form_page()       # rows all the same height: nothing stands out as a heading
    assert all(p.heading is None for p in strips.make_strips(grid, 3))


def test_join_strips_drops_lines_repeated_across_a_cut():
    first = "Title\nExample: | IBM Corp. (stock) NYSE | Purchase | 2/1/1X\n1"
    second = ("Example: IBM Corp. (stock) NYSE | Purchase | 2/1/1X | E | X | A\n1\n"
              "2 | (S) MH Built to Last LLC | Sale | 12/16/25")
    third = "7 | (S) MH Built to Last LLC | Sale | 12/16/25"
    text, dropped = extract.join_strips([first, second, third])
    assert text.split("\n") == ["Title", "Example: | IBM Corp. (stock) NYSE | Purchase | 2/1/1X", "1", "1",
                                 "2 | (S) MH Built to Last LLC | Sale | 12/16/25",
                                 "7 | (S) MH Built to Last LLC | Sale | 12/16/25"]
    assert dropped[1] == ["Example: IBM Corp. (stock) NYSE | Purchase | 2/1/1X | E | X | A"]
    assert dropped[0] == [] and dropped[2] == []


def test_scan_in_strips_saves_each_strip_and_the_notebook_can_show_them(tmp_path):
    import sqlite3
    from pile_scan import look
    from pile_scan.scanner import run_scan

    pile = tmp_path / "inbox"
    pile.mkdir()
    form, _ = ruled_form()
    form.save(pile / "form.png")
    server, port, rec = start_fake(tmp_path)
    try:
        cfg = ScanConfig(input_dir=str(pile), output_dir=str(tmp_path / "drive" / "out"),
                         local_dir=str(tmp_path / "local"), server_port=port, strips=3)
        reader = llama.LlamaPageReader(server, cfg, log=lambda *_: None)
        assert run_scan(cfg, reader, log=lambda *_: None).pages_read == 1
    finally:
        server.stop()
    reqs = sent(rec)
    assert len(reqs) == 4
    prompts = [next(c["text"] for c in r["messages"][-1]["content"] if c["type"] == "text") for r in reqs]
    assert any("grey bar" in p for p in prompts[2:])

    conn = sqlite3.connect(cfg.drive_db_path)
    stored, text = conn.execute("SELECT strips_json, ocr_text FROM pages").fetchone()
    saved = json.loads(stored)
    assert [s["part"] for s in saved["strips"]] == [1, 2, 3]
    assert all("Hello" in s["reply"] for s in saved["strips"]) and saved["meta_reply"]
    assert any(s["heading"] for s in saved["strips"])
    # Lines under 8 characters are never treated as repeats.
    assert text == "\n".join(["Hello\nWorld"] * 3) and saved["strips"][1]["dropped_lines"] == []

    view = look.page_strips(cfg, "form", 1)
    assert view.from_database and len(view.strips) == 3
    assert view.strips[1].reply == saved["strips"][1]["reply"]
    assert view.strips[2].image.size == next(
        p.image.size for p in strips.make_strips(form, 3) if p.part == 3)
    # A page not yet scanned shows how it would be cut.
    preview = look.page_strips(ScanConfig(input_dir=str(pile), local_dir=str(tmp_path / "x"),
                                          output_dir=str(tmp_path / "y"), strips=2), "form", 1)
    assert not preview.from_database and len(preview.strips) == 2 and preview.strips[0].reply is None
