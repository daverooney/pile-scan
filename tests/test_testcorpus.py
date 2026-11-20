import csv
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from pile_scan import testcorpus as tc
from tests.textpdf import write_text_pdf


def body(n: int, word: str = "text") -> list[str]:
    return [f"Some {word} line {k} of the page." for k in range(n)]


def mini_volume() -> list[list[str]]:
    """A 36-page stand-in for a volume of the United States Reports."""
    pages: list[list[str]] = []
    pages.append(["- - - - - - - -", "UNITED STATES", "REPORTS", "586",
                  "The dashed line indicates the bottom edge of the book.",
                  "This camera copy was created for books with spines up to 3 inches wide."])
    pages.append(["UNITED STATES REPORTS", "VOLUME 586", "CASES ADJUDGED", "IN", "THE SUPREME COURT"])
    pages.append(["JUSTICES", "of the", "SUPREME COURT", *body(5), "iii"])
    pages.append(["TABLE OF CASES REPORTED", "Page", "Alpha v. Beta . . . . . . 1", "Gamma v. Delta . . . 9", "xv"])
    pages.append(["xvi TABLE OF CASES REPORTED", "Page", "Kappa v. Lambda . . . . 803",
                  "Mu v. Nu . . . . . . . . . . 1021,", "804"])
    pages.append(["TABLE OF CASES REPORTED xvii", "Page", "Omicron v. Pi . . . . 805"])
    # Opinion 1, pages 1-8
    pages.append(["CASES ADJUDGED", "IN THE", "SUPREME COURT OF THE UNITED STATES", "AT",
                  "OCTOBER TERM, 2018", "ALPHA FIRE DISTRICT v. BETA et al.",
                  "certiorari to the united states court of appeals for", "No. 17-1. Decided November 6, 2018",
                  *body(8, "alpha"), "1"])
    for n in range(2, 9):
        head = f"{n} ALPHA FIRE DIST. v. BETA" if n % 2 == 0 else f"Cite as: 586 U. S. 1 (2018) {n}"
        pages.append([head, "Opinion of the Court", *body(10, f"alpha{n}")])
    pages.append([])          # a blank page inside the opinions
    # Opinion 2, pages 9-20
    pages.append(["OCTOBER TERM, 2018 9", "Syllabus", "GAMMA v. DELTA FISH AND", "WILDLIFE SERVICE et al.",
                  "on petition for writ of certiorari", "No. 17-71. Decided November 27, 2018", *body(8, "gamma")])
    for n in range(10, 21):
        head = f"{n} GAMMA v. DELTA" if n % 2 == 0 else f"Cite as: 586 U. S. 9 (2018) {n}"
        pages.append([head, *body(10, f"gamma{n}")])
    # Orders 801-806
    pages.append(["Reporter's Note", "ORDERS FOR OCTOBER 1, 2018, THROUGH", "MARCH 25, 2019", "October 1, 2018",
                  *body(6, "order"), "801"])
    for n in range(802, 807):
        if n % 2 == 0:
            pages.append([f"{n} OCTOBER TERM, 2018", "October 1, 2018 586 U. S.", *body(10, f"order{n}")])
        else:
            pages.append([f"ORDERS {n}", "586 U. S. October 1, 2018", *body(10, f"order{n}")])
    # Index 1261-1263
    pages.append(["I N D E X", "(Vol. 586 U. S.)", "AGE DISCRIMINATION.", *body(5, "index"), "1261"])
    pages.append(["1262 INDEX", *body(10, "index2")])
    pages.append(["INDEX 1263", *body(10, "index3")])
    return pages


@pytest.fixture(scope="module")
def mini_pdf(tmp_path_factory) -> Path:
    return write_text_pdf(tmp_path_factory.mktemp("src") / "vol.pdf", mini_volume())


def config(tmp_path: Path, src: Path, **kw) -> tc.CorpusConfig:
    kw.setdefault("dpi", 40)
    kw.setdefault("source_url", None)
    return tc.CorpusConfig(corpus_dir=str(tmp_path / "corpus"), source_pdf=str(src), **kw)


def test_roman_numbers():
    assert tc.roman_to_int("xlv") == 45
    assert tc.roman_to_int("cci") == 201
    assert tc.roman_to_int("mix") is None
    assert tc.roman_to_int("civil") is None
    assert tc.roman_to_int("iiii") is None


@pytest.mark.parametrize("lines, expected", [
    (["2 MOUNT LEMMON FIRE DIST. v. GUIDO", "Syllabus"], ("2", "head")),
    (["Cite as: 586 U. S. 1 (2018) 3", "x"], ("3", "head")),
    (["ORDERS 803", "586 U. S. October 1, 2018"], ("803", "head")),
    (["liv TABLE OF CASES REPORTED", "Page"], ("liv", "head")),
    (["TABLE OF CASES REPORTED lv", "Page"], ("lv", "head")),
    (["OCTOBER TERM, 2018 9", "Syllabus"], ("9", "head")),
    (["OCTOBER TERM, 2018", "MOUNT LEMMON", "body", "1"], ("1", "foot")),
    (["586 U. S. October 1, 2018", "body"], (None, None)),
    (["Just a page", "of words"], (None, None)),
])
def test_printed_page(lines, expected):
    assert tc.printed_page(lines) == expected


def test_classify_mini_volume():
    pages = tc.classify(["\n".join(p) for p in mini_volume()])
    sections = [(r["section"], r["pages"]) for r in tc.section_ranges(pages)]
    assert sections == [("cover", 1), ("front_matter", 2), ("table_of_cases", 3), ("opinions", 21),
                        ("orders", 6), ("index", 3)]
    # A case's page number at the foot of a table-of-cases page doesn't start the opinions.
    assert pages[4].section == "table_of_cases"
    opinions = [p for p in pages if p.section == "opinions"]
    assert {p.document for p in opinions} == {"opinion at p. 1", "opinion at p. 9"}
    assert opinions[0].title == "ALPHA FIRE DISTRICT v. BETA et al."
    assert opinions[-1].title == "GAMMA v. DELTA FISH AND WILDLIFE SERVICE et al."
    blank = pages[14]
    assert blank.text == "" and blank.section == "opinions" and blank.document == "opinion at p. 1"
    assert {p.document for p in pages if p.section == "orders"} == {"orders October 1, 2018"}
    assert [p.printed for p in pages if p.section == "index"] == ["1261", "1262", "1263"]


def test_select_pages():
    pages = tc.classify(["\n".join(p) for p in mini_volume()])
    chosen = tc.select_pages(pages, "1-9")
    assert [p.printed for p in chosen] == ["1", "2", "3", "4", "5", "6", "7", "8", None, "9"]
    assert [p.pdf_page for p in tc.select_pages(pages, "pdf:2-3")] == [2, 3]
    assert len(tc.select_pages(pages, "all")) == len(pages)
    with pytest.raises(ValueError):
        tc.parse_page_range("9-1")


def test_partial_shuffle_mostly_local():
    n = 300
    order, tags = tc.partial_shuffle(n, random.Random(1), block_share=0.08, stray_share=0.03)
    assert sorted(order) == list(range(n))
    assert order == tc.partial_shuffle(n, random.Random(1))[0]
    assert sum(1 for t in tags.values() if t == "stray") == 9
    blocks = sum(1 for t in tags.values() if t.startswith("block"))
    assert 20 <= blocks <= 40
    # Most neighbours in the pile are neighbours (or nearly) in the true order.
    close = sum(1 for a, b in zip(order, order[1:]) if abs(b - a) <= 2)
    follows = sum(1 for a, b in zip(order, order[1:]) if b == a + 1)
    assert close > 0.85 * (n - 1) and follows > 0.7 * (n - 1)
    # Pages in the same block stay together (a block can carry a local swap with it).
    groups: dict[str, list[int]] = {}
    for t in order:
        if tags.get(t, "").startswith("block"):
            groups.setdefault(tags[t], []).append(t)
    for members in groups.values():
        assert sorted(members) == list(range(min(members), min(members) + len(members)))


def test_make_corpus(tmp_path, mini_pdf):
    cfg = config(tmp_path, mini_pdf, pages="1-20", rotate_share=0.5, degrade_share=0.5, seed=7)
    summary = tc.make_corpus(cfg, log=lambda *_: None)
    pages_dir = Path(cfg.corpus_dir) / "pages"
    answer = Path(cfg.corpus_dir) / "answer_key"
    files = sorted(p.name for p in pages_dir.iterdir())
    assert files == [f"scan_{k:05d}.png" for k in range(1, 22)]     # 20 numbered pages + the blank one
    assert summary.pile_pages == 21
    assert summary.held_out == {"cover": 1, "table_of_cases": 3, "index": 3}

    rows = list(csv.DictReader(open(answer / "pages.csv")))
    assert [r["file"] for r in rows] == files
    assert {r["section"] for r in rows} == {"opinions"}
    true = (answer / "true_order.txt").read_text().split()
    by_file = {r["file"]: r for r in rows}
    assert [by_file[f]["pdf_page"] for f in true] == [str(k) for k in range(7, 28)]
    for r in rows:
        with Image.open(pages_dir / r["file"]) as im:
            assert im.mode == "L"
            portrait = im.height > im.width
        assert portrait == (r["rotation_cw"] in ("0", "180"))
    assert any(r["rotation_cw"] != "0" for r in rows)
    assert any(r["degraded"] == "True" for r in rows) and any(r["degraded"] == "False" for r in rows)

    texts = [json.loads(line) for line in open(answer / "pages.jsonl")]
    first = next(t for t in texts if t["printed_page"] == "1")
    assert "ALPHA FIRE DISTRICT v. BETA" in first["text"] and first["title"].startswith("ALPHA")
    assert "Some index line" in (answer / "held_out" / "index.txt").read_text()
    assert len(list((answer / "held_out" / "table_of_cases").glob("*.png"))) == 3
    manifest = json.loads((answer / "manifest.json").read_text())
    assert manifest["pile"]["pages"] == 21 and manifest["settings"]["seed"] == 7

    with pytest.raises(FileExistsError):
        tc.make_corpus(cfg, log=lambda *_: None)


def test_same_seed_same_pile(tmp_path, mini_pdf):
    a = config(tmp_path / "a", mini_pdf, pages="all", rotate_share=0.3, degrade_share=0.3, hold_out_images=False)
    b = config(tmp_path / "b", mini_pdf, pages="all", rotate_share=0.3, degrade_share=0.3, hold_out_images=False)
    tc.make_corpus(a, log=lambda *_: None)
    tc.make_corpus(b, log=lambda *_: None)
    csv_a = (Path(a.corpus_dir) / "answer_key" / "pages.csv").read_text()
    assert csv_a == (Path(b.corpus_dir) / "answer_key" / "pages.csv").read_text()
    sections = {r["section"] for r in csv.DictReader(csv_a.splitlines())}
    assert sections == {"front_matter", "opinions", "orders"}
    assert not (Path(a.corpus_dir) / "answer_key" / "held_out" / "index").exists()


def test_random_names_and_no_shuffle(tmp_path, mini_pdf):
    cfg = config(tmp_path, mini_pdf, pages="1-9", shuffle="none", filenames="random", rotate_share=0)
    tc.make_corpus(cfg, log=lambda *_: None)
    rows = list(csv.DictReader(open(Path(cfg.corpus_dir) / "answer_key" / "pages.csv")))
    assert [r["true_position"] for r in rows] == [str(k) for k in range(1, 11)]
    assert all(len(r["file"]) == len("0123456789ab.png") and not r["file"].startswith("scan_") for r in rows)


def test_download_when_missing(tmp_path, mini_pdf):
    calls = []

    def fake_download(url, dest, log=print):
        calls.append(url)
        dest.write_bytes(mini_pdf.read_bytes())

    cfg = config(tmp_path, tmp_path / "src" / "v.pdf", source_url="https://example.invalid/v.pdf")
    assert tc.ensure_source(cfg, download=fake_download, log=lambda *_: None).exists()
    tc.ensure_source(cfg, download=fake_download, log=lambda *_: None)
    assert calls == ["https://example.invalid/v.pdf"]

    cfg2 = config(tmp_path, tmp_path / "src" / "w.pdf", source_url="https://example.invalid/w.pdf")
    with pytest.raises(RuntimeError, match="isn't a PDF"):
        tc.ensure_source(cfg2, download=lambda url, dest, log=print: dest.write_text("<html>"),
                         log=lambda *_: None)


def test_settings_checked(tmp_path, mini_pdf):
    with pytest.raises(ValueError):
        config(tmp_path, mini_pdf, rotate_share=1.5).validate()
    with pytest.raises(ValueError):
        config(tmp_path, mini_pdf, hold_out=("appendix",)).validate()
    assert tc.parse_hold_out("cover, table_of_cases  index") == ("cover", "table_of_cases", "index")
