from pathlib import Path

from pile_scan import files


def test_discover_sorted_skips_hidden(pile: Path):
    found = files.discover(pile)
    rels = [f.rel_path for f in found]
    assert rels == sorted(rels)
    assert ".hidden.png" not in rels
    assert "sub/b_receipt.jpg" in rels
    assert {f.rel_path for f in found if not f.supported} == {"notes.txt"}


def test_discover_non_recursive(pile: Path):
    rels = [f.rel_path for f in files.discover(pile, recursive=False)]
    assert all("/" not in r for r in rels)


def test_hash_ignores_name(pile: Path):
    assert files.sha256_file(pile / "a_letter.png") == files.sha256_file(pile / "sub" / "a_letter_copy.png")
    assert files.sha256_file(pile / "a_letter.png") != files.sha256_file(pile / "c_report.pdf")


def test_page_counts(pile: Path):
    assert files.page_count(pile / "c_report.pdf") == 3
    assert files.page_count(pile / "d_fax.tiff") == 2
    assert files.page_count(pile / "a_letter.png") == 1


def test_pdf_pages_render_and_skip(pile: Path):
    pages = list(files.iter_pages(pile / "c_report.pdf", dpi=72, skip={1}))
    assert [p.index for p in pages] == [0, 2]
    for p in pages:
        assert p.image.mode == "RGB"
        assert p.image.width > 100


def test_tiff_frames(pile: Path):
    pages = list(files.iter_pages(pile / "d_fax.tiff"))
    assert [p.index for p in pages] == [0, 1]


def test_max_side_downscales(pile: Path):
    (page,) = files.iter_pages(pile / "a_letter.png", max_side=500)
    assert max(page.image.size) == 500


def test_transparent_png_flattened_to_white(tmp_path: Path):
    from PIL import Image

    p = tmp_path / "t.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(p)
    (page,) = files.iter_pages(p)
    assert page.image.getpixel((5, 5)) == (255, 255, 255)
