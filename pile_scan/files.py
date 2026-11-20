"""Finding files in the input folder, hashing them, and turning them into page images."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

# Large scans are legitimate; don't let Pillow refuse them as "decompression bombs".
Image.MAX_IMAGE_PIXELS = 400_000_000


@dataclass(frozen=True)
class FoundFile:
    path: Path          # absolute path
    rel_path: str       # path relative to the input folder, with forward slashes
    size: int
    mtime: float

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def supported(self) -> bool:
        return self.extension in SUPPORTED_EXTENSIONS


@dataclass
class PageImage:
    index: int
    image: Image.Image
    text_layer: str | None = None   # embedded PDF text, if any


def discover(input_dir: str | Path, *, recursive: bool = True) -> list[FoundFile]:
    """All regular files under `input_dir`, sorted by relative path.

    Sorting makes runs deterministic, so a `sample_limit` picks the same pages
    every time. Hidden files and folders (names starting with ".") are skipped.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")
    found: list[FoundFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if not recursive:
            dirnames[:] = []
        for name in filenames:
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            found.append(FoundFile(path=p, rel_path=rel, size=st.st_size, mtime=st.st_mtime))
    found.sort(key=lambda f: f.rel_path)
    return found


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _fit(image: Image.Image, max_side: int | None) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        # Flatten transparency onto white rather than black.
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.getchannel("A"))
            image = bg
        else:
            image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def page_count(path: str | Path) -> int:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    if ext in IMAGE_EXTENSIONS:
        with Image.open(path) as im:
            # Multi-page TIFFs (common from office scanners) have several frames.
            # Animated GIFs are treated as a single page.
            if ext in (".tif", ".tiff"):
                return getattr(im, "n_frames", 1)
            return 1
    raise ValueError(f"Unsupported file type: {ext}")


def iter_pages(path: str | Path, *, dpi: int = 200, max_side: int | None = 2048,
               skip: set[int] | frozenset[int] = frozenset()) -> Iterator[PageImage]:
    """Yield page images for a file, skipping page indexes in `skip`.

    Pages are rendered one at a time, so a 500-page PDF never sits in memory.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            scale = dpi / 72.0
            for i in range(len(pdf)):
                if i in skip:
                    continue
                page = pdf[i]
                try:
                    bitmap = page.render(scale=scale)
                    img = bitmap.to_pil()
                    textpage = page.get_textpage()
                    try:
                        text = textpage.get_text_range().strip() or None
                    finally:
                        textpage.close()
                finally:
                    page.close()
                yield PageImage(index=i, image=_fit(img, max_side), text_layer=text)
        finally:
            pdf.close()
        return

    if ext in IMAGE_EXTENSIONS:
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1) if ext in (".tif", ".tiff") else 1
            for i in range(n):
                if i in skip:
                    continue
                if n > 1:
                    im.seek(i)
                frame = im.copy()
                yield PageImage(index=i, image=_fit(frame, max_side))
        return

    raise ValueError(f"Unsupported file type: {ext}")
