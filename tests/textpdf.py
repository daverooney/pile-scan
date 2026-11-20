"""A minimal PDF writer with a real text layer, so tests need no PDF library beyond pypdfium2."""

from __future__ import annotations

from pathlib import Path


def _escape(line: str) -> str:
    line = line.encode("latin-1", "replace").decode("latin-1")
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_text_pdf(path: Path, pages: list[list[str]], size: tuple[int, int] = (441, 666)) -> Path:
    """Each page is a list of text lines, set top to bottom in Helvetica 10."""
    w, h = size
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")        # filled in below
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    kids = []
    for lines in pages:
        ops = ["BT", "/F1 10 Tf", "12 TL", f"40 {h - 50} Td"]
        for ln in lines:
            ops.append(f"({_escape(ln)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(add(f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 {w} {h}] "
                        f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>".encode()))
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    objects[pages_obj - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
                              f"/Count {len(kids)} >>").encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path
