"""SQLite schema and the small set of queries the scanner needs.

Design notes for the notebooks that come after this one:

* A page is identified by (file_sha256, page_index). The hash is of the file's
  bytes, so renaming or moving a file on Drive doesn't cause a rescan, and two
  copies of the same file are scanned once.
* `pages.id` is a stable integer rowid. The embeddings notebook can key a
  sqlite-vec virtual table on it (`rowid = pages.id`), and the sorting
  notebook can store group and order assignments in its own tables keyed on it.
* Ordering clues (title, printed page number, dates, first and last lines,
  running header and footer, mid-sentence flags) are real columns, so the
  sorting notebook can group and order pages with plain SQL before asking a
  model anything.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at      REAL NOT NULL,
    ended_at        REAL,
    model_id        TEXT,
    quantization    TEXT,
    max_soft_tokens INTEGER,
    gpu_name        TEXT,
    settings_json   TEXT,
    pages_done      INTEGER NOT NULL DEFAULT 0,
    pages_failed    INTEGER NOT NULL DEFAULT 0,
    end_reason      TEXT
);

-- One row per distinct file content.
CREATE TABLE IF NOT EXISTS files (
    file_sha256   TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,      -- path relative to the input folder, last seen
    file_name     TEXT NOT NULL,
    extension     TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    page_count    INTEGER,            -- NULL until the file has been opened
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | opened | unsupported | error
    error         TEXT,
    first_seen_at REAL NOT NULL,
    last_seen_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS files_by_path ON files(source_path);

-- Every path where a file was seen, including duplicates of the same content.
CREATE TABLE IF NOT EXISTS file_paths (
    source_path  TEXT PRIMARY KEY,
    file_sha256  TEXT NOT NULL REFERENCES files(file_sha256),
    size_bytes   INTEGER NOT NULL,
    mtime        REAL NOT NULL
);

-- One row per page (an image file is a single page; PDFs and multi-frame
-- TIFFs have several).
CREATE TABLE IF NOT EXISTS pages (
    id                  INTEGER PRIMARY KEY,
    file_sha256         TEXT NOT NULL REFERENCES files(file_sha256),
    page_index          INTEGER NOT NULL,          -- 0-based position in its file
    status              TEXT NOT NULL,             -- done | error
    -- What the model read.
    ocr_text            TEXT,
    pdf_text_layer      TEXT,                      -- embedded text, for born-digital PDFs
    truncated           INTEGER NOT NULL DEFAULT 0,-- 1 if generation hit max_new_tokens
    -- Ordering clues for the sorting pass.
    doc_title           TEXT,
    doc_type            TEXT,
    page_number_printed TEXT,                      -- as printed, e.g. "iv" or "3 of 7"
    page_number_int     INTEGER,                   -- best-effort integer
    page_total_int      INTEGER,                   -- from "3 of 7" style numbering
    dates_json          TEXT,                      -- JSON array of ISO-ish date strings
    primary_date        TEXT,
    first_line          TEXT,
    last_line           TEXT,
    running_header      TEXT,
    running_footer      TEXT,
    starts_mid_sentence INTEGER,                   -- 0/1/NULL
    ends_mid_sentence   INTEGER,                   -- 0/1/NULL
    language            TEXT,
    -- Answers to the configured factual questions, as a JSON object.
    answers_json        TEXT,
    meta_json           TEXT,                      -- full parsed metadata block
    raw_response        TEXT,
    -- Provenance.
    image_width         INTEGER,
    image_height        INTEGER,
    image_soft_tokens   INTEGER,                   -- visual tokens the model actually saw
    model_id            TEXT,
    quantization        TEXT,
    max_soft_tokens     INTEGER,
    prompt_version      INTEGER,
    run_id              TEXT,
    elapsed_s           REAL,
    output_tokens       INTEGER,
    error               TEXT,
    attempts            INTEGER NOT NULL DEFAULT 1,
    created_at          REAL NOT NULL,
    UNIQUE (file_sha256, page_index)
);
CREATE INDEX IF NOT EXISTS pages_by_status ON pages(status);
CREATE INDEX IF NOT EXISTS pages_by_title ON pages(doc_title);
CREATE INDEX IF NOT EXISTS pages_by_date ON pages(primary_date);
"""

PAGE_COLUMNS = (
    "status", "ocr_text", "pdf_text_layer", "truncated",
    "doc_title", "doc_type", "page_number_printed", "page_number_int", "page_total_int",
    "dates_json", "primary_date", "first_line", "last_line", "running_header",
    "running_footer", "starts_mid_sentence", "ends_mid_sentence", "language",
    "answers_json", "meta_json", "raw_response",
    "image_width", "image_height", "image_soft_tokens",
    "model_id", "quantization", "max_soft_tokens", "prompt_version", "run_id",
    "elapsed_s", "output_tokens", "error",
)


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # The working copy lives on local disk, so WAL is safe and fast. Snapshots
    # for Drive go through the backup API (see checkpoint.py), which produces a
    # single self-contained file.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    elif int(row["value"]) > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema v{row['value']} is newer than this code (v{SCHEMA_VERSION}). "
            "Update pile-scan before scanning into it."
        )
    conn.commit()


# --- runs -------------------------------------------------------------------

def start_run(conn: sqlite3.Connection, run_id: str, *, model_id: str, quantization: str,
              max_soft_tokens: int, gpu_name: str | None, settings_json: str) -> None:
    conn.execute(
        "INSERT INTO runs(run_id, started_at, model_id, quantization, max_soft_tokens, gpu_name, settings_json)"
        " VALUES (?,?,?,?,?,?,?)",
        (run_id, time.time(), model_id, quantization, max_soft_tokens, gpu_name, settings_json),
    )
    conn.commit()


def bump_run(conn: sqlite3.Connection, run_id: str, *, ok: bool) -> None:
    col = "pages_done" if ok else "pages_failed"
    conn.execute(f"UPDATE runs SET {col} = {col} + 1 WHERE run_id=?", (run_id,))


def end_run(conn: sqlite3.Connection, run_id: str, reason: str) -> None:
    conn.execute("UPDATE runs SET ended_at=?, end_reason=? WHERE run_id=?", (time.time(), reason, run_id))
    conn.commit()


# --- files ------------------------------------------------------------------

def cached_hash(conn: sqlite3.Connection, source_path: str, size: int, mtime: float) -> str | None:
    """Return the hash recorded for this path if its size and mtime are unchanged.

    Hashing means reading every byte through the Drive mount, which is slow,
    so a restart reuses earlier hashes for files that haven't changed.
    """
    row = conn.execute(
        "SELECT file_sha256 FROM file_paths WHERE source_path=? AND size_bytes=? AND mtime=?",
        (source_path, size, mtime),
    ).fetchone()
    return row["file_sha256"] if row else None


def upsert_file(conn: sqlite3.Connection, *, sha256: str, source_path: str, file_name: str,
                extension: str, size: int, mtime: float) -> None:
    now = time.time()
    conn.execute(
        """INSERT INTO files(file_sha256, source_path, file_name, extension, size_bytes, mtime,
                             first_seen_at, last_seen_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(file_sha256) DO UPDATE SET
               source_path=excluded.source_path, file_name=excluded.file_name,
               mtime=excluded.mtime, last_seen_at=excluded.last_seen_at""",
        (sha256, source_path, file_name, extension, size, mtime, now, now),
    )
    conn.execute(
        """INSERT INTO file_paths(source_path, file_sha256, size_bytes, mtime) VALUES (?,?,?,?)
           ON CONFLICT(source_path) DO UPDATE SET
               file_sha256=excluded.file_sha256, size_bytes=excluded.size_bytes, mtime=excluded.mtime""",
        (source_path, sha256, size, mtime),
    )


def set_file_status(conn: sqlite3.Connection, sha256: str, status: str, *,
                    page_count: int | None = None, error: str | None = None) -> None:
    conn.execute(
        "UPDATE files SET status=?, page_count=COALESCE(?, page_count), error=? WHERE file_sha256=?",
        (status, page_count, error, sha256),
    )


def get_file(conn: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM files WHERE file_sha256=?", (sha256,)).fetchone()


# --- pages ------------------------------------------------------------------

def finished_pages(conn: sqlite3.Connection, sha256: str, *, retry_errors: bool) -> set[int]:
    """Page indexes of this file that need no more work."""
    if retry_errors:
        q = "SELECT page_index FROM pages WHERE file_sha256=? AND status='done'"
    else:
        q = "SELECT page_index FROM pages WHERE file_sha256=?"
    return {r["page_index"] for r in conn.execute(q, (sha256,))}


def save_page(conn: sqlite3.Connection, sha256: str, page_index: int, values: dict[str, Any]) -> None:
    unknown = set(values) - set(PAGE_COLUMNS)
    if unknown:
        raise KeyError(f"unknown page columns: {sorted(unknown)}")
    cols = ["file_sha256", "page_index", "created_at", *values.keys()]
    params = [sha256, page_index, time.time(), *values.values()]
    updates = ", ".join(f"{c}=excluded.{c}" for c in ["created_at", *values.keys()])
    conn.execute(
        f"INSERT INTO pages({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        f" ON CONFLICT(file_sha256, page_index) DO UPDATE SET {updates}, attempts = pages.attempts + 1",
        params,
    )


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    out = {"files": conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]}
    for row in conn.execute("SELECT status, COUNT(*) AS n FROM pages GROUP BY status"):
        out[f"pages_{row['status']}"] = row["n"]
    for row in conn.execute("SELECT status, COUNT(*) AS n FROM files GROUP BY status"):
        out[f"files_{row['status']}"] = row["n"]
    return out


def json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def iter_rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, tuple(params))]
