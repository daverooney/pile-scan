"""End-to-end scan with a fake model: resume, checkpoints, sampling, failures."""

import sqlite3

from pile_scan import extract
from pathlib import Path

import pytest

from pile_scan import checkpoint, compare, db
from pile_scan.scanner import run_scan

from .conftest import FakeReader

# a_letter (1) + b_receipt (1) + c_report (3) + d_fax (2); the duplicate PNG adds nothing.
TOTAL_PAGES = 7


def _pages(path: Path) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT file_sha256, page_index, status FROM pages ORDER BY 1, 2").fetchall()
    finally:
        conn.close()


def test_full_scan(cfg):
    reader = FakeReader()
    s = run_scan(cfg, reader, log=lambda *_: None)
    assert s.pages_read == TOTAL_PAGES and reader.calls == TOTAL_PAGES
    assert s.files_unsupported == ["notes.txt"]
    rows = _pages(cfg.drive_db_path)
    assert len(rows) == TOTAL_PAGES and all(r[2] == "done" for r in rows)
    conn = sqlite3.connect(cfg.drive_db_path)
    assert conn.execute("SELECT COUNT(*) FROM file_paths").fetchone()[0] == 5  # includes duplicate path
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 4
    r = conn.execute("SELECT first_line, last_line, page_number_int, image_soft_tokens, prompt_version "
                     "FROM pages LIMIT 1").fetchone()
    assert r[0].startswith("First line") and r[1] == "Last line" and r[3] == 1080 and r[4] == extract.PROMPT_VERSION


def test_rerun_does_nothing(cfg):
    run_scan(cfg, FakeReader(), log=lambda *_: None)
    reader = FakeReader()
    s = run_scan(cfg, reader, log=lambda *_: None)
    assert reader.calls == 0 and s.pages_skipped == TOTAL_PAGES


def test_resume_after_disconnect_on_fresh_vm(cfg, tmp_path):
    # Die after 5 pages. checkpoint_every=2, so Drive should hold 4 pages.
    with pytest.raises(SystemExit):
        run_scan(cfg, FakeReader(die_after=5), log=lambda *_: None)
    # A hard kill skips the final checkpoint, but our SystemExit runs `finally`.
    # Simulate the real thing: roll Drive back to the last periodic checkpoint
    # by checking it has at least 4, then wipe local disk as a new VM would.
    on_drive = len(_pages(cfg.drive_db_path))
    assert on_drive >= 4
    for p in Path(cfg.local_dir).glob("*"):
        if p.is_file():
            p.unlink()

    reader = FakeReader()
    s = run_scan(cfg, reader, log=lambda *_: None)
    assert reader.calls == TOTAL_PAGES - on_drive
    rows = _pages(cfg.drive_db_path)
    assert len(rows) == TOTAL_PAGES and len(set((r[0], r[1]) for r in rows)) == TOTAL_PAGES
    assert s.pages_skipped == on_drive


def test_periodic_checkpoint_survives_hard_kill(cfg, monkeypatch):
    """If the process dies without running `finally`, Drive still has the last periodic checkpoint."""
    import pile_scan.scanner as scanner

    real_checkpoint = checkpoint.checkpoint
    calls = {"n": 0}

    def fake_checkpoint(conn, drive_db, *, scratch_dir):
        calls["n"] += 1
        if calls["n"] > 2:  # the third checkpoint is the one `finally` would run
            raise KeyboardInterrupt  # stand-in: skip writing, as a killed VM would
        return real_checkpoint(conn, drive_db, scratch_dir=scratch_dir)

    monkeypatch.setattr(scanner.checkpoint, "checkpoint", fake_checkpoint)
    with pytest.raises((SystemExit, KeyboardInterrupt)):
        run_scan(cfg, FakeReader(die_after=5), log=lambda *_: None)
    assert len(_pages(cfg.drive_db_path)) == 4  # two checkpoints of two pages each
    assert checkpoint.prev_path(cfg.drive_db_path).exists()


def test_sample_limit_is_deterministic(cfg, tmp_path):
    cfg.sample_limit = 3
    run_scan(cfg, FakeReader(), log=lambda *_: None)
    first = _pages(cfg.drive_db_path)
    cfg2 = type(cfg)(**{**cfg.__dict__, "output_dir": str(tmp_path / "out2"),
                        "local_dir": str(tmp_path / "local2")})
    run_scan(cfg2, FakeReader(), log=lambda *_: None)
    assert [r[:2] for r in _pages(cfg2.drive_db_path)] == [r[:2] for r in first]
    assert len(first) == 3
    rows = compare.compare(cfg.drive_db_path, cfg2.drive_db_path)
    assert len(rows) == 3 and all(r["text_similarity"] == 1.0 for r in rows)
    assert "3 pages compared" in compare.summarize(rows)


def test_failed_pages_recorded_and_retried(cfg):
    s = run_scan(cfg, FakeReader(fail_on={2}), log=lambda *_: None)
    assert s.pages_failed == 1 and s.pages_read == TOTAL_PAGES - 1
    # Without retry_errors the failed page is left alone.
    reader = FakeReader()
    run_scan(cfg, reader, log=lambda *_: None)
    assert reader.calls == 0
    cfg.retry_errors = True
    reader = FakeReader()
    run_scan(cfg, reader, log=lambda *_: None)
    assert reader.calls == 1
    conn = sqlite3.connect(cfg.drive_db_path)
    assert conn.execute("SELECT COUNT(*) FROM pages WHERE status='done'").fetchone()[0] == TOTAL_PAGES
    assert conn.execute("SELECT MAX(attempts) FROM pages").fetchone()[0] == 2


def test_inventory_mode_reads_nothing(cfg):
    s = run_scan(cfg, None, log=lambda *_: None)
    assert s.pages_total_known == TOTAL_PAGES and s.pages_read == 0
    conn = sqlite3.connect(cfg.drive_db_path)
    assert conn.execute("SELECT SUM(page_count) FROM files").fetchone()[0] == TOTAL_PAGES


def test_hash_cache_avoids_rehash(cfg, monkeypatch):
    run_scan(cfg, None, log=lambda *_: None)
    import pile_scan.files as files

    monkeypatch.setattr(files, "sha256_file", lambda *_: pytest.fail("rehashed an unchanged file"))
    run_scan(cfg, FakeReader(), log=lambda *_: None)


def test_unreadable_pdf_does_not_stop_run(cfg, pile):
    (pile / "zz_broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    s = run_scan(cfg, FakeReader(), log=lambda *_: None)
    assert "zz_broken.pdf" in s.files_unreadable
    assert s.pages_read == TOTAL_PAGES


def test_newer_schema_refused(tmp_path):
    p = tmp_path / "x.db"
    conn = db.connect(p)
    conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError):
        db.connect(p)


def test_old_database_is_upgraded_and_cut_off_pages_retried(cfg):
    """A v1 database (pages cut off were saved as done) gains the new columns,
    and RETRY_ERRORS reads its cut-off pages again."""
    from pile_scan import db

    run_scan(cfg, FakeReader(), log=lambda *_: None)
    conn = sqlite3.connect(cfg.drive_db_path)
    conn.execute("UPDATE pages SET truncated=1 WHERE page_index=0")
    conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    # Make it look like the v1 schema, without the two new columns.
    conn = sqlite3.connect(cfg.drive_db_path)
    conn.execute("ALTER TABLE pages DROP COLUMN loop_detected")
    conn.execute("ALTER TABLE pages DROP COLUMN retry_mode")
    conn.commit()
    conn.close()
    import shutil
    shutil.rmtree(cfg.local_dir)

    cfg.retry_errors = True
    reader = FakeReader()
    s = run_scan(cfg, reader, log=lambda *_: None)
    n_first_pages = 4   # one page 0 per distinct file
    assert reader.calls == n_first_pages and s.pages_read == n_first_pages
    conn = sqlite3.connect(cfg.drive_db_path)
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(db.SCHEMA_VERSION)
    assert conn.execute("SELECT COUNT(*) FROM pages WHERE truncated=1").fetchone()[0] == 0
