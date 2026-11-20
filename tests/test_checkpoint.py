from pathlib import Path

from pile_scan import checkpoint, db


def _db_with_pages(path: Path, n: int):
    conn = db.connect(path)
    for i in range(n):
        db.upsert_file(conn, sha256=f"h{i}", source_path=f"f{i}", file_name=f"f{i}",
                       extension=".png", size=1, mtime=0.0)
        db.save_page(conn, f"h{i}", 0, {"status": "done", "ocr_text": f"t{i}"})
    conn.commit()
    return conn


def test_checkpoint_writes_healthy_copy_and_keeps_prev(tmp_path):
    drive = tmp_path / "drive" / "scan.db"
    conn = _db_with_pages(tmp_path / "local" / "scan.db", 2)
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "scratch")
    assert checkpoint.is_healthy(drive)
    assert not checkpoint.prev_path(drive).exists()
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "scratch")
    assert checkpoint.is_healthy(checkpoint.prev_path(drive))
    assert not list(drive.parent.glob("*.tmp-*"))


def test_restore_prefers_prev_when_main_is_damaged(tmp_path):
    drive = tmp_path / "drive" / "scan.db"
    conn = _db_with_pages(tmp_path / "a" / "scan.db", 3)
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "s")
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "s")
    drive.write_bytes(b"half-written garbage")
    logs = []
    local = tmp_path / "fresh_vm" / "scan.db"
    msg = checkpoint.restore(drive, local, log=logs.append)
    assert "Restored scan.db.prev" in msg
    assert any("damaged" in m for m in logs)
    assert checkpoint.progress(local)[0] == 3


def test_restore_when_main_missing_mid_rename(tmp_path):
    drive = tmp_path / "drive" / "scan.db"
    conn = _db_with_pages(tmp_path / "a" / "scan.db", 1)
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "s")
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "s")
    drive.unlink()  # killed between the two renames
    local = tmp_path / "vm" / "scan.db"
    assert "Restored" in checkpoint.restore(drive, local)
    assert checkpoint.progress(local)[0] == 1


def test_restore_keeps_newer_local_copy(tmp_path):
    drive = tmp_path / "drive" / "scan.db"
    local = tmp_path / "local" / "scan.db"
    conn = _db_with_pages(local, 2)
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "s")
    db.upsert_file(conn, sha256="new", source_path="new", file_name="new", extension=".png", size=1, mtime=0)
    db.save_page(conn, "new", 0, {"status": "done"})
    conn.commit()  # still only in WAL / local, not yet checkpointed
    conn.close()
    assert "Kept local" in checkpoint.restore(drive, local)
    assert checkpoint.progress(local, read_only=False)[0] == 3


def test_restore_replaces_older_local_copy(tmp_path):
    drive = tmp_path / "drive" / "scan.db"
    conn = _db_with_pages(tmp_path / "other" / "scan.db", 4)
    checkpoint.checkpoint(conn, drive, scratch_dir=tmp_path / "s")
    local = tmp_path / "local" / "scan.db"
    _db_with_pages(local, 1).close()
    assert "Restored" in checkpoint.restore(drive, local)
    assert checkpoint.progress(local)[0] == 4


def test_restore_fresh(tmp_path):
    assert "starting fresh" in checkpoint.restore(tmp_path / "d" / "x.db", tmp_path / "l" / "x.db")
