"""Moving the database between local disk and Google Drive safely.

The working database lives on the Colab VM's local disk. At checkpoints we
take a consistent snapshot with SQLite's backup API, copy it to Drive under a
temporary name, check it opens cleanly, then rename it over the real file. The
previous copy is kept alongside as `<name>.prev`, so even a kill between the
two renames leaves one good copy on Drive.

What this can't protect against: Drive syncs in the background, so a
checkpoint written seconds before Colab kills the VM may not have reached
Google's servers. The previous checkpoint will still be there, so the loss is
at most `checkpoint_every` pages plus whatever was in flight.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from pathlib import Path


def prev_path(path: Path) -> Path:
    return path.with_name(path.name + ".prev")


def is_healthy(path: Path, *, read_only: bool = True) -> bool:
    """True if `path` is a SQLite database that passes a quick integrity check."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        uri = f"file:{path}?mode=ro" if read_only else f"file:{path}"
        conn = sqlite3.connect(uri, uri=True)
        try:
            result = conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
        return result is not None and result[0] == "ok"
    except sqlite3.DatabaseError:
        return False


def progress(path: Path, *, read_only: bool = True) -> tuple[int, int, float]:
    """(pages done, pages recorded, newest page timestamp); zeros if unreadable.

    Used to decide which copy holds the most work. File mtimes can't be trusted
    for this: the local copy runs in WAL mode, so recent pages may sit in the
    -wal file while the main file's mtime stays old. Pages are only ever added,
    or moved from error to done, so these counts only grow.
    """
    try:
        uri = f"file:{path}?mode=ro" if read_only else f"file:{path}"
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(status='done'), 0), COUNT(*), COALESCE(MAX(created_at), 0) FROM pages"
            ).fetchone()
        finally:
            conn.close()
        return int(row[0]), int(row[1]), float(row[2])
    except sqlite3.DatabaseError:
        return 0, 0, 0.0


def restore(drive_db: Path, local_db: Path, *, log=print) -> str:
    """Bring the newest good copy of the database onto local disk.

    Returns a short description of what happened, for the notebook's log.
    A local copy that is newer than Drive's (same VM, notebook re-run without
    a disconnect) is kept, since it may hold pages not yet checkpointed.
    """
    local_db.parent.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in (drive_db, prev_path(drive_db)) if is_healthy(p)]
    for p in (drive_db, prev_path(drive_db)):
        if p.exists() and p not in candidates:
            log(f"Ignoring damaged copy on Drive: {p}")

    candidates.sort(key=progress, reverse=True)

    if local_db.exists() and is_healthy(local_db, read_only=False):
        best_drive = progress(candidates[0]) if candidates else (0, 0, 0.0)
        if progress(local_db, read_only=False) >= best_drive:
            return f"Kept local working copy {local_db} (at least as new as Drive's)."

    if not candidates:
        if local_db.exists():
            return f"No usable copy on Drive; continuing with local {local_db}."
        return "No existing database found; starting fresh."

    source = candidates[0]
    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(local_db) + suffix)
        if stale.exists():
            stale.unlink()
    shutil.copyfile(source, local_db)
    return f"Restored {source.name} from Drive to {local_db}."


def checkpoint(conn: sqlite3.Connection, drive_db: Path, *, scratch_dir: Path) -> float:
    """Copy the live database to Drive atomically. Returns seconds taken."""
    t0 = time.time()
    conn.commit()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    snapshot = scratch_dir / (drive_db.name + ".snapshot")
    if snapshot.exists():
        snapshot.unlink()
    dest = sqlite3.connect(str(snapshot))
    try:
        conn.backup(dest)
        dest.execute("PRAGMA journal_mode=DELETE")
    finally:
        dest.close()

    drive_db.parent.mkdir(parents=True, exist_ok=True)
    tmp = drive_db.with_name(f"{drive_db.name}.tmp-{os.getpid()}")
    shutil.copyfile(snapshot, tmp)
    if not is_healthy(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint copy at {tmp} failed its integrity check; Drive copy left as it was.")
    if drive_db.exists():
        os.replace(drive_db, prev_path(drive_db))
    os.replace(tmp, drive_db)
    snapshot.unlink(missing_ok=True)
    return time.time() - t0
