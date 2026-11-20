"""The scan loop: find files, skip finished pages, read the rest, checkpoint."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from . import checkpoint, db, extract, files
from .config import ScanConfig
from .reader import PageReader


@dataclass
class ScanSummary:
    run_id: str
    end_reason: str = "finished"
    pages_read: int = 0
    pages_failed: int = 0
    pages_truncated: int = 0   # read, but still cut off after a retry
    pages_skipped: int = 0
    files_seen: int = 0
    files_unsupported: list[str] = field(default_factory=list)
    files_unreadable: list[str] = field(default_factory=list)
    files_duplicate: list[str] = field(default_factory=list)
    pages_total_known: int = 0
    checkpoints: int = 0
    seconds: float = 0.0


def _short(s: str | None, n: int = 60) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def run_scan(cfg: ScanConfig, reader: PageReader | None, *, log=print,
             gpu_name: str | None = None) -> ScanSummary:
    """Scan everything in `cfg.input_dir` that isn't already in the database.

    With `reader=None` it does an inventory only: hashes files and counts
    pages without loading a model. Useful to check Drive access and see how big
    the pile is before spending GPU time.
    """
    cfg.validate()
    t_start = time.time()
    local_db, drive_db = cfg.local_db_path, cfg.drive_db_path
    scratch = Path(cfg.local_dir) / "scratch"

    log(checkpoint.restore(drive_db, local_db, log=log))
    conn = db.connect(local_db)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    summary = ScanSummary(run_id=run_id)
    if reader is not None:
        db.start_run(conn, run_id, model_id=cfg.model_label, quantization=cfg.quantization,
                     max_soft_tokens=cfg.max_soft_tokens, gpu_name=gpu_name, settings_json=cfg.to_json())

    prompt = extract.build_prompt(cfg.questions)
    since_checkpoint = 0
    checked_budget = False

    def do_checkpoint() -> None:
        nonlocal since_checkpoint
        secs = checkpoint.checkpoint(conn, drive_db, scratch_dir=scratch)
        summary.checkpoints += 1
        since_checkpoint = 0
        c = db.counts(conn)
        log(f"  checkpoint → {drive_db} ({secs:.1f}s) · {c.get('pages_done', 0)} done, "
            f"{c.get('pages_truncated', 0)} cut off, {c.get('pages_error', 0)} failed in total")

    found = files.discover(cfg.input_dir, recursive=cfg.recursive)
    seen_hashes: set[str] = set()
    log(f"Found {len(found)} files in {cfg.input_dir}")

    try:
        for f in found:
            if cfg.sample_limit is not None and summary.pages_read + summary.pages_failed >= cfg.sample_limit:
                summary.end_reason = "sample_limit"
                break
            summary.files_seen += 1
            if not f.supported:
                summary.files_unsupported.append(f.rel_path)
                continue

            sha = db.cached_hash(conn, f.rel_path, f.size, f.mtime) or files.sha256_file(f.path)
            db.upsert_file(conn, sha256=sha, source_path=f.rel_path, file_name=f.path.name,
                           extension=f.extension, size=f.size, mtime=f.mtime)
            if sha in seen_hashes:
                summary.files_duplicate.append(f.rel_path)
                conn.commit()
                continue
            seen_hashes.add(sha)
            row = db.get_file(conn, sha)
            n_pages = row["page_count"]
            if n_pages is None:
                try:
                    n_pages = files.page_count(f.path)
                    db.set_file_status(conn, sha, "opened", page_count=n_pages)
                except Exception as exc:  # corrupt or password-protected file
                    db.set_file_status(conn, sha, "error", error=f"{type(exc).__name__}: {exc}")
                    summary.files_unreadable.append(f.rel_path)
                    log(f"! could not open {f.rel_path}: {exc}")
                    continue
            summary.pages_total_known += n_pages
            conn.commit()

            done = db.finished_pages(conn, sha, retry_errors=cfg.retry_errors)
            summary.pages_skipped += len(done)
            if reader is None or len(done) >= n_pages:
                continue

            try:
                for page in files.iter_pages(f.path, dpi=cfg.pdf_dpi, max_side=cfg.max_image_side, skip=done):
                    if cfg.sample_limit is not None and summary.pages_read + summary.pages_failed >= cfg.sample_limit:
                        summary.end_reason = "sample_limit"
                        break
                    label = f"{f.rel_path} p{page.index + 1}/{n_pages}"
                    base = {
                        "pdf_text_layer": page.text_layer,
                        "image_width": page.image.width,
                        "image_height": page.image.height,
                        "model_id": cfg.model_label,
                        "quantization": cfg.quantization,
                        "max_soft_tokens": cfg.max_soft_tokens,
                        "prompt_version": extract.PROMPT_VERSION,
                        "run_id": run_id,
                    }
                    try:
                        result = reader.read(page.image, prompt)
                        values = extract.parse_response(result.raw)
                        values["truncated"] = int(bool(values["truncated"] or result.truncated))
                        values.update(base, status="truncated" if values["truncated"] else "done",
                                      loop_detected=int(result.loop_detected), retry_mode=result.retry_mode,
                                      strips_json=getattr(result, "strips_json", None),
                                      elapsed_s=result.elapsed_s,
                                      output_tokens=result.output_tokens,
                                      image_soft_tokens=result.image_soft_tokens, error=None)
                        db.save_page(conn, sha, page.index, values)
                        db.bump_run(conn, run_id, ok=True)
                        summary.pages_read += 1
                        if not checked_budget:
                            checked_budget = True
                            log(f"  image tokens on first page: {result.image_soft_tokens} "
                                f"(budget {cfg.max_soft_tokens}; a little under is normal, "
                                f"near 280 means the setting was ignored)")
                        flag = ""
                        if result.loop_detected:
                            flag += " [loop stopped" + (f", re-read ({result.retry_mode})]" if result.retry_mode else "]")
                        elif result.retry_mode:
                            flag += f" [re-read ({result.retry_mode})]"
                        if values["truncated"]:
                            flag += " [TRUNCATED]"
                            summary.pages_truncated += 1
                        log(f"[{summary.pages_read}] {label} · {result.elapsed_s:.1f}s · "
                            f"{result.output_tokens} tok · {_short(values['doc_title'], 40)}{flag}")
                    except Exception as exc:
                        # e.g. the server ran out of GPU memory and exited: let the
                        # reader bring it back before the next page.
                        if hasattr(reader, "recover"):
                            reader.recover()
                        db.save_page(conn, sha, page.index, {
                            **base, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:2000],
                        })
                        db.bump_run(conn, run_id, ok=False)
                        summary.pages_failed += 1
                        log(f"! {label}: {type(exc).__name__}: {_short(str(exc), 200)}")
                    conn.commit()
                    since_checkpoint += 1
                    if since_checkpoint >= cfg.checkpoint_every:
                        do_checkpoint()
            except Exception as exc:  # a page that won't render: note it and move on
                db.set_file_status(conn, sha, "error", error=f"{type(exc).__name__}: {exc}")
                summary.files_unreadable.append(f.rel_path)
                log(f"! stopped reading {f.rel_path}: {exc}")
    except KeyboardInterrupt:
        summary.end_reason = "interrupted"
        log("Interrupted; saving progress.")
    finally:
        summary.seconds = time.time() - t_start
        if reader is not None:
            db.end_run(conn, run_id, summary.end_reason)
        do_checkpoint()
        conn.close()

    log(f"Run {run_id} {summary.end_reason}: read {summary.pages_read} pages, "
        f"{summary.pages_truncated} still cut off, {summary.pages_failed} failed, "
        f"{summary.pages_skipped} already done, in {summary.seconds / 60:.1f} min.")
    if summary.pages_truncated:
        log("Pages still cut off are saved with status 'truncated'. Tick RETRY_ERRORS to read them again.")
    if summary.files_duplicate:
        log(f"{len(summary.files_duplicate)} files were exact copies of others and were scanned once.")
    if summary.files_unsupported:
        log(f"Skipped {len(summary.files_unsupported)} unsupported files, e.g. {summary.files_unsupported[:3]}")
    return summary
