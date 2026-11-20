"""Settings for a scan run.

Everything a run needs lives in one dataclass, so the notebook's settings cell
is the only place a person has to edit.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

# Gemma 4's visual token budget. Google's docs list exactly these values, and
# Transformers defaults to 280. Higher keeps more resolution for small text.
ALLOWED_SOFT_TOKENS = (70, 140, 280, 560, 1120)

QUANTIZATIONS = ("4bit", "8bit", "none")

# Default factual questions asked of every page. Keys become JSON keys in the
# `answers_json` column, so keep them short and stable.
DEFAULT_QUESTIONS: dict[str, str] = {
    "summary": "One sentence describing what this page is about.",
    "author": "Who wrote or sent this document, if shown? Otherwise null.",
    "recipient": "Who is this document addressed to, if shown? Otherwise null.",
    "organizations": "List the organizations named on the page (array of strings).",
}


@dataclass
class ScanConfig:
    # Where the pile lives (a folder on the mounted Drive) and where results go.
    input_dir: str = "/content/drive/MyDrive/pile-scan/inbox"
    output_dir: str = "/content/drive/MyDrive/pile-scan/output"
    db_name: str = "pile_scan.db"
    # Working copy of the database on the VM's local disk. SQLite on a mounted
    # Drive is unreliable, so all writes happen here and get copied out.
    local_dir: str = "/content/pile_scan_work"

    recursive: bool = True
    # Copy the database back to Drive after this many pages.
    checkpoint_every: int = 10
    # Stop after this many pages have been processed in this run (None = all).
    # Pages are visited in sorted path order, so the same limit gives the same
    # sample on every run: handy for comparing 4-bit with 8-bit.
    sample_limit: int | None = None
    # Retry pages that failed on an earlier run.
    retry_errors: bool = False

    # Model.
    model_id: str = "google/gemma-4-12B-it"
    quantization: str = "4bit"  # "4bit" (free T4), "8bit" (L4/A100), "none"
    max_soft_tokens: int = 1120
    max_new_tokens: int = 3072

    # Page rendering.
    pdf_dpi: int = 200
    # Downscale rendered pages so the long side is at most this many pixels.
    # The processor resizes again to fit the token budget, so this only saves
    # memory and time on very large scans.
    max_image_side: int = 2048

    questions: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_QUESTIONS))

    def validate(self) -> None:
        if self.max_soft_tokens not in ALLOWED_SOFT_TOKENS:
            raise ValueError(
                f"max_soft_tokens must be one of {ALLOWED_SOFT_TOKENS}, got {self.max_soft_tokens}"
            )
        if self.quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {QUANTIZATIONS}, got {self.quantization!r}")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if self.sample_limit is not None and self.sample_limit < 1:
            raise ValueError("sample_limit must be None or at least 1")
        for key in self.questions:
            if not key.isidentifier():
                raise ValueError(f"question key {key!r} must be a simple identifier")

    @property
    def local_db_path(self) -> Path:
        return Path(self.local_dir) / self.db_name

    @property
    def drive_db_path(self) -> Path:
        return Path(self.output_dir) / self.db_name

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)
