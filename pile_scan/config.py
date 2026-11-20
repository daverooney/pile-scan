"""Settings for a scan run.

Everything a run needs lives in one dataclass, so the notebook's settings cell
is the only place a person has to edit.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

# Gemma 4's visual token budget. Google's docs list exactly these values;
# higher keeps more resolution for small text. 1120 is the most detail.
ALLOWED_SOFT_TOKENS = (70, 140, 280, 560, 1120)

# GGUF builds of Gemma 4 12B for llama.cpp, from
# https://huggingface.co/unsloth/gemma-4-12b-it-GGUF (file names checked against
# the repo's file list). UD-Q4_K_XL is Unsloth's recommended 4-bit build.
GGUF_REPO = "unsloth/gemma-4-12b-it-GGUF"
GGUF_FILES = {
    "4bit": "gemma-4-12b-it-UD-Q4_K_XL.gguf",  # 7.4 GB: fits a free T4 (16 GB)
    "8bit": "gemma-4-12b-it-Q8_0.gguf",        # 12.7 GB: use an L4 (24 GB)
}
# The vision projector llama.cpp needs alongside the model for image input.
MMPROJ_FILE = "mmproj-F16.gguf"
QUANTIZATIONS = tuple(GGUF_FILES)

# llama.cpp release to build. v1.0.0 (2026-11-13) has the --image-min-tokens
# and --image-max-tokens flags Gemma 4 needs. Changing it triggers a new build.
LLAMA_CPP_REF = "v1.0.0"

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
    quantization: str = "4bit"  # "4bit" (free T4) or "8bit" (L4 or bigger)
    gguf_repo: str = GGUF_REPO
    gguf_file: str | None = None  # override the file picked by `quantization`
    mmproj_file: str = MMPROJ_FILE
    max_soft_tokens: int = 1120
    max_new_tokens: int = 3072
    ctx_size: int = 8192

    # llama.cpp server. The compiled binary is cached under `cache_dir` on Drive,
    # so only the first session on each GPU type pays for the build.
    llama_cpp_ref: str = LLAMA_CPP_REF
    cache_dir: str = "/content/drive/MyDrive/pile-scan/cache"
    # Model files are downloaded here each session (fast inside Google's network).
    model_dir: str = "/content/models"
    # Also keep model files on Drive. Saves a download per session, costs
    # 7-13 GB of Drive space.
    cache_models_on_drive: bool = False
    server_port: int = 8090

    # Page rendering.
    pdf_dpi: int = 200
    # Downscale rendered pages so the long side is at most this many pixels.
    # llama.cpp resizes again to fit the token budget, so this only saves
    # memory, upload time and time on very large scans.
    max_image_side: int = 2048

    questions: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_QUESTIONS))

    def validate(self) -> None:
        if self.max_soft_tokens not in ALLOWED_SOFT_TOKENS:
            raise ValueError(
                f"max_soft_tokens must be one of {ALLOWED_SOFT_TOKENS}, got {self.max_soft_tokens}"
            )
        if self.quantization not in QUANTIZATIONS and not self.gguf_file:
            raise ValueError(f"quantization must be one of {QUANTIZATIONS}, got {self.quantization!r}")
        # Gemma 4 encodes an image's tokens in one non-causal pass, so they must
        # fit in a single ubatch; the server is started with 2048. Leave room
        # for the prompt and the reply as well.
        if self.ctx_size < self.max_soft_tokens + self.max_new_tokens + 1024:
            raise ValueError("ctx_size is too small for max_soft_tokens + max_new_tokens + prompt")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if self.sample_limit is not None and self.sample_limit < 1:
            raise ValueError("sample_limit must be None or at least 1")
        for key in self.questions:
            if not key.isidentifier():
                raise ValueError(f"question key {key!r} must be a simple identifier")

    @property
    def model_file(self) -> str:
        return self.gguf_file or GGUF_FILES[self.quantization]

    @property
    def model_label(self) -> str:
        """Stored with every page so results can be traced to the exact weights."""
        return f"{self.gguf_repo}/{self.model_file}"

    @property
    def local_db_path(self) -> Path:
        return Path(self.local_dir) / self.db_name

    @property
    def drive_db_path(self) -> Path:
        return Path(self.output_dir) / self.db_name

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)
