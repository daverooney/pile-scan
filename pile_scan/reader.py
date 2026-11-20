"""What the scan loop expects from a page reader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ReadResult:
    raw: str                         # the model's reply, unparsed
    output_tokens: int
    image_soft_tokens: int | None    # visual tokens the model actually saw, if known
    elapsed_s: float
    truncated: bool = False          # cut off at the token limit, or stopped in a loop
    loop_detected: bool = False      # a reading got stuck repeating itself
    retry_mode: str | None = None    # "dry" or "longer" if the page was read a second time
    strips_json: str | None = None   # per-strip cuts and replies, when read in strips


class PageReader(Protocol):
    def read(self, image, prompt: str) -> ReadResult: ...
