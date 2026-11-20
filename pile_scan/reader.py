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
    truncated: bool = False          # generation stopped at the token limit


class PageReader(Protocol):
    def read(self, image, prompt: str) -> ReadResult: ...
