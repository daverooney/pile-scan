"""Loading Gemma 4 with Transformers and reading one page image.

This is the only module that needs a GPU, torch and transformers. Everything
else in pile_scan works (and is tested) without them.

Sources for the settings used here:
* Model id, image-before-text order and the 70/140/280/560/1120 token budgets:
  https://huggingface.co/google/gemma-4-12B-it
* `max_soft_tokens` on the image processor, default 280:
  https://huggingface.co/docs/transformers/main/model_doc/gemma4
* Passing the budget as `processor_kwargs={"images_kwargs": {...}}` to
  `apply_chat_template` (the flat form had no effect on transformers 5.15.1):
  https://yosuke4061.com/new_toppage/briefings/brief068_en/
We set it both ways and then count the image tokens actually produced, so the
first page's log line shows whether the budget took effect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import ScanConfig


@dataclass
class ReadResult:
    raw: str
    output_tokens: int
    image_soft_tokens: int | None
    elapsed_s: float


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


class GemmaPageReader:
    """Wraps processor + model. Call `read(image, prompt)` once per page."""

    def __init__(self, cfg: ScanConfig):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForMultimodalLM as AutoModelCls
        except ImportError:  # older transformers
            from transformers import AutoModelForImageTextToText as AutoModelCls

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU (or L4).")

        self.cfg = cfg
        self.torch = torch
        # The T4 has no bfloat16 support, so fall back to float16 there.
        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        quant_config = None
        if cfg.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )
        elif cfg.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            self.processor = AutoProcessor.from_pretrained(cfg.model_id, max_soft_tokens=cfg.max_soft_tokens)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "max_soft_tokens"):
            image_processor.max_soft_tokens = cfg.max_soft_tokens

        kwargs: dict[str, Any] = {"device_map": "auto"}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        else:
            kwargs["dtype"] = self.compute_dtype
        self.model = AutoModelCls.from_pretrained(cfg.model_id, **kwargs)
        self.model.eval()

        # Used only to count image tokens, as a check that the budget took effect.
        tok = getattr(self.processor, "tokenizer", None)
        self._image_token_id = (getattr(self.processor, "image_token_id", None)
                                or getattr(self.model.config, "image_token_id", None))
        if self._image_token_id is None and tok is not None:
            image_token = getattr(self.processor, "image_token", None)
            if image_token:
                self._image_token_id = tok.convert_tokens_to_ids(image_token)

    def _inputs(self, image, prompt: str):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},   # image before text, per Google's guidance
                {"type": "text", "text": prompt},
            ],
        }]
        common = dict(tokenize=True, add_generation_prompt=True, return_dict=True,
                      return_tensors="pt", enable_thinking=False)
        try:
            return self.processor.apply_chat_template(
                messages, **common,
                processor_kwargs={"images_kwargs": {"max_soft_tokens": self.cfg.max_soft_tokens}},
            )
        except TypeError:
            return self.processor.apply_chat_template(messages, **common)

    def read(self, image, prompt: str) -> ReadResult:
        torch = self.torch
        t0 = time.time()
        inputs = self._inputs(image, prompt).to(self.model.device)
        input_ids = inputs["input_ids"]
        soft_tokens = None
        if self._image_token_id is not None:
            soft_tokens = int((input_ids == self._image_token_id).sum().item())
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,          # greedy: transcription wants the likeliest text, not variety
                repetition_penalty=1.0,
            )
        new_tokens = out[0][input_ids.shape[-1]:]
        raw = self.processor.decode(new_tokens, skip_special_tokens=False)
        del inputs, out
        return ReadResult(raw=raw, output_tokens=int(new_tokens.shape[-1]),
                          image_soft_tokens=soft_tokens, elapsed_s=time.time() - t0)

    def free_cache(self) -> None:
        self.torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name == "OutOfMemoryError" or "CUDA out of memory" in str(exc)
