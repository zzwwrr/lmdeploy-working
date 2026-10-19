# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/inference/encoding.py.
# Modified for LMDeploy: standalone inference-only protocol and input processing.
"""CPU-only decision encoding shared by Transformers and LMDeploy backends."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from .schema import DECISION_TOKEN, CompiledExample, compile_row


@dataclass(frozen=True)
class EncodedDecision:
    compiled: CompiledExample
    batch: dict[str, torch.Tensor]
    marker_positions: torch.Tensor
    candidate_ids: tuple[tuple[int, ...], ...]
    input_length: int


class DecisionEncoder:
    """Preserve the trained template, image order and candidate symbol mapping.

    This class loads tokenizer/processor files only. No model or CUDA context is
    created. Limits apply here as well as at the HTTP boundary so CLI callers
    cannot bypass them.
    """

    def __init__(self, checkpoint, processor_path=None, media_root="", max_length=8192):
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.media_root = Path(media_root or ".").resolve()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if DECISION_TOKEN not in self.tokenizer.get_added_vocab():
            raise ValueError("Checkpoint must include its trained decision tokenizer")
        self.marker_id = self.tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        self.processor = AutoProcessor.from_pretrained(processor_path or checkpoint, local_files_only=True)
        self.processor.tokenizer = self.tokenizer
        if self.tokenizer.encode(DECISION_TOKEN, add_special_tokens=False) != [self.marker_id]:
            raise ValueError('Decision marker must encode to exactly one registered token')
        if not self.tokenizer.chat_template or not self.processor.chat_template:
            raise ValueError('Decision checkpoint must supply tokenizer and processor chat templates')
        for symbol in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789':
            ids = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f'Answer symbol {symbol!r} must encode to exactly one token')
        self._candidate_ids: dict[str, int] = {}

    def _candidate_id(self, symbol: str) -> int:
        if symbol not in self._candidate_ids:
            ids = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Answer symbol {symbol!r} must encode to exactly one token")
            self._candidate_ids[symbol] = int(ids[0])
        return self._candidate_ids[symbol]

    def encode(self, row) -> EncodedDecision:
        if not isinstance(row, Mapping):
            raise ValueError("Decision input must be an object")
        questions = row.get("questions")
        if not isinstance(questions, Mapping) or not 1 <= len(questions) <= 16:
            raise ValueError("Decision input must contain 1 to 16 question fields")
        image_inputs = row.get("images")
        if image_inputs is None:
            image_inputs = []
        if not isinstance(image_inputs, (list, tuple)) or len(image_inputs) > 8:
            raise ValueError("images must be a list containing at most 8 images")
        compiled = compile_row(row, include_targets=False)
        candidate_ids = tuple(
            tuple(self._candidate_id(symbol) for symbol in compiled.symbols[field]) for field in compiled.fields
        )
        messages = copy.deepcopy(compiled.messages)
        # Both the opened file and the converted RGB image are closed even if
        # decoding or processor execution raises. Returned tensors own their data.
        with ExitStack() as resources:
            images = []
            if image_inputs:
                converted = []
                for part in messages[1]["content"]:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if not isinstance(url, (str, Path)) or not str(url):
                            raise ValueError("Each image requires a local path")
                        path = Path(url)
                        path = path if path.is_absolute() else self.media_root / path
                        with Image.open(path) as image:
                            if image.width * image.height > 16_000_000 or getattr(image, 'n_frames', 1) != 1:
                                raise ValueError('Only static images up to 16 million pixels are supported')
                            rgb = image.convert("RGB")
                        resources.callback(rgb.close)
                        images.append(rgb)
                        converted.append({"type": "image"})
                    else:
                        converted.append(part)
                messages[1]["content"] = converted
            template = self.processor if images else self.tokenizer
            text = template.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
                add_vision_id=True,
            )
            if images:
                batch = self.processor(text=[text], images=images, return_tensors="pt", padding=False)
            else:
                batch = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        batch = dict(batch)
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in batch.values()):
            raise ValueError("Decision encoder must return CPU tensors")
        ids = batch["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Decision encoder expects one unpadded sequence")
        length = int(ids.shape[-1])
        if length > self.max_length:
            raise ValueError(f"Example has {length} tokens, above {self.max_length}; truncation is forbidden")
        positions = (ids[0] == self.marker_id).nonzero().flatten().to(torch.long) - 1
        if len(positions) != len(compiled.fields) or bool((positions < 0).any()):
            raise ValueError("Decision marker count or position mismatch")
        mask = batch.get("attention_mask")
        if mask is not None and (mask.shape != ids.shape or not bool((mask == 1).all())):
            raise ValueError("Decision encoder must not pad or mask input tokens")
        return EncodedDecision(compiled, batch, positions, candidate_ids, length)
