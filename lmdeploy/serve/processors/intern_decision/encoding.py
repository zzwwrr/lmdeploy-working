# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/inference/encoding.py.
# Modified for LMDeploy: render token ids only; image placeholders are expanded by the engine's processor.
"""Render a decision request with the checkpoint's trained chat template."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

from transformers import AutoProcessor, AutoTokenizer

from .schema import ANSWER_SYMBOLS, DECISION_TOKEN, CompiledExample, compile_row


@dataclass(frozen=True)
class EncodedDecision:
    compiled: CompiledExample
    # Chat-template token ids; each image is still a single ``<|image_pad|>``.
    input_ids: list[int]
    # Tokens from the row before the first ``<decision>`` to the end. Images all precede the skeleton, so this
    # distance is the same before and after the engine expands the image placeholders.
    tail: int
    # Per field, the token ids of its answer symbols in option order.
    candidate_ids: tuple[tuple[int, ...], ...]


class DecisionEncoder:
    """Preserve the trained template, image order and candidate symbol mapping.

    Only tokenizer/processor files are loaded; no model weights.
    """

    def __init__(self, checkpoint, trust_remote_code: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
        if DECISION_TOKEN not in self.tokenizer.get_added_vocab():
            raise ValueError('Checkpoint must include its trained decision tokenizer')
        self.marker_id = self.tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        if self.tokenizer.encode(DECISION_TOKEN, add_special_tokens=False) != [self.marker_id]:
            raise ValueError('Decision marker must encode to exactly one registered token')
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
        self.processor.tokenizer = self.tokenizer
        if not self.tokenizer.chat_template or not getattr(self.processor, 'chat_template', None):
            raise ValueError('Decision checkpoint must supply tokenizer and processor chat templates')
        symbol_ids = {}
        for symbol in ANSWER_SYMBOLS:
            ids = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f'Answer symbol {symbol!r} must encode to exactly one token')
            symbol_ids[symbol] = ids[0]
        if len(set(symbol_ids.values())) != len(ANSWER_SYMBOLS) or self.marker_id in symbol_ids.values():
            raise ValueError('Answer symbols must have distinct token IDs, separate from the decision marker')
        self.symbol_ids = symbol_ids
        image_token = getattr(self.processor, 'image_token', None)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token) if image_token else None

    def encode(self, row: Mapping, num_images: int = 0) -> EncodedDecision:
        """``row`` holds ``state`` and ``questions``; images are passed to the engine separately, in order."""
        compiled = compile_row({**row, 'images': [f'image-{i}' for i in range(num_images)]})
        candidate_ids = tuple(
            tuple(self.symbol_ids[symbol] for symbol in compiled.symbols[field]) for field in compiled.fields)
        messages = copy.deepcopy(compiled.messages)
        if num_images:
            messages[1]['content'] = [{
                'type': 'image'
            } if part['type'] == 'image_url' else part for part in messages[1]['content']]
        template = self.processor if num_images else self.tokenizer
        text = template.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=False,
                                            enable_thinking=False,
                                            add_vision_id=True)
        input_ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        markers = [i for i, token in enumerate(input_ids) if token == self.marker_id]
        if len(markers) != len(compiled.fields) or markers[0] < 1:
            raise ValueError('Decision marker count or position mismatch')
        if self.image_token_id is not None:
            # Special-token text inside ``state`` would otherwise create extra image slots.
            image_slots = [i for i, token in enumerate(input_ids) if token == self.image_token_id]
            if len(image_slots) != num_images or any(i > markers[0] for i in image_slots):
                raise ValueError('Image placeholder count or position mismatch')
        return EncodedDecision(compiled=compiled,
                               input_ids=input_ids,
                               tail=len(input_ids) - (markers[0] - 1),
                               candidate_ids=candidate_ids)
