# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-Decision request adaptation using the engine's existing model resources."""
from __future__ import annotations

import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lmdeploy.intern_decision_encoding import (
    ANSWER_SYMBOLS,
    DECISION_TOKEN,
    DEFAULT_TEMPERATURE,
    DecisionOutputError,
    compile_row,
    decode_answers,
    validate_temperature,
)
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.core.decision_protocol import TASK_NAME, DecisionRequest, DecisionResponse
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError


@dataclass(frozen=True)
class DecisionInput:
    """Engine input compiled from one decision request."""

    messages: list[dict]
    gen_config: GenerationConfig
    preprocess_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionConfig:
    """Small immutable model metadata; tokenizer/processor ownership stays with the engine."""

    model_name: str
    model_names: frozenset[str]
    marker_id: int
    symbol_ids: Mapping[str, int]
    image_token_id: int | None
    supports_images: bool
    max_length: int
    temperature: float


def create_decision_config(engine, model_path: str, *, max_length: int = 8192,
                           temperature: float = DEFAULT_TEMPERATURE) -> DecisionConfig:
    """Check task capabilities once without loading another tokenizer or processor."""
    if engine.backend != 'pytorch' or not isinstance(engine.backend_config, PytorchEngineConfig):
        raise ValueError(f'{TASK_NAME} requires the PyTorch backend')
    if engine.speculative_config is not None:
        raise ValueError(f'{TASK_NAME} does not support speculative decoding')
    if engine.backend_config.role != EngineRole.Hybrid:
        raise ValueError(f'{TASK_NAME} requires the hybrid engine role')
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    tokenizer = engine.tokenizer.model.model
    if DECISION_TOKEN not in tokenizer.get_added_vocab():
        raise ValueError('Checkpoint must include its trained decision tokenizer')
    if not tokenizer.chat_template:
        raise ValueError('Decision checkpoint must supply its trained chat template')
    marker_id = tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
    if tokenizer.encode(DECISION_TOKEN, add_special_tokens=False) != [marker_id]:
        raise ValueError('Decision marker must encode to exactly one registered token')
    symbol_ids = {}
    for symbol in ANSWER_SYMBOLS:
        ids = tokenizer.encode(symbol, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'Answer symbol {symbol!r} must encode to exactly one token')
        symbol_ids[symbol] = ids[0]
    if len(set(symbol_ids.values())) != len(ANSWER_SYMBOLS) or marker_id in symbol_ids.values():
        raise ValueError('Answer symbols must have distinct token IDs, separate from the decision marker')
    vocab_size = getattr(engine.hf_cfg, 'text_config', engine.hf_cfg).vocab_size
    if any(not 0 <= token < vocab_size for token in [marker_id, *symbol_ids.values()]):
        raise ValueError('Decision tokenizer ids exceed the model vocabulary')
    vision = getattr(engine, 'vl_encoder', None)
    image_token = None
    if vision is not None:
        processor = vision.model.processor
        if not getattr(processor, 'chat_template', None):
            raise ValueError('Decision checkpoint must supply its trained processor chat template')
        # The visual processor tokenizes the rendered image prompt. Do not replace
        # its tokenizer while other requests may be using it.
        visual_tokenizer = processor.tokenizer
        for text, token in [(DECISION_TOKEN, marker_id), *symbol_ids.items()]:
            if visual_tokenizer.encode(text, add_special_tokens=False) != [token]:
                raise ValueError('Decision text and visual tokenizer mappings disagree')
        image_token = getattr(processor, 'image_token', None)
    elif '<|image_pad|>' in tokenizer.get_vocab():
        image_token = '<|image_pad|>'
    return DecisionConfig(
        model_name=engine.model_name,
        model_names=frozenset((engine.model_name, os.path.basename(os.path.normpath(model_path)))),
        marker_id=marker_id,
        symbol_ids=MappingProxyType(symbol_ids),
        image_token_id=tokenizer.convert_tokens_to_ids(image_token) if image_token else None,
        supports_images=vision is not None,
        max_length=min(max_length, engine.session_len - 1),
        temperature=validate_temperature(temperature),
    )


class DecisionParser:
    """Per-request task input/output conversion; no engine execution or image ownership."""

    def __init__(self, request, config: DecisionConfig):
        self.request = DecisionRequest.model_validate(request)
        self.config = config
        self.compiled = None
        self.candidate_ids = ()
        self.score_row_indices = ()
        self.expected_score_rows = 0
        if self.request.model is not None and self.request.model not in config.model_names:
            raise ValueError(f'Unknown model: {self.request.model}')
        if self.request.images and not config.supports_images:
            raise RequestError(ErrorCode.ENGINE_UNAVAILABLE,
                               'This engine was started without a vision encoder; image requests are unavailable')

    def prepare_input(self) -> DecisionInput:
        row = self.request.model_dump(exclude={'model', 'images'})
        self.compiled = compile_row(row)
        self.candidate_ids = tuple(
            tuple(self.config.symbol_ids[symbol] for symbol in self.compiled.symbols[field])
            for field in self.compiled.fields)
        messages = self.compiled.messages
        if self.request.images:
            user_text = messages[1]['content']
            content = []
            for image in self.request.images:
                if isinstance(image, dict):
                    data = image.get('data')
                    if not isinstance(data, str):
                        raise ValueError('Image upload objects require a string "data" field')
                    if not data.startswith('data:'):
                        media_type = image.get('type')
                        if not isinstance(media_type, str) or not media_type:
                            raise ValueError('Base64 image uploads require a "type" field, e.g. "image/png"')
                        data = f'data:{media_type};base64,{data}'
                    image = data
                elif isinstance(image, Path):
                    image = str(image)
                content.append({'type': 'image_url', 'image_url': {'url': image}})
            messages[1]['content'] = [*content, {'type': 'text', 'text': user_text}]
        return DecisionInput(
            messages=messages,
            # The candidate set comes from the checkpoint, so every decision
            # request on this engine names the same ids. Within decision-only
            # batches the engine's union of requested ids therefore equals this
            # set and no per-request column projection happens.
            gen_config=GenerationConfig(max_new_tokens=0, logprobs=0, logprob_start_len=0,
                                        logprob_extra_token_ids=sorted(set(self.config.symbol_ids.values()))),
            preprocess_kwargs=dict(scoring_input=True,
                                   chat_template_kwargs=dict(enable_thinking=False, add_vision_id=True)),
        )

    def finalize_input(self, prepared) -> None:
        ids = prepared.inputs['input_ids']
        if len(ids) > self.config.max_length:
            raise ValueError(f'Input has {len(ids)} tokens, above the limit {self.config.max_length}; '
                             'truncation is not allowed')
        markers = [i for i, token in enumerate(ids) if token == self.config.marker_id]
        if len(markers) != len(self.compiled.fields) or markers[0] < 1:
            raise ValueError('Decision marker count or position mismatch')
        if self.config.image_token_id is not None:
            slots = [i for i, token in enumerate(ids) if token == self.config.image_token_id]
            if (not self.request.images and slots) or any(i >= markers[0] for i in slots):
                raise ValueError('Image placeholder count or position mismatch')
        # Scoring row 0 predicts the token at start + 1 in the expanded input.
        start = markers[0] - 1
        self.score_row_indices = tuple(position - (start + 1) for position in markers)
        self.expected_score_rows = len(ids) - (start + 1)
        prepared.gen_config.logprob_start_len = start

    def parse_output(self, output, *, timing: dict[str, float]) -> dict:
        if output.finish_reason == 'abort':
            raise RequestError(ErrorCode.ENGINE_UNAVAILABLE, 'The decision request was aborted by the engine')
        if output.logprobs is None:
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Scoring returned no input logprobs (finish_reason={output.finish_reason})')
        if len(output.logprobs) != self.expected_score_rows:
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Expected {self.expected_score_rows} scoring rows, got {len(output.logprobs)}')
        rows = [output.logprobs[i] for i in self.score_row_indices]
        try:
            logits = [[row[token] for token in ids] for row, ids in zip(rows, self.candidate_ids)]
        except KeyError as exc:
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'Scoring rows lack a requested candidate token') from exc
        try:
            answers = decode_answers(self.request.questions, self.compiled.fields, logits,
                                     temperature=self.config.temperature)
        except DecisionOutputError as exc:
            raise RequestError(ErrorCode.INTERNAL_ERROR, str(exc)) from exc
        return DecisionResponse(
            request_id=secrets.token_hex(16), model=self.config.model_name, answers=answers,
            usage=dict(input_tokens=output.input_token_len, output_tokens=len(answers),
                       decision_count=len(answers), generated_tokens=0),
            timing=timing,
            calibration=dict(method='temperature-scaling', temperature=self.config.temperature),
        ).model_dump()
