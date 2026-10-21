# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/inputs/schema.py,
# src/inference/encoding.py and inference.py (DecisionEngine.predict, scale_result).
# Modified for LMDeploy: inference-only prompts, engine-expanded image placeholders,
# candidate-logit decoding with finite checks and temperature calibration.
"""The trained Intern-Decision prompt, token encoding and candidate-score decoding."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DECISION_TOKEN = '<decision>'
ANSWER_SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
# Temperature shipped with Intern-Decision-0.8B (inference.py, DEFAULT_TEMPERATURE).
DEFAULT_TEMPERATURE = 2.747760550703
SYSTEM_PROMPT = ('You are a careful decision assistant. Use the state and decision schema in '
                 'the user message to make the requested decisions. For every field, choose '
                 'exactly one answer symbol (e.g. A, B, C, ...) from its listed options and '
                 'return one valid JSON object mapping each field name to its chosen symbol. '
                 'Use the field names and symbols exactly as given. Do not include '
                 'explanations, Markdown, or extra text.')


@dataclass(frozen=True)
class CompiledExample:
    messages: list[dict[str, Any]]
    fields: tuple[str, ...]
    symbols: dict[str, tuple[str, ...]]


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


class DecisionOutputError(RuntimeError):
    """Candidate scores cannot be decoded into a consistent decision result."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def question_options(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return option values and descriptions in the trained answer-symbol order."""
    kind = question.get('type')
    criteria = question.get('criteria')
    if kind == 'choice':
        if not isinstance(criteria, Mapping):
            raise ValueError('choice criteria must be an object')
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == 'score':
        if isinstance(criteria, list):
            return [(str(index), str(value)) for index, value in enumerate(criteria)]
        if isinstance(criteria, Mapping):
            return [(str(key), str(value)) for key, value in criteria.items()]
        raise ValueError('score criteria must be a list or object')
    if kind == 'noul':
        descriptions = criteria if isinstance(criteria, Mapping) else {}
        yes = next(
            (str(descriptions[k]) for k in descriptions if str(k).lower() in {'yes', 'true', '1'}),
            'The answer is yes (affirmative, or align with the claim).',
        )
        no = next(
            (str(descriptions[k]) for k in descriptions if str(k).lower() in {'no', 'false', '0'}),
            'The answer is no (negative, or disagree with the claim).',
        )
        return [('no', no), ('yes', yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def _message_content(row: Mapping[str, Any], user_text: str) -> str | list[dict[str, Any]]:
    images = row.get('images') or []
    if not images:
        return user_text
    content: list[dict[str, Any]] = []
    for image in images:
        if isinstance(image, Mapping):
            url = image.get('url') or image.get('path')
            image_wh = image.get('image_wh')
        else:
            url, image_wh = str(image), None
        image_url: dict[str, Any] = {'url': url}
        if image_wh is not None:
            image_url['image_wh'] = image_wh
        content.append({'type': 'image_url', 'image_url': image_url})
    content.append({'type': 'text', 'text': user_text})
    return content


def compile_row(row: Mapping[str, Any]) -> CompiledExample:
    questions = row.get('questions')
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError(f"{row.get('id', '<missing id>')}: questions must be a non-empty object")

    fields: list[str] = []
    symbols: dict[str, tuple[str, ...]] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        if not isinstance(question, Mapping):
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} is not an object")
        options = question_options(question)
        if not options:
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} has no options")
        if len(options) > len(ANSWER_SYMBOLS):
            raise ValueError(f"At most {len(ANSWER_SYMBOLS)} single-token answer symbols are supported")
        field_name = str(field)
        fields.append(field_name)
        field_symbols = tuple(ANSWER_SYMBOLS[:len(options)])
        symbols[field_name] = field_symbols
        schema_lines.append(f"{field_name}: {question.get('instructions', '')}")
        for symbol, (value, description) in zip(field_symbols, options):
            schema_lines.append(f"    {symbol} = {value}: {description}")

    state = _json(row.get('state'))
    user_text = ("Return one answer for every field using the supplied answer symbols.\n\n"
                 "## State\n"
                 f"{state}\n"
                 "## Decision schema\n" + "\n".join(schema_lines))
    if DECISION_TOKEN in user_text or any(DECISION_TOKEN in field for field in fields):
        raise ValueError('Reserved decision marker appears in input evidence')
    skeleton = json.dumps(dict.fromkeys(fields, DECISION_TOKEN), ensure_ascii=False, indent=4)
    messages = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': _message_content(row, user_text)
        },
        {
            'role': 'assistant',
            'content': skeleton
        },
    ]
    return CompiledExample(
        messages=messages,
        fields=tuple(fields),
        symbols=symbols,
    )


class DecisionEncoder:
    """Preserve the trained template, image order and candidate symbol mapping.

    Load tokenizer/processor files and validate token mappings once per engine,
    then reuse them across requests. No model weights are loaded here.
    """

    def __init__(self, checkpoint, trust_remote_code: bool = False):
        from transformers import AutoProcessor, AutoTokenizer

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


def validate_temperature(temperature) -> float:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    return temperature


def argmax(probs: dict[str, float]) -> str:
    return min(probs, key=lambda label: (-probs[label], label))


def scale_probabilities(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Temperature-scale a candidate distribution exactly as the checkpoint's reference does."""
    if temperature == 1.0:
        return dict(probs)
    logs = {label: math.log(p) if p else -math.inf for label, p in probs.items()}
    maximum = max(logs.values())
    weights = {label: math.exp((logp - maximum) / temperature) for label, logp in logs.items()}
    total = sum(weights.values())
    result = {label: weight / total for label, weight in weights.items()}
    if argmax(result) != argmax(probs):
        raise DecisionOutputError('Floating-point temperature scaling changed argmax')
    return result


def decode_answers(questions: dict,
                   fields: tuple[str, ...],
                   candidate_logits: list[list[float]],
                   *,
                   temperature: float = DEFAULT_TEMPERATURE) -> dict:
    """Decode candidate logits in field/option order into calibrated answers."""
    import torch

    temperature = validate_temperature(temperature)
    if len(candidate_logits) != len(fields):
        raise DecisionOutputError('Decision field mapping is inconsistent')
    answers = {}
    for field, logits in zip(fields, candidate_logits):
        question = questions[field]
        options = question_options(question)
        values = [value for value, _ in options]
        logits = torch.tensor(logits, dtype=torch.float32)
        if logits.numel() != len(values) or not bool(torch.isfinite(logits).all()):
            raise DecisionOutputError('Expected one finite logit per candidate')
        probs = dict(zip(values, torch.softmax(logits, dim=-1).tolist()))
        probs = scale_probabilities(probs, temperature)
        best = argmax(probs)
        kind = question['type']
        answer = dict(type=kind, probabilities=probs, confidence=probs[best])
        if kind == 'noul':
            answer['noul'] = probs['yes']
        elif kind == 'score':
            answer['score'] = sum(float(value) * probs[value] for value in values)
            answer['legend'] = dict(options)
        else:
            answer['choice'] = best
        answer['source'] = 'local'
        answer['decision'] = best
        answers[field] = answer
    return answers
