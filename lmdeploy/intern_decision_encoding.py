# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/inputs/schema.py,
# src/inference/encoding.py and inference.py (DecisionEngine.predict, scale_result).
# Modified for LMDeploy: inference-only prompts, engine-expanded image placeholders,
# candidate-logit decoding with finite checks and temperature calibration.
"""The trained Intern-Decision prompt, token encoding and candidate-score decoding."""

from __future__ import annotations

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
