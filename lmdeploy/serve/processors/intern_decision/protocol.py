# Copyright (c) OpenMMLab. All rights reserved.
"""Shared Python/HTTP contract for decision protocol v1."""
import json
import math
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DecisionError(RuntimeError):
    status_code = 500


class DecisionUnavailable(DecisionError):
    status_code = 503


class DecisionQueueFull(DecisionError):
    status_code = 429


class DecisionTimeout(DecisionError):
    status_code = 504


class ThinkingOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    enabled: bool = False
    confidence_threshold: float = 0.7

    @field_validator('enabled')
    @classmethod
    def disabled(cls, value):
        if value:
            raise ValueError('thinking.enabled=true is not supported')
        return value

    @field_validator('confidence_threshold', mode='before')
    @classmethod
    def threshold(cls, value):
        if isinstance(value, bool) or not isinstance(value,
                                                     (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('confidence_threshold must be a finite number in [0, 1]')
        return float(value)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    state: Any
    questions: dict[str, dict[str, Any]]
    images: list[Any] = Field(default_factory=list, max_length=8)
    model: str | None = None
    thinking: ThinkingOptions = Field(default_factory=ThinkingOptions)
    # Training-only metadata is accepted but never enters the inference prompt.
    targets: Any = Field(default=None, exclude=True)
    answer: Any = Field(default=None, exclude=True)
    id: Any = Field(default=None, exclude=True)

    @field_validator('state')
    @classmethod
    def json_state(cls, value):
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('state must contain finite JSON data') from exc
        return value

    @field_validator('questions')
    @classmethod
    def valid_questions(cls, questions):
        if not 1 <= len(questions) <= 16:
            raise ValueError('Supply 1–16 questions')
        clean = {}
        for name, question in questions.items():
            if not name:
                raise ValueError('Question names must be nonempty strings')
            kind, criteria = question.get('type'), question.get('criteria')
            if kind not in ('choice', 'score', 'noul'):
                raise ValueError('Question type must be choice, score, or noul')
            if kind == 'choice' and not isinstance(criteria, dict):
                raise ValueError('Choice criteria must be an object')
            if kind == 'score':
                if not isinstance(criteria, (list, dict)):
                    raise ValueError('Score criteria must be a list or object')
                if isinstance(criteria, dict):
                    try:
                        if any(isinstance(key, bool) or not math.isfinite(float(key)) for key in criteria):
                            raise ValueError('Score option keys must be finite numbers')
                    except (TypeError, ValueError) as exc:
                        raise ValueError('Score option keys must be finite numbers') from exc
            if kind != 'noul' and not 1 <= len(criteria) <= 62:
                raise ValueError('Supply 1–62 options')
            if isinstance(criteria, dict) and any(not isinstance(key, str) for key in criteria):
                raise ValueError('Option keys must be strings')
            clean[name] = {key: question[key] for key in ('type', 'instructions', 'criteria') if key in question}
        return clean


@dataclass(frozen=True)
class PreparedDecision:
    input_ids: list[int]
    multimodal: list[dict] | None
    read_positions: tuple[int, ...]
    candidate_ids: tuple[tuple[int, ...], ...]
    fields: tuple[str, ...]
    option_values: tuple[tuple[str, ...], ...]
    question_types: tuple[str, ...]
    legends: tuple[dict[str, str], ...]
    input_length: int


class DecisionResponse(BaseModel):
    request_id: str
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    timing: dict[str, float]
    thinking: dict[str, Any]
