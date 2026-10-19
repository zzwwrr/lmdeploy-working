# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), inference.py (DecisionEngine.predict, scale_result).
# Modified for LMDeploy: candidate logits come from engine logprob rows; finite checks; zero-generation usage.
"""Turn candidate logits into calibrated choice / score / noul answers."""
import math

import torch

from .protocol import DecisionError
from .schema import _options

# Temperature shipped with Intern-Decision-0.8B (inference.py, DEFAULT_TEMPERATURE).
DEFAULT_TEMPERATURE = 2.747760550703


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
        raise DecisionError('Floating-point temperature scaling changed argmax')
    return result


class DecisionOutputProcessor:

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        self.temperature = validate_temperature(temperature)

    def __call__(self, questions: dict, fields: tuple[str, ...], candidate_logits: list[list[float]]) -> dict:
        """``candidate_logits[i]`` holds the logits of field ``i``'s answer symbols, in option order."""
        if len(candidate_logits) != len(fields):
            raise DecisionError('Decision field mapping is inconsistent')
        answers = {}
        for field, logits in zip(fields, candidate_logits):
            question = questions[field]
            options = _options(question)
            values = [value for value, _ in options]
            logits = torch.tensor(logits, dtype=torch.float32)
            if logits.numel() != len(values) or not bool(torch.isfinite(logits).all()):
                raise DecisionError('Expected one finite logit per candidate')
            probs = dict(zip(values, torch.softmax(logits, dim=-1).tolist()))
            probs = scale_probabilities(probs, self.temperature)
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

    @property
    def calibration(self) -> dict:
        return {'method': 'temperature-scaling', 'temperature': self.temperature}
