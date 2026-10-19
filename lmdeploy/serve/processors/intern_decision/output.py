# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/inference/engine.py and pipeline.py.
# Modified for LMDeploy: frozen input mapping, finite checks and zero-generation usage.
"""Decode next-token scores over each question's legal candidates."""
import math

import torch

from .protocol import DecisionError, DecisionResponse


class DecisionOutputProcessor:

    def __call__(self, prepared, logits, *, request_id, model, thinking, timing):
        if (not 1 <= len(prepared.fields) <= 16 or not isinstance(logits, torch.Tensor) or logits.ndim != 2
                or logits.shape[0] != len(prepared.fields) or not logits.is_floating_point()
                or not bool(torch.isfinite(logits).all())):
            raise DecisionError('Expected finite floating decision logits with shape [questions, vocabulary]')
        if not (len(prepared.fields) == len(prepared.candidate_ids) == len(prepared.option_values) == len(
                prepared.question_types) == len(prepared.legends) == len(prepared.read_positions)):
            raise DecisionError('Decision field mapping is inconsistent')
        answers = {}
        for i, field in enumerate(prepared.fields):
            ids, values = prepared.candidate_ids[i], prepared.option_values[i]
            if (not ids or len(ids) != len(values) or len(set(values)) != len(values)
                    or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= logits.shape[1] for x in ids)
                    or not 0 <= prepared.read_positions[i] < prepared.input_length):
                raise DecisionError('Invalid decision candidate or position mapping')
            candidate_logits = logits[i, list(ids)].float()
            if not bool(torch.isfinite(candidate_logits).all()):
                raise DecisionError('Candidate logits are not finite in float32')
            probabilities = torch.softmax(candidate_logits, dim=-1).cpu().tolist()
            probs = dict(zip(values, probabilities))
            best = min(values, key=lambda value: (-probs[value], value))
            kind = prepared.question_types[i]
            answer = dict(type=kind, probabilities=probs, confidence=probs[best], decision=best, source='local')
            if kind == 'choice':
                answer['choice'] = best
            elif kind == 'score':
                answer['score'] = sum(float(value) * probs[value] for value in values)
                if not math.isfinite(answer['score']):
                    raise DecisionError('Score expectation is not finite')
                answer['legend'] = prepared.legends[i]
            elif kind == 'noul' and values == ('no', 'yes'):
                answer['noul'] = probs['yes']
            else:
                raise DecisionError('Invalid decision question type')
            answers[field] = answer
        count = len(answers)
        return DecisionResponse(request_id=request_id,
                                model=model,
                                answers=answers,
                                usage=dict(input_tokens=prepared.input_length,
                                           output_tokens=count,
                                           decision_count=count,
                                           generated_tokens=0),
                                timing=timing,
                                thinking=dict(enabled=False,
                                              confidence_threshold=thinking.confidence_threshold,
                                              tasks=[]))
