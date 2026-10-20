# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Fixture cases adapted from Intern-Decision (0a8a2388), verify_lmdeploy_decisions.py.
# Modified for LMDeploy: frozen expectations with no external source dependency.
"""Golden checks against ``fixtures/intern_decision_v1.json``.

The fixture was recorded with the v1 integration (uncalibrated probabilities, fully expanded HF encodings). v2 renders
unexpanded token ids and calibrates, so text encodings are compared exactly, image cases through their expanded
lengths, and probabilities after applying the same calibration to the fixture.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from lmdeploy.intern_decision_encoding import DEFAULT_TEMPERATURE, scale_probabilities

GOLDEN = Path(__file__).parent / 'fixtures/intern_decision_v1.json'


def cases(directory):
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    for name, size, color in (('red', (256, 256), (240, 20, 20)), ('blue', (320, 384), (20, 20, 240))):
        with Image.new('RGB', size, color) as image:
            image.save(directory / f"{name}.png")
    questions = {
        'action': {
            'type': 'choice',
            'instructions': 'Select the safest action.',
            'criteria': {
                'stop': 'Stop at the red traffic light.',
                'go': 'Drive through the red traffic light.'
            }
        },
        'red': {
            'type': 'noul',
            'instructions': 'Is the traffic light red?'
        },
        'risk': {
            'type': 'score',
            'instructions': 'How dangerous is driving through this light?',
            'criteria': {
                '0': 'Safe',
                '1': 'Some risk',
                '2': 'Dangerous'
            }
        },
    }
    result = [
        ('text', {
            'state': 'The traffic light is red. There are pedestrians crossing.',
            'questions': questions
        }),
        ('long_text', {
            'state': 'Observe the road carefully. ' * 260 + 'The traffic light is red.',
            'questions': questions
        }),
    ]
    color_question = {
        'color': {
            'type': 'choice',
            'instructions': 'What is the main color of the FIRST image?',
            'criteria': {
                'red': 'Red',
                'blue': 'Blue',
                'green': 'Green'
            }
        }
    }
    for name, names in (('red_image', ['red']), ('blue_image', ['blue']), ('two_images', ['red', 'blue']),
                        ('two_images_reversed', ['blue', 'red']), ('eight_images', ['red', 'blue'] * 4)):
        result.append((name, {
            'state': 'Inspect the supplied images in their given order.',
            'images': [str(directory / f"{n}.png") for n in names],
            'questions': color_question
        }))
    states = [
        'The traffic light is green and the road is empty.', 'The traffic light is yellow.',
        'The light is red and a child is crossing.', '红灯亮着，行人正在过马路。', '绿灯亮着，前方道路畅通。',
        'The traffic light is red, but there are no pedestrians.', 'A police officer signals STOP.',
        'The vehicle is already stationary at a red light.', 'There is heavy rain and low visibility.',
        'An ambulance is approaching the crossing.', 'The traffic signal is not working.',
        'A bus is blocking the view of the crossing.', 'The pedestrian signal says WALK.',
        'A red light is visible in the distance.', 'A car ahead has stopped suddenly.',
        'The road is closed for repairs.', 'The driver is approaching a school zone.',
        'The road is icy near the intersection.', 'A cyclist enters the crossing.',
        'There is a stop sign before the intersection.'
    ]
    result.extend((f"fixed_{i:02d}", {
        'state': state,
        'questions': copy.deepcopy(questions)
    }) for i, state in enumerate(states))
    result.append(('sixteen_fields', {
        'state': 'The light is red.',
        'questions': {
            f"field_{i}": {
                'type': 'noul',
                'instructions': 'Is the light red?'
            }
            for i in range(16)
        }
    }))
    return result


def _sha256(ids):
    return hashlib.sha256(torch.tensor([ids], dtype=torch.int64).view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.fixture(scope='module')
def checkpoint():
    path = os.getenv('DECISION_TEST_CHECKPOINT')
    if not path:
        pytest.skip('Set DECISION_TEST_CHECKPOINT to run the Intern-Decision golden tests')
    return path


def test_encoding_matches_golden(checkpoint, tmp_path):
    from lmdeploy.intern_decision_encoding import DecisionEncoder
    encoder = DecisionEncoder(checkpoint)
    golden = json.loads(GOLDEN.read_text())['cases']
    for name, row in cases(tmp_path):
        expected = golden[name]
        num_images = len(row.get('images', []))
        encoded = encoder.encode({k: v for k, v in row.items() if k != 'images'}, num_images=num_images)
        assert list(encoded.compiled.fields) == expected['fields'], name
        assert [list(ids) for ids in encoded.candidate_ids] == expected['candidate_ids'], name
        positions = expected['encoding']['marker_positions']
        # The first scored row sits ``tail`` tokens before the end, before and after image expansion.
        assert expected['input_tokens'] - encoded.tail == positions[0], name
        if num_images == 0:
            assert len(encoded.input_ids) == expected['input_tokens'], name
            assert _sha256(encoded.input_ids) == expected['encoding']['tensors']['input_ids']['sha256'], name


def test_real_checkpoint_gpu(checkpoint, tmp_path):
    if os.getenv('DECISION_TEST_GPU') != '1':
        pytest.skip('Set DECISION_TEST_GPU=1 on an idle GPU')
    from lmdeploy import PytorchEngineConfig, pipeline
    golden = json.loads(GOLDEN.read_text())['cases']
    pipe = pipeline(checkpoint, task='intern-decision', backend_config=PytorchEngineConfig(session_len=8192))
    try:
        for name, row in cases(tmp_path):
            result = pipe.decide(copy.deepcopy(row))
            expected = golden[name]
            assert list(result['answers']) == expected['fields'], name
            assert result['usage']['input_tokens'] == expected['input_tokens'], name
            assert result['usage']['generated_tokens'] == 0
            for answer, probs in zip(result['answers'].values(), expected['probabilities']):
                calibrated = scale_probabilities(dict(zip(answer['probabilities'], probs)), DEFAULT_TEMPERATURE)
                # BF16 kernels differ between the v1 full-logits path and the v2 scoring path.
                torch.testing.assert_close(torch.tensor(list(answer['probabilities'].values())),
                                           torch.tensor(list(calibrated.values())),
                                           rtol=0,
                                           atol=2e-2)
        assert not pipe.async_engine.session_mgr.sessions
    finally:
        pipe.close()
