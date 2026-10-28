# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Fixture cases adapted from Intern-Decision (0a8a2388), verify_lmdeploy_decisions.py.
# Modified for LMDeploy: frozen expectations with no external source dependency.
"""Golden checks of the shared preprocessing against frozen reference outputs."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


@pytest.fixture(scope='module')
def preprocessing_engine(checkpoint):
    from transformers import AutoConfig

    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.model import BaseChatTemplate
    from lmdeploy.serve.processors import MultimodalProcessor
    from lmdeploy.tokenizer import Tokenizer
    from lmdeploy.vl.engine import ImageEncoder
    # Only CPU preprocessing is exercised here; avoid a CUDA BF16 capability probe.
    backend_config = PytorchEngineConfig(dtype='float16')
    tokenizer = Tokenizer(checkpoint)
    vision = ImageEncoder(checkpoint, 'pytorch', backend_config=backend_config)
    engine = SimpleNamespace(backend='pytorch', backend_config=backend_config, speculative_config=None,
                             session_len=8192, hf_cfg=AutoConfig.from_pretrained(checkpoint),
                             model_name=checkpoint, tokenizer=tokenizer, vl_encoder=vision)
    engine.prompt_processor = MultimodalProcessor(tokenizer, BaseChatTemplate(), vision, backend='pytorch')
    yield engine
    vision.executor.shutdown(wait=True)


def test_encoding_matches_golden(checkpoint, tmp_path, preprocessing_engine):
    from lmdeploy.serve.parsers.decision_parser import DecisionParser, create_decision_config
    config = create_decision_config(preprocessing_engine, checkpoint)
    golden = json.loads(GOLDEN.read_text())['cases']

    async def run():
        for name, row in cases(tmp_path):
            parser = DecisionParser(row, config)
            inputs = parser.prepare_input()
            result = await preprocessing_engine.prompt_processor.get_prompt_input(
                inputs.messages, do_preprocess=True, **inputs.preprocess_kwargs)
            ids = result['input_ids']
            prepared = SimpleNamespace(inputs=result, input_token_len=len(ids), gen_config=inputs.gen_config)
            parser.finalize_input(prepared)
            expected = golden[name]
            assert list(parser.compiled.fields) == expected['fields'], name
            assert [list(ids) for ids in parser.candidate_ids] == expected['candidate_ids'], name
            assert len(ids) == expected['input_tokens'], name
            assert prepared.gen_config.logprob_start_len == expected['encoding']['marker_positions'][0], name
            assert _sha256(ids) == expected['encoding']['tensors']['input_ids']['sha256'], name

    asyncio.run(run())
