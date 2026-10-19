# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Fixture cases adapted from Intern-Decision (0a8a2388), verify_lmdeploy_decisions.py.
# Modified for LMDeploy: frozen expectations with no external source dependency.
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from lmdeploy.serve.processors.intern_decision.input import DecisionInputProcessor


def cases(directory):
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    for name, size, color in (("red", (256, 256), (240, 20, 20)), ("blue", (320, 384), (20, 20, 240))):
        with Image.new("RGB", size, color) as image:
            image.save(directory / f"{name}.png")
    questions = {
        "action": {
            "type": "choice",
            "instructions": "Select the safest action.",
            "criteria": {
                "stop": "Stop at the red traffic light.",
                "go": "Drive through the red traffic light."
            }
        },
        "red": {
            "type": "noul",
            "instructions": "Is the traffic light red?"
        },
        "risk": {
            "type": "score",
            "instructions": "How dangerous is driving through this light?",
            "criteria": {
                "0": "Safe",
                "1": "Some risk",
                "2": "Dangerous"
            }
        },
    }
    result = [
        ("text", {
            "state": "The traffic light is red. There are pedestrians crossing.",
            "questions": questions
        }),
        ("long_text", {
            "state": "Observe the road carefully. " * 260 + "The traffic light is red.",
            "questions": questions
        }),
    ]
    color_question = {
        "color": {
            "type": "choice",
            "instructions": "What is the main color of the FIRST image?",
            "criteria": {
                "red": "Red",
                "blue": "Blue",
                "green": "Green"
            }
        }
    }
    for name, names in (("red_image", ["red"]), ("blue_image", ["blue"]), ("two_images", ["red", "blue"]),
                        ("two_images_reversed", ["blue", "red"]), ("eight_images", ["red", "blue"] * 4)):
        result.append((name, {
            "state": "Inspect the supplied images in their given order.",
            "images": [str(directory / f"{n}.png") for n in names],
            "questions": color_question
        }))
    states = [
        "The traffic light is green and the road is empty.", "The traffic light is yellow.",
        "The light is red and a child is crossing.", "红灯亮着，行人正在过马路。", "绿灯亮着，前方道路畅通。",
        "The traffic light is red, but there are no pedestrians.", "A police officer signals STOP.",
        "The vehicle is already stationary at a red light.", "There is heavy rain and low visibility.",
        "An ambulance is approaching the crossing.", "The traffic signal is not working.",
        "A bus is blocking the view of the crossing.", "The pedestrian signal says WALK.",
        "A red light is visible in the distance.", "A car ahead has stopped suddenly.",
        "The road is closed for repairs.", "The driver is approaching a school zone.",
        "The road is icy near the intersection.", "A cyclist enters the crossing.",
        "There is a stop sign before the intersection."
    ]
    result.extend((f"fixed_{i:02d}", {
        "state": state,
        "questions": copy.deepcopy(questions)
    }) for i, state in enumerate(states))
    result.append(("sixteen_fields", {
        "state": "The light is red.",
        "questions": {
            f"field_{i}": {
                "type": "noul",
                "instructions": "Is the light red?"
            }
            for i in range(16)
        }
    }))
    return result


def encoding_evidence(batch, positions):
    import torch
    return {
        "marker_positions": positions.tolist(),
        "tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256":
                hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            }
            for key, value in batch.items()
        }
    }


@pytest.fixture(scope='module')
def processor():
    checkpoint = os.getenv('DECISION_TEST_CHECKPOINT')
    if not checkpoint:
        pytest.skip('Set DECISION_TEST_CHECKPOINT for CPU tokenizer/processor golden tests')
    torch.set_num_threads(2)
    return DecisionInputProcessor(checkpoint, AutoConfig.from_pretrained(checkpoint, local_files_only=True))


def test_frozen_encoding_golden(processor, tmp_path):
    golden = json.loads((Path(__file__).parent / 'fixtures/intern_decision_v1.json').read_text())['cases']
    for name, row in cases(tmp_path):
        result = processor(row)
        encoded = processor.encoder.encode(row)
        expected = golden[name]
        assert encoding_evidence(encoded.batch, encoded.marker_positions) == expected['encoding'], name
        assert result.input_length == expected['input_tokens']
        assert list(result.fields) == expected['fields']
        assert [list(ids) for ids in result.candidate_ids] == expected['candidate_ids']
        if result.multimodal:
            assert sum(item['pixel_values'].shape[0]
                       for item in result.multimodal) == len(encoded.batch['pixel_values'])


def test_length_limits_no_truncation(processor):
    row = {'state': '', 'questions': {'a': {'type': 'noul'}}}
    result = processor(row)
    with pytest.raises(ValueError, match='capacity'):
        processor(row, result.input_length - 1)
    row['state'] = 'red ' * 9000
    with pytest.raises(ValueError, match='truncation'):
        processor(row)


def test_duplicate_images(processor, tmp_path):
    row = cases(tmp_path)[2][1]
    row['images'] *= 2
    result = processor(row)
    assert len(result.multimodal) == 2
    assert torch.equal(result.multimodal[0]['pixel_values'], result.multimodal[1]['pixel_values'])


def test_real_checkpoint_gpu(tmp_path):
    if os.getenv('DECISION_TEST_GPU') != '1' or not os.getenv('DECISION_TEST_CHECKPOINT'):
        pytest.skip('Set DECISION_TEST_GPU=1 and DECISION_TEST_CHECKPOINT on an authorized idle GPU')
    from lmdeploy import PytorchEngineConfig, pipeline
    golden = json.loads((Path(__file__).parent / 'fixtures/intern_decision_v1.json').read_text())['cases']
    pipe = pipeline(os.environ['DECISION_TEST_CHECKPOINT'],
                    task='intern-decision',
                    backend_config=PytorchEngineConfig(dtype='bfloat16',
                                                       eager_mode=True,
                                                       max_batch_size=1,
                                                       max_prefill_token_num=8192,
                                                       session_len=8192,
                                                       cache_max_entry_count=0.05,
                                                       enable_metrics=False))
    try:
        for name, row in cases(tmp_path):
            result = pipe.decide(row)
            assert list(result['answers']) == golden[name]['fields']
            assert result['usage']['generated_tokens'] == 0
            assert result['usage']['input_tokens'] == golden[name]['input_tokens']
            actual = [list(answer['probabilities'].values()) for answer in result['answers'].values()]
            for probabilities, expected in zip(actual, golden[name]['probabilities']):
                # Migration golden for the documented pinned BF16 environment.
                torch.testing.assert_close(torch.tensor(probabilities), torch.tensor(expected), rtol=0, atol=1e-6)
            assert not pipe.async_engine.session_mgr.sessions
            assert not pipe.async_engine.engine.scheduler.sessions
            assert pipe.async_engine.session_mgr.request_handle_pool.pool.qsize() == 1
    finally:
        pipe.close()


def test_exact_8192_and_8193_cpu(processor):
    row = {'state': '', 'questions': {'red': {'type': 'noul', 'instructions': 'Is the light red?'}}}
    original = processor.encoder.max_length
    try:
        processor.encoder.max_length = 9000
        initial = processor.encoder.encode(row).input_length
        found = {}
        for target in (8192, 8193):
            for count in range(target - initial - 6, target - initial + 7):
                candidate = row | {'state': 'red ' * count}
                if processor.encoder.encode(candidate).input_length == target:
                    found[target] = candidate
                    break
            assert target in found
    finally:
        processor.encoder.max_length = original
    assert processor(found[8192]).input_length == 8192
    with pytest.raises(ValueError, match='truncation'):
        processor(found[8193])


def test_max_questions_and_candidates(processor):
    row = {
        'state': 'choose',
        'questions': {
            str(i): {
                'type': 'choice',
                'criteria': {
                    str(j): str(j)
                    for j in range(62)
                }
            }
            for i in range(16)
        }
    }
    # The protocol allows these counts, while expanded length remains authoritative.
    from lmdeploy.serve.processors.intern_decision.protocol import DecisionRequest
    request = DecisionRequest.model_validate(row)
    row['questions'] = {'one': request.questions['0']}
    result = processor(row)
    assert len(result.candidate_ids[0]) == 62
    assert len(set(result.candidate_ids[0])) == 62


def test_http_exif_and_trusted_path_semantics(processor, tmp_path):
    import base64
    import shutil

    from PIL import Image

    from lmdeploy.serve.processors.intern_decision.uploads import materialize_images
    path = tmp_path / 'oriented.jpg'
    with Image.new('RGB', (64, 32), 'red') as image:
        exif = image.getexif()
        exif[274] = 6  # A clockwise quarter turn, independently visible in dimensions.
        image.save(path, exif=exif)
    row = {'state': 'inspect', 'questions': {'red': {'type': 'noul'}}, 'images': [str(path)]}
    local = processor(row)
    upload = {'type': 'image/jpeg', 'data': base64.b64encode(path.read_bytes()).decode()}
    normalized, directory = materialize_images({'images': [upload]})
    try:
        with Image.open(normalized['images'][0]) as image:
            assert image.size == (32, 64)
            assert not image.getexif()
    finally:
        shutil.rmtree(directory)
    http = processor(row | {'images': [upload]})
    local_grid = local.multimodal[0]['image_grid_thw'][0].tolist()
    http_grid = http.multimodal[0]['image_grid_thw'][0].tolist()
    assert local_grid[1] < local_grid[2]
    assert http_grid == [local_grid[0], local_grid[2], local_grid[1]]
