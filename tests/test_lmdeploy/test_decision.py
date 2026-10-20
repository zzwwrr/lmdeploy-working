# Copyright (c) OpenMMLab. All rights reserved.
"""CPU tests for the Intern-Decision task; no checkpoint or GPU required."""
import asyncio
import base64
import io
import math
import threading
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from lmdeploy.intern_decision_encoding import (
    DecisionOutputError,
    DecisionOutputProcessor,
    EncodedDecision,
    compile_row,
    scale_probabilities,
)
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.serve.core.decide_runner import DecideRunner
from lmdeploy.serve.processors.intern_decision import DecisionError, DecisionQueueFull, DecisionRequest
from lmdeploy.serve.processors.intern_decision.images import load_images
from lmdeploy.serve.processors.intern_decision.runtime import DecisionRuntime

QUESTIONS = {
    'action': {
        'type': 'choice',
        'instructions': 'Select an action.',
        'criteria': {
            'stop': 'Stop.',
            'go': 'Go.'
        }
    },
    'risk': {
        'type': 'score',
        'criteria': ['low', 'mid', 'high']
    },
    'red': {
        'type': 'noul'
    },
}


def _png(size=(4, 3), color=(255, 0, 0), fmt='PNG', **save_kwargs):
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def _upload(payload, mime='image/png'):
    return {'type': mime, 'data': base64.b64encode(payload).decode()}


# ---------------------------------------------------------------- engine config


def test_logprob_token_ids_requires_input_logprobs():
    GenerationConfig(max_new_tokens=0, logprobs=0, logprob_start_len=0, logprob_token_ids=[1, 2])
    with pytest.raises(ValueError, match='logprob_start_len'):
        GenerationConfig(logprob_token_ids=[1])
    with pytest.raises(ValueError, match='non-negative'):
        GenerationConfig(max_new_tokens=0, logprobs=0, logprob_start_len=0, logprob_token_ids=[-1])


# ---------------------------------------------------------------- protocol


@pytest.mark.parametrize('change', [
    {'questions': {}},
    {'questions': {'q': {'type': 'rank'}}},
    {'questions': {'q': {'type': 'choice', 'criteria': ['a']}}},
    {'questions': {'q': {'type': 'score', 'criteria': {'x': 'not a number'}}}},
    {'questions': {f'q{i}': {'type': 'noul'} for i in range(17)}},
    {'state': float('nan')},
    {'images': ['a.png'] * 9},
    {'thinking': {'enabled': True}},
    {'unknown': 1},
])
def test_invalid_request(change):
    with pytest.raises(ValueError):
        DecisionRequest.model_validate({'state': {}, 'questions': QUESTIONS, **change})


def test_training_metadata_is_dropped():
    request = DecisionRequest.model_validate({'state': 1, 'questions': QUESTIONS, 'targets': {'a': 'A'}, 'id': 3})
    assert 'targets' not in request.model_dump() and 'id' not in request.model_dump()


def test_reserved_marker_rejected():
    with pytest.raises(ValueError, match='Reserved'):
        compile_row({'state': 'say <decision>', 'questions': QUESTIONS})


# ---------------------------------------------------------------- images


def test_upload_and_path_load_the_same_pixels(tmp_path):
    payload = _png()
    path = tmp_path / 'red.png'
    path.write_bytes(payload)
    upload, = load_images([_upload(payload)], allow_paths=False)
    local, = load_images([str(path)], allow_paths=True)
    assert upload.mode == local.mode == 'RGB' and upload.size == local.size == (4, 3)
    assert upload.tobytes() == local.tobytes()


def test_exif_orientation_is_not_applied():
    # Matches the checkpoint's inference.py, which feeds pixels as stored.
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    image, = load_images([_upload(_png(fmt='JPEG', exif=exif), 'image/jpeg')], allow_paths=False)
    assert image.size == (4, 3) and not image.info


@pytest.mark.parametrize('item, allow_paths, match', [
    ('local.png', False, 'server-side paths'),
    ('https://example.com/a.png', True, 'Remote'),
    ('missing.png', True, 'not found'),
    ({'type': 'image/jpeg', 'data': base64.b64encode(_png()).decode()}, False, 'declared format'),
    ({'type': 'image/bmp', 'data': 'AAAA'}, False, 'Supported image formats'),
    ({'type': 'image/png', 'data': '***'}, False, 'base64'),
    ({'data': 'data:image/png;base64,' + base64.b64encode(_png()).decode(), 'type': 'image/jpeg'}, False, 'disagree'),
    (3, True, 'local path or an uploaded'),
])
def test_invalid_images(tmp_path, item, allow_paths, match):
    with pytest.raises(ValueError, match=match):
        load_images([item], allow_paths=allow_paths, media_root=tmp_path)


def test_animated_gif_rejected():
    buffer = io.BytesIO()
    frames = [Image.new('RGB', (2, 2), c) for c in ((255, 0, 0), (0, 255, 0))]
    frames[0].save(buffer, format='GIF', save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match='Animated'):
        load_images([_upload(buffer.getvalue(), 'image/gif')], allow_paths=False)


# ---------------------------------------------------------------- output


@pytest.mark.parametrize('temperature', [1.0, 2.747760550703])
def test_output_semantics_and_calibration(temperature):
    request = DecisionRequest.model_validate({'state': 1, 'questions': QUESTIONS})
    logits = [[2.0, 0.5], [0.1, 0.7, 1.9], [1.2, -0.3]]
    answers = DecisionOutputProcessor(temperature)(request.questions, ('action', 'risk', 'red'), logits)

    def expected(values, row):
        probs = dict(zip(values, torch.softmax(torch.tensor(row), -1).tolist()))
        logs = {k: math.log(v) for k, v in probs.items()}
        top = max(logs.values())
        weights = {k: math.exp((v - top) / temperature) for k, v in logs.items()}
        return {k: w / sum(weights.values()) for k, w in weights.items()}

    action = expected(['stop', 'go'], logits[0])
    risk = expected(['0', '1', '2'], logits[1])
    red = expected(['no', 'yes'], logits[2])
    assert answers['action']['choice'] == answers['action']['decision'] == 'stop'
    assert answers['action']['probabilities'] == pytest.approx(action)
    assert answers['risk']['score'] == pytest.approx(sum(float(k) * v for k, v in risk.items()))
    assert answers['risk']['legend'] == {'0': 'low', '1': 'mid', '2': 'high'}
    assert answers['red']['noul'] == pytest.approx(red['yes'])
    assert answers['red']['confidence'] == pytest.approx(max(red.values()))


def test_temperature_flattens_but_keeps_argmax():
    probs = {'a': 0.9, 'b': 0.1}
    scaled = scale_probabilities(probs, 3.0)
    assert scaled['a'] > scaled['b'] and scaled['a'] < probs['a']
    assert scale_probabilities(probs, 1.0) == probs


@pytest.mark.parametrize('logits', [[[1.0, float('nan')]], [[1.0]], [[1.0, 2.0], [0.0, 1.0]]])
def test_output_rejects_bad_logits(logits):
    questions = {'q': {'type': 'noul'}}
    with pytest.raises(DecisionOutputError):
        DecisionOutputProcessor()(questions, ('q', ), logits)


# ---------------------------------------------------------------- runner

MARKER, IMAGE_PAD = 900, 800
SYMBOLS = {'A': 32, 'B': 33, 'C': 34}


class FakeSessionManager:

    def __init__(self):
        self.sessions = {}
        self.count = 0

    def get(self):
        self.count += 1
        session = SimpleNamespace(session_id=self.count, epoch=None)
        self.sessions[session.session_id] = session
        return session

    def remove(self, session):
        self.sessions.pop(session.session_id, None)


class FakeEngine:
    """Expands each image pad to ``expand`` tokens and scores every row after logprob_start_len."""

    backend = 'pytorch'
    speculative_config = None
    session_len = 8192
    model_name = 'intern-decision'
    epoch = 0

    def __init__(self, expand=4, fail=None, abort=False):
        self.backend_config = PytorchEngineConfig()
        self.session_mgr = FakeSessionManager()
        self.expand = expand
        self.fail = fail
        self.abort = abort
        self.requests = []

    async def preprocess(self, messages, session, gen_config=None, input_ids=None, do_preprocess=True):
        if messages is not None:
            assert do_preprocess is False
            content = messages[0]['content']
            input_ids = content[0]['text']
            assert len(content) - 1 == input_ids.count(IMAGE_PAD)
        ids = []
        for token in input_ids:
            ids.extend([token] * (self.expand if token == IMAGE_PAD else 1))
        request = SimpleNamespace(session=session, ids=ids, input_token_len=len(ids), gen_config=gen_config)
        self.requests.append(request)
        return request

    async def generate(self, request, stream_response=True):
        if self.fail:
            raise self.fail
        if self.abort:
            yield SimpleNamespace(logprobs=None, logprob_token_ids=None, finish_reason='abort')
            return
        start = request.gen_config.logprob_start_len
        targets = request.ids[start + 1:]
        rows = [{t: -1.0, **{s: float(s) for s in request.gen_config.logprob_token_ids}} for t in targets]
        yield SimpleNamespace(logprobs=rows, logprob_token_ids=targets, finish_reason='length')


def _runtime(engine):
    runtime = DecisionRuntime.__new__(DecisionRuntime)
    runtime.model_names = {engine.model_name, 'Intern-Decision-0.8B'}
    runtime.max_length = 8191
    runtime.max_pending_requests = 2
    runtime.supports_images = True
    runtime.encoder = SimpleNamespace(marker_id=MARKER, symbol_ids=SYMBOLS)
    runtime.output = DecisionOutputProcessor()
    runtime.pending = 0
    return runtime


def _encoded(images=0):
    ids = [1] + [IMAGE_PAD] * images + [5, 6, 7, MARKER, 8, 9, MARKER, 10]
    first = ids.index(MARKER)
    compiled = compile_row({'state': 1, 'questions': {'a': QUESTIONS['action'], 'r': QUESTIONS['risk']}})
    return EncodedDecision(compiled=compiled, input_ids=ids, tail=len(ids) - (first - 1), candidate_ids=((32, 33),
                                                                                                         (32, 33, 34)))


@pytest.mark.parametrize('images', [0, 2])
def test_score_reads_marker_rows_after_image_expansion(images):
    engine = FakeEngine()
    runner = DecideRunner(engine, _runtime(engine))
    logits, length = asyncio.run(runner._score(_encoded(images), [object()] * images))
    request = engine.requests[0]
    assert length == len(request.ids)
    # The first scored row is the token right before the first <decision>, after expansion.
    assert request.ids[request.gen_config.logprob_start_len + 1] == MARKER
    assert request.gen_config.max_new_tokens == 0
    assert request.gen_config.logprob_token_ids == [32, 33, 34]
    assert logits == [[32.0, 33.0], [32.0, 33.0, 34.0]]
    assert not engine.session_mgr.sessions


def test_score_failure_cleans_up_session():
    engine = FakeEngine(fail=RuntimeError('boom'))
    with pytest.raises(RuntimeError, match='boom'):
        asyncio.run(DecideRunner(engine, _runtime(engine))._score(_encoded(), []))
    assert not engine.session_mgr.sessions


def test_aborted_score_is_unavailable():
    from lmdeploy.serve.processors.intern_decision import DecisionUnavailable
    engine = FakeEngine(abort=True)
    with pytest.raises(DecisionUnavailable, match='aborted'):
        asyncio.run(DecideRunner(engine, _runtime(engine))._score(_encoded(), []))
    assert not engine.session_mgr.sessions


def test_score_rejects_over_length_without_running():
    engine = FakeEngine(expand=10_000)
    with pytest.raises(ValueError, match='truncation'):
        asyncio.run(DecideRunner(engine, _runtime(engine))._score(_encoded(1), [object()]))
    assert not engine.session_mgr.sessions


def test_decide_end_to_end_and_admission():
    engine = FakeEngine()
    runtime = _runtime(engine)
    runner = DecideRunner(engine, runtime)
    encoded = _encoded()
    runner._prepare = lambda request, allow_local_paths: ([], encoded)
    request = {'state': 1, 'questions': {'a': QUESTIONS['action'], 'r': QUESTIONS['risk']}}
    result = asyncio.run(runner.run({**request, 'model': 'Intern-Decision-0.8B'}))
    assert list(result['answers']) == ['a', 'r']
    assert result['usage']['generated_tokens'] == 0 and result['usage']['decision_count'] == 2
    assert result['calibration'] == {'method': 'temperature-scaling', 'temperature': 2.747760550703}
    assert runtime.pending == 0

    runtime.pending = runtime.max_pending_requests
    with pytest.raises(DecisionQueueFull):
        asyncio.run(DecideRunner(engine, runtime).run(request))
    with pytest.raises(ValueError, match='Unknown model'):
        asyncio.run(DecideRunner(engine, _runtime(engine)).run({**request, 'model': 'other'}))


@pytest.mark.parametrize('failure', ['preparation', 'engine', 'output'])
def test_request_failure_releases_images_and_admission(failure):
    engine = FakeEngine(fail=RuntimeError('failed') if failure == 'engine' else None)
    runtime = _runtime(engine)
    runner = DecideRunner(engine, runtime)
    closed = []
    image = SimpleNamespace(close=lambda: closed.append(True))

    def fail(*args, **kwargs):
        raise ValueError('failed')

    if failure == 'preparation':
        runtime.encoder.encode = fail
        # Exercise the real _prepare's image ownership on encoder failure.
        request = {'state': 1, 'questions': QUESTIONS, 'images': [_upload(_png())]}
    else:
        runner._prepare = lambda *args: ([image], _encoded(1))
        request = {'state': 1, 'questions': {'a': QUESTIONS['action'], 'r': QUESTIONS['risk']}}
        if failure == 'output':
            runtime.output = fail
    with pytest.raises((ValueError, RuntimeError), match='failed'):
        asyncio.run(runner.run(request))
    assert runtime.pending == 0
    assert not engine.session_mgr.sessions
    if failure != 'preparation':
        assert closed == [True]


def test_cancelled_preparation_keeps_admission_until_worker_releases_images():
    engine = FakeEngine()
    runtime = _runtime(engine)
    runtime.max_pending_requests = 1
    runner = DecideRunner(engine, runtime)
    release_worker = threading.Event()
    closed = []
    image = SimpleNamespace(close=lambda: closed.append(True))
    request = {'state': 1, 'questions': QUESTIONS}

    async def run():
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def prepare(*args):
            loop.call_soon_threadsafe(started.set)
            assert release_worker.wait(timeout=10)
            return [image], _encoded()

        runner._prepare = prepare
        task = asyncio.create_task(runner.run(request))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            # Allow the request to enter shielded cleanup, then cancel again.
            while not hasattr(runner, '_request_cleanup_task'):
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime.pending == 1
            with pytest.raises(DecisionQueueFull):
                await DecideRunner(engine, runtime).run(request)
        finally:
            release_worker.set()
            await asyncio.wait_for(runner._request_cleanup_task, timeout=5)
        assert closed == [True]
        assert runtime.pending == 0
        assert not engine.requests

    asyncio.run(run())


@pytest.mark.parametrize('case, message', [
    ('empty', 'no input logprobs'),
    ('missing', 'no input logprobs'),
    ('misaligned', 'misaligned'),
    ('no_markers', 'Expected 2 decision rows'),
    ('missing_candidate', 'lack a requested candidate'),
])
def test_invalid_scoring_output_releases_session(case, message):
    engine = FakeEngine()

    async def generate(*args, **kwargs):
        if case == 'empty':
            return
        rows = [{32: 1.0, 33: 2.0, 34: 3.0}] * 2
        targets = [MARKER, MARKER]
        if case == 'missing':
            rows = None
        elif case == 'misaligned':
            targets = [MARKER]
        elif case == 'no_markers':
            targets = [1, 2]
        elif case == 'missing_candidate':
            rows = [{32: 1.0}] * 2
        yield SimpleNamespace(logprobs=rows, logprob_token_ids=targets, finish_reason='length')

    engine.generate = generate
    with pytest.raises(DecisionError, match=message):
        asyncio.run(DecideRunner(engine, _runtime(engine))._score(_encoded(), []))
    assert not engine.session_mgr.sessions


def test_pipeline_decide_uses_request_runner_with_local_paths(monkeypatch):
    from concurrent.futures import Future

    from lmdeploy.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.async_engine = FakeEngine()
    pipeline.decision_runtime = _runtime(pipeline.async_engine)
    observed = []

    def prepare(self, request, allow_local_paths):
        observed.append(allow_local_paths)
        return [], _encoded()

    def run(coro):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    monkeypatch.setattr(DecideRunner, '_prepare', prepare)
    pipeline._run = run
    result = pipeline.decide({'state': 1, 'questions': {'a': QUESTIONS['action'], 'r': QUESTIONS['risk']}})
    assert result['usage']['generated_tokens'] == 0
    assert observed == [True]
    assert pipeline.decision_runtime.pending == 0


# ---------------------------------------------------------------- HTTP


def test_http_routes(monkeypatch):
    from lmdeploy.serve.openai.api_server import ServerContext
    from lmdeploy.serve.openai.endpoints import create_openai_router

    context = ServerContext()
    app = FastAPI()
    app.include_router(create_openai_router(context))
    client = TestClient(app)
    body = {'state': 1, 'questions': {'q': {'type': 'noul'}}}
    assert client.post('/v1/decisions', json=body).status_code == 503

    calls = []

    class Runner:

        def __init__(self, engine, runtime):
            assert engine is context.async_engine
            assert runtime is context.decision_runtime

        async def run(self, request, allow_local_paths, is_disconnected):
            assert callable(is_disconnected)
            calls.append(allow_local_paths)
            if request.state == 'busy':
                raise DecisionQueueFull('busy')
            if request.state == 'bad':
                raise ValueError('bad image')
            return {'ok': True}

    monkeypatch.setattr(import_module('lmdeploy.serve.openai.endpoints.decisions'), 'DecideRunner', Runner)
    context.decision_runtime = object()
    assert client.post('/v1/decisions', json=body).json() == {'ok': True}
    assert client.post('/v1/jev', json=body).json() == {'ok': True}
    assert client.post('/v1/decisions', json={**body, 'state': 'busy'}).status_code == 429
    assert client.post('/v1/decisions', json={**body, 'state': 'bad'}).status_code == 400
    assert client.post('/v1/decisions', json={'state': 1, 'questions': {}}).status_code == 422
    assert calls and not any(calls)
