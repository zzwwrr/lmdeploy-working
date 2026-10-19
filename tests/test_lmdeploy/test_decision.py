# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import base64
import copy
import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.serve.core import AsyncEngine
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.processors.intern_decision.output import DecisionOutputProcessor
from lmdeploy.serve.processors.intern_decision.protocol import (DecisionError, DecisionQueueFull, DecisionRequest,
                                                                DecisionUnavailable, PreparedDecision, ThinkingOptions)
from lmdeploy.serve.processors.intern_decision.runtime import DecisionRuntime
from lmdeploy.serve.processors.intern_decision.schema import compile_row
from lmdeploy.serve.processors.intern_decision.uploads import materialize_images


def row():
    return dict(
        state={'signal': '红灯'},
        questions={'action': {
            'type': 'choice',
            'instructions': '选择行动',
            'criteria': {
                'stop': '停车',
                'go': '通行'
            }
        }})


def prepared(kind='choice', values=('z', 'a')):
    return PreparedDecision([0, 1, 2], None, (1, ), ((0, 1), ), ('action', ), (values, ), (kind, ),
                            (dict(zip(values, values)), ), 3)


@pytest.mark.parametrize('change', [
    {
        'questions': {}
    },
    {
        'questions': {
            str(i): {
                'type': 'noul'
            }
            for i in range(17)
        }
    },
    {
        'images': [{}] * 9
    },
    {
        'thinking': {
            'enabled': True
        }
    },
    {
        'thinking': {
            'enabled': 1
        }
    },
    {
        'thinking': {
            'confidence_threshold': True
        }
    },
    {
        'thinking': {
            'confidence_threshold': float('nan')
        }
    },
    {
        'thinking': {
            'confidence_threshold': -0.01
        }
    },
    {
        'thinking': {
            'confidence_threshold': 1.01
        }
    },
    {
        'thinking': {
            'unexpected': False
        }
    },
    {
        'calibration': {}
    },
    {
        'state': float('inf')
    },
    {
        'questions': {
            'a': {
                'type': 'score',
                'criteria': {
                    'nan': 'bad'
                }
            }
        }
    },
    {
        'questions': {
            'a': {
                'type': 'choice',
                'criteria': {}
            }
        }
    },
    {
        'questions': {
            'a': {
                'type': 'choice',
                'criteria': {
                    str(i): 'x'
                    for i in range(63)
                }
            }
        }
    },
    {
        'questions': {
            'a': {
                'type': 'unknown'
            }
        }
    },
    {
        'questions': {
            '': {
                'type': 'noul'
            }
        }
    },
])
def test_invalid_request(change):
    with pytest.raises(ValueError):
        DecisionRequest.model_validate(row() | change)


def test_required_state_and_training_metadata():
    with pytest.raises(ValueError):
        DecisionRequest.model_validate({'questions': row()['questions']})
    request = row()
    reference = compile_row(request).messages
    request['questions']['action']['answer'] = {'choice': '<decision>'}
    request['targets'] = {'action': '<decision>'}
    clean = DecisionRequest.model_validate(request).model_dump()
    assert compile_row(clean).messages == reference
    assert list(DecisionRequest.model_validate(row()).questions['action']['criteria']) == ['stop', 'go']


@pytest.mark.parametrize('where', ['state', 'name', 'instructions', 'description', 'value'])
def test_reserved_marker(where):
    request = row()
    if where == 'state':
        request['state'] = '<decision>'
    elif where == 'name':
        request['questions']['<decision>'] = request['questions'].pop('action')
    elif where == 'instructions':
        request['questions']['action']['instructions'] = '<decision>'
    elif where == 'description':
        request['questions']['action']['criteria']['stop'] = '<decision>'
    else:
        request['questions']['action']['criteria']['<decision>'] = 'stop'
    with pytest.raises(ValueError, match='Reserved'):
        compile_row(request)


@pytest.mark.parametrize('kind,values,key,expected', [
    ('choice', ('z', 'a'), 'choice', 'a'),
    ('score', ('0', '2'), 'score', 1.0),
    ('noul', ('no', 'yes'), 'noul', 0.5),
])
def test_output_semantics(kind, values, key, expected):
    result = DecisionOutputProcessor()(prepared(kind, values),
                                       torch.tensor([[0., 0., 1000.]]),
                                       request_id='r',
                                       model='m',
                                       thinking=ThinkingOptions(),
                                       timing={})
    answer = result.answers['action']
    assert answer[key] == expected
    assert sum(answer['probabilities'].values()) == 1
    assert answer['confidence'] == 0.5
    assert answer['source'] == 'local'
    assert result.usage == dict(input_tokens=3, output_tokens=1, decision_count=1, generated_tokens=0)
    if kind == 'score':
        assert answer['decision'] == '0' and answer['legend'] == {'0': '0', '2': '2'}


@pytest.mark.parametrize('logits', [
    torch.zeros(2, 3),
    torch.zeros(1),
    torch.ones(1, 2, dtype=torch.long),
    torch.tensor([[float('nan'), 0.]]),
    torch.tensor([[float('inf'), 0.]])
])
def test_output_rejects_bad_logits(logits):
    with pytest.raises(DecisionError):
        DecisionOutputProcessor()(prepared(), logits, request_id='r', model='m', thinking=ThinkingOptions(), timing={})


def image_upload(fmt='PNG'):
    buffer = io.BytesIO()
    with Image.new('RGB', (16, 12), 'red') as image:
        image.save(buffer, format=fmt)
    return dict(type='image/png', name='../../escape.png', data=base64.b64encode(buffer.getvalue()).decode())


def test_upload_private_paths_and_metadata(tmp_path):
    import shutil
    request, directory = materialize_images({'images': [image_upload()]})
    try:
        assert directory.stat().st_mode & 0o777 == 0o700
        assert request['images'] == [str(directory / '00.png')]
        with Image.open(request['images'][0]) as image:
            assert image.mode == 'RGB' and not image.info
    finally:
        shutil.rmtree(directory)


@pytest.mark.parametrize('item', [
    'file:///etc/passwd', '/etc/passwd', {
        'url': 'https://example.com'
    }, {
        'type': 'image/png',
        'data': '%%%'
    }, {
        'type': 'image/png',
        'data': 'data:image/jpeg;base64,eA=='
    }, {
        'type': 'image/png',
        'data': 'eA=='
    }
])
def test_upload_rejects_invalid(item):
    with pytest.raises(ValueError):
        materialize_images({'images': [item]})


class Handle:

    def __init__(self):
        self.events = []
        self.mode = 'ok'
        self.started = asyncio.Event()
        self.end_gate = None
        self.end_error = False

    async def async_stream_infer(self, session_id, input_ids, gen_config, **kwargs):
        self.events.append(('infer', session_id, gen_config, kwargs))
        self.started.set()
        try:
            if self.mode == 'block':
                await asyncio.Event().wait()
            if self.mode == 'raise':
                raise RuntimeError('model failed')
            if self.mode == 'empty':
                return
            if self.mode == 'success':
                yield EngineOutput(ResponseType.SUCCESS, [])
                return
            status = ResponseType.CANCEL if self.mode == 'cancel' else ResponseType.FINISH
            logits = torch.zeros(len(input_ids), 8)
            if self.mode == 'nan':
                logits[0, 0] = float('nan')
            if self.mode == 'shape':
                logits = logits[:-1]
            yield EngineOutput(status, [], logits=logits)
        finally:
            self.events.append(('generator_closed', session_id))

    async def async_cancel(self, session_id):
        self.events.append(('cancel', session_id))
        return ResponseType.SUCCESS

    async def async_end(self, session_id):
        self.events.append(('end_sent', session_id))
        if self.end_gate is not None:
            await self.end_gate.wait()
        if self.end_error:
            raise RuntimeError('end failed')
        self.events.append(('end_confirmed', session_id))
        return ResponseType.SUCCESS


def fake_engine(handle=None, config=None):
    handle = handle or Handle()
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.task, engine.backend, engine.model_name = 'intern-decision', 'pytorch', 'intern-decision'
    engine.hf_cfg = SimpleNamespace(text_config=SimpleNamespace(vocab_size=8))
    engine.backend_config = SimpleNamespace(adapters=None)
    engine.session_len = 8192
    engine.engine = SimpleNamespace(create_instance=lambda: handle,
                                    get_max_session_len=lambda: 8192,
                                    close=lambda: None)
    engine.session_mgr = SessionManager()
    engine.session_mgr.build_request_handle_pool(engine.engine, 1)
    engine.decision_runtime = DecisionRuntime(config)
    engine.decision_input = lambda request, limit: prepared()
    engine.decision_output = DecisionOutputProcessor()
    return engine, handle


def assert_released(engine):
    assert not engine.session_mgr.sessions
    assert engine.session_mgr.request_handle_pool.pool.qsize() == 1


def test_prefill_config_and_confirmed_cleanup():

    async def run():
        engine, handle = fake_engine()
        handle.end_gate = asyncio.Event()
        task = asyncio.create_task(engine.async_decide(row()))
        while not any(e[0] == 'end_sent' for e in handle.events):
            await asyncio.sleep(0)
        assert not task.done()
        assert engine.session_mgr.request_handle_pool.pool.qsize() == 0
        handle.end_gate.set()
        result = await task
        assert result['usage']['generated_tokens'] == 0
        gen = handle.events[0][2]
        assert gen.max_new_tokens == 0 and gen.top_k == 1 and gen.output_logits == 'all'
        assert gen.with_cache is False and gen.preserve_cache is False
        assert_released(engine)

    asyncio.run(run())


@pytest.mark.parametrize('mode', ['raise', 'cancel', 'empty', 'success', 'nan', 'shape'])
def test_scoring_errors_propagate_and_recover(mode):

    async def run():
        engine, handle = fake_engine()
        handle.mode = mode
        with pytest.raises(RuntimeError):
            await engine.async_decide(row())
        assert_released(engine)
        handle.mode = 'ok'
        assert (await engine.async_decide(row()))['answers']
        assert_released(engine)

    asyncio.run(run())


def test_running_cancellation_and_queued_cancellation():

    async def run():
        engine, handle = fake_engine()
        handle.mode = 'block'
        first = asyncio.create_task(engine.async_decide(row()))
        await handle.started.wait()
        queued = asyncio.create_task(engine.async_decide(row()))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert any(e[0] == 'cancel' for e in handle.events)
        assert any(e[0] == 'end_confirmed' for e in handle.events)
        assert_released(engine)
        assert engine.decision_runtime.pending == 0
        handle.mode = 'ok'
        await engine.async_decide(row())

    asyncio.run(run())


@pytest.mark.parametrize('timeout', [False, True])
def test_cleanup_failure_quarantines_handle(timeout):

    async def run():
        engine, handle = fake_engine(config={'cleanup_timeout': 0.01})
        if timeout:
            handle.end_gate = asyncio.Event()
        else:
            handle.end_error = True
        with pytest.raises((RuntimeError, asyncio.TimeoutError)):
            await engine.async_decide(row())
        assert engine.session_mgr.request_handle_pool.pool.qsize() == 0
        assert not engine.session_mgr.sessions
        with pytest.raises(DecisionUnavailable):
            await engine.async_decide(row())

    asyncio.run(run())


def test_queue_full_timeout_and_close():

    async def run():
        engine, handle = fake_engine(config={'max_pending_requests': 1})
        handle.mode = 'block'
        running = asyncio.create_task(engine.async_decide(row()))
        await handle.started.wait()
        with pytest.raises(DecisionQueueFull):
            await engine.async_decide(row())
        await engine.async_shutdown_decisions()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert_released(engine)
        with pytest.raises(DecisionUnavailable):
            await engine.async_decide(row())

    asyncio.run(run())


def test_caller_session_ownership_and_generic_logits():

    async def run():
        engine, handle = fake_engine()
        session = engine.session_mgr.get()
        result = await engine.async_get_logits([[0, 1, 2]], sessions=[session], sequence_end=False)
        assert result[0].shape == (3, 8)
        assert engine.session_mgr.has(session.session_id)
        assert session._engine_session_active
        assert not any(e[0] == 'end_sent' for e in handle.events)
        await session.async_close()  # step remains zero; must still send END_SESSION
        assert any(e[0] == 'end_confirmed' for e in handle.events)
        engine.session_mgr.remove(session)

    asyncio.run(run())


def test_http_aliases_and_errors(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lmdeploy.serve.openai.api_server import VariableInterface, router
    from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware
    engine, _ = fake_engine()
    monkeypatch.setattr(VariableInterface, 'async_engine', engine)
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(AuthenticationMiddleware, tokens=['secret'])
    headers = {'Authorization': 'Bearer secret'}
    with TestClient(app) as client:
        assert client.post('/v1/decisions', json=row()).status_code == 401
        a = client.post('/v1/decisions', json=row(), headers=headers)
        b = client.post('/v1/jev', json=row(), headers=headers)
        assert a.status_code == b.status_code == 200
        assert a.json()['answers'] == b.json()['answers']
        assert client.post('/v1/decisions', json=row() | {'model': 'other'}, headers=headers).status_code == 404
        assert client.post('/v1/decisions', json=row() | {
            'images': ['/etc/passwd']
        }, headers=headers).status_code == 400
        assert client.post('/v1/decisions', json={'questions': {}}, headers=headers).status_code == 422
        engine.task = None
        assert client.post('/v1/decisions', json=row(), headers=headers).status_code == 503
        assert client.get('/v1/models', headers=headers).status_code == 200


def test_queue_wait_timeout_and_inference_timeout():

    async def run():
        engine, handle = fake_engine(config={'queue_timeout': 0.01, 'inference_timeout': 0.1})
        handle.mode = 'block'
        first = asyncio.create_task(engine.async_decide(row()))
        await handle.started.wait()
        with pytest.raises(DecisionQueueFull, match='wait timed out'):
            await engine.async_decide(row())
        from lmdeploy.serve.processors.intern_decision.protocol import DecisionTimeout
        with pytest.raises(DecisionTimeout):
            await first
        assert_released(engine)
        handle.mode = 'ok'
        await engine.async_decide(row())

    asyncio.run(run())


def test_cancellation_during_end_ack_wait():

    async def run():
        engine, handle = fake_engine()
        handle.end_gate = asyncio.Event()
        task = asyncio.create_task(engine.async_decide(row()))
        while not any(e[0] == 'end_sent' for e in handle.events):
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        handle.end_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert_released(engine)
        await engine.async_decide(row())

    asyncio.run(run())


def test_multimodal_forward_and_actual_capacity():

    async def run():
        engine, handle = fake_engine()
        engine.engine.get_max_session_len = lambda: 2

        def prepare(request, limit):
            assert limit == 2
            raise ValueError('above actual capacity')

        engine.decision_input = prepare
        with pytest.raises(ValueError, match='actual capacity'):
            await engine.async_decide(row())
        assert not handle.events and not engine.decision_runtime.failure
        await engine.async_get_logits([[0, 1, 2]], multimodal=[[{'test': 'pixels'}]], read_positions=[(1, )])
        assert handle.events[0][3]['multimodal'] == [{'test': 'pixels'}]

    asyncio.run(run())


def test_parallel_scoring_failure_cancels_sibling():

    async def run():
        engine, failing = fake_engine()
        blocked = Handle()
        blocked.mode = 'block'
        failing.mode = 'raise'
        handles = iter([blocked, failing])
        engine.engine.create_instance = lambda: next(handles)
        engine.session_mgr.build_request_handle_pool(engine.engine, 2)
        with pytest.raises(RuntimeError, match='model failed'):
            await engine.async_get_logits([[0, 1, 2], [0, 1, 2]])
        assert not engine.session_mgr.sessions
        assert engine.session_mgr.request_handle_pool.pool.qsize() == 2
        assert any(e[0] == 'end_confirmed' for e in blocked.events)

    asyncio.run(run())


def test_http_disconnect_cancels_and_recovers():
    from lmdeploy.serve.openai.serving_decision import serve_decision

    async def run():
        engine, handle = fake_engine()
        handle.mode = 'block'

        class Request:

            async def is_disconnected(self):
                return handle.started.is_set()

        result = await serve_decision(DecisionRequest.model_validate(row()), Request(), engine)
        assert result.status_code == 499
        assert_released(engine)
        handle.mode = 'ok'
        await engine.async_decide(row())

    asyncio.run(run())


def test_client_serialization_and_http_error(tmp_path, monkeypatch):
    import requests

    from lmdeploy.serve.openai.api_client import APIClient
    path = tmp_path / 'image.png'
    path.write_bytes(base64.b64decode(image_upload()['data']))
    request = row() | {'images': [str(path)]}
    saved = copy.deepcopy(request)

    def post(url, headers, json, timeout):
        assert url == 'http://localhost/v1/decisions'
        assert headers['Authorization'] == 'Bearer key'
        assert timeout == 42
        assert json['images'][0]['type'] == 'image/png'
        assert base64.b64decode(json['images'][0]['data']) == path.read_bytes()

        class Response:

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                raise requests.HTTPError('429 overloaded')

        return Response()

    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(requests.HTTPError):
        APIClient('http://localhost', api_key='key').decide(request, timeout=42)
    assert request == saved


def test_task_selection_and_configuration(monkeypatch):
    from lmdeploy import archs
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.serve.processors.intern_decision import task
    cfg = SimpleNamespace(text_config=SimpleNamespace(hidden_size=1024, num_hidden_layers=24, vocab_size=248320))
    monkeypatch.setattr(task, 'get_model_arch', lambda _: ('Qwen3_5ForConditionalGeneration', cfg))
    assert archs.get_task('unused', task='intern-decision') == ('intern-decision', AsyncEngine)
    with pytest.raises(ValueError):
        archs.get_task('unused', task='typo')
    config = task.configure_task('intern-decision', 'unused', 'pytorch', None)
    assert config.eager_mode and config.dtype == 'bfloat16' and config.max_batch_size == 1
    assert not config.enable_prefix_caching and config.max_prefill_token_num == 8192
    for change in [
            dict(tp=2),
            dict(dp=2),
            dict(dtype='float16'),
            dict(enable_prefix_caching=True),
            dict(eager_mode=False),
            dict(max_batch_size=2),
            dict(max_prefill_token_num=4096),
            dict(quant_policy=8),
            dict(enable_mp_engine=True),
            dict(attn_tp_size=2)
    ]:
        data = dict(dtype='bfloat16', eager_mode=True, max_prefill_token_num=8192) | change
        with pytest.raises(ValueError):
            task.configure_task('intern-decision', 'unused', 'pytorch', PytorchEngineConfig(**data))
    with pytest.raises(ValueError):
        task.configure_task('intern-decision', 'unused', 'turbomind', None)
    monkeypatch.setattr(task, 'get_model_arch', lambda _: ('LlamaForCausalLM', cfg))
    with pytest.raises(ValueError):
        task.configure_task('intern-decision', 'unused', 'pytorch', None)


def test_animated_upload_and_failure_cleanup(tmp_path, monkeypatch):
    from lmdeploy.serve.processors.intern_decision import uploads
    buffer = io.BytesIO()
    with Image.new('RGB', (8, 8), 'red') as a, Image.new('RGB', (8, 8), 'blue') as b:
        a.save(buffer, format='GIF', save_all=True, append_images=[b])
    directory = tmp_path / 'private'
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(uploads.tempfile, 'mkdtemp', lambda **kwargs: str(directory))
    with pytest.raises(ValueError, match='Animated'):
        materialize_images({'images': [dict(type='image/gif', data=base64.b64encode(buffer.getvalue()).decode())]})
    assert not directory.exists()


@pytest.mark.parametrize('limit', ['MAX_UPLOAD_BYTES', 'MAX_UPLOAD_TOTAL', 'MAX_IMAGE_PIXELS'])
def test_upload_size_limits(limit, monkeypatch):
    from lmdeploy.serve.processors.intern_decision import uploads
    monkeypatch.setattr(uploads, limit, 1)
    with pytest.raises(ValueError):
        materialize_images({'images': [image_upload()]})


def test_step_zero_close_confirmation_failure_is_visible():

    async def run():
        engine, handle = fake_engine()
        session = engine.session_mgr.get()
        await engine.async_get_logits([[0, 1, 2]], sessions=[session], sequence_end=False)
        handle.end_error = True
        with pytest.raises(RuntimeError, match='end failed'):
            await session.async_close()
        assert engine.session_mgr.request_handle_pool.pool.qsize() == 0

    asyncio.run(run())


def test_preprocessing_error_and_cancel_own_worker():
    import threading

    async def run():
        engine, _ = fake_engine()
        started, release = threading.Event(), threading.Event()

        def prepare(request, limit):
            started.set()
            release.wait(2)
            return prepared()

        engine.decision_input = prepare
        task = asyncio.create_task(engine.async_decide(row()))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not engine.decision_runtime.pending and not engine.decision_runtime.workers
        assert not engine.decision_runtime.failure

    asyncio.run(run())


def test_preprocessing_timeout_quarantines_until_worker_finishes():
    import threading

    async def run():
        from lmdeploy.serve.processors.intern_decision.protocol import DecisionTimeout
        engine, _ = fake_engine(config={'inference_timeout': 0.01, 'cleanup_timeout': 0.01})
        release = threading.Event()

        def prepare(request, limit):
            release.wait(2)
            return prepared()

        engine.decision_input = prepare
        try:
            with pytest.raises(DecisionTimeout):
                await engine.async_decide(row())
            with pytest.raises(DecisionUnavailable):
                await engine.async_decide(row())
        finally:
            release.set()
        await asyncio.gather(*engine.decision_runtime.workers)

    asyncio.run(run())


def test_end_session_uses_request_receipt():
    from lmdeploy.pytorch.engine.engine_instance import EngineInstance
    from lmdeploy.pytorch.engine.request import RequestType

    async def run():
        sent = []

        class Sender:

            async def async_send(self, kind, payload):
                sent.append((kind, payload))
                return SimpleNamespace(type=ResponseType.SUCCESS)

        instance = SimpleNamespace(req_sender=Sender())
        assert await EngineInstance.async_end(instance, 123) == ResponseType.SUCCESS
        assert sent == [(RequestType.END_SESSION, {'session_id': 123, 'response': True})]

    asyncio.run(run())


def test_ordinary_routing_unchanged(monkeypatch):
    from lmdeploy import archs
    config = SimpleNamespace(to_dict=lambda: {'architectures': ['Qwen3_5ForConditionalGeneration']})
    monkeypatch.setattr(archs, 'get_model_arch', lambda _: ('Qwen3_5ForConditionalGeneration', config))
    assert archs.get_task('ordinary-qwen') == ('llm', AsyncEngine)
    assert not archs.check_vl_llm(config.to_dict())


def test_reward_and_ppl_score_shape_compatibility():

    async def run():
        engine, handle = fake_engine()
        engine.arch = 'Qwen2ForRewardModel'
        original = handle.async_stream_infer

        async def reward(*args, **kwargs):
            async for output in original(*args, **kwargs):
                output.logits = torch.ones(3, 1)
                yield output

        handle.async_stream_infer = reward
        assert await engine.async_get_reward_score([[0, 1, 2]]) == [1.0]
        assert_released(engine)

    asyncio.run(run())


def test_cli_task_flags_and_early_rejection(monkeypatch):
    from lmdeploy.cli.cli import CLI
    from lmdeploy.cli.serve import SubCliServe
    from lmdeploy.serve.openai import api_server
    from lmdeploy.serve.processors.intern_decision import task
    SubCliServe.add_parser_api_server()
    arguments = [
        'serve', 'api_server', 'unused', '--backend', 'pytorch', '--task', 'intern-decision', '--eager-mode', '--dtype',
        'bfloat16', '--max-batch-size', '1', '--max-prefill-token-num', '8192'
    ]
    args = CLI.parser.parse_args(arguments)
    cfg = SimpleNamespace(text_config=SimpleNamespace(hidden_size=1024, num_hidden_layers=24, vocab_size=248320))
    monkeypatch.setattr(task, 'get_model_arch', lambda _: ('Qwen3_5ForConditionalGeneration', cfg))
    called = []
    monkeypatch.setattr(api_server, 'serve', lambda *args, **kwargs: called.append(kwargs))
    SubCliServe.api_server(args)
    assert called[0]['task'] == 'intern-decision'
    assert called[0]['backend_config'].max_prefill_token_num == 8192
    args.dp = 2
    with pytest.raises(ValueError, match='TP/DP/EP=1'):
        SubCliServe.api_server(args)
    assert len(called) == 1


def test_primary_error_keeps_cleanup_failure_context():

    async def run():
        engine, handle = fake_engine()
        handle.mode = 'raise'
        handle.end_error = True
        with pytest.raises(RuntimeError, match='model failed') as caught:
            await engine.async_decide(row())
        assert 'cleanup also failed' in '\n'.join(getattr(caught.value, '__notes__', [])) or caught.value.__context__
        assert engine.decision_runtime.failure
        assert engine.session_mgr.request_handle_pool.pool.qsize() == 0

    asyncio.run(run())


def test_metrics_balance_on_failure():
    from lmdeploy.metrics.metrics_processor import metrics_processor

    async def run():
        before = metrics_processor.scheduler_stats.num_api_routed_reqs
        engine, handle = fake_engine()
        handle.mode = 'raise'
        with pytest.raises(RuntimeError):
            await engine.async_decide(row())
        assert metrics_processor.scheduler_stats.num_api_routed_reqs == before

    asyncio.run(run())


def test_output_rejects_fp32_overflow():
    with pytest.raises(DecisionError, match='float32'):
        DecisionOutputProcessor()(prepared(),
                                  torch.tensor([[1e300, 0.]], dtype=torch.float64),
                                  request_id='r',
                                  model='m',
                                  thinking=ThinkingOptions(),
                                  timing={})


def test_initialization_rejects_unimplemented_calibration():
    for key in ('calibration', 'calibration_path', 'temperature'):
        with pytest.raises(ValueError, match='calibration'):
            AsyncEngine('unused', task='intern-decision', backend='pytorch', **{key: 'unsupported'})
