# Copyright (c) OpenMMLab. All rights reserved.
"""Lifecycle regressions for the shared request execution path."""
import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.base_runner import BaseRunner
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError


class SessionManager:

    def __init__(self):
        self.session = SimpleNamespace(session_id=10, epoch=None, _handle=None)
        self.removed = []
        self.mapped = []

    def get(self, session_id=None):
        assert session_id in (None, 10)
        return self.session

    def map_user_session_id(self, session_id):
        self.mapped.append(session_id)
        return 10

    def remove(self, session):
        self.removed.append(session)


class Engine:

    epoch = 5

    def __init__(self):
        self.session_mgr = SessionManager()
        self.closed = False
        self.generated = False
        self.output = SimpleNamespace(logprobs=[{7: 0.5}], logprob_token_ids=[9])

    async def preprocess(self, messages, session, **kwargs):
        return SimpleNamespace(session=session, gen_config=kwargs['gen_config'])

    async def generate(self, prepared, **kwargs):
        self.generated = True
        try:
            yield self.output
        finally:
            self.closed = True


@pytest.mark.parametrize('user_id', [None, -1, 123])
def test_session_mapping_epoch_and_raw_output(user_id):
    engine = Engine()

    async def run():
        runner = BaseRunner(engine)
        await runner.prepare_engine(None, gen_config=GenerationConfig(), user_session_id=user_id)
        outputs = [out async for out in runner.iter_outputs()]
        await runner.close()
        return outputs

    assert asyncio.run(run()) == [engine.output]
    assert engine.session_mgr.session.epoch == engine.epoch
    assert engine.session_mgr.mapped == ([123] if user_id == 123 else [])
    assert engine.session_mgr.removed == [engine.session_mgr.session]
    assert engine.closed


@pytest.mark.parametrize('stage', ['preprocess', 'after_preprocess'])
def test_preparation_failure_releases_session_without_generating(stage):
    engine = Engine()
    runner = BaseRunner(engine)

    def fail(*args, **kwargs):
        raise ValueError('preparation failed')

    kwargs = {}
    if stage == 'preprocess':

        async def fail_preprocess(*args, **kwargs):
            fail()

        engine.preprocess = fail_preprocess
    else:
        kwargs['after_preprocess'] = fail
    with pytest.raises(ValueError, match='preparation failed'):
        asyncio.run(runner.prepare_engine(None, gen_config=GenerationConfig(), **kwargs))
    assert engine.session_mgr.removed == [engine.session_mgr.session]
    assert not engine.generated


def test_active_session_conflict_does_not_remove_or_restamp_session():
    engine = Engine()
    engine.session_mgr.session._handle = object()
    engine.session_mgr.session.epoch = 1
    with pytest.raises(RequestError) as exc:
        asyncio.run(BaseRunner(engine).prepare_engine(None, gen_config=GenerationConfig(), user_session_id=123))
    assert exc.value.code == ErrorCode.REQUEST_CONFLICT
    assert not engine.session_mgr.removed
    assert engine.session_mgr.session.epoch == 1


def test_preprocess_conflict_preserves_other_request_session():
    engine = Engine()

    async def conflict(*args, **kwargs):
        raise RequestError(ErrorCode.REQUEST_CONFLICT)

    engine.preprocess = conflict
    with pytest.raises(RequestError):
        asyncio.run(BaseRunner(engine).prepare_engine(None, gen_config=GenerationConfig()))
    assert not engine.session_mgr.removed


def test_disconnect_aborts_and_cleans_session():
    engine = Engine()
    aborted = []

    async def abort():
        aborted.append(True)

    async def disconnected():
        return True

    engine.session_mgr.session.async_abort = abort

    async def run():
        runner = BaseRunner(engine)
        await runner.prepare_engine(None, gen_config=GenerationConfig())
        return [out async for out in runner.iter_outputs(is_disconnected=disconnected)]

    with pytest.raises(RequestError, match='Client disconnected'):
        asyncio.run(run())
    assert aborted == [True]
    assert engine.closed
    assert engine.session_mgr.removed == [engine.session_mgr.session]


def test_cancel_while_waiting_for_output_cleans_generator():
    engine = Engine()

    async def run():
        entered = asyncio.Event()

        async def generate(*args, **kwargs):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield engine.output
            finally:
                engine.closed = True

        engine.generate = generate
        runner = BaseRunner(engine)
        await runner.prepare_engine(None, gen_config=GenerationConfig())

        async def consume():
            async for _ in runner.iter_outputs():
                pass

        consumer = asyncio.create_task(consume())
        await entered.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    asyncio.run(run())
    assert engine.closed
    assert engine.session_mgr.removed == [engine.session_mgr.session]


def test_cleanup_survives_cancellation_and_can_be_awaited_again():
    engine = Engine()

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        class Generator:
            async def aclose(self):
                entered.set()
                await release.wait()
                engine.closed = True

        runner = BaseRunner(engine)
        runner.session = engine.session_mgr.session
        runner.result_generator = Generator()
        waiter = asyncio.create_task(runner.close())
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not engine.closed
        release.set()
        await runner.close()

    asyncio.run(run())
    assert engine.closed
    assert engine.session_mgr.removed == [engine.session_mgr.session]
