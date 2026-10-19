# Copyright (c) OpenMMLab. All rights reserved.
"""Bounded admission and the shared asynchronous decision operation."""
import asyncio
import secrets
import time

from pydantic import BaseModel, ConfigDict, Field

from .protocol import DecisionQueueFull, DecisionRequest, DecisionTimeout, DecisionUnavailable


class DecisionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_pending_requests: int = Field(default=32, ge=1, le=1024)
    queue_timeout: float = Field(default=120, gt=0, allow_inf_nan=False)
    inference_timeout: float = Field(default=120, gt=0, allow_inf_nan=False)
    cleanup_timeout: float = Field(default=10, gt=0, allow_inf_nan=False)


class DecisionRuntime:

    def __init__(self, config=None):
        self.config = DecisionConfig.model_validate(config or {})
        self.lock = asyncio.Lock()
        self.pending = 0
        self.failure = None
        self.closing = False
        self.tasks = set()
        self.workers = set()

    def check_ready(self):
        if self.closing or self.failure:
            raise DecisionUnavailable(self.failure or 'Decision engine is closing')

    async def close(self):
        self.closing = True
        tasks = list(self.tasks - {asyncio.current_task()})
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def decide(self, engine, request):
        started = time.perf_counter()
        self.check_ready()
        request = DecisionRequest.model_validate(request)
        if request.model is not None and request.model != engine.model_name:
            raise ValueError(f'Unknown model: {request.model}')
        if self.pending >= self.config.max_pending_requests:
            raise DecisionQueueFull('Decision queue is full')
        self.pending += 1
        task = asyncio.current_task()
        self.tasks.add(task)
        acquired = False
        worker = None
        try:
            try:
                await asyncio.wait_for(self.lock.acquire(), self.config.queue_timeout)
            except asyncio.TimeoutError as exc:
                raise DecisionQueueFull('Decision queue wait timed out') from exc
            acquired = True
            self.check_ready()
            admitted = time.perf_counter()
            # Capacity is queried through the engine API, including process proxies.
            limit = min(8192, engine.session_len, engine.engine.get_max_session_len())
            worker = asyncio.create_task(asyncio.to_thread(engine.decision_input, request, limit))
            self.workers.add(worker)
            worker.add_done_callback(self.workers.discard)
            worker.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            try:
                prepared = await asyncio.wait_for(asyncio.shield(worker), self.config.inference_timeout)
                remaining = max(0., self.config.inference_timeout - (time.perf_counter() - admitted))
                logits = (await asyncio.wait_for(
                    engine.async_get_logits([prepared.input_ids],
                                            multimodal=[prepared.multimodal],
                                            read_positions=[prepared.read_positions]), remaining))[0]
            except asyncio.TimeoutError as exc:
                raise DecisionTimeout('Decision execution timed out') from exc
            ended = time.perf_counter()
            response = engine.decision_output(prepared,
                                              logits,
                                              request_id=secrets.token_hex(16),
                                              model=engine.model_name,
                                              thinking=request.thinking,
                                              timing=dict(queue_ms=(admitted - started) * 1000,
                                                          inference_ms=(ended - admitted) * 1000,
                                                          server_ms=(ended - started) * 1000))
            response.timing['server_ms'] = (time.perf_counter() - started) * 1000
            return response.model_dump()
        finally:
            cleanup_error = None
            # Worker owns its uploads until it exits, even after caller timeout.
            if worker is not None and not worker.done():
                from lmdeploy.serve.core.scoring import await_bounded_cleanup
                try:
                    await await_bounded_cleanup(asyncio.shield(worker), self.config.cleanup_timeout)
                except BaseException as exc:
                    cleanup_error = exc
                    if not (isinstance(exc, asyncio.CancelledError) and getattr(exc, 'cleanup_completed', False)):
                        self.failure = 'Decision preprocessing cleanup failed; restart the engine'
            if acquired:
                self.lock.release()
            self.pending -= 1
            self.tasks.discard(task)
            if cleanup_error is not None and isinstance(cleanup_error, asyncio.CancelledError):
                raise cleanup_error
