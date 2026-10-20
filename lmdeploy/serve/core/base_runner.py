# Copyright (c) OpenMMLab. All rights reserved.
"""Request-scoped engine execution shared by serving tasks."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators

from .exceptions import ErrorCode, RequestError

if TYPE_CHECKING:
    from lmdeploy.messages import GenerationConfig
    from lmdeploy.serve.managers import Session

    from .async_engine import AsyncEngine, GenOut, PreprocessedRequest

DisconnectCheck = Callable[[], Awaitable[bool]]


def create_session(engine: AsyncEngine, user_session_id: int | None = None) -> Session:
    """Resolve an API session and stamp its engine epoch."""
    manager = engine.session_mgr
    if user_session_id is None or user_session_id == -1:
        session = manager.get()
    else:
        session = manager.get(manager.map_user_session_id(user_session_id))
    # A rejected concurrent request must not change or remove the active session.
    if getattr(session, '_handle', None) is not None:
        raise RequestError(ErrorCode.REQUEST_CONFLICT, f'Session {session.session_id} already has an active request.')
    session.epoch = engine.epoch
    return session


class BaseRunner:
    """Own one engine request from preparation through cancellation and cleanup.

    Task runners build inputs/configuration and interpret raw engine outputs.
    They must close the runner when output parsing fails or consumption stops
    early, either with a context manager or a ``finally`` block.
    """

    def __init__(self, engine: AsyncEngine):
        self.async_engine = engine
        self.session_manager = engine.session_mgr
        self.session = None
        self.result_generator = None
        self._started = False
        self._cleanup_task = None

    async def prepare_engine(self,
                             messages,
                             *,
                             gen_config: GenerationConfig,
                             user_session_id: int | None = None,
                             stream_response: bool = True,
                             **preprocess_kwargs) -> PreprocessedRequest:
        if self._started or self._cleanup_task is not None:
            raise RuntimeError('A runner can only execute one request.')
        self._started = True
        try:
            self.session = create_session(self.async_engine, user_session_id)
            prepared = await self.async_engine.preprocess(
                messages, self.session, gen_config=gen_config, **preprocess_kwargs)
            await self.after_preprocess(prepared)
            self.result_generator = self.async_engine.generate(prepared, stream_response=stream_response)
            return prepared
        except BaseException as exc:
            if isinstance(exc, RequestError) and exc.code == ErrorCode.REQUEST_CONFLICT:
                # Another request owns this session's handle; leave it intact.
                self.session = None
            await self.close()
            raise

    async def after_preprocess(self, prepared: PreprocessedRequest) -> None:
        """Validate/adjust the processed input before starting engine execution."""

    async def iter_outputs(self, *, is_disconnected: DisconnectCheck | None = None) -> AsyncIterator[GenOut]:
        if self.result_generator is None:
            raise RuntimeError('Prepare the runner before consuming outputs.')
        try:
            async for result in self.result_generator:
                if is_disconnected is not None and await is_disconnected():
                    await self.session.async_abort()
                    raise RequestError(ErrorCode.INVALID_REQUEST, 'Client disconnected')
                yield result
        finally:
            await self.close()

    async def close(self) -> None:
        """Schedule cleanup once; cancellation of a waiter cannot cancel it."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(
                cleanup_result_generators(
                    [self.result_generator] if self.result_generator is not None else [],
                    [self.session] if self.session is not None else [],
                    self.session_manager,
                ),
                name=f'{type(self).__name__}_cleanup',
            )
        await asyncio.shield(self._cleanup_task)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
