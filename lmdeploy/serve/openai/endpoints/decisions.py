# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-Decision endpoint: one scoring pass per request."""
from __future__ import annotations

import time
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.errors import create_error_response, create_request_error_response
from lmdeploy.serve.openai.protocol import DecisionRequest
from lmdeploy.serve.parsers.decision_parser import DecisionParser
from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


async def _score_request(server_context, parser: DecisionParser, raw_request=None) -> dict:
    """Run one scoring pass without generating tokens.

    Owns the engine session and result generator; the parser only converts request and output data.
    """
    started_at = time.perf_counter()
    inputs = parser.prepare_input()

    session = server_context.create_session()
    generator = None
    output = None
    preprocess_ms = 0.
    inference_ms = 0.
    try:
        started = time.perf_counter()
        prepared = await server_context.async_engine.preprocess(
            inputs.messages, session, gen_config=inputs.gen_config, **inputs.preprocess_kwargs)
        # Marker positions and logprob_start_len depend on the expanded multimodal input.
        parser.finalize_input(prepared)
        preprocess_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        generator = server_context.async_engine.generate(prepared, stream_response=False)
        async for result in generator:
            if raw_request is not None and await raw_request.is_disconnected():
                await session.async_abort()
                raise RequestError(ErrorCode.INVALID_REQUEST, 'Client disconnected')
            # Scoring yields a single final GenOut carrying the complete logprob snapshot.
            output = result
        inference_ms = (time.perf_counter() - started) * 1000
    finally:
        await cleanup_result_generators(
            [generator] if generator is not None else [],
            [session],
            server_context.session_manager)

    if output is None:
        raise RequestError(ErrorCode.INTERNAL_ERROR, 'No generation output from engine.')
    return parser.parse_output(output,
                               timing=dict(preprocess_ms=preprocess_ms,
                                           inference_ms=inference_ms,
                                           server_ms=(time.perf_counter() - started_at) * 1000))


def register(router: APIRouter, server_context) -> None:

    async def decide(request: DecisionRequest, raw_request: Request = None):
        """Score structured questions in one forward pass; no tokens are generated.

        The body follows the Intern-Decision (Jev) protocol: ``state``, ``questions`` and optional ``images``. Over
        HTTP images use the same strings or parameter dictionaries as ``/generate``: a URL,
        complete data URL or server-local path, or ``{"url": "..."}``, supplied singly or as a list.
        Loading follows the shared multimodal pipeline.
        """
        config = server_context.decision_config
        if config is None:
            return create_error_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                'Start the server with --task <Intern-Decision checkpoint name> to use decisions.')
        try:
            return await _score_request(server_context, DecisionParser(request, config), raw_request)
        except RequestError as e:
            return create_request_error_response(e)
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        except Exception:
            logger.exception('[decisions] decision request failed')
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'Decision scoring failed.')

    dependencies = [Depends(validate_json_request)]
    router.add_api_route('/v1/decisions', decide, methods=['POST'], dependencies=dependencies)
    # Compatibility alias for clients of the original Intern-Decision service.
    router.add_api_route('/v1/jev', decide, methods=['POST'], dependencies=dependencies)
