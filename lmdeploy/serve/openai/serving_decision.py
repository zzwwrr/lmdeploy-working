# Copyright (c) OpenMMLab. All rights reserved.
"""Thin HTTP transport for the existing engine's decision task."""
import asyncio
from contextlib import suppress

from fastapi.responses import JSONResponse

from lmdeploy.serve.processors.intern_decision.protocol import DecisionError

from .protocol import ErrorResponse


def error_response(code, message):
    return JSONResponse(ErrorResponse(message=message, type='decision_error', code=code).model_dump(), status_code=code)


async def serve_decision(request, raw_request, engine):
    if getattr(engine, 'task', None) != 'intern-decision':
        return error_response(503, 'Enable task=intern-decision to use decisions')
    if request.model is not None and request.model != engine.model_name:
        return error_response(404, f'Unknown model: {request.model}')
    # Python paths are an explicitly trusted input. HTTP gets bytes only.
    if any(not isinstance(item, dict) or not isinstance(item.get('data'), str) for item in request.images):
        return error_response(400, 'HTTP images must contain uploaded base64 data; paths and URLs are unsupported')

    async def disconnected():
        while not await raw_request.is_disconnected():
            await asyncio.sleep(0.05)

    operation = asyncio.create_task(engine.async_decide(request))
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((operation, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            operation.cancel()
            with suppress(asyncio.CancelledError):
                await operation
            return error_response(499, 'Client disconnected')
        return await operation
    except DecisionError as exc:
        return error_response(exc.status_code, str(exc))
    except ValueError as exc:
        return error_response(400, str(exc))
    except Exception:
        return error_response(500, 'Decision scoring failed')
    finally:
        watcher.cancel()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(watcher, operation, return_exceptions=True)
