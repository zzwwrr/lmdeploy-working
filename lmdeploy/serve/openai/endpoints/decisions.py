# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-Decision endpoints: a thin HTTP transport over ``DecisionService``."""
from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.errors import create_error_response, create_request_error_response
from lmdeploy.serve.processors.intern_decision import DecisionError, DecisionRequest
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def register(router: APIRouter, server_context) -> None:

    async def decide(request: DecisionRequest, raw_request: Request = None):
        """Score structured questions in one forward pass; no tokens are generated.

        The body follows the Intern-Decision (Jev) protocol: ``state``, ``questions`` and optional ``images``. Over
        HTTP every image must be an upload object ``{"data": "<base64 or data URL>", "type": "image/png"}``;
        server-side paths and remote URLs are rejected.
        """
        service = server_context.decision_service
        if service is None:
            return create_error_response(HTTPStatus.SERVICE_UNAVAILABLE,
                                         'Start the server with --task intern-decision to use decisions.')
        try:
            return await service.decide(request, allow_local_paths=False)
        except RequestError as e:
            return create_request_error_response(e)
        except DecisionError as e:
            return create_error_response(e.status_code, str(e))
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        except Exception:
            logger.exception('[decisions] decision request failed')
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'Decision scoring failed.')

    dependencies = [Depends(validate_json_request)]
    router.add_api_route('/v1/decisions', decide, methods=['POST'], dependencies=dependencies)
    # Compatibility alias for clients of the original Intern-Decision service.
    router.add_api_route('/v1/jev', decide, methods=['POST'], dependencies=dependencies)
