# Copyright (c) OpenMMLab. All rights reserved.
# Image upload policy adapted from Intern-Decision (0a8a2388), src/service/uploads.py.
# Copyright 2026 Shanghai AI Laboratory; SPDX-License-Identifier: Apache-2.0.
"""Shared decision components and per-request scoring for HTTP and Python callers."""
from __future__ import annotations

import asyncio
import os
import secrets
import time
from functools import partial
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from lmdeploy.intern_decision_encoding import (
    DEFAULT_TEMPERATURE,
    DecisionEncoder,
    DecisionOutputError,
    DecisionOutputProcessor,
)
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.utils import get_logger
from lmdeploy.vl.media.image import ImageMediaIO

from .base_runner import BaseRunner, DisconnectCheck
from .decision_protocol import MAX_IMAGES, DecisionRequest, DecisionResponse
from .exceptions import ErrorCode, RequestError

logger = get_logger('lmdeploy')
TASK_NAME = 'intern-decision'

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_TOTAL = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
FORMATS = {'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP', 'image/gif': 'GIF'}


def _decode_upload(item: dict) -> tuple[bytes, str]:
    """Return the raw bytes and PIL format name of a base64 upload object."""
    data = item.get('data')
    if not isinstance(data, str):
        raise ValueError('Uploaded images need a base64 "data" string.')
    if len(data) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 128:
        raise ValueError('Each image is limited to 12 MB.')
    if data.startswith('data:'):
        header, separator, encoded = data.partition(',')
        if not separator or not header.endswith(';base64'):
            raise ValueError('Images must use base64 data URLs.')
        content_type = header[5:].split(';')[0].lower()
    else:
        encoded, content_type = data, str(item.get('type', '')).lower()
    declared = item.get('type')
    if declared is not None and str(declared).lower() != content_type:
        raise ValueError('Image MIME and data URL disagree')
    if content_type not in FORMATS:
        raise ValueError('Supported image formats: JPEG, PNG, WebP and static GIF.')
    payload = ImageMediaIO(strict_base64=True, max_bytes=MAX_UPLOAD_BYTES).decode_base64(content_type, encoded)
    return payload, FORMATS[content_type]


def _validate_image(source: Image.Image, expected_format: str | None) -> None:
    if expected_format is not None and source.format != expected_format:
        raise ValueError('Image content does not match its declared format.')
    if source.width * source.height > MAX_IMAGE_PIXELS:
        raise ValueError('Each image is limited to 16 million pixels.')
    if getattr(source, 'n_frames', 1) != 1:
        raise ValueError('Animated images are not supported; upload a still image.')


def _load_images(items: list, *, allow_paths: bool, media_root: str | Path = '.') -> list[Image.Image]:
    """Load up to 8 images, in order.

    Upload objects (``{"data": ..., "type": ...}``) are always accepted. Plain strings are local file paths and are
    accepted only when ``allow_paths`` is set (trusted Python callers); remote URLs are never fetched.
    """
    if not isinstance(items, list) or len(items) > MAX_IMAGES:
        raise ValueError(f'images must be a list of at most {MAX_IMAGES} items.')
    images, total = [], 0
    try:
        for item in items:
            if isinstance(item, dict):
                payload, expected = _decode_upload(item)
                total += len(payload)
                if total > MAX_UPLOAD_TOTAL:
                    raise ValueError('Combined image upload is limited to 32 MB.')
                source = payload
            elif isinstance(item, (str, Path)) and str(item):
                if not allow_paths:
                    raise ValueError('Upload image bytes; server-side paths are not accepted.')
                if '://' in str(item):
                    raise ValueError('Remote image URLs are not supported; pass a local path or upload the bytes.')
                path = Path(item)
                source, expected = (path if path.is_absolute() else Path(media_root) / path), None
                if not source.is_file():
                    raise ValueError(f'Image file not found: {item}')
            else:
                raise ValueError('Each image must be a local path or an uploaded image object.')
            try:
                # Validation sees original headers/frames, before RGB conversion.
                # Keep the checkpoint's stored pixel orientation and drop metadata.
                loader = ImageMediaIO(validator=partial(_validate_image, expected_format=expected), clear_metadata=True)
                images.append(loader.load_bytes(source) if isinstance(source, bytes) else loader.load_file(source))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError('Invalid, incomplete or oversized image.') from exc
    except BaseException:
        for image in images:
            image.close()
        raise
    return images


class DecideRunner:
    """Initialize once per engine and run independent scoring requests.

    Encoder, calibration and admission are shared. Sessions, processed inputs,
    generators and images belong to a single run() call. All calls and cleanup
    execute on the engine's serving event loop (including Pipeline.decide()).
    """

    def __init__(self,
                 async_engine,
                 model_path: str,
                 *,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_length: int = 8192,
                 max_pending_requests: int = 64,
                 trust_remote_code: bool = False):
        self.async_engine = async_engine
        config = async_engine.backend_config
        if async_engine.backend != 'pytorch' or not isinstance(config, PytorchEngineConfig):
            raise ValueError(f'{TASK_NAME} requires the PyTorch backend (input logprobs are PyTorch-only)')
        if async_engine.speculative_config is not None:
            raise ValueError(f'{TASK_NAME} does not support speculative decoding')
        if config.role != EngineRole.Hybrid:
            raise ValueError(f'{TASK_NAME} requires the hybrid engine role')
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError('max_length must be a positive integer')
        self.max_length = min(max_length, async_engine.session_len - 1)
        if (isinstance(max_pending_requests, bool) or not isinstance(max_pending_requests, int)
                or max_pending_requests < 1):
            raise ValueError('max_pending_requests must be a positive integer')
        self.max_pending_requests = max_pending_requests
        # Clients of the original service send the checkpoint name (e.g. "Intern-Decision-0.8B").
        self.model_names = {async_engine.model_name, os.path.basename(os.path.normpath(model_path))}
        self.supports_images = getattr(async_engine, 'vl_encoder', None) is not None
        self.encoder = DecisionEncoder(model_path, trust_remote_code=trust_remote_code)
        hf_cfg = async_engine.hf_cfg
        vocab_size = getattr(hf_cfg, 'text_config', hf_cfg).vocab_size
        if any(not 0 <= token < vocab_size for token in [self.encoder.marker_id, *self.encoder.symbol_ids.values()]):
            raise ValueError('Decision tokenizer ids exceed the model vocabulary')
        self.output = DecisionOutputProcessor(temperature)
        self.pending = 0
        self._cleanup_tasks: set[asyncio.Task] = set()

    def _prepare(self, request: DecisionRequest, allow_local_paths: bool):
        """CPU work: decode images and render the prompt."""
        images = _load_images(request.images, allow_paths=allow_local_paths)
        try:
            row = request.model_dump(exclude={'model', 'thinking', 'images'})
            return images, self.encoder.encode(row, num_images=len(images))
        except BaseException:
            for image in images:
                image.close()
            raise

    async def _score(self,
                     execution: BaseRunner,
                     encoded,
                     images,
                     *,
                     is_disconnected=None) -> tuple[list[list[float]], int]:
        def configure_scoring(prepared):
            length = prepared.input_token_len
            if length > self.max_length:
                raise ValueError(f'Input has {length} tokens, above the limit {self.max_length}; '
                                 'truncation is not allowed')
            start = length - encoded.tail
            if not 0 <= start < length:
                raise RequestError(ErrorCode.INTERNAL_ERROR, 'Decision skeleton is outside the processed input')
            # Capture this request's tail; image expansion precedes the skeleton.
            prepared.gen_config.logprob_start_len = start

        gen_config = GenerationConfig(max_new_tokens=0,
                                      logprobs=0,
                                      logprob_start_len=0,
                                      logprob_token_ids=sorted(set(self.encoder.symbol_ids.values())))
        async with execution:
            if images:
                content = [{'type': 'text', 'text': encoded.input_ids}]
                content += [{'type': 'image', 'image': image} for image in images]
                prepared = await execution.prepare_engine([{'role': 'user', 'content': content}],
                                                          gen_config=gen_config,
                                                          do_preprocess=False,
                                                          stream_response=False,
                                                          after_preprocess=configure_scoring)
            else:
                prepared = await execution.prepare_engine(None,
                                                          gen_config=gen_config,
                                                          input_ids=encoded.input_ids,
                                                          stream_response=False,
                                                          after_preprocess=configure_scoring)
            length = prepared.input_token_len
            out = None
            async for out in execution.iter_outputs(is_disconnected=is_disconnected):
                pass

        if out is not None and out.finish_reason == 'abort':
            raise RequestError(ErrorCode.ENGINE_UNAVAILABLE, 'The decision request was aborted by the engine')
        if out is None or out.logprobs is None or out.logprob_token_ids is None:
            reason = getattr(out, 'finish_reason', None)
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Scoring returned no input logprobs (finish_reason={reason})')
        if len(out.logprobs) != len(out.logprob_token_ids):
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'Scoring rows and target tokens are misaligned')
        # Row j holds the logits that predict logprob_token_ids[j].
        rows = [
            row for target, row in zip(out.logprob_token_ids, out.logprobs)
            if target == self.encoder.marker_id
        ]
        if len(rows) != len(encoded.candidate_ids):
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Expected {len(encoded.candidate_ids)} decision rows, got {len(rows)}')
        try:
            logits = [[row[token] for token in ids] for row, ids in zip(rows, encoded.candidate_ids)]
        except KeyError as exc:
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'Scoring rows lack a requested candidate token') from exc
        return logits, length

    async def run(self,
                  request,
                  *,
                  allow_local_paths: bool = False,
                  is_disconnected: DisconnectCheck | None = None) -> dict:
        """Score one request; clean up even when CPU preparation is cancelled."""
        started = time.perf_counter()
        request = DecisionRequest.model_validate(request)
        if request.model is not None and request.model not in self.model_names:
            raise ValueError(f'Unknown model: {request.model}')
        if request.images and not self.supports_images:
            raise RequestError(ErrorCode.ENGINE_UNAVAILABLE,
                               'This engine was started without a vision encoder; image requests are unavailable')
        execution = BaseRunner(self.async_engine)
        if self.pending >= self.max_pending_requests:
            raise RequestError(ErrorCode.OVERLOADED, 'Too many pending decision requests')
        self.pending += 1
        prepare_task = None
        try:
            prepare_task = asyncio.create_task(asyncio.to_thread(self._prepare, request, allow_local_paths))
            # Cancelling the request cannot stop the worker thread. Keep its
            # result accessible so cleanup can close images when it finishes.
            images, encoded = await asyncio.shield(prepare_task)
            prepared = time.perf_counter()
            logits, length = await self._score(execution, encoded, images, is_disconnected=is_disconnected)
            scored = time.perf_counter()
            try:
                answers = self.output(request.questions, encoded.compiled.fields, logits)
            except DecisionOutputError as exc:
                raise RequestError(ErrorCode.INTERNAL_ERROR, str(exc)) from exc
            response = DecisionResponse(request_id=secrets.token_hex(16),
                                        model=self.async_engine.model_name,
                                        answers=answers,
                                        usage=dict(input_tokens=length,
                                                   output_tokens=len(answers),
                                                   decision_count=len(answers),
                                                   generated_tokens=0),
                                        timing=dict(preprocess_ms=(prepared - started) * 1000,
                                                    inference_ms=(scored - prepared) * 1000,
                                                    server_ms=(time.perf_counter() - started) * 1000),
                                        thinking=dict(enabled=False,
                                                      confidence_threshold=request.thinking.confidence_threshold,
                                                      tasks=[]),
                                        calibration=self.output.calibration)
            return response.model_dump()
        finally:

            async def cleanup():
                try:
                    await execution.close()
                finally:
                    try:
                        if prepare_task is not None:
                            try:
                                images, _ = await prepare_task
                            except BaseException:
                                # _prepare closes its images on failure.
                                pass
                            else:
                                for image in images:
                                    image.close()
                    finally:
                        self.pending -= 1

            # Keep admission until the worker has finished and released images,
            # even if the HTTP task is cancelled again while waiting here.
            cleanup_task = asyncio.create_task(cleanup(), name='decision_request_cleanup')
            self._cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._cleanup_finished)
            await asyncio.shield(cleanup_task)

    def _cleanup_finished(self, task: asyncio.Task) -> None:
        self._cleanup_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error('Decision request cleanup failed', exc_info=(type(error), error, error.__traceback__))
