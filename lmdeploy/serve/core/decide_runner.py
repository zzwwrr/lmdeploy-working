# Copyright (c) OpenMMLab. All rights reserved.
"""Request-scoped Intern-Decision scoring for HTTP and Python callers."""
from __future__ import annotations

import asyncio
import secrets
import time

from lmdeploy.intern_decision_encoding import DecisionOutputError
from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.processors.intern_decision.images import load_images
from lmdeploy.serve.processors.intern_decision.runtime import DecisionRuntime

from .base_runner import BaseRunner, DisconnectCheck
from .decision_protocol import DecisionRequest, DecisionResponse
from .exceptions import ErrorCode, RequestError


class DecideRunner(BaseRunner):
    """Execute one scoring-only prefill using shared decision components."""

    def __init__(self, engine, runtime: DecisionRuntime):
        super().__init__(engine)
        self.runtime = runtime
        self.encoded = None
        self._run_started = False

    def _prepare(self, request: DecisionRequest, allow_local_paths: bool):
        """CPU work: decode images and render the prompt."""
        images = load_images(request.images, allow_paths=allow_local_paths)
        try:
            row = request.model_dump(exclude={'model', 'thinking', 'images'})
            return images, self.runtime.encoder.encode(row, num_images=len(images))
        except BaseException:
            for image in images:
                image.close()
            raise

    async def after_preprocess(self, prepared) -> None:
        length = prepared.input_token_len
        if length > self.runtime.max_length:
            raise ValueError(f'Input has {length} tokens, above the limit {self.runtime.max_length}; '
                             'truncation is not allowed')
        start = length - self.encoded.tail
        if not 0 <= start < length:
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'Decision skeleton is outside the processed input')
        # Image expansion occurs before the skeleton; score only its rows.
        prepared.gen_config.logprob_start_len = start

    async def _score(self, encoded, images, *, is_disconnected=None) -> tuple[list[list[float]], int]:
        self.encoded = encoded
        gen_config = GenerationConfig(max_new_tokens=0,
                                      logprobs=0,
                                      logprob_start_len=0,
                                      logprob_token_ids=sorted(set(self.runtime.encoder.symbol_ids.values())))
        async with self:
            if images:
                content = [{'type': 'text', 'text': encoded.input_ids}]
                content += [{'type': 'image', 'image': image} for image in images]
                prepared = await self.prepare_engine([{'role': 'user', 'content': content}],
                                                     gen_config=gen_config,
                                                     do_preprocess=False,
                                                     stream_response=False)
            else:
                prepared = await self.prepare_engine(None, gen_config=gen_config,
                                                     input_ids=encoded.input_ids, stream_response=False)
            length = prepared.input_token_len
            out = None
            async for out in self.iter_outputs(is_disconnected=is_disconnected):
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
            if target == self.runtime.encoder.marker_id
        ]
        if len(rows) != len(encoded.candidate_ids):
            raise RequestError(ErrorCode.INTERNAL_ERROR,
                               f'Expected {len(encoded.candidate_ids)} decision rows, got {len(rows)}')
        try:
            logits = [[row[token] for token in ids] for row, ids in zip(rows, encoded.candidate_ids)]
        except KeyError as exc:
            raise RequestError(ErrorCode.INTERNAL_ERROR, 'Scoring rows lack a requested candidate token') from exc
        return logits, length

    async def run(self, request, *, allow_local_paths: bool = False,
                  is_disconnected: DisconnectCheck | None = None) -> dict:
        """Score one request; clean up even when CPU preparation is cancelled."""
        if self._run_started:
            raise RuntimeError('A runner can only execute one request.')
        self._run_started = True
        started = time.perf_counter()
        request = DecisionRequest.model_validate(request)
        runtime = self.runtime
        if request.model is not None and request.model not in runtime.model_names:
            raise ValueError(f'Unknown model: {request.model}')
        if request.images and not runtime.supports_images:
            raise RequestError(ErrorCode.ENGINE_UNAVAILABLE,
                               'This engine was started without a vision encoder; image requests are unavailable')
        runtime.acquire()
        prepare_task = None
        try:
            prepare_task = asyncio.create_task(asyncio.to_thread(self._prepare, request, allow_local_paths))
            # Cancelling the request cannot stop the worker thread. Keep its
            # result accessible so cleanup can close images when it finishes.
            images, encoded = await asyncio.shield(prepare_task)
            prepared = time.perf_counter()
            logits, length = await self._score(encoded, images, is_disconnected=is_disconnected)
            scored = time.perf_counter()
            try:
                answers = runtime.output(request.questions, encoded.compiled.fields, logits)
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
                                        calibration=runtime.output.calibration)
            return response.model_dump()
        finally:
            async def cleanup():
                try:
                    await self.close()
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
                        runtime.release()

            # Keep admission until the worker has finished and released images,
            # even if the HTTP task is cancelled again while waiting here.
            self._request_cleanup_task = asyncio.create_task(cleanup(), name='decision_request_cleanup')
            await asyncio.shield(self._request_cleanup_task)
