# Copyright (c) OpenMMLab. All rights reserved.
"""The decision operation shared by ``Pipeline.decide`` and ``POST /v1/decisions``.

A decision is one scoring-only prefill: ``max_new_tokens=0`` plus ``logprob_start_len`` makes the engine project
the LM head only on the answer skeleton, and ``logprob_token_ids`` returns the logits of the answer symbols on
every scored row. The rows whose target token is ``<decision>`` are the decision rows.
"""
import asyncio
import os
import secrets
import time

from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators

from .encoding import DecisionEncoder
from .images import load_images
from .output import DEFAULT_TEMPERATURE, DecisionOutputProcessor
from .protocol import DecisionError, DecisionQueueFull, DecisionRequest, DecisionResponse, DecisionUnavailable

TASK_NAME = 'intern-decision'


class DecisionService:
    """Attach the Intern-Decision protocol to an existing PyTorch ``AsyncEngine``.

    Args:
        async_engine: the engine that serves the checkpoint (``VLAsyncEngine`` for image requests).
        model_path: local checkpoint directory; its tokenizer and processor render the trained prompt.
        temperature: calibration temperature applied to every candidate distribution.
        max_length: largest accepted prompt, in tokens after image expansion. Longer prompts are rejected, never
            truncated.
        max_pending_requests: decisions admitted at once; further requests fail with ``DecisionQueueFull``.
    """

    def __init__(self,
                 async_engine,
                 model_path: str,
                 *,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_length: int = 8192,
                 max_pending_requests: int = 64,
                 trust_remote_code: bool = False):
        config = async_engine.backend_config
        if async_engine.backend != 'pytorch' or not isinstance(config, PytorchEngineConfig):
            raise ValueError(f'{TASK_NAME} requires the PyTorch backend (input logprobs are PyTorch-only)')
        if async_engine.speculative_config is not None:
            raise ValueError(f'{TASK_NAME} does not support speculative decoding')
        if config.role != EngineRole.Hybrid:
            raise ValueError(f'{TASK_NAME} requires the hybrid engine role')
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError('max_length must be a positive integer')
        self.async_engine = async_engine
        self.max_length = min(max_length, async_engine.session_len - 1)
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

    def _prepare(self, request: DecisionRequest, allow_local_paths: bool):
        """CPU work: decode images and render the prompt."""
        images = load_images(request.images, allow_paths=allow_local_paths)
        try:
            row = request.model_dump(exclude={'model', 'thinking', 'images'})
            return images, self.encoder.encode(row, num_images=len(images))
        except BaseException:
            for image in images:
                image.close()
            raise

    async def _score(self, encoded, images) -> tuple[list[list[float]], int]:
        """Run one scoring-only prefill and return the candidate logits of every decision row."""
        engine = self.async_engine
        gen_config = GenerationConfig(max_new_tokens=0,
                                      logprobs=0,
                                      logprob_start_len=0,
                                      logprob_token_ids=sorted(set(self.encoder.symbol_ids.values())))
        session = engine.session_mgr.get()
        # Like ServerContext.create_session: a stop_all_session() after this point aborts the request.
        session.epoch = engine.epoch
        generator = None
        try:
            if images:
                # Token ids in the text slot take the raw (no chat template) path; the engine's processor expands
                # each <|image_pad|> to the image's patch count, in order.
                content = [{'type': 'text', 'text': encoded.input_ids}]
                content += [{'type': 'image', 'image': image} for image in images]
                request = await engine.preprocess([{
                    'role': 'user',
                    'content': content
                }],
                                                  session,
                                                  gen_config=gen_config,
                                                  do_preprocess=False)
            else:
                request = await engine.preprocess(None, session, gen_config=gen_config, input_ids=encoded.input_ids)
            length = request.input_token_len
            if length > self.max_length:
                raise ValueError(f'Input has {length} tokens, above the limit {self.max_length}; '
                                 'truncation is not allowed')
            start = length - encoded.tail
            if not 0 <= start < length:
                raise DecisionError('Decision skeleton is outside the processed input')
            # Only the skeleton rows go through the LM head.
            request.gen_config.logprob_start_len = start

            generator = engine.generate(request, stream_response=False)
            out = None
            async for out in generator:
                pass
        finally:
            # Idempotent with generate()'s own cleanup; also covers failures before generate() starts.
            await asyncio.shield(
                cleanup_result_generators([generator] if generator is not None else [], [session],
                                          engine.session_mgr))

        if out is not None and out.finish_reason == 'abort':
            raise DecisionUnavailable('The decision request was aborted by the engine')
        if out is None or out.logprobs is None or out.logprob_token_ids is None:
            reason = getattr(out, 'finish_reason', None)
            raise DecisionError(f'Scoring returned no input logprobs (finish_reason={reason})')
        if len(out.logprobs) != len(out.logprob_token_ids):
            raise DecisionError('Scoring rows and target tokens are misaligned')
        # Row j holds the logits that predict logprob_token_ids[j].
        rows = [row for target, row in zip(out.logprob_token_ids, out.logprobs) if target == self.encoder.marker_id]
        if len(rows) != len(encoded.candidate_ids):
            raise DecisionError(f'Expected {len(encoded.candidate_ids)} decision rows, got {len(rows)}')
        try:
            logits = [[row[token] for token in ids] for row, ids in zip(rows, encoded.candidate_ids)]
        except KeyError as exc:
            raise DecisionError('Scoring rows lack a requested candidate token') from exc
        return logits, length

    async def decide(self, request, *, allow_local_paths: bool = False) -> dict:
        """Score one request and return a JSON-compatible dict. No tokens are generated."""
        started = time.perf_counter()
        request = DecisionRequest.model_validate(request)
        if request.model is not None and request.model not in self.model_names:
            raise ValueError(f'Unknown model: {request.model}')
        if request.images and not self.supports_images:
            raise DecisionUnavailable('This engine was started without a vision encoder; image requests are '
                                      'unavailable')
        if self.pending >= self.max_pending_requests:
            raise DecisionQueueFull('Too many pending decision requests')
        self.pending += 1
        try:
            images, encoded = await asyncio.to_thread(self._prepare, request, allow_local_paths)
            prepared = time.perf_counter()
            try:
                logits, length = await self._score(encoded, images)
            finally:
                for image in images:
                    image.close()
            scored = time.perf_counter()
            answers = self.output(request.questions, encoded.compiled.fields, logits)
        finally:
            self.pending -= 1
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
