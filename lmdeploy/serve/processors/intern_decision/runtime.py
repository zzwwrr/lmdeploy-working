# Copyright (c) OpenMMLab. All rights reserved.
"""Shared model components and admission state for Intern-Decision requests."""
import os

from lmdeploy.intern_decision_encoding import DEFAULT_TEMPERATURE, DecisionEncoder, DecisionOutputProcessor
from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole

from .protocol import DecisionQueueFull

TASK_NAME = 'intern-decision'


class DecisionRuntime:
    """Initialize once per engine; used by request-scoped DecideRunner objects.

    Admission methods run on the engine's serving event loop, including calls
    scheduled by Pipeline.decide(). Tokenizers and calibration are shared.
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

    def acquire(self) -> None:
        if self.pending >= self.max_pending_requests:
            raise DecisionQueueFull('Too many pending decision requests')
        self.pending += 1

    def release(self) -> None:
        self.pending -= 1
