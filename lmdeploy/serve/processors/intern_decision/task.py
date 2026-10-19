# Copyright (c) OpenMMLab. All rights reserved.
"""Explicit task selection; network registration remains unchanged."""
from copy import deepcopy

from lmdeploy.archs import get_model_arch
from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole


def default_backend_config():
    return PytorchEngineConfig(dtype='bfloat16',
                               session_len=8192,
                               max_batch_size=1,
                               max_prefill_token_num=8192,
                               eager_mode=True,
                               cache_max_entry_count=0.05,
                               enable_prefix_caching=False)


def configure_task(task, model_path, backend, config, speculative_config=None):
    if task != 'intern-decision':
        raise ValueError(f'Unsupported task: {task}')
    if backend != 'pytorch' or (config is not None and not isinstance(config, PytorchEngineConfig)):
        raise ValueError('intern-decision requires the PyTorch backend')
    arch, hf_config = get_model_arch(model_path)
    if arch != 'Qwen3_5ForConditionalGeneration':
        raise ValueError('intern-decision requires Qwen3_5ForConditionalGeneration')
    text = hf_config.text_config
    if (text.hidden_size, text.num_hidden_layers, text.vocab_size) != (1024, 24, 248320):
        raise ValueError('Only the validated Intern-Decision-0.8B configuration is supported')
    if getattr(hf_config, 'quantization_config', None):
        raise ValueError('Quantized decision checkpoints are not supported')
    config = deepcopy(config) if config is not None else default_backend_config()
    if (config.tp != 1 or config.dp != 1 or config.ep != 1 or config.quant_policy != 0
            or config.dtype not in ('auto', 'bfloat16') or config.enable_prefix_caching
            or config.max_batch_size not in (None, 1) or not config.eager_mode or config.max_prefill_token_num != 8192
            or config.device_type != 'cuda' or config.adapters or config.empty_init or config.enable_microbatch
            or config.disable_vision_encoder or config.model_format or speculative_config or config.custom_module_map
            or config.hf_overrides or config.dllm_block_length or config.dllm_denoising_steps or config.enable_eplb
            or config.role != EngineRole.Hybrid or config.distributed_executor_backend not in (None, 'uni')
            or config.attn_tp_size not in (None, 1) or config.mlp_tp_size not in (None, 1)
            or config.moe_tp_size not in (None, 1) or config.enable_mp_engine or config.mp_engine_backend != 'mp'):
        raise ValueError('intern-decision requires BF16, TP/DP/EP=1, batch=1, eager mode, '
                         'prefill=8192, in-process engine, no prefix caching, adapters, quantization or speculation')
    if config.session_len is not None and (type(config.session_len) is not int or config.session_len < 1):
        raise ValueError('Decision session_len must be a positive integer')
    config.dtype = 'bfloat16'
    config.max_batch_size = 1
    config.session_len = config.session_len or 8192
    return config
