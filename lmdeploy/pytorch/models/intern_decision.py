# Copyright (c) OpenMMLab. All rights reserved.
"""Dense Qwen3.5 with FP32 zero-centered norm scales for Intern-Decision."""
from collections.abc import Iterable

import torch
from transformers import PretrainedConfig

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import RMSNorm

from .patch import get_build_model_context
from .qwen3_5 import Qwen3_5ForConditionalGeneration
from .utils.vision_cuda_graph import VisionCudaGraphs


class InternDecisionForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Reuse Qwen3.5 execution, preserving only text norm scales in FP32."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        if (QuantizationConfig.from_config(config).quant_method is not None
                or get_build_model_context().quant_config.quant_method is not None):
            raise ValueError('Intern-Decision FP32 norms require an unquantized dense Qwen3.5 model.')
        super().__init__(config, ctx_mgr, dtype=dtype, device=device, prefix=prefix)
        allow_decision_prefill = (
            (_envs.decision_share_prefix or _envs.decision_score_suffix) and dtype == torch.bfloat16 and device is not None
            and torch.device(device).type == 'cuda' and torch.cuda.get_device_capability(device) == (9, 0)
            and get_tp_world_rank()[0] == 1 and not get_build_model_context().enable_deterministic)
        self.model.language_model.share_decision_prefix = allow_decision_prefill and _envs.decision_share_prefix
        self.model.language_model.use_decision_suffix = allow_decision_prefill and _envs.decision_score_suffix
        if (get_build_model_context().allow_vision_cuda_graph and dtype == torch.bfloat16 and device is not None
                and torch.device(device).type == 'cuda' and get_tp_world_rank()[0] == 1
                and not getattr(self.model.visual, '_is_dummy_mod', False)
                and torch.cuda.get_device_capability(device) == (9, 0)):
            self.model.visual.vision_graphs = VisionCudaGraphs()
        self.supports_piecewise_inputs_embeds = (getattr(self.model.visual, 'vision_graphs', None) is not None
                                                 and not self.is_spec_decoding)

        # Gated norms use direct scales and are not RMSNorm instances. Restrict
        # this policy to the text model so vision norms keep their original dtype.
        self._fp32_norm_weights = set()
        for name, module in self.model.language_model.named_modules():
            if isinstance(module, RMSNorm):
                module.float()
                self._fp32_norm_weights.add(f'model.language_model.{name}.weight')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Cast offsets before the parent loader folds +1 and copies
        weights."""
        self.reset_vision_graphs()
        super().load_weights((name, weight.float() if name in self._fp32_norm_weights else weight)
                             for name, weight in weights)

    def reset_vision_graphs(self):
        graphs = getattr(self.model.visual, 'vision_graphs', None)
        if graphs is not None:
            graphs.clear()

    def prepare_piecewise_inputs(self, inputs):
        """Finish visual encoding before the standard text prefill graph binds its inputs."""
        if self.is_spec_decoding or inputs['ts_values'] is not None:
            return inputs
        embeds = inputs['inputs_embeds']
        if embeds is None:
            embeds = self.get_input_embeddings()(inputs['input_ids'])
            if inputs['pixel_values'] is not None:
                embeds = self.model.embed_image_tokens(embeds, inputs['pixel_values'], inputs['vis_cu_seqlens'],
                                                       inputs['vis_pos_emb'], inputs['pos_embeds'], inputs['grid_thw'],
                                                       inputs['multimodal_mask'])
        return dict(inputs, inputs_embeds=embeds, pixel_values=None, vis_cu_seqlens=None, vis_pos_emb=None,
                    pos_embeds=None, grid_thw=None, multimodal_mask=None)
