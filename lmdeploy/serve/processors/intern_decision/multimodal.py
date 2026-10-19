# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/inference/lmdeploy_inputs.py.
# Modified for LMDeploy: standalone inference-only protocol and input processing.
"""Validate and bridge decision processor tensors to LMDeploy raw image inputs.

No LMDeploy import is required here. The engine's own InputProcessor converts
these dictionaries to MultiModalTensor objects and computes MRoPE positions.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .encoding import EncodedDecision


def _config_value(config, name):
    value = config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)
    if value is None:
        raise ValueError(f"Model config must define {name}")
    return value


def _integer(config, name):
    value = _config_value(config, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Model config {name} must be an integer")
    return value


def to_lmdeploy_multimodal(encoded: EncodedDecision, hf_config) -> list[dict] | None:
    """Convert 0..8 static images, preserving patch and token boundaries.

    ``hf_config`` accepts a Transformers config or its nested ``to_dict()`` form.
    Pixels are split by unmerged T*H*W patches, while image token spans have
    T*H*W/merge**2 entries. All validation happens on CPU, before engine submission.
    """
    batch = encoded.batch
    ids = batch.get("input_ids")
    if (not isinstance(ids, torch.Tensor) or ids.device.type != "cpu" or ids.dtype != torch.long or ids.ndim != 2
            or ids.shape != (1, encoded.input_length)):
        raise ValueError("Expected CPU int64 input_ids with shape [1, input_length]")
    image_id = _integer(hf_config, "image_token_id")
    start_id = _integer(hf_config, "vision_start_token_id")
    end_id = _integer(hf_config, "vision_end_token_id")
    if len({image_id, start_id, end_id}) != 3:
        raise ValueError("Image/start/end token IDs must be distinct")
    content = encoded.compiled.messages[1]["content"]
    image_count = sum(part.get("type") == "image_url" for part in content) if isinstance(content, list) else 0
    if image_count > 8:
        raise ValueError("At most 8 static images are supported")
    tokens = ids[0].tolist()
    spans = []
    cursor = 0
    while cursor < len(tokens):
        if tokens[cursor] != image_id:
            cursor += 1
            continue
        begin = cursor
        while cursor < len(tokens) and tokens[cursor] == image_id:
            cursor += 1
        spans.append((begin, cursor))
    starts = [i for i, token in enumerate(tokens) if token == start_id]
    ends = [i for i, token in enumerate(tokens) if token == end_id]
    if len(spans) != image_count or len(starts) != image_count or len(ends) != image_count:
        raise ValueError("Image spans and vision delimiters must match the input image count")
    if image_count == 0:
        if "pixel_values" in batch or "image_grid_thw" in batch:
            raise ValueError("Image tensors supplied without input images")
        return None
    merge = _integer(_config_value(hf_config, "vision_config"), "spatial_merge_size")
    if merge != 2:
        raise ValueError("This LMDeploy Qwen3.5 adapter requires spatial_merge_size=2")
    grids = batch.get("image_grid_thw")
    pixels = batch.get("pixel_values")
    if (not isinstance(grids, torch.Tensor) or grids.device.type != "cpu"
            or grids.dtype not in (torch.int32, torch.int64) or grids.shape != (image_count, 3)):
        raise ValueError("image_grid_thw must be a CPU integer tensor with shape [image_count, 3]")
    if (not isinstance(pixels, torch.Tensor) or pixels.device.type != "cpu" or pixels.ndim != 2 or pixels.shape[1] < 1
            or not pixels.is_floating_point() or not bool(torch.isfinite(pixels).all())):
        raise ValueError("pixel_values must be a finite CPU floating tensor of flattened patches")
    result = []
    patch_offset = 0
    for index, ((begin, end), (t, h, w)) in enumerate(zip(spans, grids.tolist())):
        if begin == 0 or end == len(tokens) or starts[index] != begin - 1 or ends[index] != end:
            raise ValueError("Each image span must be enclosed by vision_start and vision_end")
        if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge:
            raise ValueError("Static image grid requires T=1 and positive H/W divisible by merge=2")
        patches = t * h * w
        image_tokens = patches // (merge * merge)
        if end - begin != image_tokens:
            raise ValueError("Image token span does not match its merged grid size")
        next_offset = patch_offset + patches
        if next_offset > len(pixels):
            raise ValueError("Image grid requires more pixel patches than supplied")
        result.append({
            "pixel_values": pixels[patch_offset:next_offset],
            "image_grid_thw": grids[index:index + 1],
            "offset": begin,
            "image_tokens": image_tokens,
            "image_token_id": image_id,
        })
        patch_offset = next_offset
    if patch_offset != len(pixels):
        raise ValueError("Image grids must consume all pixel patches exactly once")
    return result
