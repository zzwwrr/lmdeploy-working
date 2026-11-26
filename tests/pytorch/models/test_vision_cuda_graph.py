# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.models.utils.vision_cuda_graph import VisionCudaGraphs


def _encode(pixels, ends, rotary, positions):
    return pixels + ends[0] + rotary[0] - rotary[1] + positions


def _inputs(device, n, first, offset=0):
    data = torch.arange(n * 2, device=device, dtype=torch.float32).add_(offset)[::2]
    return data, torch.tensor([first, n], device=device), (data + 3, data * 2), data + 1


def test_cpu_inputs_take_eager_path():
    cache = VisionCudaGraphs()
    args = _inputs('cpu', 6, 2)
    with torch.inference_mode():
        torch.testing.assert_close(cache(_encode, *args), _encode(*args), rtol=0, atol=0)
    assert not cache.pending and not cache.captured


def test_piecewise_embedding_opt_in_descriptor_and_padding():
    from lmdeploy.pytorch.backends.cuda.graph_runner.standard import StandardDecoderPiecewiseGraphRuntime

    context = SimpleNamespace(enable_microbatch=False, local_adapter_ids=None, is_dummy=False)
    inputs = dict(input_ids=torch.zeros(1, 3, dtype=torch.long), position_ids=torch.arange(3)[None],
                  inputs_embeds=torch.full((1, 3, 8), 2.5))
    stock = StandardDecoderPiecewiseGraphRuntime(SimpleNamespace(), 8, token_stride=4)
    assert stock.get_piecewise_graph_descriptor(context, inputs) is None
    runtime = StandardDecoderPiecewiseGraphRuntime(SimpleNamespace(supports_piecewise_inputs_embeds=True),
                                                  8, token_stride=4)
    desc = runtime.get_piecewise_graph_descriptor(context, inputs)
    assert desc.token_bucket == 4
    index = desc.graph_input_names.index('inputs_embeds')
    buffer = runtime._make_bucket_inputs(inputs, desc)[index]
    torch.testing.assert_close(buffer[:, :3], inputs['inputs_embeds'])
    assert buffer.shape == (1, 4, 8) and torch.count_nonzero(buffer[:, 3]) == 0
    for alt in (torch.ones(1, 3, 4), inputs['inputs_embeds'].half()):
        assert runtime.get_piecewise_graph_descriptor(context, dict(inputs, inputs_embeds=alt)) != desc
    two_rows = {name: value.expand(2, *value.shape[1:]) for name, value in inputs.items()}
    assert runtime.get_piecewise_graph_descriptor(context, two_rows) != desc
    assert runtime.get_piecewise_graph_descriptor(context, dict(inputs, inputs_embeds=torch.zeros(1, 2, 8))) is None
    assert runtime.get_piecewise_graph_descriptor(context, dict(inputs, pixel_values=torch.zeros(1))) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
@torch.inference_mode()
def test_replay_with_new_data_and_boundaries_owns_each_output():
    cache = VisionCudaGraphs(max_shapes=2, max_patches=12, max_frames=2)
    old = []
    for n, first, offset in [(6, 2, 0)] * 3 + [(10, 8, 1)] * 3 + [(6, 5, -2), (10, 2, 4), (6, 1, 7)]:
        args = _inputs('cuda', n, first, offset)
        old.append((cache(_encode, *args), _encode(*args)))
    assert len(cache.captured) == 2
    for got, expected in old:
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    owner = torch.cuda.current_stream()
    other = torch.cuda.Stream()
    other.wait_stream(owner)
    count = 0

    def counted(*args):
        nonlocal count
        count += 1
        return _encode(*args)

    with torch.cuda.stream(other):
        args = _inputs('cuda', 6, 4, 7)
        got = cache(counted, *args)
        expect = _encode(*args)
    owner.wait_stream(other)
    torch.testing.assert_close(got, expect, rtol=0, atol=0)
    cache(counted, *_inputs('cuda', 14, 3))
    cache(counted, *_inputs('cuda', 8, 3))
    assert count == 3
    cache.clear()
    assert not cache.captured and cache.owner is None and cache.pool is None
    for _ in range(3):
        cache(_encode, *_inputs('cuda', 8, 3))
    assert len(cache.captured) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
@torch.inference_mode()
def test_capture_allocation_failure_returns_eager_result(monkeypatch):
    cache = VisionCudaGraphs()
    args = _inputs('cuda', 6, 3)
    calls = 0

    def oom(*_):
        nonlocal calls
        calls += 1
        raise torch.OutOfMemoryError('simulated capture allocation failure')

    monkeypatch.setattr(cache, '_capture', oom)
    for _ in range(5):
        torch.testing.assert_close(cache(_encode, *args), _encode(*args), rtol=0, atol=0)
    assert cache.out_of_memory and calls == 1
    cache.clear()
    assert not cache.out_of_memory
