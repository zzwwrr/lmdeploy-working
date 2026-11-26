# Copyright (c) OpenMMLab. All rights reserved.
"""Fused width-four depthwise convolution and GDN prefill input preparation."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=['length'])
def _conv_prepare(x, weight, history, row_ids, starts, input_ids, b, a, dt, a_scale,
                  q_out, k_out, v_out, beta_out, g_out, length,
                  SX: tl.constexpr, SH0: tl.constexpr, SH1: tl.constexpr, SH2: tl.constexpr,
                  SB: tl.constexpr, SA: tl.constexpr, UNITS: tl.constexpr,
                  H: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    head = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    c = head * D + d
    active = t < length
    row = tl.load(row_ids + t, active, other=0)
    start = tl.load(starts + row, active, other=0)
    distance = t - start
    acc = tl.full((BT, D), 0, tl.float32)
    # A tile can straddle requests. Use history only for their first three tokens.
    interior = tl.min(tl.where(active, distance, 3), 0) >= 3
    if interior:
        for tap in tl.static_range(4):
            z = tl.load(x + (t + tap - 3)[:, None] * SX + c[None, :], active[:, None], other=0)
            w = tl.load(weight + 4 * c + tap)
            acc = tl.fma(z.to(tl.float32), w[None, :].to(tl.float32), acc)
    else:
        slot = tl.load(input_ids + row, active, other=0)
        for tap in tl.static_range(4):
            from_x = distance + tap >= 3
            z = tl.load(x + (t + tap - 3)[:, None] * SX + c[None, :],
                        (active & from_x)[:, None], other=0)
            if tap < 3:
                past = tl.load(history + slot[:, None].to(tl.int64) * SH0 + c[None, :] * SH1
                               + (distance + tap)[:, None] * SH2,
                               (active & ~from_x & (slot >= 0) & (slot < UNITS))[:, None], other=0)
                z = tl.where(from_x[:, None], z, past)
            w = tl.load(weight + 4 * c + tap)
            acc = tl.fma(z.to(tl.float32), w[None, :].to(tl.float32), acc)
    # Match the original BF16 convolution output before the FP32 Q/K norm.
    conv = (tl.sigmoid(acc) * acc).to(tl.bfloat16)
    if head < 2 * H:
        raw = conv.to(tl.float32)
        normalized = raw * (1.0 / tl.sqrt(tl.sum(raw * raw, 1) + 1e-6))[:, None]
        target = q_out
        src_head = head
        if head >= H:
            target = k_out
            src_head = head - H
        base = t[:, None] * (H * R * D) + src_head * (R * D) + d[None, :]
        for rep in tl.static_range(R):
            tl.store(target + base + rep * D, normalized, active[:, None])
    else:
        v_head = head - 2 * H
        tl.store(v_out + t[:, None] * H * R * D + v_head * D + d[None, :], conv, active[:, None])
        gate = tl.load(a + t * SA + v_head, active, other=0).to(tl.float32)
        decay = gate + tl.load(dt + v_head).to(tl.float32)
        log_decay = tl.load(a_scale + v_head).to(tl.float32) * tl.where(
            decay > 20.0, decay, libdevice.log1p(libdevice.exp(decay)))
        beta = tl.sigmoid(tl.load(b + t * SB + v_head, active, other=0).to(tl.float32))
        tl.store(beta_out + t * H * R + v_head, beta, active)
        tl.store(g_out + t * H * R + v_head, log_decay, active)


def conv_prepare(x, weight, history, row_ids, starts, input_ids, b, a, dt, a_scale, *, head_dim, ratio):
    """Return contiguous expanded Q/K, V, beta and log-decay for one packed prefill."""
    _, length, channels = x.shape
    heads = channels // ((2 + ratio) * head_dim)
    shape = (1, length, heads * ratio, head_dim)
    q = x.new_empty(shape)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    beta = b.new_empty(shape[:-1])
    g = torch.empty_like(beta, dtype=torch.float32)
    block_t = 8
    _conv_prepare[(heads * (2 + ratio), triton.cdiv(length, block_t))](
        x, weight, history, row_ids, starts, input_ids, b, a, dt, a_scale, q, k, v, beta, g, length,
        x.stride(1), *history.stride(), b.stride(1), a.stride(1), history.size(0),
        heads, ratio, head_dim, block_t, num_warps=4)
    return q, k, v, beta, g


def indexed_gdn_prefill(layer, x, b, a, history, state, meta):
    from .causal_conv1d import _export_conv_history
    from .indexed_chunk_gated_delta import indexed_chunk_gated_delta_rule
    ids, prefill = meta.ssm_state_meta, meta.ssm_prefill_meta
    input_ids, final_ids = ids.input_state_ids, ids.output_state_ids[:, 0]
    q, k, v, beta, g = conv_prepare(
        x, layer.conv1d.weight, history, prefill.seq_idx, prefill.cu_seqlens, input_ids, b, a,
        layer.dt_bias, layer.get_A_log_exp(), head_dim=layer.head_k_dim, ratio=layer.kv_ratio)
    # An input can also be a final destination. Export only after all readers complete.
    export = _export_conv_history(x.size(-1), 4, x.dtype, x.stride(1), history.stride(),
                                  input_ids.dtype, prefill.checkpoint_ids is not None)
    export(x[0].T, history, input_ids, final_ids, prefill.cu_seqlens, prefill.checkpoint_ids,
           prefill.checkpoint_offsets)
    return indexed_chunk_gated_delta_rule(
        q, k, v, g, beta, state, input_ids, final_ids, state_v_first=True,
        cu_seqlens=prefill.cu_seqlens, chunk_indices=prefill.chunk_indices, chunk_offsets=prefill.chunk_offsets,
        checkpoint_ids=prefill.checkpoint_ids, checkpoint_offsets=prefill.checkpoint_offsets)


def try_indexed_gdn_prefill(layer, x, b, a, history, state, meta):
    from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import get_piecewise_graph_execution

    if (meta.is_decoding or getattr(meta, 'ssm_prefill_meta', None) is None
            or getattr(meta, 'ssm_state_meta', None) is None
            or get_piecewise_graph_execution() is not None or x.dtype != torch.bfloat16
            or x.ndim != 3 or x.size(0) != 1 or not 2048 < x.size(1) <= 32768 or x.stride(-1) != 1
            or b.shape != a.shape or b.ndim != 3 or b.stride(-1) != 1 or a.stride(-1) != 1
            or b.dtype != x.dtype or a.dtype != x.dtype or history.dtype != x.dtype
            or history.ndim != 3 or history.size(-1) != 3 or not layer.conv1d.weight.is_contiguous()):
        return None
    return indexed_gdn_prefill(layer, x, b, a, history, state, meta)
