# Copyright (c) OpenMMLab. All rights reserved.
"""Final-layer attention for one-shot, text-only Intern-Decision scoring."""
import torch.nn.functional as F


def forward_decision_scores(layer, hidden_states, residual, rotary, metadata, rows):
    attention = layer.self_attn
    impl = attention.attn_fwd.impl
    projection = attention.qkv_proj
    query_width, key_width, value_width = projection.all_out_features
    if (not hasattr(impl, 'flash_attn_varlen_func_v3') or metadata.quant_policy or projection.bias is not None
            or key_width != value_width or not impl.causal):
        return None

    x, residual = layer.input_layernorm(hidden_states, residual)
    weight = projection.weight
    cos, sin = rotary
    # All keys and values contribute to the scoring suffix. Only suffix queries are consumed.
    kv = F.linear(x, weight[query_width:]).view(x.size(1), -1, attention.head_dim)
    k, v = kv.chunk(2, dim=1)
    k = attention.k_norm(k)
    empty_q = k.new_empty((k.size(0), 0, attention.head_dim))
    _, k = attention.apply_rotary_pos_emb(empty_q, k, cos, sin, inplace=True)

    padded_rows = F.pad(rows.queries, (0, max(0, 512 - rows.queries.numel())))
    q_gate = F.linear(x[0].index_select(0, padded_rows), weight[:query_width])[:rows.queries.numel()]
    q, gate = q_gate.view(q_gate.size(0), -1, 2 * attention.head_dim).chunk(2, dim=-1)
    q = attention.q_norm(q)
    empty_k = q.new_empty((q.size(0), 0, attention.head_dim))
    q, _ = attention.apply_rotary_pos_emb(q, empty_k, cos.index_select(0, rows.queries),
                                          sin.index_select(0, rows.queries), inplace=True)

    # FA3 aligns the causal mask at the right end of each query/key sequence.
    out = impl.flash_attn_varlen_func_v3(q=q, k=k, v=v, cu_seqlens_q=rows.cu_query_lengths,
                                         cu_seqlens_k=metadata.cu_seqlens_k, max_seqlen_q=rows.max_query_length,
                                         max_seqlen_k=metadata.max_kv_seqlen, softmax_scale=impl.scale,
                                         causal=True, window_size=impl._normalize_sliding_window(impl.sliding_window),
                                         softcap=impl.logit_softcapping)
    gated = out.flatten(1) * gate.flatten(1).sigmoid()
    out = attention.o_proj(gated.index_select(0, rows.scores_in_suffix)[None])
    out, residual = layer.post_attention_layernorm(out, residual.index_select(1, rows.scores_in_prompt))
    return layer.mlp(out), residual
