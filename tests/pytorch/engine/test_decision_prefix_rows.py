from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.engine.inputs_maker import _decision_score_suffix, _shared_decision_rows, fill_logits_indices


def test_prefix_map_stops_at_first_different_token():
    # Two independent template groups. Later equal tokens must retain their own histories.
    base = np.arange(160, dtype=np.int64)
    seqs = [base.copy(), base.copy(), base + 1000, base + 1000, np.array([7, 8])]
    seqs[1][100] = -1
    seqs[3][69] = -2
    ids = np.concatenate(seqs)
    kept, inverse = _shared_decision_rows(ids, [len(seq) for seq in seqs])
    source = kept[inverse].numpy()
    np.testing.assert_array_equal(ids[source], ids)
    np.testing.assert_array_equal(source[160:260], np.arange(100))
    np.testing.assert_array_equal(source[260:320], np.arange(260, 320))
    np.testing.assert_array_equal(source[480:549], np.arange(320, 389))
    np.testing.assert_array_equal(source[549:], np.arange(549, len(ids)))


@pytest.mark.parametrize('mode', ['native', 'media', 'chunk', 'history', 'adapter', 'generic', 'short', 'low_savings'])
def test_score_prefill_scope(monkeypatch, mode):
    monkeypatch.setattr(envs, 'decision_share_prefix', True)
    monkeypatch.setattr(envs, 'decision_score_suffix', False)
    length = 100 if mode == 'short' else 3000 if mode == 'low_savings' else 768
    seqs = [np.arange(length, dtype=np.int64) for _ in range(4)]
    for i, seq in enumerate(seqs):
        seq[96:] += i + 1
    inputs = SimpleNamespace(input_ids=torch.tensor(np.concatenate(seqs))[None], is_chunk=mode == 'chunk',
                             is_decoding=False, is_dummy=False, enable_microbatch=False,
                             decision_prefix_rows=None, decision_prefix_inverse=None)
    msgs = [SimpleNamespace(num_history_ids=int(mode == 'history'), input_start_pos=0, logprob_start_pos=length - 3,
                            input_end_pos=length, adapter_name='lora' if mode == 'adapter' else None,
                            num_images=int(mode == 'media'), history_embeddings=[],
                            prefix_cache=SimpleNamespace(multimodal_spans=[]),
                            sampling_param=SimpleNamespace(decision_text_scoring=mode != 'generic')) for _ in seqs]
    fill_logits_indices(inputs, msgs, [length] * len(seqs))
    if mode == 'native':
        assert inputs.decision_prefix_rows.numel() == len(seqs) * length - 3 * 96
        assert inputs.decision_prefix_inverse.numel() == len(seqs) * length
    else:
        assert inputs.decision_prefix_rows is None


def test_score_queries_preserve_causal_end_alignment_and_order():
    lengths = [2500, 3072]
    scores = [2494, 2497, 5561, 5562, 5569]
    plan = _decision_score_suffix(lengths, scores, [2, 3]).to_device('cpu')
    assert plan.queries.tolist() == [*range(2494, 2500), *range(5561, 5572)]
    assert plan.cu_query_lengths.tolist() == [0, 6, 17]
    assert plan.max_query_length == 11
    assert plan.num_scores == len(scores)
    assert plan.scores_in_suffix.numel() == 512
    assert plan.queries[plan.scores_in_suffix].tolist() == plan.scores_in_prompt.tolist()
    assert plan.scores_in_prompt[:plan.num_scores].tolist() == scores
    assert _decision_score_suffix(lengths, [10, 5562], [1, 1]) is None
