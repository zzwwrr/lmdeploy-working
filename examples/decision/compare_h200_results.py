"""Compare portable H200 benchmark results without requiring CUDA or Torch."""
import argparse
import json
from pathlib import Path


def compare(reference, other):
    for data in (reference, other):
        assert data['complete'], data.get('error')
        assert not data['args']['profile'], 'Profile runs are not performance measurements'
    assert reference['inputs_sha256'] == other['inputs_sha256']
    assert reference['args']['concurrency'] == other['args']['concurrency']
    for name in ('tp', 'max_batch_size', 'dtype', 'session_len', 'max_prefill_token_num',
                 'cache_max_entry_count', 'enable_prefix_caching'):
        assert reference['config'][name] == other['config'][name], name
    if reference['args']['backend'] == other['args']['backend']:
        assert reference['config'] == other['config']
    assert reference['temperature'] == other['temperature']
    assert reference['args']['model'] == other['args']['model']
    assert reference['gpu_uuid'] == other['gpu_uuid'], 'Use the same physical GPU for both runs'
    left = {(row['iteration'], row['case_index']): row for row in reference['records']}
    right = {(row['iteration'], row['case_index']): row for row in other['records']}
    assert len(left) == len(reference['records']) and len(right) == len(other['records'])
    assert left.keys() == right.keys()
    fields = flips = 0
    examples = []
    maximum = 0.0
    for key, row in left.items():
        candidate = right[key]
        assert row['input_ids_sha256'] == candidate['input_ids_sha256']
        assert row['answers'].keys() == candidate['answers'].keys()
        assert len(row['candidate_scores']) == len(candidate['candidate_scores'])
        for a, b in zip(row['candidate_scores'], candidate['candidate_scores']):
            assert len(a) == len(b)
            maximum = max(maximum, max(abs(x - y) for x, y in zip(a, b)))
        for name, answer in row['answers'].items():
            fields += 1
            if answer['decision'] != candidate['answers'][name]['decision']:
                flips += 1
                if len(examples) < 20:
                    examples.append(dict(iteration=key[0], case_index=key[1], field=name,
                                         reference=answer, other=candidate['answers'][name]))
    return dict(requests=len(left), compared_fields=fields, decision_flips=flips,
                flip_examples=examples,
                maximum_candidate_logit_difference=maximum,
                per_length={tokens: dict(reference_ms=value['p50_ms'], other_ms=other['summary'][tokens]['p50_ms'],
                                         latency_reduction_percent=100 * (
                                             1 - other['summary'][tokens]['p50_ms'] / value['p50_ms']))
                            for tokens, value in reference['summary'].items()},
                reference_driver_req_per_s=reference['driver_req_per_s'],
                other_driver_req_per_s=other['driver_req_per_s'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('other')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = compare(json.loads(Path(args.reference).read_text()), json.loads(Path(args.other).read_text()))
    result.update(reference=args.reference, other=args.other)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
