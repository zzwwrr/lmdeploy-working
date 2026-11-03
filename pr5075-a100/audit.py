"""Audit the published quality artifacts without loading weights or using a GPU."""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha((ROOT / name).read_bytes()) == digest, name
    summary = json.loads((ROOT / 'summary.json').read_text())
    fingerprints = []
    for size, result in summary['results'].items():
        path = ROOT / size
        meta = json.loads((path / 'metadata.json').read_text())
        metrics = json.loads((path / 'metrics.json').read_text())
        raw = gzip.decompress((path / 'responses.jsonl.gz').read_bytes())
        assert sha(raw) == result['evidence_sha256']['responses.jsonl']
        for name in ('metadata.json', 'metrics.json', 'source.patch'):
            assert sha((path / name).read_bytes()) == result['evidence_sha256'][name]
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == meta['records'] == 10751
        assert len({(r['suite'], r['source_line']) for r in rows}) == len(rows)
        assert meta['complete'] and meta['engine_closed'] and not meta.get('error')
        assert sha((ROOT / 'historical_eval.py').read_bytes()) == meta['script_sha256']
        counts = Counter(r['suite'] for r in rows)
        assert len(counts) == len(metrics['suites']) == 7
        assert sum(len(r['response']['answers']) for r in rows) == metrics['fields'] == 12351
        assert sum(s['correct'] for s in metrics['suites'].values()) == metrics['correct']
        assert sum(s['total'] for s in metrics['suites'].values()) == metrics['fields']
        macro = sum(s['accuracy'] for s in metrics['suites'].values()) / 7
        assert math.isclose(macro, metrics['macro7_accuracy'])
        assert math.isclose(macro * 100, result['actual']['macro7_percent'])
        assert metrics['hard_brier'] == result['actual']['hard_brier']
        assert metrics['hard_ece'] == result['actual']['hard_ece']
        for rel, digest in meta['sources'].items():
            assert sha((ROOT.parent / rel).read_bytes()) == digest, rel
        fingerprints.append(meta['sources'])
        print(size, 'PASS', len(rows), 'requests;', metrics['fields'], 'fields;', dict(counts))
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    print('Artifact hashes, coverage, aggregate metrics and historical production source hashes verified.')
    print('This audit does not recompute accuracy against gold labels; use replay.py for official rescoring.')


if __name__ == '__main__':
    main()
