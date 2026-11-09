"""Audit all raw responses, gold accuracy, Hard calibration and frozen sources."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference'))
from jevbench.metrics import brier_score, ece_top_label
from jevbench.scoring import score_task
from types import SimpleNamespace


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, expected in manifest.items():
        assert digest((ROOT / name).read_bytes()) == expected, name
    admission = json.loads((ROOT / 'idle-admission.json').read_text())
    for key in ('first', 'second'):
        device = admission[key]
        assert not device['pids'] and device['memory_mib'] < 1024 and device['utilization'] < 10
    monitor = [json.loads(line) for line in (ROOT / 'gpu-monitor.jsonl').read_text().splitlines()]
    assert monitor and all(not row['external_pids'] for row in monitor)
    source = json.loads((ROOT / 'source-manifest.json').read_text())
    with tarfile.open(ROOT / 'source-7801fd7c.tar.gz') as tar:
        for name, expected in source['files'].items():
            assert digest(tar.extractfile(name).read()) == expected, name
    with gzip.open(ROOT / 'gold-labels.jsonl.gz', 'rt') as stream:
        gold = list(map(json.loads, stream))
    assert len(gold) == 10751 and sum(len(r['fields']) for r in gold) == 12351
    frozen_models = json.loads((ROOT / 'model-reference-manifest.json').read_text())
    for name in ('__init__.py', 'metrics.py', 'scoring.py'):
        assert digest((ROOT / 'reference/jevbench' / name).read_bytes()) == frozen_models['jevbench_files']['jevbench/' + name]
    results = {}
    for size in ('0.8B', '2B', '4B'):
        path = ROOT / 'results' / size
        meta = json.loads((path / 'metadata.json').read_text())
        saved = json.loads((path / 'metrics.json').read_text())
        assert meta['complete'] and meta['engine_closed'] and not meta.get('error')
        assert meta['revision'] == source['revision']
        assert meta['reference_revision'] == frozen_models['reference_revision']
        for name, value in meta['model_files'].items():
            assert value == frozen_models['models'][size]['files'][name]
        assert meta['datasets'] == {name: values['data_sha256'] for name, values in saved['suites'].items()}
        assert meta['sources'] == meta['sources_after'] == source['files']
        assert meta['script_sha256'] == digest((ROOT / 'evaluate.py').read_bytes())
        raw = gzip.decompress((path / 'responses.jsonl.gz').read_bytes())
        assert digest(raw) == json.loads((path / 'raw-sha256.json').read_text())['responses.jsonl']
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == meta['records'] == 10751
        assert len({(r['suite'], r['source_line'], r['id']) for r in rows}) == 10751
        totals = defaultdict(lambda: dict(correct=0, fields=0))
        hard_briers, hard_pairs = [], []
        for row, target in zip(rows, gold, strict=True):
            for key in ('suite', 'source_line', 'id'):
                assert row[key] == target[key]
            response = row['response']
            assert response['model'] == 'Intern-Decision-' + size
            assert set(response['answers']) == set(target['fields'])
            for field, task in target['fields'].items():
                answer = response['answers'][field]
                probabilities = answer.get('probabilities')
                if answer['type'] == 'noul':
                    probabilities = {'yes': answer['noul'], 'no': 1 - answer['noul']}
                result = score_task(probabilities, SimpleNamespace(
                    labels=task['labels'], expected=task['expected'], question={'type': task['type']}))
                assert result['valid'] and result['strict_valid'] and not result['renormalized']
                correct = result['correct']
                totals[row['suite']]['fields'] += 1
                totals[row['suite']]['correct'] += int(correct)
                if row['suite'] == 'jevbench-hard':
                    hard_briers.append(brier_score(result['probs'], task['expected'], task['labels']))
                    hard_pairs.append((max(result['probs'].values()), correct))
        for suite, total in totals.items():
            assert total['correct'] == saved['suites'][suite]['correct']
            assert total['fields'] == saved['suites'][suite]['total']
            assert total['correct'] / total['fields'] == saved['suites'][suite]['accuracy']
        macro = statistics.mean(t['correct'] / t['fields'] for t in totals.values())
        brier = sum(hard_briers) / len(hard_briers)
        ece = ece_top_label(hard_pairs)['ece']
        assert len(hard_briers) == 111
        assert math.isclose(macro, saved['macro7_accuracy'], abs_tol=1e-14)
        assert math.isclose(brier, saved['hard_brier'], abs_tol=1e-14)
        assert math.isclose(ece, saved['hard_ece'], abs_tol=1e-14)
        results[size] = dict(requests=len(rows), fields=sum(t['fields'] for t in totals.values()),
                             correct=sum(t['correct'] for t in totals.values()), macro7_accuracy=macro,
                             hard_brier=brier, hard_ece=ece, source_files=len(source['files']))
    print(json.dumps(dict(passed=True, verified_files=len(manifest), gpu_monitor_samples=len(monitor), results=results), indent=2))


if __name__ == '__main__':
    main()
