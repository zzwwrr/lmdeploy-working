"""Re-score saved responses with the pinned official scorer; no weights or GPU needed."""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--jevbench', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    meta = json.loads((root / '0.8B/metadata.json').read_text())
    revision = subprocess.check_output(
        ['git', '-C', str(args.reference), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == meta['reference_revision'], 'Use the recorded official scorer revision'
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.reference.resolve()), str(args.jevbench.resolve())]
    from src.eval import jev

    for size in ('0.8B', '2B', '4B'):
        path = root / size
        meta = json.loads((path / 'metadata.json').read_text())
        expected_metrics = json.loads((path / 'metrics.json').read_text())
        with gzip.open(path / 'responses.jsonl.gz', 'rt') as stream:
            records = [json.loads(line) for line in stream]
        suites = {}
        for name, data, expected_count in jev.suite_datasets():
            assert jev.sha256(data) == meta['datasets'][name], name
            rows = [r for r in records if r['suite'] == name]
            assert len(rows) == expected_count

            class Replay:
                checkpoint = meta['model']
                backend_name = 'lmdeploy-pytorch'
                temperature = meta['temperature']

                def __init__(self):
                    self.rows = iter(rows)

                def predict(self, row):
                    saved = next(self.rows)
                    assert saved['id'] == row['id']
                    response = json.loads(json.dumps(saved['response']))
                    for answer in response['answers'].values():
                        if answer['type'] == 'noul':
                            answer['probabilities'] = {'yes': answer['noul'], 'no': 1 - answer['noul']}
                    return response

            result = jev.evaluate(Replay(), data, args.output / size / f'{name}.jsonl')
            for key in ('correct', 'total', 'accuracy'):
                assert result[key] == expected_metrics['suites'][name][key], (size, name, key)
            suites[name] = result
        jev.verify_suite_counts(suites)
        hard = suites['jevbench-hard']
        assert math.isclose(hard['brier'], expected_metrics['hard_brier'], abs_tol=1e-12)
        assert math.isclose(hard['ece']['ece'], expected_metrics['hard_ece'], abs_tol=1e-12)
        (args.output / size / 'metrics.json').write_text(json.dumps(suites, indent=2) + '\n')
        print(size, 'official scorer replay PASS')


if __name__ == '__main__':
    main()
