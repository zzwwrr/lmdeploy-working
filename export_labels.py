"""Export public-suite IDs and gold labels for a GPU-free independent audit."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

BASE = Path('/workspace/decision-sandbox')
sys.path[:0] = [str(BASE / 'Intern-Decision'), str(BASE / 'third_party/jevbench-7ce310c7262ed49cc85853339a8a42459298e3f3')]
from src.eval import jev
from src.inputs.schema import _answer_value, _options
from jevbench.tasks import Task

root = Path(__file__).resolve().parent
with gzip.open(root / 'gold-labels.jsonl.gz', 'wt') as stream:
    for suite, path, count in jev.suite_datasets():
        rows = list(jev.read_rows(path))
        assert len(rows) == count
        for index, raw in enumerate(rows):
            row = jev._canonical_public(raw)
            fields = {}
            for field, question in row['questions'].items():
                if 'question' in raw:
                    task = Task.from_dict(raw)
                    expected, labels = task.expected, task.labels
                else:
                    expected = _answer_value(question, row.get('targets', {}).get(field))
                    labels = [v for v, _ in _options(question)]
                assert expected is not None and str(expected) in labels
                fields[field] = dict(expected=str(expected), labels=labels, type=question['type'])
            payload = {k: row[k] for k in ('state', 'questions')}
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            stream.write(json.dumps(dict(suite=suite, source_line=index+1, id=raw['id'], fields=fields, input_sha256=digest), ensure_ascii=False)+'\n')
print('Exported gold labels for 10,751 public requests; no inference performed.')
