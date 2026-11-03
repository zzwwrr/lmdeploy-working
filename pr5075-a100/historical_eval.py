"""Reproducible PR 5075 accuracy and same-GPU latency checks; no gold in requests."""
import argparse
import ast
import asyncio
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path('/workspace')
REPO = ROOT / 'dev/lmdeploy-decision-parser'
REFERENCE = ROOT / 'decision-sandbox/Intern-Decision'
JEV = ROOT / 'decision-sandbox/third_party/jevbench-7ce310c7262ed49cc85853339a8a42459298e3f3'
sys.meta_path = [f for f in sys.meta_path if 'editable' not in repr(f).lower()]
sys.path[:0] = [str(REPO), str(REFERENCE), str(JEV)]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={REPO}',
                                    f'--git-dir={ROOT}/dev/lmdeploy/.git/worktrees/lmdeploy-decision-parser',
                                    f'--work-tree={REPO}', *args], text=True)


def latency_row(model):
    # Use the published checkpoint's quick-start request without adding padding.
    for code in re.findall(r'```python\n(.*?)```', (model / 'README.md').read_text(), re.S):
        for node in ast.parse(code).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'request' for t in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError('No published quick-start request found')


def stats(values):
    from jevbench.metrics import percentile
    return dict(n=len(values), mean_ms=statistics.mean(values), p50_ms=percentile(values, .5),
                p95_ms=percentile(values, .95), samples_ms=values)


def replay_metrics(records, model, temperature, output):
    from src.eval import jev
    suites = {}
    for name, path, expected in jev.suite_datasets():
        rows = [r for r in records if r['suite'] == name]
        assert len(rows) == expected

        class Replay:
            checkpoint = str(model)
            backend_name = 'lmdeploy-pytorch'

            def __init__(self):
                self.temperature = temperature
                self.rows = iter(rows)

            def predict(self, row):
                saved = next(self.rows)
                assert saved['id'] == row['id']
                response = json.loads(json.dumps(saved['response']))
                for answer in response['answers'].values():
                    if answer['type'] == 'noul':
                        answer['probabilities'] = {'yes': answer['noul'], 'no': 1 - answer['noul']}
                return response

        suites[name] = jev.evaluate(Replay(), path, output / 'scored' / f'{name}.jsonl')
    jev.verify_suite_counts(suites)
    fields = sum(s['total'] for s in suites.values())
    correct = sum(s['correct'] for s in suites.values())
    return dict(suites=suites, macro7_accuracy=statistics.mean(s['accuracy'] for s in suites.values()),
                fields=fields, correct=correct, weighted_accuracy=correct / fields,
                hard_brier=suites['jevbench-hard']['brier'], hard_ece=suites['jevbench-hard']['ece']['ece'])


async def run_lmdeploy(args, model):
    import torch
    from lmdeploy.archs import get_task
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.serve.openai.api_server import ServerContext
    from lmdeploy.serve.openai.decision_protocol import SystemoneRequest
    from lmdeploy.serve.openai.endpoints.decisions import _score_request
    from lmdeploy.serve.parsers.decision_parser import DecisionParser, create_decision_context
    from src.eval import jev

    cfg = PytorchEngineConfig(tp=1, dtype='bfloat16', max_batch_size=args.concurrency,
                             session_len=8192, max_prefill_token_num=8192,
                             cache_max_entry_count=.15, enable_prefix_caching=False,
                             eager_mode=False, piecewise_cudagraph_max_tokens=8192,
                             enable_mp_engine=True, distributed_executor_backend='uni', logprobs_mode='raw_logits')
    meta = dict(revision=git('rev-parse', 'HEAD').strip(), diff_sha256=hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
                script_sha256=sha(__file__), model=str(model), temperature=None, config=asdict(cfg),
                gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), argv=sys.argv,
                reference_revision=subprocess.check_output(['git', '-c', f'safe.directory={REFERENCE}', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip(),
                versions={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'flash-linear-attention', 'triton', 'tilelang']},
                model_files={p: sha(model / p) for p in ['config.json', 'inference.py', 'tokenizer_config.json']},
                sources={str(p.relative_to(REPO)): sha(p) for p in (REPO / 'lmdeploy').rglob('*.py')},
                datasets={n: sha(p) for n, p, _ in jev.suite_datasets()}, complete=False, records=0,
                scope='Production Systemone schema, parser and _score_request with real weights; no HTTP network. '
                      'Accuracy uses bounded pure-decision concurrency; latency uses serial requests, cache off.')
    write(args.output / 'metadata.json', meta)
    (args.output / 'source.patch').write_text(git('diff', 'HEAD'))
    engine = None
    records = []
    try:
        _, cls = get_task('pytorch', str(model), backend_config=cfg)
        engine = cls(str(model), backend='pytorch', backend_config=cfg, model_name=model.name)
        ctx = ServerContext()
        ctx.async_engine = engine
        ctx.decision_context = create_decision_context(engine, str(model))
        meta['temperature'] = ctx.decision_context.temperature

        async def score(row):
            request = SystemoneRequest.model_validate({k: row[k] for k in ('state', 'questions')})
            return await _score_request(ctx, DecisionParser(request, ctx.decision_context))

        row = latency_row(model)
        write(args.output / 'latency-request.json', row)
        for _ in range(args.warmup):
            await score(row)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = await score(row)
            times.append((time.perf_counter() - start) * 1000)
        latency = dict(**stats(times), warmup=args.warmup, input_tokens=result['usage']['input_tokens'],
                       question_count=len(row['questions']), response=result)
        write(args.output / 'latency-lmdeploy.json', latency)
        print(json.dumps(dict(event='latency', model=model.name, **{k:v for k,v in latency.items() if k not in ('samples_ms','response')})), flush=True)

        async def one(name, index, raw):
            row = jev._canonical_public(raw)
            assert not row.get('images'), 'This test suite is expected to be text-only'
            result = await score(row)
            return dict(suite=name, source_line=index + 1, id=raw['id'], response=result)

        start = time.perf_counter()
        with (args.output / 'responses.jsonl').open('x') as stream:
            for name, data_path, expected in jev.suite_datasets():
                rows = list(jev.read_rows(data_path))
                assert len(rows) == expected
                for offset in range(0, len(rows), args.concurrency):
                    current = await asyncio.gather(*(one(name, i, rows[i]) for i in range(offset, min(offset + args.concurrency, len(rows)))))
                    assert not ctx.session_manager.sessions, 'Sessions leaked after a completed batch'
                    for record in current:
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    meta['records'] = len(records)
                    if len(records) // 200 != (len(records) - len(current)) // 200:
                        stream.flush()
                        write(args.output / 'metadata.json', meta)
                        print(json.dumps(dict(event='progress', model=model.name, records=len(records), elapsed_s=time.perf_counter()-start)), flush=True)
        assert len(records) == 10751
        meta['inference_seconds'] = time.perf_counter() - start
        meta['inference_complete'] = True
    except BaseException as exc:
        meta['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if engine is not None:
            engine.close()
            meta['engine_closed'] = True
        write(args.output / 'metadata.json', meta)
    metrics = replay_metrics(records, model, meta['temperature'], args.output)
    write(args.output / 'metrics.json', metrics)
    meta['complete'] = True
    write(args.output / 'metadata.json', meta)
    print(json.dumps(dict(event='complete', model=model.name, **{k:v for k,v in metrics.items() if k!='suites'})), flush=True)


def run_hf(args, model):
    import torch
    from lmdeploy.intern_decision_encoding import resolve_temperature
    from src.inference.engine import DecisionEngine
    engine = DecisionEngine(str(model), backend='hf', device='cuda', dtype='bfloat16', attn_implementation='sdpa')
    engine.temperature = resolve_temperature(model)
    row = latency_row(model)
    for _ in range(args.warmup):
        engine.predict(row)
    times = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = engine.predict(row)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    data = dict(**stats(times), warmup=args.warmup, input_tokens=result['usage']['input_tokens'],
                gpu=torch.cuda.get_device_name(), temperature=engine.temperature, response=result,
                scope='Same-GPU HF SDPA reference, same published request; preprocessing and scoring included, no HTTP network.')
    write(args.output / 'latency-hf.json', data)
    print(json.dumps({k:v for k,v in data.items() if k not in ('samples_ms','response')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', required=True, choices=['0.8B', '2B', '4B'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--hf-latency', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = ROOT / 'decision-sandbox/models' / f'Intern-Decision-{args.size}'
    if args.hf_latency:
        run_hf(args, model)
    else:
        asyncio.run(run_lmdeploy(args, model))
