"""Portable backend benchmark for TM legacy/SM90 and the current PyTorch tree.

Inputs are synthetic decision requests, not an accuracy evaluation corpus.
Each backend runs in its own process. Profile runs are separate from timing.
"""
import argparse
import ast
import asyncio
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def temperature(model):
    """Read the checkpoint constant without importing its remote code."""
    tree = ast.parse((Path(model) / 'inference.py').read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'DEFAULT_TEMPERATURE'
                      for target in node.targets)]
    if len(values) != 1:
        raise ValueError('Expected one checkpoint DEFAULT_TEMPERATURE constant')
    value = ast.literal_eval(values[0])
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid checkpoint temperature')
    return value


def summary(records):
    result = {}
    for tokens in sorted({row['tokens'] for row in records}):
        values = sorted(row['api_ms'] for row in records if row['tokens'] == tokens)
        result[str(tokens)] = dict(count=len(values), p50_ms=statistics.median(values),
                                  p95_ms=values[min(len(values) - 1, int(len(values) * .95))])
    return result


async def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = Path(args.tree).resolve()
    # Select exactly one source tree, including when another editable install exists.
    sys.meta_path = [finder for finder in sys.meta_path if 'editable' not in repr(finder).lower()]
    sys.path.insert(0, str(tree))
    os.environ['TM_GDR_FORCE_LEGACY'] = str(args.legacy)
    if args.profile and args.backend == 'pytorch':
        os.environ.update(LMDEPLOY_PROFILE_CPU='1', LMDEPLOY_PROFILE_CUDA='1',
                          LMDEPLOY_PROFILE_DELAY='0', LMDEPLOY_PROFILE_DURATION='-1',
                          LMDEPLOY_PROFILE_OUT_PREFIX=str(output.with_suffix('.worker')),
                          LMDEPLOY_PROFILE_USE_GZIP='1')
    import torch
    import lmdeploy
    from transformers import AutoTokenizer
    from lmdeploy.intern_decision_encoding import ANSWER_SYMBOLS, DECISION_TOKEN, compile_row, decode_answers
    from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, ResponseType, TurbomindEngineConfig

    assert Path(lmdeploy.__file__).resolve().is_relative_to(tree), lmdeploy.__file__
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1, 'Select one GPU with CUDA_VISIBLE_DEVICES'
    capability = torch.cuda.get_device_capability()
    if args.require_sm90 and capability != (9, 0):
        raise ValueError(f'Expected SM90, got {capability}')
    inputs = json.loads(Path(args.inputs).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    extra = sorted({tokenizer.convert_tokens_to_ids(symbol) for symbol in ANSWER_SYMBOLS})
    marker = tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
    calibrated_temperature = temperature(args.model)
    for row in inputs:
        compiled = compile_row(row['request'])
        text = tokenizer.apply_chat_template(compiled.messages, tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False, add_vision_id=True)
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        assert ids == row['ids'], 'Rendered tokens differ from the persisted input'
        assert [i for i, token in enumerate(ids) if token == marker] == row['markers']
        assert list(compiled.fields) == row['fields']
        assert [[tokenizer.convert_tokens_to_ids(symbol) for symbol in compiled.symbols[field]]
                for field in compiled.fields] == row['candidate_ids']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == row['input_ids_sha256']
    common = dict(tp=1, max_batch_size=args.concurrency, dtype='bfloat16', session_len=16384,
                  max_prefill_token_num=16384, cache_max_entry_count=.1, enable_prefix_caching=False)
    extension = None
    if args.backend == 'pytorch':
        from lmdeploy.pytorch.engine import Engine
        config = PytorchEngineConfig(**common, eager_mode=False, piecewise_cudagraph_max_tokens=8192,
                                    enable_mp_engine=True, distributed_executor_backend='uni',
                                    logprobs_mode='raw_logits')
    else:
        from lmdeploy.turbomind import TurboMind, _tm
        extension = Path(_tm.__file__).resolve()
        assert extension.is_relative_to(tree), extension
        config = TurbomindEngineConfig(**common)
    git = ['git', '-c', f'safe.directory={tree}', '-C', str(tree)]
    if args.git_dir:
        git += [f'--git-dir={Path(args.git_dir).resolve()}', f'--work-tree={tree}']
    result = dict(args=vars(args), commit=subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output([*git, 'diff', 'HEAD'])).hexdigest(),
                  source=lmdeploy.__file__, extension=str(extension) if extension else None,
                  tm_gdr_force_legacy=os.environ['TM_GDR_FORCE_LEGACY'],
                  tm_gdr_cp_level=os.environ.get('TM_GDR_CP_LEVEL', '2'),
                  extension_sha256=sha(extension) if extension else None, harness_sha256=sha(__file__),
                  inputs_sha256=sha(args.inputs), config=asdict(config), temperature=calibrated_temperature,
                  gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  capability=capability, torch=torch.__version__,
                  cuda=torch.version.cuda, complete=False, warmups=[], records=[], rounds=[],
                  gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
                                                          '--format=csv,noheader'], text=True),
                  timing_scope='API latency: async_stream_infer through FINISH, no forced GPU drain. '
                               'Round wall includes result extraction and driver scheduling. '
                               'Both exclude initialization, tokenization, HTTP and warmup. '
                               'Profile runs must not be used for performance comparisons.',
                  scoring_scope='PT: zero generation and candidate columns. TM: suffix full-vocabulary '
                                'logits and generation budget 1. Synthetic four-input benchmark; '
                                'not full-corpus accuracy or HTTP throughput.')

    def save():
        result['summary'] = summary(result['records'])
        wall = sum(item['wall_seconds'] for item in result['rounds'])
        result['driver_req_per_s'] = len(result['records']) / wall if wall else None
        temporary = output.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(result, indent=2, default=str) + '\n')
        temporary.replace(output)

    engine = None
    try:
        if args.backend == 'pytorch':
            engine = Engine.from_pretrained(args.model, engine_config=config, trust_remote_code=True)
            ready = engine.start_loop()
            if inspect.isawaitable(ready):
                await ready
        else:
            engine = TurboMind(args.model, engine_config=config)
        instances = [engine.create_instance() for _ in range(args.concurrency)]
        session = 0

        async def execute(instance, row, iteration, case_index):
            nonlocal session
            session += 1
            session_id = session
            start = row['markers'][0] - 1
            count = len(row['ids']) - start - 1
            gen = (GenerationConfig(max_new_tokens=0, logprobs=0, logprob_start_len=start,
                                    logprob_extra_token_ids=extra) if args.backend == 'pytorch' else
                   GenerationConfig(max_new_tokens=1, logprobs=0, logprob_start_len=start,
                                    output_logits='all', top_k=1, ignore_eos=True))
            last = None
            begun = time.perf_counter()
            async for chunk in instance.async_stream_infer(session_id=session_id, input_ids=row['ids'].copy(),
                                                           gen_config=gen, stream_output=False):
                last = chunk
            elapsed = (time.perf_counter() - begun) * 1000
            assert last is not None and last.status == ResponseType.FINISH, last
            offsets = [position - 1 - start for position in row['markers']]
            if args.backend == 'pytorch':
                assert last.logprobs is not None and len(last.logprobs) == count
                scores = [[float(last.logprobs[i][token]) for token in candidates]
                          for i, candidates in zip(offsets, row['candidate_ids'])]
            else:
                assert last.logits is not None and last.logits.shape[0] == count
                scores = [[float(last.logits[i, token]) for token in candidates]
                          for i, candidates in zip(offsets, row['candidate_ids'])]
            answers = decode_answers(row['request']['questions'], row['fields'], scores,
                                     temperature=calibrated_temperature)
            return dict(iteration=iteration, case_index=case_index, tokens=len(row['ids']), api_ms=elapsed,
                        input_ids_sha256=row['input_ids_sha256'], candidate_scores=scores, answers=answers)

        rng = random.Random(20261030)

        async def round_run(iterations):
            jobs = [(iteration, index, row) for iteration in iterations for index, row in enumerate(inputs)]
            rng.shuffle(jobs)
            pending = iter(jobs)

            async def worker(instance):
                rows = []
                for iteration, index, row in pending:
                    rows.append(await execute(instance, row, iteration, index))
                return rows

            begun = time.perf_counter()
            rows = await asyncio.gather(*(worker(instance) for instance in instances))
            wall = time.perf_counter() - begun
            return [row for batch in rows for row in batch], wall

        result['warmups'], _ = await round_run(range(-args.warmup, 0))
        save()
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        context = torch.profiler.profile(activities=activities) if args.profile and args.backend == 'turbomind' \
            else nullcontext()
        with context as profiler:
            result['records'], wall = await round_run(range(args.repeat))
            result['rounds'].append(dict(wall_seconds=wall, requests=len(result['records'])))
            if args.profile and args.backend == 'turbomind':
                torch.cuda.synchronize()  # Outside both the per-request and round timing windows.
        if args.profile and args.backend == 'turbomind':
            trace_path = output.with_suffix('.trace.json')
            profiler.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
            names = sorted({event['name'] for event in trace['traceEvents'] if event.get('cat') == 'kernel'})
            result['gdr_cuda_kernels'] = [name for name in names if any(
                term in name for term in ('Gdr', 'GDR', 'KktSolve', 'CorrectInitialStates'))]
            assert result['gdr_cuda_kernels'], 'No GDR CUDA kernels captured; inspect the profiler setup'
            if args.require_sm90:
                expected = 'ChunkedGdrKernel' if args.legacy else 'Sm90FusedGdrFwdKernel'
                assert any(expected in name for name in names), f'Expected {expected}; inspect {trace_path}'
                if args.legacy:
                    assert not any('Sm90FusedGdrFwdKernel' in name for name in names)
            result['trace'] = str(trace_path)
        result['complete'] = True
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if engine is not None:
            engine.close()
            result['engine_closed'] = True
        save()
    print(json.dumps(dict(output=str(output), summary=result['summary'],
                          driver_req_per_s=result['driver_req_per_s']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['turbomind', 'pytorch'], required=True)
    parser.add_argument('--tree', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--git-dir', help='Metadata directory for a relocated linked worktree')
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', default=str(Path(__file__).with_name('data') / 'gdr_inputs.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--legacy', type=int, choices=[0, 1], default=0)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5, help='Warmup corpus passes')
    parser.add_argument('--repeat', type=int, default=20, help='Measured corpus passes')
    parser.add_argument('--require-sm90', action='store_true')
    parser.add_argument('--profile', action='store_true')
    arguments = parser.parse_args()
    if arguments.concurrency < 1 or arguments.repeat < 1 or arguments.warmup < 0:
        parser.error('concurrency/repeat must be positive; warmup must be nonnegative')
    if arguments.backend == 'pytorch' and arguments.legacy:
        parser.error('--legacy only applies to TurboMind')
    asyncio.run(run(arguments))
