"""Run three official-suite quality evaluations from the frozen PR head."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CONTAINER_ROOT = Path('/workspace') / ROOT.relative_to('/nvme/zhuwenrui/qwen35-lmdeploy')
GPU = os.environ['EVAL_GPU']


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)
    (ROOT / 'gpu-before.csv').write_text(devices)
    env = dict(CUDA_VISIBLE_DEVICES=GPU, CUDA_HOME='/usr/local/cuda', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               EVAL_SOURCE=str(CONTAINER_ROOT / 'source'), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=':'.join((str(CONTAINER_ROOT / 'source'), '/tmp/lmdeploy-rebase-main-20261030-site',
                                    '/workspace/dev/cache/uv/archive-v0/H54tyfQwHbSJOSGYJch_7')),
               LD_LIBRARY_PATH='/tmp/lmdeploy-rebase-main-20261030-site/z3/lib:/usr/local/cuda/lib64')
    write('environment.json', env)
    for size in ('0.8B', '2B', '4B'):
        result = ROOT / 'results' / size
        result.mkdir(parents=True, exist_ok=False)
        command = ['docker', 'exec']
        for key, value in env.items():
            command += ['-e', f'{key}={value}']
        command += ['-w', str(CONTAINER_ROOT / 'source'), 'zwr-qwen35-lmdeploy', '/opt/py3/bin/python',
                    str(CONTAINER_ROOT / 'evaluate.py'), '--size', size, '--output',
                    str(CONTAINER_ROOT / 'results' / size), '--concurrency', '8']
        write(f'command-{size}.json', command)
        write('status.json', dict(phase='running', size=size, pid=os.getpid(), gpu=GPU, started_at=time.time()))
        with (result / 'run.log').open('x') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        (result / 'exit_code').write_text(f'{code}\n')
        if code:
            write('status.json', dict(phase='failed', size=size, exit_code=code))
            return code
    write('status.json', dict(phase='complete', sizes=['0.8B', '2B', '4B'], completed_at=time.time()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
