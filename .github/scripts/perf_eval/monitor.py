#!/usr/bin/env python3
"""Run a command and sample the resource usage of its whole process tree.

Writes a JSON summary (wall time, peak total RSS, peak process counts, the largest single process seen) and an optional
per-second CSV time series. Requires psutil. The exit code of the wrapped command is propagated.
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
import time

import psutil

HEAVY = ('cicc', 'ptxas', 'cc1plus', 'nvcc', 'fatbinary', 'ld', 'ld.bfd', 'ld.gold', 'lld')


def sample(root):
    procs = [root]
    try:
        procs += root.children(recursive=True)
    except psutil.Error:
        pass
    total = 0
    counts = {}
    biggest = (0, '')
    for p in procs:
        try:
            rss = p.memory_info().rss
            name = p.name()
            total += rss
            if name in HEAVY:
                counts[name] = counts.get(name, 0) + 1
            if rss > biggest[0]:
                cmd = ' '.join(p.cmdline())
                biggest = (rss, f'{name}: {cmd[-200:]}')
        except psutil.Error:
            continue
    return total, counts, biggest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--series', default='')
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd
    if not cmd:
        ap.error('missing command')

    t0 = time.time()
    child = subprocess.Popen(cmd)
    root = psutil.Process(child.pid)
    done = threading.Event()
    state = {'peak_rss': 0, 'peak_procs': 0, 'peak_heavy': 0, 'biggest': (0, ''), 'rows': []}

    def loop():
        while not done.is_set():
            total, counts, biggest = sample(root)
            n_heavy = sum(counts.get(k, 0) for k in ('cicc', 'ptxas', 'cc1plus'))
            state['peak_rss'] = max(state['peak_rss'], total)
            state['peak_heavy'] = max(state['peak_heavy'], n_heavy)
            if biggest[0] > state['biggest'][0]:
                state['biggest'] = biggest
            state['rows'].append((round(time.time() - t0, 1), total, n_heavy))
            done.wait(args.interval)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    rc = child.wait()
    done.set()
    th.join()
    wall = time.time() - t0

    gib = 1024**3
    summary = {
        'exit_code': rc,
        'wall_s': round(wall, 1),
        'peak_tree_rss_gib': round(state['peak_rss'] / gib, 2),
        'peak_concurrent_compile_procs': state['peak_heavy'],
        'largest_single_process_gib': round(state['biggest'][0] / gib, 2),
        'largest_single_process': state['biggest'][1],
        'samples': len(state['rows']),
    }
    with open(args.summary, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    if args.series:
        with open(args.series, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['t_s', 'tree_rss_bytes', 'compile_procs'])
            w.writerows(state['rows'])
    return rc


if __name__ == '__main__':
    sys.exit(main())
