#!/usr/bin/env python3
"""Analyze a ninja .ninja_log to attribute build wall time to files/targets.

ninja log v5/v7 row: start_ms  end_ms  mtime_ns  output  hash Times are milliseconds since ninja start, so they are
directly comparable across edges and let us reconstruct the critical path.
"""
import argparse
import collections
import sys

BUCKETS = (
    ('gemm kernels', 'kernels/gemm/'),
    ('attention kernels', 'kernels/attention/'),
    ('linear_attn kernels', 'kernels/linear_attn/'),
    ('comm (nccl/ipc/deepep)', 'comm/'),
    ('model impls', 'models/'),
    ('other kernels', 'kernels/'),
    ('python bindings', 'python/'),
)


def bucket_of(path):
    for name, needle in BUCKETS:
        if name == 'other kernels' and 'kernels/' in path and any(
                n in path for _, n in BUCKETS[:3]):
            continue
        if needle in path:
            return name
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--only', default='', help='substring filter for top list')
    args = ap.parse_args()

    rows = []
    with open(args.log, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            rows.append((start, end, parts[3]))

    if not rows:
        print('no parseable rows', file=sys.stderr)
        return 1

    total_start = min(r[0] for r in rows)
    total_end = max(r[1] for r in rows)
    span = (total_end - total_start) / 1000.0

    print(f'edges            : {len(rows)}')
    print(f'span             : {span:.1f} s  ({span/60:.1f} min)')
    print(f'cpu-seconds      : {sum(r[1]-r[0] for r in rows)/1000.0:.1f} s')
    avg_par = (sum(r[1] - r[0] for r in rows) / 1000.0) / span if span else 0
    print(f'avg parallelism  : {avg_par:.2f}x')

    # occupancy: how many edges were running at each second
    events = []
    for s, e, _ in rows:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    print(f'peak concurrency : {peak}')

    print('\n--- wall time by bucket (self time, i.e. summed edge duration) ---')
    per = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    for s, e, out in rows:
        b = bucket_of(out)
        per[b] += (e - s) / 1000.0
        cnt[b] += 1
    for name in sorted(per, key=lambda k: -per[k]):
        share = per[name] / sum(per.values()) * 100
        print(f'{name:<24} {per[name]:9.1f} s  {share:5.1f}%  ({cnt[name]} edges)')

    top = sorted(rows, key=lambda r: -(r[1] - r[0]))[:args.top]
    print(f'\n--- top {args.top} slowest edges ---')
    for s, e, out in top:
        print(f'{(e-s)/1000.0:8.1f} s  @{s/1000.0:7.1f}s  {out}')

    if args.only:
        filt = [r for r in rows if args.only in r[2]]
        print(f'\n--- "{args.only}" : {len(filt)} edges, '
              f'{sum(r[1]-r[0] for r in filt)/1000.0:.1f} s self ---')
        return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
