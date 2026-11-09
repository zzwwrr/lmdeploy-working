# PR 5075: A100 text quality at 7801fd7c

New full-weight measurements on 2026-11-09 use clean committed source
`7801fd7cdad7495633bf0c3d6be978eec32f0b1a`, including the indexed GDN gate,
Qwen3.5 zero-centered FP32 norm scales, and model-output event waiting fixes.
Uncommitted workspace edits are excluded. Each of the three sizes completed
10,751 requests / 12,351 fields without failures. Requests invoke the production
Systemone schema, parser and scoring handler directly; HTTP is excluded.
Gold labels are used only for scoring, never submitted to the inference handler.

One A100-SXM4-80GB, BF16 activations/weights with the source's FP32 norm
scales, TP1, concurrency/max batch 8, session/prefill 8192, prefix cache OFF,
piecewise CUDA graphs, cache_max_entry_count=0.15. The selected GPU had no compute processes, less than 1 GiB used and utilization
below 10% in two admission checks. GPU occupancy is monitored throughout the run;
new external compute processes invalidate the formal result. These runs report quality only. No latency, throughput or speedup is
claimed. PyTorch 2.10.0+cu128, Transformers 5.6.2, FLA 0.5.2, Triton 3.6.0,
TileLang 0.1.11. See each metadata.json and run.log.gz for effective configuration.

| Model | Official macro | LMDeploy macro | Difference (pp) | Correct fields / 12,351 |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 79.38% | 79.2854% | -0.0946 | 10,157 |
| 2B | 84.68% | 84.7906% | +0.1106 | 10,631 |
| 4B | 90.02% | 90.0667% | +0.0467 | 11,000 |

Per-suite accuracy (%):

| Model | Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailbreak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 97.9167 | 79.1667 | 53.1532 | 77.4000 | 94.5161 | 88.5921 | 64.2534 |
| 2B | 100.0000 | 84.7222 | 64.8649 | 79.1000 | 96.4516 | 89.9342 | 78.4615 |
| 4B | 100.0000 | 98.6111 | 73.8739 | 80.6500 | 96.7742 | 90.8289 | 89.7285 |

Jevbench-Hard calibration only (111 fields; lower is better):

| Model | Official Brier | LMDeploy Brier | Official ECE | LMDeploy ECE |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 0.529519 | 0.529201 | 0.065710 | 0.090511 |
| 2B | 0.437256 | 0.438570 | 0.100208 | 0.074415 |
| 4B | 0.346762 | 0.348156 | 0.065267 | 0.087249 |

Published official baselines: [fixed official README](https://github.com/InternLM/Intern-Decision/blob/565350244af860e1c276768cd80bdd9b8be93913/README.md#benchmark-accuracy).
These deltas compare published values and do not isolate backend error or prove
numerical equivalence. Historical A100 measurements remain available separately.

## Offline audit

```bash
python3 audit.py
```

The audit uses the bundled pinned official pure-Python scorer. It verifies all
published file hashes, 764 production-file hashes from the frozen source archive,
the before/after run fingerprints, raw-byte hashes, all 32,253 response identities,
37,053 decision fields, all 21 suite accuracies and all three Hard Brier/ECE pairs
against exported public gold labels. Gold-label export includes IDs, candidate
labels and canonical input hashes; full benchmark prompts and weights are excluded.

For an independent replay from the original public datasets, install the official
reference evaluation dependencies, obtain the original layout and run:

```bash
python3 replay.py --reference /path/to/Intern-Decision \
  --jevbench /path/to/jevbench --output /tmp/pr5075-a100-rescored
```

The reference repository must be at `3572c8a68b5df5dafe02d0e093989ba8ec0183bc`;
Jevbench is pinned to `7ce310c7262ed49cc85853339a8a42459298e3f3`. Replay verifies
the reference revision and seven dataset hashes before scoring saved responses.

## Reproduce inference

Extract source-7801fd7c.tar.gz into a fresh source directory and set EVAL_SOURCE to
its absolute path. evaluate.py expects the original container layout: models at
/workspace/decision-sandbox/models/Intern-Decision-{size}, official reference at
/workspace/decision-sandbox/Intern-Decision and the pinned Jevbench vendor directory
under /workspace/decision-sandbox/third_party. See environment.json and the three
command-*.json files for the exact launch commands. Output directories must be new.
run.py preserves the host/container paths used in the original experiment; adapt
those paths for a different machine and retain that adaptation with the new run.
model-reference-manifest.json records all checkpoint file hashes, including weights.
The original byte-for-byte production sources are also embedded in the archive.

This evidence does not establish image GPU accuracy, HTTP behavior, disconnect
cancellation, mixed generation/scoring safety, or performance on an idle GPU.
