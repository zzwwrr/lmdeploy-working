# PR 5075: historical A100 text quality evidence

This evidence branch preserves the evaluated LMDeploy source at commit
`cda4c3b5`. It is separate from the implementation PR to keep raw benchmark artifacts
out of the upstream source diff. Results are from 2026-11-03 and are not a rerun of
the latest PR head. No speedup or latency comparison is made here.

## Scope and environment

All three sizes completed 10,751 text requests / 12,351 decision fields each,
with zero failed requests. The harness calls the production Systemone schema,
parser and scoring handler with real weights; it does not traverse HTTP or the
concurrency middleware. Gold labels are used only by the official scoring pass,
never in inference requests.

One A100-SXM4-80GB, sequential model runs, BF16, TP1, cache OFF, concurrency 8,
max batch 8, session/prefill limits 8192 and piecewise CUDA graphs. Temperatures:
0.8B = 2.747760550703; 2B = 2.100509348278; 4B = 1.99241824.
PyTorch 2.10.0+cu128, Transformers 5.6.2, flash-linear-attention 0.5.2,
Triton 3.6.0 and TileLang 0.1.11. Full engine configuration is in metadata.json.

The original metadata records HEAD `3df76b5b` plus an uncommitted patch. That patch
was subsequently committed as `cda4c3b5`. All 753 production Python file hashes
were checked against that commit, and are identical across the three runs.
The branch includes those sources; audit.py checks them again.
Model config/tokenizer metadata hashes, dataset hashes and official scorer revision
are retained in each model's original metadata.json. No weights or benchmark input
texts are redistributed here. The metadata does not provide a complete weight-file
manifest, so it should not be treated as proof of every weight file's identity.

## Results

Accuracy below is the unweighted mean of seven suites. Brier/ECE below are Jevbench-Hard only (111 fields), calibrated with the checkpoint temperature; they are not the 96-case distribution pilot. Official averages are published to two decimal places.

| Model | Official average | LMDeploy average | Difference (pp) | Official Hard Brier / ECE | LMDeploy Hard Brier / ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 79.38% | 79.1617% | -0.2183 | 0.529519 / 0.065710 | 0.529083 / 0.091346 |
| 2B | 84.68% | 84.9129% | +0.2329 | 0.437256 / 0.100208 | 0.437578 / 0.071510 |
| 4B | 90.02% | 89.8408% | -0.1792 | 0.346762 / 0.065267 | 0.347182 / 0.081564 |

Official values: https://github.com/InternLM/Intern-Decision/blob/565350244af860e1c276768cd80bdd9b8be93913/README.md#benchmark-accuracy (README blob c7c2535c29830cb69495c8e66a598303ac2cdcf7). Differences are comparisons with published values, not a controlled attribution of backend errors.

Per-suite LMDeploy accuracy (%):

| Model | Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailbreak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 97.9167 | 79.1667 | 53.1532 | 77.4500 | 94.5161 | 88.6711 | 63.2579 |
| 2B | 100.0000 | 84.7222 | 65.7658 | 79.3000 | 96.4516 | 89.9605 | 78.1900 |
| 4B | 100.0000 | 98.6111 | 72.9730 | 80.3000 | 96.4516 | 90.7763 | 89.7738 |

The observed average-accuracy differences are within 0.24 percentage points, but calibration differences remain. These measurements support reporting the numbers above, not a claim of full numerical or API equivalence. This run is text-only; it does not establish full-image GPU accuracy or mixed generation/scoring safety. 


## Download and audit (standard-library Python only)

```bash
git clone --branch evidence/pr5075-a100-20261103 --single-branch \
  https://github.com/zzwwrr/lmdeploy-working.git pr5075-evidence
cd pr5075-evidence
python3 pr5075-a100/audit.py
```

Each model directory contains the original metadata.json, metrics.json,
source.patch and a losslessly compressed responses.jsonl.gz (all 10,751 raw
responses, including candidate probabilities). summary.json gives the comparison
and original uncompressed artifact hashes. manifest.json covers published files.
The audit validates integrity, coverage, source identity and metric aggregation;
it does not itself compare answers with gold labels.

## Recompute scores without inference

Use the reference scorer revision recorded by the run and the original Jevbench
revision. Install the reference evaluation dependencies in an isolated environment;
the recorded dependency versions above describe the evaluated inference environment.

```bash
git clone https://github.com/InternLM/Intern-Decision.git /tmp/pr5075-reference
git -C /tmp/pr5075-reference checkout 3572c8a68b5df5dafe02d0e093989ba8ec0183bc
# Obtain Jevbench revision 7ce310c7262ed49cc85853339a8a42459298e3f3
# following the reference repository's evaluation setup instructions.
python3 pr5075-a100/replay.py \
  --reference /tmp/pr5075-reference \
  --jevbench /path/to/jevbench \
  --output /tmp/pr5075-rescored
```

The output directory must be new. replay.py verifies the reference revision and
all dataset hashes, then replays saved responses through the official evaluator.
It checks all seven suite accuracies and Hard Brier/ECE against published artifacts.
No model weights or GPU are required for replay. Publication validation reran this replay for all three sizes: all 21 suite accuracies and all three Hard Brier/ECE pairs matched the saved results.

## Repeat inference in the original layout

historical_eval.py and historical_run.sh are the original harness and launch script,
not a portable benchmark API. They retain the original paths and include a separate
latency exercise; those timings are excluded from this quality report.
For an exact harness rerun, use this evidence branch's source at
`/workspace/dev/lmdeploy-decision-parser`, reference revision above at
`/workspace/decision-sandbox/Intern-Decision`, Jevbench at
`/workspace/decision-sandbox/third_party/jevbench-7ce310c7262ed49cc85853339a8a42459298e3f3`,
and checkpoints at `/workspace/decision-sandbox/models/Intern-Decision-{size}`.
The historical git helper also assumes the original main-repository/worktree layout;
adapt it to `git -C REPO ...` if using a standalone clone. Preserve these changes in
your new run's patch and do not describe adapted reruns as the original measurement.

```bash
cp pr5075-a100/historical_eval.py scratch/pr5075_live_eval.py
cp pr5075-a100/historical_run.sh scratch/run_pr5075_live_eval.sh
# In the original container, with GPU 4 available and recorded dependencies installed:
bash scratch/run_pr5075_live_eval.sh scratch/results/pr5075-new-run
```

The launcher rejects an existing output directory. It selects GPU 4 and the original
isolated dependency directories; review these paths before using a different host.
The preserved raw responses plus official scorer replay are the CPU-only route to
verify the published quality numbers without recreating that machine layout.

## Limits

Text only, pure decision batches, no HTTP transport. This is not image-GPU validation,
client-disconnect validation, mixed generation/scoring validation, or full API/numerical
equivalence. The official published baselines use their documented reference setup;
these deltas do not isolate backend error. Quality and performance remain separate.
