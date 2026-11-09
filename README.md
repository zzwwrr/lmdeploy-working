# PR 5075: H200 HTTP capacity and quality at 7801fd7c

New 2026-11-09 measurements use clean source `7801fd7cdad7495633bf0c3d6be978eec32f0b1a`, including the indexed GDN
first-gate fix, Qwen3.5 zero-centered FP32 norm scales and model-output event wait.
This updates the previous H200 measurements at `4446d416`.

## Download and audit

- [Full compact review bundle](pr5075-h200-current-7801fd7c-summary.tar.gz): 17,315,546 bytes.
- [SHA256 checksum](pr5075-h200-current-7801fd7c-summary.tar.gz.sha256).
- SHA256: `d725c1c065544ad56573b8fd36b4d97629e22ddbbdcbe1de36bad03e3fc5ff6e`.

```bash
sha256sum -c pr5075-h200-current-7801fd7c-summary.tar.gz.sha256
tar -xzf pr5075-h200-current-7801fd7c-summary.tar.gz
python3 -I pr5075-h200-current-7801fd7c/audit.py
```

The standard-library offline audit passed. It verifies archive contents, compressed
originals, the frozen 764 production files, idle admission, periodic GPU-process
observations, request counts/timing/P95, all seven suite accuracies and Hard
Brier/ECE against frozen gold labels with the bundled pinned official scorer.
Original production file Git blobs and all four model-weight hashes match both
before and after inference. Model weights are not redistributed.

## Capacity

AIPerf 0.13.0 sends open-loop Poisson arrivals to production `POST /v1/decisions`.
There is no profiling concurrency or HTTP connection-pool ceiling. The same frozen
RACE/MultiRC validation pool contains 2,000 grouped requests / 10,600 fields, with
600 requests each at 1/4/8 fields and 100 each at 12/16. Prefix caching is ON and
the fixed pool is replayed; each point records actual cached tokens.

80,579 requests across 14 formal points and 1320 injection
seconds passed HTTP and response validation, with no skipped planned arrivals.
The search starts at 50/70 RPS, explores for 60 seconds per point and confirms
boundaries for 120 seconds. Isolated short-window backlog slopes are rechecked
at 120 seconds; an inconsistent bracket is refined using longer observations.

| Target Poisson RPS | Seconds | Actual send RPS | Completed within window RPS | P95 response ms | Tail backlog slope / s | Observation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50 | 120 | 50.67 | 50.51 | 303 | 0.0271 | Stable |
| 57.5 | 120 | 57.26 | 57.18 | 476 | -0.0662 | Stable |
| 58.75 | 120 | 58.15 | 58.02 | 497 | 0.1568 | Tail-slope flag |
| 60 | 120 | 61.03 | 60.88 | 540 | 0.2174 | Tail-slope flag |
| 65 | 120 | 65.34 | 63.90 | 2,367 | 1.2109 | Tail-slope flag |
| 70 | 60 | 69.73 | 64.27 | 4,861 | 4.2238 | Tail-slope flag, completion rate below arrival rate |

Overloaded observations complete 63.62–64.27 RPS, approximately 64 RPS. The lower slope-rule interval below is a conservative finite-window classification; it is not a new observed physical saturation plateau.

The 60 RPS / 120-second observation has actual send/completion rates 61.03/60.88 RPS and P95 540 ms. Its tail backlog slope is 0.2174 requests/s, above the fixed 0.1200 threshold, so it is classified as unstable. The configured-rate bracket is conservative under this finite-window slope criterion; the table retains actual send/completion rates and latency for interpreting that classification. The 58.75 RPS endpoint also has only the tail-slope flag (actual send/completion 58.15/58.02 RPS, P95 497 ms).

The final search interval under the fixed stability rule, expressed as configured Poisson means, is **57.5–58.75
RPS**. Finite-window arrival rates differ from configured means as shown above.
These fixed-pool short observations do not establish a production SLA or
long-duration capacity. All points, errors, backlog slopes and one-second goodput
statistics are included in summary.json and capacity.csv inside the bundle.

## Text quality

One complete pass through production `POST /v1/systemone`, concurrency 8:

- 10,751 / 10,751 successful requests.
- 10,991 / 12,351 correct fields.
- Seven-suite unweighted accuracy: **89.8629313663%**.
- Field-weighted accuracy: **88.9887458505%**.
- Jevbench-Hard (111 fields): Brier **0.3457996899773662**,
  ECE **0.0647707689169727**.
- Compared with saved H200 `4446d416` responses: 34
  changed top-1 fields and 8198 fields with
  changed candidate probabilities out of 12,351. This is saved-answer comparison,
  not a causal attribution of performance changes or numerical equivalence.
- Seven text/image protocol smoke cases pass; full image accuracy suites were
  not repeated in this experiment.

## Configuration and provenance

One H200 GPU 2, verified idle before launch, BF16 activations with the source's
FP32 norm scales, TP1, raw logits, temperature 1.99241824. PyTorch backend with
Triton attention; max batch 128, prefill 8192, session 16384, prefix cache ON,
cache fraction 0.1. Actual dependency versions are in server-effective.json.
Weights match the fixed Intern-Decision-4B reference revision
`38c9dafbbef52dd04bae943c0c6ac918419adfd5` by all four shard SHA256 values.
The official scorer is pinned to `7ce310c7262ed49cc85853339a8a42459298e3f3`.

The remote environment uses a PID namespace, while NVIDIA reports host PIDs.
GPU monitoring records both namespaces and establishes host-PID baselines after
HTTP readiness; it does not directly equate host PIDs with visible process IDs.
Two startup/preflight failures were preserved separately and are excluded from
all formal points and published quality metrics.

The compact bundle contains frozen inputs/source, all verified request timings,
complete quality responses, official scoring outputs, scheduler records and
launchers. Full AIPerf payload/transport exports remain on H200, indexed by path,
size and SHA256 in summary.json. No production code is changed by this evidence
branch. This experiment does not validate client-disconnect cancellation or
mixed generation/decision scheduling.
