# PR 5075: current H200 capacity and quality evidence

This bundle records new HTTP measurements of
[PR 5075](https://github.com/InternLM/lmdeploy/pull/5075) at fixed source commit
[`4446d416789988d0429443ff88e3faf038a9965b`](https://github.com/InternLM/lmdeploy/commit/4446d416789988d0429443ff88e3faf038a9965b).
The run uses one H200 and Intern-Decision-4B. It retains the indexed GDN gate
fix and excludes the separately scoped RMSNorm experiments.

## Download and verify

- [Review bundle](pr5075-h200-current-4446d416-review.tar.gz):
  16,979,231 bytes (16.19 MiB).
- [SHA256 checksum](pr5075-h200-current-4446d416-review.tar.gz.sha256).
- SHA256: `10622378fb4f5f8d9a2bb192119690ca5da620ca157a4b1d3fcbfbc390d85314`.

Download both files into the same directory, then run:

```bash
sha256sum -c pr5075-h200-current-4446d416-review.tar.gz.sha256
tar -xzf pr5075-h200-current-4446d416-review.tar.gz
python3 -I pr5075-h200-current-4446d416/audit.py
```

The audit uses Python 3.12 and its standard library, without a GPU or model.
It verifies 232 file hashes and compressed original bytes, recomputes the
per-request timing summaries, and independently scores the complete quality
responses using the bundled official scorer. The local audit passed before
publication: 70,051 stress requests, 10,751 quality requests and 12,351 fields.

## Capacity

AIPerf 0.13.0 sends open-loop Poisson arrivals to the production
`POST /v1/decisions` endpoint. There is no profiling concurrency limit; the
client connection pool is unlimited. The frozen RACE/MultiRC public-validation
pool contains 2,000 grouped requests and 10,600 fields: 600 requests each with
1, 4 or 8 fields, and 100 each with 12 or 16 fields. Prefix caching is enabled;
the fixed pool is replayed without cache busting. Actual cache hits are recorded
per point. This measures that workload rather than a fresh-input workload.

Thirteen formal points contain 70,051 requests over 1,140 seconds of injection.
The search starts at 50/70 RPS and bisects around the previous 60 RPS region.
Exploratory windows last 60 seconds; boundary confirmations last 120 seconds.
All formal requests pass HTTP and response validation, with no skipped planned
arrivals. The following rows are separate 120-second observations:

| Target Poisson RPS | Actual send RPS | Completed within window RPS | P95 response ms | Observed status |
| ------------------ | --------------- | --------------------------- | --------------- | --------------- |
| 50                 | 50.47           | 50.44                       | 269             | Stable          |
| 60                 | 60.19           | 60.10                       | 648             | Stable          |
| 62.5               | 60.87           | 60.68                       | 646             | Stable          |
| 63.75              | 63.72           | 63.24                       | 1,575           | Growing backlog |
| 65                 | 65.65           | 64.04                       | 3,676           | Growing backlog |

The observed completion plateau is about 64 RPS. The final bracket of configured
Poisson means is 62.5 to 63.75 RPS; random finite-window arrival counts differ
from those means, as the table shows. A target of 60 RPS leaves some margin in
this run: P95 is 648 ms and every response completes within one second.
These short observations do not establish a long-duration capacity or SLA.
`capacity.csv` and `summary.json` include every point, backlog slopes, errors,
actual arrivals, latency distributions, and the one-second goodput statistic.

## Quality

The seven official text suites run once through the production
`POST /v1/systemone` endpoint at concurrency 8:

- 10,751 requests succeed; 10,997 of 12,351 fields are correct.
- Equal-weight seven-suite accuracy: **89.8893203024%**.
- Field-weighted accuracy: **89.0373249130%**.
- JevBench Hard Brier: **0.3465291004245039**; ECE: **0.0818989263554503**.

Matching by suite, source line and ID against the historical gate-only Triton
C8 run gives zero changed top-1 answers and zero probability differences across
all fields. Those runs have different serving settings, so this agreement does
not constitute a controlled performance comparison. The new run also passes
seven text/image protocol smoke cases; it does not repeat the historical image
accuracy suites.

## Frozen configuration and scope

- Model revision: `38c9dafbbef52dd04bae943c0c6ac918419adfd5`;
  all four weight-shard SHA256 values match the historical fixed model.
- BF16, TP1, raw logits, temperature `1.99241824`; PyTorch backend with Triton
  attention (`use_fa3=false`).
- Maximum batch size 128, prefill 8,192 tokens, session length 16,384,
  cache fraction 0.1, prefix caching on and default state budget.
- Official scorer revision: `7ce310c7262ed49cc85853339a8a42459298e3f3`.
- The 764 tracked production-file Git blobs match the frozen source before and
  after the run. The source archive, effective settings, harness, preparation
  records, scheduler state and timing-analysis corrections are included.

The compact bundle includes frozen inputs, all verified per-request timings,
complete quality responses, scoring outputs, AIPerf metric exports, source and
the offline audit. The full AIPerf payload/transport exports total 426,245,719
bytes and remain on H200; `summary.json` records their paths and SHA256 values.
Those exports are not included in this public bundle. Model weights, GPU traces
and TurboMind benchmark results are not included.

Historical A100 and H200 measurements remain separate reference artifacts;
they are not measurements of this source revision. No production-code changes
are made by this evidence branch. The repository's complete source snapshot is
included only to identify and reproduce the measured version.
