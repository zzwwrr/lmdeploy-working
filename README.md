# PR 5075: historical H200 quality reference

This bundle preserves archived Intern-Decision-4B results from 2026-10-30 to
2026-11-03. Use it as historical reference for the H200 tables. It does not
measure the current PR head or establish its capacity, throughput or latency.
The runs predate the indexed GDN gate fix and the proposed RMSNorm changes.

## Contents

- [Reference bundle](pr5075-h200-20261106.tar.gz): 8,934,498 bytes (8.52 MiB).
- [SHA256 checksum](pr5075-h200-20261106.tar.gz.sha256).
- 69,854 original response rows / 73,054 decision fields: two complete text runs
  and four complete POPE/AI2D image runs, with launchers, configurations,
  preparation records, selected production source snapshots and a GPU-free audit.
- TurboMind execution records and unrelated full-repository material are excluded.

The raw response bytes are preserved losslessly. The smaller bundle uses XZ
compression for JSONL files and selected source snapshots. Model weights, full
input corpora and dataset images are not included.

## Evaluated versions

| Component                | Revision                                   |
| ------------------------ | ------------------------------------------ |
| Text LMDeploy            | `7d2ef19115c9306e2d57437ec30a994d5b942fd1` |
| Image LMDeploy           | `3df76b5bda652d4f8ba5fc056e1acf773bc2ab36` |
| vLLM serving adapter     | `a8fe089c73472acc728102ac3f271f96f2398352` |
| Intern-Decision-4B model | `38c9dafbbef52dd04bae943c0c6ac918419adfd5` |
| Official scorer          | `7ce310c7262ed49cc85853339a8a42459298e3f3` |

Both backends use BF16, TP1 and temperature `1.99241824`. Effective settings,
wrapper overrides and dataset revisions are recorded inside the bundle.

## Verify

Download the bundle and its checksum into the same directory, then run:

```bash
sha256sum -c pr5075-h200-20261106.tar.gz.sha256
tar -xzf pr5075-h200-20261106.tar.gz
python3 pr5075-h200/audit.py --output h200-verification.json
```

Python 3.12 and its standard library suffice. The bundled README describes
optional replay with the pinned official scorer and input preparation.
Before publication, integrity checks, original-byte checks and official-scorer
replay reproduced the saved text/image quality metrics and all 34 guide values.

## Historical limits

Each historical image run did not save its own production-file hash manifest.
The frozen source revision, launchers and original output manifests are retained;
collection-time repository cleanliness is recorded separately. The input
re-preparation recipe is pinned but was not rerun by downloading all datasets.
The bundled README and provenance files describe these limits in detail.
