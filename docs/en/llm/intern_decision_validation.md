# Intern-Decision native task validation

This report distinguishes native task delivery, migration equivalence, and
cross-backend numerical acceptance. The native implementation is based on
LMDeploy `0f925435eb6afdea945ed91d91d26bfada00b7d4` (network baseline
`17ed9e546aaa12b373bb3ac713653f14ff57b3d2`, package 0.12.1).
Steps 01–09 are implemented on the `feat/native-intern-decision` branch;
step 10 (projection before the LM head) remains deferred by design.

## Environment and provenance

Validation date: 2026-10-19. Checkpoint: Intern-Decision-0.8B, revision
`85a0cc5a99d67ea8d56dfe98115689212867171d`. All 20 checkpoint files match the
original SHA256 manifest. No weights, tokenizer files, calibration or temperature
were changed. PyTorch 2.10.0+cu128, Transformers 5.3.0, FLA 0.4.1, Python 3.12,
BF16, TP/DP/EP=1, eager mode, batch 1, prefill/session limit 8192, cache fraction
0.05, prefix caching disabled. Validation used the explicitly authorized physical
GPU 7, UUID `GPU-cf298831-61c8-9603-28b3-463bcbeb4831`, with no competing process.

The old external LMDeploy backend was run against an isolated, unchanged worktree
at `0f925435`. Its private-protocol SHA256 checks were preserved. The native
session changes deliberately invalidate that external backend's protocol hashes
in this worktree; restarting it against the native tree requires a separate
compatibility update. Existing external source changes were left intact.

## Results

| Check | Result |
| --- | --- |
| CPU protocol, image, lifecycle, routing, MRoPE, and selected existing regressions | 129 passed; one explicit GPU golden test skipped in CPU run; one existing remote-config test deselected |
| CPU final token/marker/candidate/patch/grid hashes vs frozen old encoder | Exact agreement in all 28 fixture cases |
| Native vs freshly run old LMDeploy scoring | 28 fresh cases plus frozen 8192-token boundary: 29 cases / 88 fields, candidate logits and probabilities exactly equal |
| Python vs actual HTTP and APIClient | Text/image answers equal; both aliases, authentication, error statuses pass |
| Existing interfaces on the same actual model | HTTP/Python chat, encode, generic logits [2,248320] and finite PPL pass |
| Native vs official HF (fresh + frozen boundary) | 29 cases / 88 fields: 56 fields within 0.02 probability/TVD gate; maximum probability error 0.06217653; FAIL |
| Full seven-suite native evaluation | Complete: 10,751 rows / 12,351 fields; exact probability agreement with frozen old LMDeploy; HF accuracy gates pass 6/7, FAIL overall |
| GPU golden and ownership | 28 standalone frozen cases pass; exact 8192 accepted / 8193 rejected; real in-flight cancellation and subsequent requests pass; tied embeddings verified |
| Sparse LM-head projection / throughput acceptance | Deferred / not claimed |

The excluded existing test is `test_utils.py::test_get_and_verify_max_len`: its InternVL-AWQ config is absent from the local model cache. It was attempted and failed for that missing dependency. A real reward-model checkpoint was unavailable; reward scoring is covered with the shared CPU execution contract, rather than a claimed real-weight reward acceptance.

The small comparison uses 28 freshly regenerated HF cases (exactly matching their archived logits/encoding) plus the archived 8192-token HF boundary. The native boundary was rerun on the final implementation, with raw candidate logits retained.

The small HF comparison has one near-tie argmax flip, in
`sixteen_fields / field_10`: HF candidate logits `[27.125, 27.125]` give
`[0.5, 0.5]`; native logits `[27.0, 27.125]` give approximately
`[0.46879065, 0.53120941]`. Every field retains raw candidate logits and original
probabilities in the validation artifacts. The old and native LMDeploy paths
agree, so this failure was not introduced by protocol migration.

The original investigation had already observed upstream hidden-state differences
and BF16 output rounding. This adaptation verifies exact input and MRoPE alignment
and preserves the original Qwen3.5 network, tied embedding/loading and operators.
It does not establish a unique numerical root cause or silently change operators
to satisfy the gate. FP32 diagnostic reprojection is not used as production output.

The fixed accuracy-v1 ToolACE split contains 310 rows but 307 unique IDs: three
identical pairs at source lines 10/220, 11/110, and 12/273. Evaluation preserves
this published split and uses `(source_line, id)` identities consistently for HF
and native inference. It does not silently deduplicate and change the benchmark.
All other split IDs are unique; dataset hashes, physical line counts and complete
prediction identities were checked against the frozen HF outputs.

## Complete seven-suite comparison

The native run preserves the complete published accuracy-v1 split and exactly
matches the frozen old LMDeploy probabilities across all 12,351 fields: maximum
probability error and TVD are zero, with no label flips. Against frozen HF,
76 labels differ (30 reference near ties). Six accuracy gates pass; Jevbench-hard
fails the 1 percentage point drop gate. These are accuracy/label gates, not a
claim that full-suite probabilities meet the 0.02 numerical threshold.

| Suite | Reference accuracy | Candidate accuracy | Change (pp) | Label agreement | Non-near agreement | Max probability error | Max TVD | Flips (near) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| jevbench-easy | 97.9167% | 97.9167% | +0.0000 | 100.0000% | 100.0000% | 0.029793 | 0.029793 | 0 (0) | PASSED |
| jevbench-original | 80.5556% | 80.5556% | +0.0000 | 100.0000% | 100.0000% | 0.118808 | 0.118808 | 0 (0) | PASSED |
| jevbench-hard | 54.0541% | 52.2523% | -1.8018 | 98.1982% | 99.0741% | 0.056719 | 0.056719 | 2 (1) | FAILED |
| agnews-test | 88.6447% | 88.5921% | -0.0526 | 99.6316% | 99.7881% | 0.154089 | 0.154089 | 28 (12) | PASSED |
| toolace-test | 94.5161% | 94.5161% | +0.0000 | 100.0000% | 100.0000% | 0.054428 | 0.054428 | 0 (0) | PASSED |
| typed_decisions-test | 77.6000% | 77.5500% | -0.0500 | 98.7500% | 99.2386% | 0.093386 | 0.093386 | 25 (10) | PASSED |
| wildjailbreak-test | 64.4796% | 64.1629% | -0.3167 | 99.0498% | 99.3548% | 0.062419 | 0.062419 | 21 (7) | PASSED |

Every flip is retained in the full comparison artifacts. Maximum full-suite
probability error is 0.154089. The final 8192-token/cancellation/PPL process peaked at 9.19 GiB of PyTorch CUDA allocations; this excludes non-PyTorch allocations and is not a serving memory guarantee. Full-suite artifacts retain original probabilities,
identities and provenance; raw candidate logits are retained for the controlled
small-case numerical comparison, rather than claimed for every accuracy-suite
record. No result was renormalized, calibrated, or replaced by FP32 reprojection.

## Code paths

Python: `lmdeploy.api.pipeline` → `Pipeline.decide` → existing pipeline event loop
→ `AsyncEngine.async_decide` → `DecisionRuntime.decide` → `DecisionInputProcessor`
→ `AsyncEngine.async_get_logits` → shared `serve/core/scoring.py` → existing engine
instance and Qwen3.5 network → `DecisionOutputProcessor`.

HTTP: existing server router and authentication → `serving_decision.serve_decision`
→ the same `AsyncEngine.async_decide`. `/v1/jev` and `/v1/decisions` use the same
route handler. The engine's actual cache-derived token capacity is read through an explicit
engine capability method.

Session scoring holds the handle through cancellation/stream close and confirmed
END_SESSION. `Session.request_handle(strict=True)` propagates errors/cancellation.
Cleanup failure quarantines the handle and disables subsequent scoring. Caller-owned
sessions survive successful `sequence_end=False` calls and can be explicitly closed,
even with step zero. Concurrent scoring failures cancel and await siblings.

## Reproduction

No external project is required for these tests or the example:

```shell
CUDA_VISIBLE_DEVICES= DECISION_TEST_CHECKPOINT=/path/to/Intern-Decision-0.8B \
  python -m pytest tests/test_lmdeploy/test_decision*.py \
  tests/pytorch/paging/test_block_trie.py tests/pytorch/paging/test_scheduler.py \
  tests/pytorch/paging/test_block_manager.py -q
```

The checkpoint is optional for the pure protocol/lifecycle tests. Tokenizer,
processor and MRoPE integration checks explicitly skip without it. To run the
separate real-weight golden test, select an authorized idle GPU and add
`DECISION_TEST_GPU=1`.

```shell
CUDA_VISIBLE_DEVICES=<authorized GPU> \
  python examples/decision/decision.py /path/to/Intern-Decision-0.8B
```

The frozen CPU fixture records source, network and checkpoint revisions. External
benchmark datasets and full prediction files are intentionally not committed.
The validation workspace retains harnesses, environment/module provenance,
checkpoint hashes, all comparison fields and every failure/flip. The full-suite
harness uses the original frozen benchmark scorer only for evaluation; production
inference enters LMDeploy directly. These artifacts do not become runtime dependencies.

## Acceptance boundary

The first release provides a reviewable native implementation with explicit model
and resource limits. Numerical parity with HF remains a failed gate. Neither
full numerical acceptance nor throughput acceptance should be inferred from
interface correctness, migration equivalence, or unit-test success.


## Process proxy restriction

Success-path text/image tests originally passed through the existing process
proxy, but code review found its session-readiness event is set before the remote
stream is started, and cancellation does not drain that RPC stream. An early
SESSION_NOT_EXIST receipt therefore cannot establish safe ownership cleanup. The
initial native decision task rejects `enable_mp_engine=True`, and HTTP serving
uses the same supported PyTorch engine directly on the server's existing loop.
The final HTTP test passed with this configuration: actual engine class
`lmdeploy.pytorch.engine.engine.Engine`, actual token capacity 8192, both aliases,
text/image Python equivalence, APIClient, authentication, input/model/thinking
errors, ordinary chat and encode. The temporary server exited and released GPU memory. Ordinary model task routing
continues to use its original process-proxy configuration. This restriction avoids
claiming proxy cancellation support from success-path tests alone.


## Per-item Git archives

| Item | Commit |
| --- | --- |
| 01 task selection | `a12fe50f` |
| 02 shared protocol | `ce7ef16a` |
| 03 CPU encoding and images | `e2f14162` |
| 04 probability and response semantics | `e61a80da` |
| 05 native async scoring | `2d67756e` |
| 06 acknowledged resource cleanup | `d83dbf2f` |
| 07 Python/HTTP interfaces | `71572db3` |
| 08 API client | `1b3fdbb5` |
| 09 tests, integration fixes and documentation | This commit (`git log -1`) |
| 10 pre-projection optimization | Deferred by the original plan; marker retained |
