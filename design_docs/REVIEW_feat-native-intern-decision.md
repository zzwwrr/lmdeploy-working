# Review: `feat/native-intern-decision` (commits [01] → last)

Scope: `a12fe50f` ([01]) through `bcb4b1b5` (docs for [06]), 11 commits, ~5.7k lines
(≈1.2k source, rest tests/fixtures/docs). Baseline for comparison: `0f925435` (the TODO annotations).

## TL;DR

The structure is sound: an explicit `task='intern-decision'` instead of guessing from the
architecture name, the trained prompt/encoding ported verbatim, strict input validation, and
careful thought about session cleanup. The four things worth changing first:

| # | Issue | Impact |
|---|-------|--------|
| 1 | Temperature calibration (T≈2.7478) is dropped | `probabilities`, `confidence`, `score`, `noul` differ from the official `DecisionEngine` output |
| 2 | Full `L × V` logits are materialized, then `Q` rows are picked | Up to ~4 GB (bf16) per 8k-token request. This is the reason for most of the restrictive engine settings |
| 3 | A global lock plus `max_batch_size=1` serializes all decisions | Continuous batching is unused, so throughput is one request at a time |
| 4 | Shared engine/session code changes behaviour for **all** users | `async_close` and `async_cancel` now raise, `end()` blocks, and one cleanup timeout disables scoring engine-wide |

---

## High priority

### 1. Calibration is silently dropped
`processors/intern_decision/output.py` applies a plain softmax over the candidates. The
checkpoint's own `inference.py` uses `DEFAULT_TEMPERATURE = 2.747760550703` and rescales the
probabilities (`scale_result`). The docs say "temperature 1; calibration is unsupported", and
`AsyncEngine.__init__` rejects `temperature`/`calibration` kwargs outright.

- The argmax `decision` is unchanged. Everything probabilistic changes: `confidence`, the
  `score` expectation, the `noul` probability, and any threshold logic a client builds on
  `confidence_threshold`.
- A client switching from the official engine to LMDeploy gets different numbers with no
  warning.

**Better:** apply the same temperature scaling in `DecisionOutputProcessor`, with the default
taken from the checkpoint. Echo `calibration: {method, temperature}` in the response as the
reference does. If uncalibrated output is wanted, make it an explicit opt-in (`temperature=1`).

### 2. Full-vocabulary logits for every position
`serve/core/scoring.py::score_logits` requests `output_logits='all'`. It receives a
`[L, 248320]` tensor, runs `isfinite` over it chunk by chunk, and only then does
`index_select(read_positions)`. For an 8192-token input that is about 8192 × 248320 × 2 B ≈ **4 GB**
in bf16 (8 GB in fp32). Only about `Q × 62` numbers are actually needed (≤ 16 fields × 62 symbols).

This cost explains most of the restrictions in `task.py`: `max_batch_size=1`,
`max_prefill_token_num == 8192` (no chunked prefill), and `cache_max_entry_count=0.05`.

**Better** (in increasing order of effort):
1. Use the scoring-only prefill that the engine already has: `max_new_tokens=0` +
   `logprob_start_len = first_marker_pos - 1`. The lm-head then runs only over the skeleton
   tail, not the whole prompt.
2. Add a `logprob_token_ids` field to `GenerationConfig`. It appends `input_logits.gather(-1, ids)`
   columns in `_get_outputs_with_logprobs`, so only the 62 candidate logits per row come back.
   I implemented this in the learning repo and it matched HF on 6/6 decisions, with a max
   probability difference of 0.0099 and about 76 ms per warm text request.
3. Longer term, pass explicit `read_positions` down into the model agent so the lm-head is
   evaluated only on those rows.

Any of these removes the need for the single-chunk/8192 prefill rule and the tiny KV cache.

### 3. Decisions are fully serialized
`runtime.py:24` creates one `asyncio.Lock`, and line 58 holds it for the whole decide call.
`configure_task` also forces `max_batch_size=1`. Concurrent requests queue behind each other even
though the engine supports continuous batching. In the learning repo, 8 concurrent text
decisions took 311 ms in total, versus about 600 ms back to back.

**Better:** replace the lock with an `asyncio.Semaphore(engine max_batch_size)` and allow
`max_batch_size > 1`. If the lock exists to keep the golden fixtures bit-exact, have those tests
compare with a tolerance instead. Batched bf16 kernels are not bit-identical anyway, and HF vs
LMDeploy already differs by about 1e-2.

### 4. Shared-code behaviour changes that reach non-decision users
These were made to support the strict cleanup protocol, but they are not gated on the task:

- **`Session.async_close`** now re-raises every failure (previously logged and swallowed) and
  has a hard 10 s bound. Chat and completion finalizers that `await session.async_close()` can
  now raise inside `finally` blocks.
- **`engine_instance.end()`** (sync) changed from fire-and-forget `send_async(response=False)`
  to a blocking `send(..., response=True)`. A sync caller now waits for the engine loop. If it is
  ever called from the loop's own thread, it deadlocks.
- **`async_cancel`** now raises `RuntimeError` on any status other than SUCCESS or
  SESSION_NOT_EXIST. Before, it went through `_check_resp_success`. It is also called from the
  generic `request_handle` cancellation path.
- **`async_get_logits` on PyTorch** is now completely replaced by `score_logits`, for reward
  models and `get_ppl` as well. The `safe_run`/`SafeRunException` semantics those callers relied
  on are gone.
- **Engine-wide poisoning:** a single cleanup timeout sets `engine._scoring_failure`. From then
  on *every* `async_get_logits` call fails until restart, including reward and perplexity
  scoring.

**Better:** keep the strict primitives opt-in (`strict=True`, as `request_handle` already does)
and leave the default paths as they were. Scope quarantine to the decision runtime or to the
single handle, which `_handle_reusable=False` already does well, not to the whole engine.

---

## Medium priority

### 5. Layering in `Session.finish_score`
`session_manager.py` now imports `serve.core.scoring`, receives the `engine`, and writes
`engine._scoring_failure` and `engine.decision_runtime.failure`. Its `timeout=` parameter actually
receives a `DecisionConfig` (`timeout.cleanup_timeout`). The session manager has become aware of
the decision task.

**Better:** `finish_score(handle, generator, *, submitted, finished, end, cleanup_timeout)`
should raise a typed `ScoringCleanupError`. `score_logits` or `DecisionRuntime` catches it and
decides on quarantine. Move `await_bounded_cleanup` to a neutral utility module
(e.g. `serve/utils`).

### 6. `configure_task` is rigid and runs up to three times
- It hard-codes `(hidden_size, num_hidden_layers, vocab_size) == (1024, 24, 248320)`, so any
  other Intern-Decision size is rejected even though nothing in the code depends on those numbers.
- One ~25-clause boolean produces a single generic error message, so users cannot tell which
  setting is wrong. `max_prefill_token_num` must be **exactly** 8192, so even a larger value is
  rejected.
- It is called from the CLI (`cli/serve.py`), indirectly via `default_backend_config` in
  `Pipeline`/`serve`, and again in `AsyncEngine.__init__`.

**Better:** validate once in `AsyncEngine`, from a table of `(field, allowed, reason)` rows so each
failure gets its own message. Detect a decision checkpoint from what the code actually needs: the
`<decision>` token, single-token answer symbols, and the vision merge size. `DecisionInputProcessor`
already checks all of these. Once #2 and #3 are fixed, most of the clauses can be dropped.

### 7. Python and HTTP preprocess the same image differently
- HTTP uploads go through `uploads.materialize_images`, which applies `ImageOps.exif_transpose`,
  strips metadata, re-encodes to PNG in a temp dir, and reopens it.
- Python local paths go straight to `encoding.py`, which only calls `.convert('RGB')` with no EXIF
  transpose.

A rotated phone photo can therefore get a different answer through `pipe.decide()` than through
`/v1/decisions`.

**Better:** run both paths through one `load_image()` function. Also skip the disk round-trip
(decode → PNG → temp file → decode again) and pass the decoded `PIL.Image` objects to the encoder
directly.

### 8. Validation and limits are scattered
- The request is validated three times: FastAPI, `DecisionRuntime.decide`, and
  `DecisionInputProcessor.__call__`.
- The limits (16 questions, 8 images, 62 options, 16 MP, 12/32 MB, 8192 tokens) are repeated as
  literals in `protocol.py`, `encoding.py`, `multimodal.py`, `uploads.py`, `runtime.py` and
  `input.py`. Changing a limit will leave the copies disagreeing.

**Better:** keep one `limits.py` (or `DecisionConfig`) and validate once at the boundary. Inner
layers should accept the already-validated `DecisionRequest`.

### 9. HTTP handler swallows unexpected errors
`serving_decision.py` uses `except Exception: return error_response(500, 'Decision scoring failed')`
with no logging, so production failures leave no trace. Add `logger.exception(...)`.

### 10. Redundant work on the hot path
- The full-tensor `isfinite` check in `score_logits` is repeated on the selected rows, and again
  in `DecisionOutputProcessor`. After #2, a single check on the `Q × 62` candidate logits is
  enough.
- `timing['server_ms']` is computed and then immediately overwritten (`runtime.py:85-88`).

---

## Low priority / nits
- `MPEngine.get_max_session_len` and the worker RPC were added, but the decision task forbids the
  MP engine. This is harmless plumbing, but it is unrelated to the feature.
- `Pipeline.close()` calls `async_shutdown_decisions()` for every task. It is a no-op, but a
  `self.task == 'intern-decision'` guard would make the intent clearer.
- `APIClient.decide` opens the image twice: once with PIL to sniff the MIME type and once to read
  the bytes. Reading the bytes once and sniffing from them is enough.
- `/v1/jev` as an alias is fine, but document it as deprecated so there is one canonical path.

## What is good and worth keeping
- **Explicit task selection:** `--task` / `task=` instead of inferring from
  `Qwen3_5ForConditionalGeneration`. Ordinary Qwen3.5 routing is unaffected.
- **Prompt encoding ported from the trained repo** with attribution. Marker and candidate token
  checks run at startup, not per request.
- **`to_lmdeploy_multimodal`:** thorough span/grid/patch consistency checks before anything reaches
  the GPU.
- **Upload hardening:** MIME-vs-content check, decompression-bomb guard, private temp dirs,
  client filenames never used as paths.
- **`block_trie.py`:** skipping prefix-cache matching for `out_logits` sequences is a genuine
  general bug fix, because a prefix hit cannot provide the logits for the cached positions. It is
  worth upstreaming separately.
- **`_handle_reusable` quarantine** of a handle whose state is unknown, and confirmed
  `END_SESSION` before a handle is reused.
- Golden fixtures and validation docs.

## Suggested order of work
1. Restore calibration (#1). This is small, and it fixes how the results are interpreted.
2. Switch to `logprob_start_len` + candidate-id logprobs (#2), then relax batch/prefill/cache limits (#3, #6).
3. Gate the shared-session changes behind `strict` and scope the failure state (#4, #5).
4. Unify image loading and limits (#7, #8). Then the smaller items.
