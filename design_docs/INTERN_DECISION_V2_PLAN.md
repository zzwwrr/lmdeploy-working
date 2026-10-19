# Intern-Decision v2: Plan

Branch `feat/intern-decision-v2`, based on `origin/main` (`56a6c797`). It replaces `feat/native-intern-decision` (v1),
which was based on `17ed9e54` (March 2026, 371 commits behind).

Goal of this round: **functional support** on current LMDeploy, built on its scoring-only prefill.
Performance tuning comes later.

## Why port instead of rebasing commit by commit

Since v1 was cut, `main` has changed the files v1 touches a great deal:
- `api_server.py` is now split into `serve/openai/endpoints/*`;
- `api_client.py` and the old `paging/block_trie.py` are deleted;
- `AsyncEngine` uses `preprocess()` → `generate()`.

Most of v1's engine-side code (full-logits scoring, session cleanup rework, prefix-cache guard)
is either already provided by `main` or no longer needed. Replaying the 12 commits would mean
resolving conflicts in code that would be deleted straight away. So v2 is a **squashed port**:
- the self-contained protocol files (`schema.py`, `protocol.py`, and parts of `encoding.py`, `output.py`, `uploads.py`) are carried over;
- the integration is rewritten against `main`'s APIs.

v1 stays intact on its own branch for history.

## What `main` already gives us

| Need | v1 built it as | `main` provides |
|---|---|---|
| Run the model without generating | `output_logits='all'` (full L×V table) | `max_new_tokens=0` + `logprob_start_len` (lm-head only on chosen rows) |
| Prefix cache must not hide scored rows | `block_trie.py` patch | `get_prefix_cache_max_candidate_step` caps reuse at `logprob_start_pos` |
| Cancel/cleanup of a request's session | `finish_score`, strict handles, END ack | `generate()` + `safe_run` + `with_request_cleanup` |
| Image token expansion | HF processor + `multimodal.py` bridge | `VLAsyncEngine` preprocess expands `<|image_pad|>` from token-id prompts |

## The one engine gap

`main` returns, per scored row, the target token plus the top-k. Intern-Decision needs the logits of
**fixed candidate tokens** (the answer letters `A`…`9`). So we add:
`GenerationConfig.logprob_token_ids: list[int] | None`.
- It is plumbed through `SamplingParam` → `SamplingInputs` (the sorted union over the batch).
- `_get_outputs_with_logprobs` appends `input_logits.gather(-1, ids)` columns.

This is ported from `lmdeploy-learning`, where it matched HF on 6/6 decisions.

## Layers after v2

```
Service   pipeline(task='intern-decision').decide()     serve(task=...) → POST /v1/decisions, /v1/jev
             │                                                  │   (endpoints/decisions.py)
Protocol  DecisionService.decide(request)  ── serve/processors/intern_decision/
             ├ protocol.py  request/response models, errors
             ├ schema.py    trained prompt + <decision> skeleton (unchanged from v1)
             ├ images.py    paths (Python only) / base64 uploads → in-memory PIL, one loader for both
             ├ encoding.py  chat template → token ids (unexpanded), tail length, candidate ids
             ├ output.py    candidate softmax + official temperature calibration (T≈2.7478)
             └ service.py   preprocess → set logprob_start_len → generate → pick marker rows
Engine    AsyncEngine.preprocess / generate (unchanged), PyTorch engine + logprob_token_ids
```

## Commits

1. `feat(pytorch): return logits of requested token ids in input-logprob rows`: `messages.py`,
   `pytorch/messages.py`, `logits_process.py`, `strategies/ar/sampling.py`, `model_agent/agent.py`, and a unit test.
2. `feat(decision): port Intern-Decision protocol onto scoring-only prefill`: the
   `intern_decision` package (schema, protocol, images, encoding, output, service).
3. `feat(decision): expose decisions via pipeline, api_server and CLI`: `api.py`, `pipeline.py`,
   `api_server.py` (`ServerContext.decision_service`), `endpoints/decisions.py`, `endpoints/router.py`, `cli/serve.py`.
4. `docs/test(decision)`: user docs (en/zh), example, CPU unit tests, the GPU golden test against the v1
   fixture (calibrated, with tolerance), and the updated tutorial and performance analysis.

## Behaviour changes vs v1 (deliberate)

- **Calibration on by default**, matching the checkpoint's `inference.py`. The response carries
  `calibration`. `temperature=1.0` reproduces v1's numbers.
- **No engine restrictions** beyond what scoring needs: PyTorch backend, no speculative decoding,
  hybrid role, vision encoder enabled. Batching, chunked prefill, prefix caching, TP and the MP
  engine are no longer blocked.
- **No custom session/cleanup code.** v1's changes to `session_manager.py`, `engine_instance.py`,
  `mp_engine` and `block_trie.py` are gone.
- **Images are decoded in memory**, and the EXIF rotation applies to both Python paths and HTTP uploads.
- `APIClient.decide` is gone with `api_client.py` (deleted on `main`). HTTP clients post JSON directly.
- Admission control is a simple pending-request cap (HTTP 429). v1's global lock and timeouts are removed.

## Known limitations (to state in the docs)

- Rows are aligned by "target token == `<decision>`". Image expansion happens before the skeleton,
  so the skeleton's distance to the end is stable, and `logprob_start_len = expanded_len - tail`.
- With mixed batches, a request's rows may carry extra candidate columns requested by other
  requests (the union). This is harmless to `/generate` top-k (sorted, truncated), but it is not
  per-request filtered yet.
- TurboMind is unsupported, because `logprob_start_len` is PyTorch-only.
- Not run in this round (as agreed). Verification is by self-review plus a subagent review.
