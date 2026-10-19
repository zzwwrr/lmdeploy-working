# Why `feat/native-intern-decision` Is Slow, and How to Fix It

Companion to `TUTORIAL_three_layers.md`. Read that first if terms like *prefill*, *logits* or
*lm-head* are new to you.

> **How these conclusions were reached.** This is based on reading the branch's code paths and on
> arithmetic about tensor sizes. Where numbers are *measured*, they come from the alternative
> implementation in `lmdeploy-learning` on the same machine, and they are labelled as such. The
> branch itself was not benchmarked here: it needs `tilelang`, which is not available on Windows.
> Section 6 gives a recipe for measuring it on Linux.

---

## 1. Summary

| # | Cause | Kind | Effect |
|---|-------|------|--------|
| 1 | Scores computed, copied and checked for **every token × all 248,320 vocab entries** | Wasted work | Latency grows with prompt length; hundreds of MB to several GB per request |
| 2 | **One decision at a time** (global lock + `max_batch_size=1`) | No parallelism | Throughput = 1 / latency; queue wait grows linearly |
| 3 | Heavy CPU work runs **on the event loop** shared with the engine | Blocking | GPU sits idle while the CPU checks the score table |
| 4 | HTTP images are decoded, re-encoded to PNG, written to disk, then decoded again | Wasted work | Extra 100s of ms for large photos |
| 5 | Prefix caching disabled | Missed reuse | The identical system prompt is recomputed for every request |
| 6 | (Windows only) Conv3d patch-embed in the ViT | Environment | ~13 s per image request in our tests |

Causes 1, 2 and 3 have **one root**. The branch is based on LMDeploy from March 2026 (`17ed9e54`),
which has no "score-only" prefill. It therefore has to request *all* logits, and then has to
restrict everything else so that this stays affordable.

---

## 2. Cause 1: the giant score table

### What happens
`score_logits()` submits the request with `output_logits='all'`. In the engine:

1. `ModelAgent._async_model_forward` sees `return_logits=True` and **skips**
   `_postprocess_forward_output`. That function normally keeps just the last position of each
   sequence. Because it is skipped, the lm-head runs on **every** position.
2. The result is a `[L, 248320]` tensor on the GPU.
3. `engine_loop` → `SchedulerSequence.append_logits` copies it to CPU as a numpy `int16` view
   (the bf16 bits) into the sequence's logits buffer.
4. `score_logits` gets it back as a torch tensor and runs `torch.isfinite` over all of it in
   256-row chunks.
5. Then `index_select` keeps the **Q** rows (Q = number of questions, ≤ 16) and checks those for
   finiteness again.
6. `DecisionOutputProcessor` uses **62 columns at most** from each row, and checks finiteness a third time.

### How big is it?

`L` = prompt length in tokens, `V` = 248,320, bf16 = 2 bytes.

| Prompt | Full table (bf16) | lm-head FLOPs (2·L·1024·V) | Numbers actually needed (Q × ≤62) |
|---|---|---|---|
| Text example, L = 336 | **167 MB** | 0.17 TFLOP | 2 × 62 = 124 |
| One-image example, L = 700 | **348 MB** | 0.36 TFLOP | 124 |
| Max allowed, L = 8192 | **4.07 GB** | 4.2 TFLOP | ≤ 992 |

For the 700-token request that is about **174 million numbers produced to use 124**.

### Where the time goes
- **GPU:** the lm-head FLOPs are actually *not* the main problem. A modern GPU does 0.36 TFLOP in
  a few milliseconds. The problems are that the 348 MB–4 GB output has to be allocated, and it
  spikes GPU memory.
- **GPU → CPU copy:** about 20 GB/s over PCIe, so roughly 17 ms for 348 MB and 200 ms for 4 GB,
  plus another host-side copy into the numpy buffer.
- **CPU scan:** `isfinite` over 174M bf16 values in Python-driven chunks is the most expensive
  step. Expect hundreds of ms for the 700-token case and seconds near 8192 tokens. *(Estimate;
  measure it with section 6.)*

### Why it also forces the other restrictions
Look at `configure_task` (`intern_decision/task.py`):
- `max_batch_size=1`: in `inputs_maker.__need_logits`, if *any* sequence in a batch wants logits,
  the whole batch is computed with full logits. Batching decisions, or mixing them with chat,
  would multiply the table.
- `max_prefill_token_num == 8192` and no chunked prefill: this keeps the whole table in one step.
- `cache_max_entry_count=0.05`: leaves GPU memory free for the table.
- `enable_prefix_caching=False`: a cached prefix has no logits (see `block_trie.py`).

### Fix
Rebase onto current `main` and use its **scoring-only prefill**:

```python
GenerationConfig(max_new_tokens=0,
                 logprob_start_len=first_marker_pos - 1)  # lm-head only on the skeleton tail
```

Then add a small engine feature that returns only the candidate columns. This is done in
`lmdeploy-learning` as `GenerationConfig.logprob_token_ids`:

```python
# model_agent/agent.py, _get_outputs_with_logprobs
cand = logprob_token_ids[None].expand(input_logits.size(0), -1)
vals = torch.cat([vals, input_logits.gather(-1, cand)], dim=-1)
```

The result is a few rows × 62 numbers instead of `L × 248,320`. Measured in `lmdeploy-learning`:
- it matched HF on 6/6 decisions, with a max probability difference of 0.0099;
- one text request took **76 ms** warm;
- one image request took **118 ms** warm.

The branch already has a TODO for an even deeper version, `[10]` in `agent.py`, which would
select hidden states before the lm-head.

---

## 3. Cause 2: everything is serialized

```python
# runtime.py
self.lock = asyncio.Lock()
...
await asyncio.wait_for(self.lock.acquire(), self.config.queue_timeout)   # held for the whole decision
```

Together with `max_batch_size=1`, exactly **one** decision is in the engine at any moment. The CPU
preprocessing for the *next* request also waits, because it runs inside the lock.

Consequences:
- Throughput is 1 / latency. If one decision takes 1 s, the server does 1 decision/s no matter how
  big the GPU is.
- The k-th request in the queue waits about k × latency. With `max_pending_requests=32`, the last
  one waits 32× as long as the first. With `queue_timeout=120 s`, requests start failing with 429
  once latency × queue length passes two minutes.
- The GPU is idle during every request's CPU phases: preprocessing, the logits scan, and building
  the response.

Measured in `lmdeploy-learning`, where batching is allowed: 8 concurrent text decisions took
**311 ms in total**, against about 600 ms (8 × 76 ms) back to back. The engine packed them into
one prefill.

### Fix
After fixing cause 1, the reason for the lock is gone:
- allow `max_batch_size > 1`;
- replace the lock with `asyncio.Semaphore(max_batch_size)`, or just rely on the engine's scheduler;
- run preprocessing *outside* the semaphore, so request N+1's CPU work overlaps request N's GPU work.

If the lock is there to make golden tests bit-exact, compare with a tolerance instead. Batched
kernels are not bit-identical anyway.

---

## 4. Cause 3: CPU work blocks the engine

The branch forces the **in-process** engine (`enable_mp_engine` is turned off for this task in
`api_server.serve`, and `configure_task` rejects it). The PyTorch engine's loop then runs on the
**same asyncio event loop** as the HTTP and scoring code.

`score_logits` runs `torch.isfinite` over the whole score table on that loop. The
`await asyncio.sleep(0)` between 256-row chunks lets other tasks in, but each chunk is still
63.6M values of blocking CPU work. While it runs, the engine loop cannot schedule the next forward
pass and HTTP cannot accept requests.

**Fix:** with cause 1 fixed there is nothing big left to scan. Check finiteness once, on the
`Q × 62` candidate logits, in the output processor.

---

## 5. Smaller causes

### 5.1 Image round-trip for HTTP uploads (`uploads.py`)
Per image: base64 decode → PIL decode → EXIF rotate → **PNG encode** → write to a temp dir →
(later, in `encoding.py`) open → decode again → convert. PNG encoding a 12-megapixel photo is
slow, often hundreds of ms. Then the HF image processor resizes it anyway.
**Fix:** keep the decoded `PIL.Image` in memory and pass it to the encoder. No temp files are
needed, and the Python and HTTP paths then do the same thing.

### 5.2 Prefix caching is off
Every request starts with the same system prompt and template, typically 100+ tokens. They are
recomputed every time. With tail-only scoring (cause 1 fixed), logits for the cached prefix are no
longer needed, so prefix caching becomes safe to enable. Check first that your LMDeploy version
supports prefix caching for hybrid linear-attention models like Qwen3.5.

### 5.3 Validation repeated three times
The request is parsed by FastAPI, then by `DecisionRuntime.decide`, then by
`DecisionInputProcessor`. Each uses strict pydantic plus a `json.dumps` of the state. This is small
(milliseconds), but easy to drop.

### 5.4 Disconnect polling
`serve_decision` polls `is_disconnected()` every 50 ms per request. That is cheap, but with 32
queued requests it is 640 wake-ups per second doing nothing useful.

### 5.5 Windows only: the ViT patch-embed
The branch's `Qwen3_5VisionPatchEmbed` uses `nn.Conv3d`. On our Windows setup (torch 2.9.1 + cuDNN,
bf16) that one layer took **~13 s per image request**. Rewriting it as an equivalent `linear` took
0.3 ms (done in `lmdeploy-learning`). The kernel moves over non-overlapping patches, so the two
computations are identical. On Linux, measure it before assuming it is fine. Note also that the
branch's `gated_delta_rule.py` / `causal_conv1d.py` require `tilelang`, so the branch does not run
on Windows at all without fallbacks.

### 5.6 Not a cause: `eager_mode=True`
CUDA graphs mainly speed up *decode* steps, which use small fixed shapes. A decision is a single
prefill with no decode, so eager mode costs little here.

---

## 6. How to measure it yourself (Linux, branch checked out)

The response already carries timing: `timing.queue_ms`, `timing.inference_ms`, `timing.server_ms`.

```python
# bench_branch.py: run with the server started via
#   lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task intern-decision
import time, concurrent.futures as cf
from lmdeploy.serve.openai.api_client import APIClient

client = APIClient('http://0.0.0.0:23333')
req = {'state': {'hp': 30}, 'questions': {'act': {'type': 'choice',
       'criteria': {'attack': 'hit it', 'flee': 'run'}}}}

client.decide(req)                                   # warm-up
t = time.perf_counter(); r = client.decide(req)
print('single', (time.perf_counter() - t) * 1000, 'ms', r['timing'])

for n in (4, 16):
    t = time.perf_counter()
    with cf.ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(client.decide, [req] * n))
    print(n, 'concurrent', (time.perf_counter() - t) * 1000, 'ms',
          'max queue_ms', max(x['timing']['queue_ms'] for x in rs))
```

What to look for:
- `inference_ms` growing roughly in proportion to prompt length confirms cause 1. Try padding
  `state` to 2k, 4k and 8k tokens.
- `queue_ms` growing linearly with concurrency confirms cause 2.
- `nvidia-smi` during an 8k-token request shows the GB-sized memory spike from cause 1.
- `py-spy dump --pid <server>` during a request shows time in `torch.isfinite` / `score_logits`
  (causes 1 and 3).
- `torch.profiler` around `ModelAgent._async_model_forward` shows lm-head and D2H copy time.

---

## 7. Fix plan, in order

| Step | Change | Expected effect |
|---|---|---|
| 1 | Rebase onto current `main` | Gets scoring-only prefill (`logprob_start_len`) and many upstream fixes |
| 2 | Score with `max_new_tokens=0` + `logprob_start_len` + candidate-id logprobs; drop `output_logits='all'` | Removes the L × 248,320 table. Per-request latency should drop to roughly one forward pass (76–118 ms measured on the alternative) |
| 3 | Remove the lock, allow `max_batch_size > 1`, move preprocessing out of the critical section | Throughput scales with batching (8 requests: 311 ms total measured, not 8× the latency) |
| 4 | One finiteness check on candidate logits only | No more event-loop blocking |
| 5 | Pass PIL images in memory, skip the PNG/temp-file round-trip | Saves 100s of ms per large uploaded photo |
| 6 | Re-enable prefix caching (if supported for hybrid models) and relax `configure_task` | Shared system prompt computed once |
| 7 | Windows: linear patch-embed, plus torch fallbacks for `tilelang` kernels | Makes Windows usable at all |

Steps 1 and 2 are the important ones. Almost everything else follows from them.
