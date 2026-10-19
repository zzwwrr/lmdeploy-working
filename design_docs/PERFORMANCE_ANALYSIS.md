# Intern-Decision Performance: v1 vs v2

Companion to `TUTORIAL_three_layers.md`. It covers why v1 (`feat/native-intern-decision`) was slow, what v2
(`feat/intern-decision-v2`) changes by design, and where the remaining cost is.

> **How these numbers were obtained.** v2 was built for functional support and **was not benchmarked on a GPU**
> in this round.
> - **Row counts and tensor sizes** are computed exactly from the real tokenizer on the 28 fixture cases.
> - **Latencies** marked *(learning repo)* were measured with the same mechanism (scoring-only prefill + candidate
>   ids) in `lmdeploy-learning`, on Windows, with torch fallbacks for missing kernels.
> - Everything else is an estimate and is labelled as such. Section 5 gives a measurement recipe.

---

## 1. Summary

| # | v1 cause | v2 status |
|---|---|---|
| 1 | Logits for **every token × all 248,320 vocab entries**, copied to CPU, scanned 3× | **Removed.** The LM head runs only on the skeleton tail, and 63 numbers per row come back. |
| 2 | One decision at a time (global lock + `max_batch_size=1`) | **Removed.** Decisions batch with each other and with chat. |
| 3 | A CPU scan of the full table on the event loop shared with the engine | **Removed.** The remaining CPU work is O(rows × 63). |
| 4 | Uploads re-encoded to PNG, written to disk, decoded again | **Removed.** Images are decoded once, in memory. |
| 5 | Prefix caching forced off | **Allowed.** `main` caps reuse at `logprob_start_len`. |
| 6 | (Windows) Conv3d patch-embed, ~13 s per image request | **Not addressed.** Linux-first. See §4.4. |

The root cause of 1–3 and 5 was v1's base version, which had no scoring-only prefill. v2 is based on current `main`,
which has it.

---

## 2. What the GPU and the host handle per request

`L` = prompt tokens after image expansion. `tail rows` = rows scored in v2 = tokens from the row before the first
`<decision>` to the end, minus one. bf16 = 2 bytes.

| Fixture case | L | Questions | v1 full table (L × V) | v2 tail rows | v2 LM-head output (rows × V) | v2 returned to host (rows × 63) |
|---|---|---|---|---|---|---|
| text | 279 | 3 | 132 MB | 22 | 10.4 MB | ~11 KB |
| long_text | 1,835 | 3 | 869 MB | 22 | 10.4 MB | ~11 KB |
| red_image | 248 | 1 | 118 MB | 6 | 2.8 MB | ~3 KB |
| eight_images | 969 | 1 | 459 MB | 6 | 2.8 MB | ~3 KB |
| sixteen_fields | 1,102 | 16 | 522 MB | 162 | 76.7 MB | ~80 KB |
| *max prompt* | 8,192 | 16 | 4.07 GB | ~160 | ~77 MB | ~80 KB |

The v2 cost of the LM head **no longer grows with the prompt**: it depends only on the size of the skeleton. The
prompt length now only affects the transformer layers, as it does for any prefill.

---

## 3. Throughput: batching is back

v1 had a global `asyncio.Lock` plus `max_batch_size=1`, so throughput was 1 / latency. v2 has no lock: `DecisionService`
only caps the number of pending requests (`max_pending_requests=64`, beyond which it returns HTTP 429). Each request
is an ordinary prefill that the scheduler packs with others. The engine's normal settings apply: `max_batch_size`,
`max_prefill_token_num`, chunked prefill and prefix caching.

*(learning repo)*:
- 8 concurrent text decisions finished in **311 ms in total**, against ~600 ms back to back (8 × 76 ms);
- a single warm text decision took **76 ms**, and a single image decision **118 ms**;
- the HF reference took ~890 ms and ~16 s for the same requests (the 16 s was dominated by the Windows Conv3d issue, §4.4).

---

## 4. Remaining costs in v2 (next optimisation targets)

These are deliberately left for a later round. The goal of this round was functional support.

### 4.1 The LM head still runs over every tail row
Only the Q rows predicting `<decision>` are needed, but every row from the first decision row to the end is
projected: skeleton punctuation, field names, and the closing template tokens. For `sixteen_fields` that is 162 rows,
of which 16 are needed (~10×).
**Option:** extend `fill_logits_indices` to take an explicit list of positions (one per marker). This is the TODO `[10]`
idea from v1, now cheap because the plumbing (`logits_indices`, `seq_logit_length`) already exists.

### 4.2 Union of candidate columns across a batch
All rows in a batch share one column layout, so the GPU gathers the union of all requested candidate ids (just the
62 letters when only decisions are batched). `engine_loop._append_input_logprobs` then trims each row to its own
request's columns. The gather is negligible, and the trim is one `index_select` per request, only when the request
asked for fewer ids than the union.
**Option:** none needed for decision-only batches.

### 4.3 Long prompts and chunked prefill
When a scoring prompt is longer than `max_prefill_token_num`, it is prefilled in chunks. The engine's cross-chunk
logit carry requires such a chunk to run **alone** (`fill_logits_indices` asserts a single request). For Intern-Decision
the scored rows are always in the last chunk, so this only matters for very long prompts. If many long decisions are
expected, raise `max_prefill_token_num`.

### 4.4 Image preprocessing
Per image, the image is decoded once (`images.py`, in a worker thread), then the engine's Qwen3.5 processor resizes,
normalises and patchifies it (`VLAsyncEngine.preprocess`). Large photos cost tens of ms of CPU each *(estimate)*.
On **Windows** only, the ViT patch embedding (`nn.Conv3d`, bf16) took ~13 s per request in our tests. An equivalent
`linear` takes 0.3 ms. That fix, and the torch fallbacks needed when `tilelang` is missing, exist in
`lmdeploy-learning` but are out of scope for this Linux-first branch.

### 4.5 Prompt rendering per request
Each request renders the chat template and tokenises it on CPU (`encoding.py`, in `asyncio.to_thread`). That is
milliseconds per request *(estimate)*. The shared system prompt is identical every time, so with prefix caching on,
its KV cache is reused on the GPU side.

### 4.6 Not a cost: `eager_mode`
CUDA graphs mainly accelerate decode steps. A decision is a single prefill, so eager mode costs little here.

---

## 5. How to measure (Linux, idle GPU)

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task intern-decision --max-batch-size 32
```

```python
# bench.py
import time, concurrent.futures as cf, requests
URL = 'http://127.0.0.1:23333/v1/decisions'
req = {'state': {'light': 'red'}, 'questions': {'act': {'type': 'choice', 'criteria': {'stop': 'Stop', 'go': 'Go'}}}}
requests.post(URL, json=req).raise_for_status()                      # warm-up
t = time.perf_counter(); r = requests.post(URL, json=req).json()
print('single', round((time.perf_counter() - t) * 1e3, 1), 'ms', r['timing'])
for n in (8, 32):
    t = time.perf_counter()
    with cf.ThreadPoolExecutor(n) as ex:
        list(ex.map(lambda _: requests.post(URL, json=req).json(), range(n)))
    print(n, 'concurrent', round((time.perf_counter() - t) * 1e3, 1), 'ms total')
```

What to check:
- `timing.inference_ms` should stay roughly flat when you pad `state` from 300 to 4,000 tokens, apart from the
  normal growth of prefill time. In v1 it grew with the full table.
- The total for n concurrent requests should be well below n × the single latency (batching).
- `nvidia-smi` should show no GB-sized spikes on long prompts.
- For correctness on the GPU, run
  `DECISION_TEST_CHECKPOINT=/path DECISION_TEST_GPU=1 pytest tests/test_lmdeploy/test_decision_golden.py`.
