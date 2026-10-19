# Intern-Decision on LMDeploy, Explained in Three Layers

A tutorial for beginners. It explains how LMDeploy is organised, how the Intern-Decision model is
supported at each level, and **why** the `feat/native-intern-decision` branch changes what it changes.

> You only need a rough idea of "a request comes in, the model runs, an answer goes out".
> Everything else is explained here.

---

## 0. The one-paragraph version

LMDeploy can be read as three layers stacked on top of each other:

| Layer | Question it answers | Analogy (a restaurant) |
|---|---|---|
| **Service layer** | *How does a request reach us, and how does the answer leave?* | The front door, the waiter, the phone line |
| **Protocol layer** | *What does the request mean, and how do we turn the model's numbers into an answer?* | The menu and the order ticket: what a "burger, no onions" means to the kitchen |
| **Execution engine layer** | *How do we actually run the neural network on the GPU, efficiently, for many requests at once?* | The kitchen: stoves, the queue of tickets, the cooks |

For an ordinary chat model, all three layers already exist and the model just plugs in.
**Intern-Decision is not a chat model.** It does not write text; it *picks answers*. Its kitchen
(the network) is already supported, but it needs a **new menu** (protocol), a **new door**
(service endpoint), and a **slightly different way of cooking** (one forward pass, no text
generation). That is what the branch adds.

---

## 1. What makes Intern-Decision special

A normal chat request looks like this:

```
"What is 2+2?"  →  model writes  →  "4"   (one token at a time, generation)
```

An Intern-Decision request looks like this (the "Jev" format):

```json
{
  "state":     {"hp": 30, "enemy": "dragon"},
  "questions": {
    "action": {"type": "choice", "criteria": {"attack": "...", "flee": "..."}},
    "danger": {"type": "score",  "criteria": ["low", "mid", "high"]}
  },
  "images": []
}
```

The model is given a prompt that **already contains the answer skeleton**:

```
...state and options...
{
    "action": <decision>,
    "danger": <decision>
}
```

It runs **once** over this whole text (one *forward pass*). At the position just before each
`<decision>`, it reads the model's scores for the allowed answer letters (`A`, `B`, `C`...),
turns them into probabilities, and that is the answer. **No text is generated.**

So the model needs:
1. The **exact** prompt format it was trained on (otherwise the answers are wrong).
2. The scores at **specific positions** (one per question) for **specific tokens** (the answer letters).
3. Zero generated tokens.

Normal LMDeploy chat is built around "generate text", so points 2 and 3 are where all the work is.

---

## 2. The three layers in LMDeploy's code

```
┌─────────────────────────── SERVICE LAYER ───────────────────────────┐
│  CLI:  lmdeploy serve api_server ... --task intern-decision          │  cli/serve.py
│  HTTP: POST /v1/decisions  (alias /v1/jev)                           │  serve/openai/api_server.py
│        → serving_decision.serve_decision()                           │  serve/openai/serving_decision.py
│  Python: pipe = pipeline(path, task='intern-decision'); pipe.decide()│  api.py, pipeline.py
│  Client: APIClient(url).decide(request)                              │  serve/openai/api_client.py
└───────────────────────────────┬─────────────────────────────────────┘
                                │  all paths meet here ↓
                    AsyncEngine.async_decide(request)                    serve/core/async_engine.py
┌─────────────────────────── PROTOCOL LAYER ──────────────────────────┐
│  DecisionRequest / DecisionResponse (what the JSON may contain)      │  intern_decision/protocol.py
│  DecisionRuntime: queue limit, timeouts, one-at-a-time lock          │  intern_decision/runtime.py
│  DecisionInputProcessor: request → PreparedDecision                  │  intern_decision/input.py
│     ├ uploads.py   base64 images → temp files                        │
│     ├ schema.py    request → chat messages + "<decision>" skeleton   │
│     ├ encoding.py  messages → token ids, image pixels, read positions│
│     └ multimodal.py pixels → LMDeploy image dicts                    │
│  DecisionOutputProcessor: logits → probabilities → answers           │  intern_decision/output.py
└───────────────────────────────┬─────────────────────────────────────┘
                                │  async_get_logits(ids, multimodal, read_positions)
┌─────────────────────────── EXECUTION ENGINE LAYER ──────────────────┐
│  score_logits(): one forward per request, gets logits, cleans up     │  serve/core/scoring.py
│  Session / request_handle: borrow an engine "instance", give it back │  serve/managers/session_manager.py
│  EngineInstance.async_stream_infer → RequestSender                   │  pytorch/engine/engine_instance.py
│  Engine loop: scheduler, KV-cache blocks, batching                   │  pytorch/engine/engine.py, engine_loop.py
│  ModelAgent: build inputs, run the model, compute logits             │  pytorch/engine/model_agent/agent.py
│  Qwen3.5 network (DeltaNet + attention + ViT)                        │  pytorch/models/qwen3_5.py
└─────────────────────────────────────────────────────────────────────┘
```

> **Note on naming.** LMDeploy does not literally have folders called "service", "protocol" and
> "engine". `AsyncEngine` (in `serve/core`) sits on the border: it is part of the service code
> but is the gateway into the real engine. The three-layer view is a way to *think*, not a folder
> structure.

---

## 3. Following one request from start to finish

Take an HTTP request with one image:

1. **(Service)** The client sends `POST /v1/decisions` with JSON. FastAPI checks the JSON against
   `DecisionRequest` and calls `serve_decision`.
2. **(Service)** `serve_decision` makes sure the server was started with
   `--task intern-decision` and that images are uploaded bytes, not server file paths. It then
   starts `engine.async_decide(request)` and also watches whether the client hangs up.
3. **(Protocol)** `DecisionRuntime.decide` checks that the queue isn't full (≤ 32 waiting) and
   waits for its lock, so only one decision runs at a time.
4. **(Protocol)** In a background thread, `DecisionInputProcessor`:
   - saves the uploaded image to a private temp folder (`uploads.py`);
   - builds the trained prompt with the answer skeleton (`schema.py`);
   - runs the Hugging Face tokenizer and image processor, giving token ids, pixel patches, and the
     **read positions** (the index just before every `<decision>`) (`encoding.py`);
   - converts the pixel patches to the format LMDeploy's Qwen3.5 expects (`multimodal.py`).
5. **(Engine)** `async_get_logits` → `score_logits` borrows a session handle and submits the
   tokens + images with `max_new_tokens=0, output_logits='all'`.
6. **(Engine)** The scheduler puts the sequence in a batch. The model agent runs the ViT on the
   image, puts the image features into the token sequence, and runs the 24 language layers. It
   then applies the **lm-head** (the final matrix that gives a score for each of the 248,320
   vocabulary entries) **at every position**.
7. **(Engine)** The full `[tokens × 248,320]` score table travels back to the CPU.
   `score_logits` checks it for NaN/Inf and keeps only the rows at the read positions. It then
   ends the session and waits for the engine to confirm it, so that the KV cache is freed.
8. **(Protocol)** `DecisionOutputProcessor` takes each row, keeps only the allowed letters
   (`A`, `B`, ...), applies softmax, and builds `{"action": {"choice": "attack", "confidence": 0.81, ...}}`.
9. **(Service)** The JSON response goes back to the client.

---

## 4. Layer by layer: what already existed, what was added, and why

### 4.1 Execution engine layer: mostly already there

**What already existed**
- `pytorch/models/qwen3_5.py`: the Qwen3.5 network, its vision encoder, its weight loading, and
  the linear-attention state cache. Intern-Decision's architecture name is
  `Qwen3_5ForConditionalGeneration`, so **the network itself needed no new code**.
- Paged KV cache, scheduler, batching.
- A way to get raw scores: `output_logits='all'`. This existed for reward models and for
  perplexity (`get_ppl`).

**What the branch changed, and why**

| Change | Commit | Why it was done |
|---|---|---|
| `async_get_logits(..., multimodal=, read_positions=)` → new `score_logits()` | [05] | The old function only accepted text token ids. Decisions need images, and need only certain rows. |
| Did **not** use `generate(max_new_tokens=0)` | [05] | In this version of LMDeploy, `generate()` returns immediately without running the model when `max_new_tokens == 0` (`async_engine.py`, the "run out of tokens" branch). So the only way to make the model run "without generating" was the logits path. |
| `Session.finish_score`, `request_handle(strict=True)`, `END_SESSION` waits for confirmation | [06] | Every request occupies an engine *session* (with KV-cache memory). If a client disconnects or times out halfway, the old code could hand the session back before the engine had actually freed it, or swallow the error. The branch makes cleanup **confirmed**, and marks a handle as unusable if its state is unknown. |
| `block_trie.py`: skip prefix caching when full logits are requested | [09a] | Prefix caching reuses already-computed KV for a shared beginning of a prompt. But then those positions are *not recomputed*, so their logits don't exist. A real bug fix for any "give me all logits" caller. |
| `get_max_session_len()` RPC | [06]/[09a] | The real token capacity can be smaller than configured after the cache is sized. The runtime checks it before submitting. |

### 4.2 Protocol layer: almost entirely new

This is the largest part, and it is genuinely necessary. The chat protocol (OpenAI messages) cannot
express "state + questions + options", and it has no notion of "read position" or "answer letters".

| File | Commit | What it does | Why it must exist |
|---|---|---|---|
| `protocol.py` | [02] | `DecisionRequest`, `DecisionResponse`, error types with HTTP codes | One definition shared by Python and HTTP, so both accept exactly the same input. |
| `schema.py` | [03] | Builds the system prompt, the option list (`A = attack: ...`), and the `<decision>` skeleton | Copied from the model authors' training code. **If one character differs, the model sees a prompt it was never trained on.** |
| `encoding.py` | [03] | Runs the HF tokenizer/processor, finds the read positions, checks every answer letter is exactly one token | It uses the checkpoint's own tokenizer and template, again so the result matches training. |
| `multimodal.py` | [03] | Splits HF pixel tensors into LMDeploy's per-image dicts (`pixel_values`, `image_grid_thw`, `offset`) | HF and LMDeploy represent images differently, so this bridges the two. It checks every image span carefully. |
| `uploads.py` | [03] | Turns base64 uploads into safe temp PNGs | Security: an HTTP client must never be able to say "read `/etc/passwd`". |
| `input.py` | [03] | Glues the above together → `PreparedDecision` | |
| `output.py` | [04] | Picks the candidate scores, softmax, and builds `choice` / `score` (expected value) / `noul` (yes probability) | Turns numbers back into business answers. |
| `runtime.py` | [05]/[09a] | Queue limit, timeouts, a lock | Protects the server from overload, and makes timeouts behave predictably. |

### 4.3 Service layer: new doors into the same room

| Entry point | Commit | Who uses it |
|---|---|---|
| `--task intern-decision` (CLI), `task=` in `pipeline()` and `serve()` | [01] | Decides whether this server is a decision server. **Explicit** on purpose: plain Qwen3.5 has the same architecture name, so LMDeploy must not guess. |
| `Pipeline.decide(request)` | [07] | Python users in the same process |
| `POST /v1/decisions` and `/v1/jev` | [07] | Remote HTTP clients. `/v1/jev` is the old name kept for compatibility. |
| `APIClient.decide()` | [08] | A Python helper for calling the HTTP API. It reads local images and uploads them as base64. |

---

## 5. "Why does it support so many request paths?"

It looks like many paths, but there is really **one** path with several doors:

```
CLI flag ─┐
pipeline()┼─► AsyncEngine(task='intern-decision')   ← configuration, done once at start-up
serve()  ─┘

Pipeline.decide ───────┐
POST /v1/decisions ────┼─► AsyncEngine.async_decide ─► DecisionRuntime ─► score_logits ─► engine
POST /v1/jev ──────────┘
APIClient.decide ─► (HTTP) ─► POST /v1/decisions
```

- **Python vs HTTP** is standard in LMDeploy: every feature (chat, embeddings, reward) has both.
  Without both, half of the users can't use the feature.
- **The CLI flag** is just how `lmdeploy serve` passes `task` to `serve()`.
- **`/v1/jev`** exists only so older clients keep working.
- **`APIClient.decide`** runs on the *client* machine. It is not a server path.

What *does* make the diff confusing is something else: the branch **also modifies shared code**
that every other feature uses (`Session.async_close`, `request_handle`, `EngineInstance.end`,
`async_cancel`, `async_get_logits`). That is where the "why is this here?" feeling comes from.
Those changes were made so that cancelling a decision is always cleaned up correctly. But they also
change behaviour for ordinary chat, reward and perplexity requests. See `REVIEW_feat-native-intern-decision.md`, item 4.

---

## 6. Sorting the changes: essential, robustness, optional

A guide for reading the diff without getting lost:

| Category | Changes | Could a first version live without it? |
|---|---|---|
| **Essential** (no answer without it) | `schema.py`, `encoding.py`, `multimodal.py`, `output.py`, `input.py`, the scoring call, `task` flag, one entry point | No |
| **Product** (useful API) | `protocol.py` pydantic models, HTTP route, `Pipeline.decide`, `APIClient.decide`, `uploads.py` | You need one of Python/HTTP. Uploads are needed for HTTP. |
| **Robustness** (behaviour under failure) | Session cleanup rework [06], strict handles, quarantine, `await_bounded_cleanup`, the disconnect watcher, queue/timeouts | Yes for a demo. No for production. |
| **Guard rails** (refuse unsupported setups) | `configure_task` checks (TP=1, batch=1, BF16, exact model size...) | Yes, and many of them exist only because of the logits approach (see the performance doc). |
| **General fixes** | `block_trie.py`, `text_config.vocab_size` in `get_ppl` | Independent of decisions. They are worth sending upstream separately. |

For comparison: a *minimal* working integration, done on newer LMDeploy in
`lmdeploy-learning/examples/intern_decision/`, is ~200 lines of example code plus one small engine
feature (`logprob_token_ids`). It reuses the checkpoint's own `inference.py` for the protocol part.
That shows the essential row above is small. Most of this branch's size is robustness, guard rails,
and tests.

---

## 7. The version problem behind many design choices

This branch is built on **LMDeploy from March 2026** (`17ed9e54`, 371 commits behind `main`).
That older version has **no "score only" prefill**. Newer LMDeploy has it:
`max_new_tokens=0` together with `logprob_start_len` runs the model once and computes the
lm-head only for the rows you ask for.

Without that feature, the branch had to:
- use `output_logits='all'`, which means scores for **every** token × **every** vocabulary entry;
- force `max_batch_size=1`, one prefill chunk, and a small cache, so this giant table stays manageable;
- add a lock so only one decision runs at a time.

Most of the confusing restrictions trace back to this. Rebasing onto current `main` would let the
branch delete a lot of code (see `PERFORMANCE_ANALYSIS.md`).

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Token** | A piece of text (a word or part of a word) given a number. The model only sees numbers. |
| **Vocabulary (V)** | All tokens the model knows: 248,320 for this model. |
| **Forward pass / prefill** | Running the network once over the whole input. |
| **Generation / decode** | Producing new tokens one at a time after the prefill. Intern-Decision does **none**. |
| **Logits** | The raw score the model gives to every vocabulary token at a position. |
| **lm-head** | The last matrix multiply that turns a position's hidden state into its V logits. |
| **Softmax** | Turns scores into probabilities that add up to 1. |
| **Temperature** | Divides scores before softmax. Above 1 makes probabilities less extreme. The official model uses ≈ 2.75. |
| **KV cache** | Memory the engine keeps for each request's tokens. It is freed when the session ends. |
| **Session** | LMDeploy's record of one conversation or request inside the engine. |
| **Prefix caching** | Reusing the KV cache of an identical beginning of a prompt from an earlier request. |
| **Continuous batching** | Running several users' requests on the GPU together, which is much more efficient than one at a time. |
| **ViT** | The vision encoder that turns images into features the language model can read. |
| **Read position** | The index just before a `<decision>` marker. The answer is read from there. |
