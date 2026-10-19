# Intern-Decision on LMDeploy, Explained in Three Layers (v2)

A beginner's tutorial. It explains how LMDeploy is organised, how Intern-Decision is supported at each level on the
`feat/intern-decision-v2` branch, and why v2 looks so much smaller than v1 (`feat/native-intern-decision`).

> You only need a rough idea of "a request comes in, the model runs, an answer goes out".

---

## 0. The one-paragraph version

LMDeploy can be read as three layers:

| Layer | Question it answers | Restaurant analogy |
|---|---|---|
| **Service** | How does a request reach us and how does the answer leave? | The front door, the waiter, the phone line |
| **Protocol** | What does the request mean, and how do the model's numbers become an answer? | The menu and the order ticket |
| **Execution engine** | How do we run the network on the GPU efficiently, for many requests at once? | The kitchen |

Intern-Decision does not write text, it *picks answers*. Its network is an ordinary Qwen3.5 vision-language model that
LMDeploy already runs. So supporting it means a **new menu** (protocol), a **new door** (endpoint), and asking the
kitchen to **taste, not cook** (score the prompt, generate nothing). In v2 the kitchen already knows how to "taste";
we only had to teach it to report the taste of 62 specific ingredients.

---

## 1. What makes Intern-Decision special

A chat request: `"What is 2+2?" → the model writes "4"`, one token at a time.

An Intern-Decision request (the "Jev" format):

```json
{"state": {"hp": 30, "enemy": "dragon"},
 "questions": {"action": {"type": "choice", "criteria": {"attack": "...", "flee": "..."}},
               "danger": {"type": "score",  "criteria": ["low", "mid", "high"]}}}
```

The model is given a prompt that already contains the answer skeleton:

```
...state, and the options listed as  A = attack: ...  B = flee: ...
{
    "action": <decision>,
    "danger": <decision>
}
```

It runs **once** over this text. At the position right before each `<decision>`, we read the model's scores for the
allowed letters (`A`, `B`, ...), turn them into probabilities, and calibrate them with a fixed temperature (≈ 2.75)
shipped with the model. No text is generated.

So the model needs: (1) the **exact** trained prompt, (2) scores at **specific positions** for **specific tokens**,
(3) zero generated tokens.

---

## 2. The three layers in v2

```
┌──────────────────────────── SERVICE LAYER ─────────────────────────────┐
│ CLI:    lmdeploy serve api_server CKPT --task intern-decision          │ cli/serve.py
│ Server: serve(task=...) → ServerContext.decision_service               │ serve/openai/api_server.py
│ HTTP:   POST /v1/decisions  (alias /v1/jev)                            │ serve/openai/endpoints/decisions.py
│ Python: pipe = pipeline(CKPT, task='intern-decision'); pipe.decide(r)  │ api.py, pipeline.py
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  every door ends here ↓
                     DecisionService.decide(request)
┌──────────────────────────── PROTOCOL LAYER ────────────────────────────┐
│ protocol.py  what a request/response may contain; error → HTTP code    │ serve/processors/intern_decision/
│ images.py    local paths (Python) / base64 uploads → in-memory images  │
│ schema.py    request → trained prompt + "<decision>" skeleton           │
│ encoding.py  prompt → token ids, "tail" length, answer-letter token ids│
│ service.py   ask the engine to score the skeleton; pick decision rows  │
│ output.py    letter logits → softmax → temperature → choice/score/noul │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  AsyncEngine.preprocess() → generate()   (unchanged)
┌──────────────────────────── EXECUTION ENGINE LAYER ────────────────────┐
│ AsyncEngine / VLAsyncEngine: sessions, image expansion, cleanup        │ serve/core/
│ Scheduler, paged KV cache, continuous batching, prefix cache           │ pytorch/engine, pytorch/paging
│ ModelAgent: forward, LM head on the selected rows only                 │ pytorch/engine/model_agent/agent.py
│   + NEW: logprob_token_ids → also return these tokens' logits          │
│ Qwen3.5 network (Gated DeltaNet + attention + ViT)                     │ pytorch/models/qwen3_5.py
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Following one request

An HTTP request with one image:

1. **Service.** `POST /v1/decisions`. FastAPI checks the body against `DecisionRequest` and calls
   `DecisionService.decide(request, allow_local_paths=False)`. HTTP may only upload bytes, so it can't read the
   server's files.
2. **Protocol.** In a worker thread (so the event loop stays free):
   - `images.py` decodes the base64 image in memory and converts it to RGB. Like the official code, it uses the
     pixels as stored and does not apply EXIF rotation.
   - `schema.py` builds the trained prompt.
   - `encoding.py` renders it with the checkpoint's own chat template. The image is still a single `<|image_pad|>`
     token at this point. The encoder records the **tail**: how many tokens there are from the row before the first
     `<decision>` to the end.
3. **Protocol → engine.** `service.py` calls `engine.preprocess(...)` with the token ids and the image. The engine's
   Qwen3.5 processor expands `<|image_pad|>` into the image's real number of patch tokens. Images always come
   *before* the skeleton, so the tail is unchanged, and the first decision row is at `expanded_length − tail`.
4. **Engine.** The service sets three things:
   - `max_new_tokens=0`: generate nothing.
   - `logprob_start_len = expanded_length − tail`: only score from here on.
   - `logprob_token_ids = [ids of A…9]`: report these tokens' logits.

   It then calls `engine.generate(...)`. The scheduler batches this prefill with other work. The model agent runs the
   ViT and the 24 language layers over the whole prompt, but applies the **LM head** (the 1024 × 248,320 projection)
   only to the ~6–160 skeleton rows. For every row it returns the target token's logit plus the 62 requested ones.
5. **Engine → protocol.** `generate()` cleans up its own session, including on errors and cancellation. The service
   keeps the rows whose target token is `<decision>`, one per question, and takes each question's own letters.
6. **Protocol.** `output.py` applies softmax over the letters, then the calibration temperature, then builds
   `choice` / `score` / `noul`.
7. **Service.** The JSON goes back. The response includes `calibration` and `usage.generated_tokens = 0`.

---

## 4. Layer by layer: what existed, what v2 adds, and why

### 4.1 Execution engine: one small addition

**Already in `main`:**
- Qwen3.5 network, ViT, and weight loading. The checkpoint's architecture `Qwen3_5ForConditionalGeneration` is
  already registered, so **no model code changed**.
- **Scoring-only prefill.** `max_new_tokens=0` + `logprob_start_len` computes the LM head only on chosen rows
  (`inputs_maker.fill_logits_indices`).
- **Prefix-cache safety.** A cached prefix is never reused past `logprob_start_len`
  (`get_prefix_cache_max_candidate_step`).
- **Request lifecycle.** `generate()` + `safe_run` + `request_cleanup` cancel engine work and remove sessions on
  error or disconnect.

**Added (commit 1):** `GenerationConfig.logprob_token_ids`.

| File | Change |
|---|---|
| `messages.py` | the new field and its validation (needs `logprob_start_len >= 0`) |
| `pytorch/messages.py` | carried on `SamplingParam` |
| `strategies/ar/sampling.py` | sorted **union** over the batch: all rows share one column layout on the GPU |
| `engine/engine_loop.py` | trims each request's rows back to the ids **it** asked for, so other requests in the batch (e.g. `/generate` top-k) see no extra columns |
| `engine/logits_process.py` | `SamplingInputs.logprob_token_ids` |
| `model_agent/agent.py` | `_get_outputs_with_logprobs` appends `input_logits.gather(-1, ids)` columns |

*Why it's needed:* without it, a scored row only returns the target token plus the top-k. The answer letters are
usually not in the top-k, and we need their exact logits to compute probabilities.

### 4.2 Protocol: the real work (commit 2)

| File | What it does | Why it must exist |
|---|---|---|
| `protocol.py` | `DecisionRequest`/`DecisionResponse`, errors with HTTP codes | One contract for Python and HTTP |
| `schema.py` | trained system prompt, option list, `<decision>` skeleton | Copied from the authors' code. A one-character difference gives the model a prompt it never saw in training. |
| `encoding.py` | checkpoint chat template → ids; tail; letter ids; guards (marker count, image slot count) | Same template and tokenizer as training. Verified to match the official encoder token for token. |
| `images.py` | one loader for local paths and uploads, in memory | Security for HTTP. The same pixels whichever door they came through. |
| `output.py` | softmax + **temperature calibration** + answer building | Copies `inference.py` exactly, so probabilities match the official engine |
| `service.py` | the scoring call and row picking | The only place that talks to the engine |

### 4.3 Service: doors into the same room (commit 3)

| Entry | Who uses it |
|---|---|
| `pipeline(..., task='intern-decision')` → `Pipeline.decide()` | Python users in the same process. Local image paths are allowed because the caller is trusted. |
| `serve(task=...)`, `--task intern-decision` | starts a server whose `ServerContext` holds the `DecisionService` |
| `POST /v1/decisions`, `/v1/jev` | remote clients. `/v1/jev` is the original service's name, kept for compatibility. |

The task is **explicit**. A plain Qwen3.5 checkpoint has the same architecture name, so LMDeploy must not guess.
Routing is otherwise untouched: the checkpoint still gets `VLAsyncEngine`, so images work, and the server still serves
the OpenAI endpoints on the same engine.

---

## 5. "Why are there so many request paths?"

There is one path with several doors:

```
pipeline(task=…) ─┐                                      ┌─ Pipeline.decide()
serve(task=…)    ─┼─ build DecisionService(engine) once ─┤
--task (CLI)     ─┘                                      └─ POST /v1/decisions, /v1/jev
                                         all → DecisionService.decide → engine.preprocess → engine.generate
```

Python and HTTP both exist because every LMDeploy feature has both. `/v1/jev` is an alias. The CLI flag is just how
`lmdeploy serve` passes `task` to `serve()`. Unlike v1, **v2 does not modify shared session or engine-instance
code**, so ordinary chat, reward and perplexity requests behave exactly as on `main`.

---

## 6. v1 → v2: what disappeared and why

| v1 had | Why v1 needed it | v2 |
|---|---|---|
| `output_logits='all'` + `scoring.py` | its base had no scoring-only prefill | `logprob_start_len` + `logprob_token_ids` |
| Batch size 1, a global lock, prefill fixed at 8192, cache 0.05 | to keep the L × 248,320 table affordable | removed; normal engine settings |
| `Session.finish_score`, strict handles, END acknowledgements, quarantine | its custom scoring path had to clean up by itself | `generate()` already does this |
| `block_trie.py` prefix-cache guard | full logits are lost on a cache hit | `main` caps reuse at `logprob_start_len` |
| `multimodal.py` HF→LMDeploy pixel bridge | it ran the HF processor itself | the engine's processor expands images |
| `uploads.py` temp files | the HF processor read paths | in-memory `images.py` |
| no calibration | (documented as unsupported) | official temperature applied and reported |
| `APIClient.decide` | client helper | `api_client.py` no longer exists on `main`, so clients post JSON (see the example) |

Rough size: v1 changed ~1.2k lines of source, including shared engine/session files. v2 changes ~30 lines of engine
code, adds a self-contained package of ~700 lines (docstrings included), and ~120 lines of service wiring.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| Token | a piece of text turned into a number |
| Vocabulary (V) | all tokens: 248,320 here |
| Prefill / forward pass | running the network once over the whole input |
| Decode / generation | producing new tokens one by one (Intern-Decision does none) |
| Logits | the model's raw score for every vocabulary token at a position |
| LM head | the last matrix that turns a position's hidden state into V logits |
| Scoring-only prefill | a prefill that returns scores for chosen rows and generates nothing |
| `logprob_start_len` | where in the prompt the scored rows start |
| `logprob_token_ids` | extra tokens whose scores are returned on every scored row (new in v2) |
| Tail | the tokens from the first decision row to the end of the prompt |
| Softmax | turns scores into probabilities summing to 1 |
| Temperature | divides log-probabilities before re-normalising. ≈ 2.75 makes the model's confidence honest. |
| KV cache / prefix cache | memory of processed tokens / reusing it for identical prompt beginnings |
| Continuous batching | running many requests on the GPU together |
