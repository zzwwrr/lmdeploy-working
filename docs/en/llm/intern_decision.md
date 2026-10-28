# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) answers structured questions by
**scoring** instead of generating. A request holds a `state`, up to 16 `questions` and optional images. The model
reads one prompt that ends in a JSON skeleton with a `<decision>` marker per question, and the logits right before
each marker, restricted to that question's answer symbols (`A`, `B`, ...), give the answer distribution.

LMDeploy serves it with the `intern-decision` task on the PyTorch engine. The checkpoint itself is an ordinary
Qwen3.5 vision-language model; the task adds the protocol on top. The decision protocol and preprocessing are
adapted from the Apache-2.0 [Intern-Decision project](https://github.com/InternLM/Intern-Decision); adapted files
keep their attribution.

Decision requests are currently supported through HTTP only; the Python Pipeline has no Decision entry point.

## HTTP

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task Intern-Decision-0.8B
```

`--task` takes the checkpoint name, which selects both the calibrated temperature and the PyTorch backend. All PyTorch engine options (batch size, TP, prefix caching,
...) remain available. Decision requests use input scoring and generate no tokens.

`POST /v1/decisions` (alias `POST /v1/jev`) takes a JSON body containing `state`, `questions`, and optional `images`. For example, upload a client-local image:

```shell
IMG=$(base64 -w0 crossing.png)
curl http://127.0.0.1:23333/v1/decisions -H 'Content-Type: application/json' -d @- <<EOF
{"state": "Inspect the image.",
 "images": ["data:image/png;base64,$IMG"],
 "questions": {"red": {"type": "noul", "instructions": "Is a red light visible?"}}}
EOF
```

`images` uses the same `ImageDataFormat` as `/generate.image_data`: a string, a parameter dictionary such as
`{"url": "..."}`, or a list mixing both. Strings may be HTTP(S) URLs, complete base64 data URLs, or local paths.
Omit `images` or pass `[]` for text-only input. Dictionary contents are checked by shared multimodal preprocessing,
rather than a separate Decision image schema. Additional parameters are forwarded to that path; their effects
depend on the existing processor.
**Migration:** pass references such as `"https://..."` or `{"url": "https://..."}`, not complete
`{"type": "image_url", "image_url": {...}}` message blocks; the compiler wraps each reference once.
The upstream `{"type": "image/png", "data": "<raw base64>"}` upload format is not adapted: encode uploads as
complete data URLs instead.
Local paths refer to files on the machine running LMDeploy. Decision HTTP requests use the same multimodal
loader as chat, including its URL checks and `allowed_media_domains` configuration. Image formats, base64 decoding,
and animation handling follow that loader; there are no additional Decision-specific image count, upload-byte,
or original-pixel limits. The checkpoint processor controls image resizing and visual token expansion. Pixels use
their stored orientation; the loader does not apply EXIF rotation.

Errors use the OpenAI error envelope: 400 for invalid input, including schema violations, and 503 when the task is
not enabled or images are sent to an engine without a vision encoder.

See also [the HTTP client example](../../../examples/decision/decision.py).

## Response

Shape of a response (values are illustrative):

```json
{
  "request_id": "...", "model": "...",
  "answers": {
    "action": {"type": "choice", "probabilities": {"stop": 0.83, "go": 0.17}, "confidence": 0.83,
               "choice": "stop", "source": "local", "decision": "stop"},
    "risk":   {"type": "score", "probabilities": {"0": 0.12, "1": 0.30, "2": 0.58}, "confidence": 0.58,
               "score": 1.46, "legend": {"0": "Safe", "1": "Some risk", "2": "Dangerous"}, ...},
    "red":    {"type": "noul", "probabilities": {"no": 0.08, "yes": 0.92}, "confidence": 0.92, "noul": 0.92, ...}
  },
  "usage": {"input_tokens": 412, "output_tokens": 3, "decision_count": 3, "generated_tokens": 0},
  "timing": {"preprocess_ms": 4.1, "inference_ms": 38.0, "server_ms": 43.2},
  "calibration": {"method": "temperature-scaling", "temperature": 2.747760550703}
}
```

Probabilities use the same candidate softmax and temperature calibration as the checkpoint's `inference.py`
(`2.747760550703` for Intern-Decision-0.8B). Different inference kernels and floating-point dtypes can introduce
small numerical differences from the official `DecisionEngine`; this is not a claim of bitwise-equivalent model
outputs. `score` is the probability-weighted mean of the numeric option keys (or list indices); `noul` is the
probability of `yes`. `usage.output_tokens` is the number of answered questions for protocol compatibility;
`usage.generated_tokens` is always zero.

## How it runs

A decision runs in the decision endpoint, which drives the engine directly and owns the request session. `DecisionParser`
converts the decision protocol to engine input and interprets the scoring output; it performs no engine execution.
There is no separate `DecideRunner`, no shared-runner hooks, no decision-owned vision pipeline, and no independently
loaded `DecisionEncoder`:

1. `compile_row` compiles `state` and `questions` into messages with the trained assistant skeleton, wraps image
   references as `image_url` blocks as `/generate` does, and places them before the user text.
   `DecisionParser` passes the complete messages to preprocessing.
   Shared startup metadata validates marker/answer token IDs using the
   engine's existing tokenizer; it does not load a second tokenizer or processor.
2. The shared preprocessing path renders the checkpoint template with `enable_thinking=False`, `add_vision_id=True`
   and no generation prompt. It preserves the assistant turn's closing tokens. Text uses the engine tokenizer without
   extra BOS/special tokens; images use the already loaded Qwen3.5 processor to tokenize and expand visual tokens.
3. After preprocessing, the parser locates `<decision>` in the final input IDs and sets `logprob_start_len` to the
   position immediately before the first marker. It saves each question's scoring row index and the expected
   row count. Image token counts need not be predicted beforehand.
4. `GenerationConfig(max_new_tokens=0, logprobs=0, logprob_extra_token_ids=...)` requests input scoring. The engine projects
   the LM head for the selected suffix and returns the requested 62 candidate scores on those rows, independent of
   top-k rank. This limits the returned scores, not the vocabulary dimension of the LM-head projection.
5. The parser validates the returned row count, selects rows using the saved indices and each question's candidate
   symbols, applies
   softmax and temperature calibration, and constructs the response. Raw logits and raw log probabilities both
   support this candidate normalization.

This is prefill-only execution, possibly split into prefill chunks; it never enters autoregressive decode. Scheduling,
session cleanup and cancellation reuse `ServerContext.create_session` and the shared request-cleanup helper.
Prefix reuse stops before the requested scoring positions so those scores are recomputed.

Decision image references are wrapped into message blocks without decoding them; shared preprocessing loads them.
The loader closes images that it creates after visual preprocessing, including partial failures, and waits for
outstanding worker threads before closing images on cancellation.

## Limitations

- PyTorch backend only; speculative decoding and non-hybrid (disaggregated) roles are rejected.
- After image expansion, input must be shorter than the engine's session length,
  which is the model's own context window unless overridden by `--session-len`.
  Longer inputs are rejected, never truncated.
- Requests contain 1–16 questions. Each `choice` or `score` question has 1–62 options, assigned
  answer symbols in request order. The original option labels are restored in the response.
- The `model` field may be omitted, or be the served model name (`--model-name`, default: the model path) or the
  checkpoint directory name (e.g. `Intern-Decision-0.8B`, as clients of the original service send).
- With `--dp > 1`, call each rank's server directly: the DP proxy does not route `/v1/decisions`.
