# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) answers structured questions by
**scoring** instead of generating. A request holds a `state`, up to 16 `questions` and up to 8 images. The model
reads one prompt that ends in a JSON skeleton with a `<decision>` marker per question, and the logits right before
each marker, restricted to that question's answer symbols (`A`, `B`, ...), give the answer distribution.

LMDeploy serves it with the `intern-decision` task on the PyTorch engine. The checkpoint itself is an ordinary
Qwen3.5 vision-language model; the task adds the protocol on top. The decision protocol and preprocessing are
adapted from the Apache-2.0 [Intern-Decision project](https://github.com/InternLM/Intern-Decision); adapted files
keep their attribution.

## Python

```python
from lmdeploy import pipeline

pipe = pipeline('/path/to/Intern-Decision-0.8B', task='intern-decision')
result = pipe.decide({
    'state': {'signal': 'red', 'pedestrians': True},
    'images': ['/trusted/local/crossing.jpg'],          # optional, local paths or upload objects
    'questions': {
        'action': {'type': 'choice', 'instructions': 'Select an action.',
                   'criteria': {'stop': 'Stop at the light.', 'go': 'Drive through.'}},
        'risk': {'type': 'score', 'instructions': 'How risky is driving through?',
                 'criteria': {'0': 'Safe', '1': 'Some risk', '2': 'Dangerous'}},
        'red': {'type': 'noul', 'instructions': 'Is the signal red?'},
    },
})
print(result['answers']['action']['choice'], result['answers']['risk']['score'])
pipe.close()
```

See also [the runnable example](../../../examples/decision/decision.py).

## HTTP

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task intern-decision
```

`--task intern-decision` selects the PyTorch backend. All PyTorch engine options (batch size, TP, prefix caching,
...) remain available, and the server keeps serving the usual OpenAI endpoints on the same engine.

`POST /v1/decisions` (alias `POST /v1/jev`) takes the same JSON body. Over HTTP every image must be uploaded:

```shell
IMG=$(base64 -w0 crossing.png)
curl http://127.0.0.1:23333/v1/decisions -H 'Content-Type: application/json' -d @- <<EOF
{"state": "Inspect the image.",
 "images": [{"type": "image/png", "data": "$IMG"}],
 "questions": {"red": {"type": "noul", "instructions": "Is a red light visible?"}}}
EOF
```

`data` may also be a data URL (`data:image/png;base64,...`). Server-side paths and remote URLs are rejected. JPEG,
PNG, WebP and static GIF are accepted, up to 12 MB each, 32 MB in total and 16 megapixels per image. EXIF rotation is
applied and metadata is dropped; the same loader handles local paths in Python.

Errors use the OpenAI error envelope: 400 for invalid input, 422 for schema violations, 429 when too many decisions
are pending, 503 when the task is not enabled or images are sent to an engine without a vision encoder.

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
  "thinking": {"enabled": false, "confidence_threshold": 0.7, "tasks": []},
  "calibration": {"method": "temperature-scaling", "temperature": 2.747760550703}
}
```

Probabilities are calibrated with the temperature shipped in the checkpoint's `inference.py`, so they match the
official `DecisionEngine`. `score` is the probability-weighted mean of the numeric option keys (or list indices);
`noul` is the probability of `yes`.

## How it runs

A decision is one scoring-only prefill on the shared engine:

1. The request is rendered with the checkpoint's chat template (`enable_thinking=False`, `add_vision_id=True`, no
   generation prompt). Images stay in memory and are expanded by the engine's Qwen3.5 processor.
2. `GenerationConfig(max_new_tokens=0, logprob_start_len=...)` makes the engine project the LM head only on the
   answer skeleton, and `logprob_token_ids` returns the logits of the 62 answer symbols on those rows.
3. The rows whose target token is `<decision>` are the decision rows; each question takes its own symbols, applies
   softmax and the calibration temperature.

Decisions are batched with other requests by the scheduler like any prefill. Prefix caching is safe: the engine never
reuses a cached prefix past `logprob_start_len`.

## Limitations

- PyTorch backend only; speculative decoding and non-hybrid (disaggregated) roles are rejected.
- `thinking.enabled=true` is not supported.
- Inputs above 8192 tokens (after image expansion) or the engine's session length are rejected, never truncated.
- In a mixed batch, a request's input-logprob rows may carry extra candidate columns requested by others;
  `/generate` top-k output is unaffected because it sorts and truncates.
