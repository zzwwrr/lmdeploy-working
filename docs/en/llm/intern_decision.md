# Intern-Decision

LMDeploy provides an explicit `intern-decision` task for Intern-Decision-0.8B.
It uses the existing Qwen3.5 visual and language networks in the PyTorch engine.
The decision protocol and CPU preprocessing are adapted from the Apache-2.0
[Intern-Decision project](https://github.com/InternLM/Intern-Decision), revision
`0a8a23882e04918d430b85b6c34fd13e9a32f9a5`; adapted files retain attribution.
No external Intern-Decision installation or `src` package is needed at runtime.

## Python

```python
from lmdeploy import pipeline

pipe = pipeline('/path/to/Intern-Decision-0.8B', task='intern-decision')
try:
    result = pipe.decide({
        'state': {'signal': 'red'},
        'questions': {
            'action': {
                'type': 'choice',
                'instructions': 'Select an action.',
                'criteria': {'stop': 'Stop at the red light.', 'go': 'Drive through the red light.'},
            },
        },
    })
    print(result['answers'])
finally:
    pipe.close()
```

`Pipeline.decide` submits to the pipeline's existing event loop. Applications that
already own an `AsyncEngine` can `await engine.async_decide(request)` on its owning
loop. Both return JSON-compatible dictionaries. `task=None` preserves ordinary
model routing; architecture names and `<decision>` tokens do not select this task
automatically. No decision metadata auto-detection is implemented.

Task defaults are BF16, TP/DP/EP=1, eager mode, batch size 1, prefill size 8192,
session length 8192, prefix caching disabled, and cache fraction 0.05. Decision serving uses the existing PyTorch engine
in the API-server process; `enable_mp_engine=True` is rejected.
Explicit backend configurations must satisfy these constraints; unsupported
combinations fail during initialization. A smaller session length is permitted
and further restricts inputs. Current support is limited to the validated 0.8B
configuration; quantization, adapters, speculative decoding, video, and larger
model sizes are unsupported. Full vocabulary logits can require substantial
memory for long inputs; requests are never shortened to fit memory.

See [the runnable example](../../../examples/decision/decision.py).

## HTTP and client

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B \
  --backend pytorch --task intern-decision --dtype bfloat16 \
  --eager-mode --max-batch-size 1 --max-prefill-token-num 8192 \
  --session-len 8192 --cache-max-entry-count 0.05 \
  --server-name 127.0.0.1 --server-port 23333
```

`POST /v1/decisions` and `POST /v1/jev` invoke the same handler. These are custom
APIs; they are not OpenAI-standard decision endpoints. Normal LMDeploy server
API key authentication also applies to both routes. The routes share the existing
model and engine with chat, encode, and other supported interfaces.

```python
from lmdeploy.serve.openai.api_client import APIClient

client = APIClient('http://127.0.0.1:23333', api_key='your-server-key')
result = client.decide({
    'state': 'Inspect the image.',
    'images': ['/trusted/client/local/image.png'],
    'questions': {'red': {'type': 'noul', 'instructions': 'Is a red light visible?'}},
})
```

The client reads image paths on the **client** and uploads bytes. It returns
ordinary JSON and raises `requests.HTTPError` for unsuccessful HTTP responses.
Its default HTTP timeout is 300 seconds, covering queue and execution defaults.
Direct HTTP accepts image objects such as
`{"type": "image/png", "data": "data:image/png;base64,..."}` (replace `...` with
actual encoded bytes). Raw base64 with an explicit MIME type is also accepted.
HTTP rejects server paths and remote URLs. Optional image names are never paths.
Python `Pipeline.decide` additionally accepts trusted local paths. Remote URLs
are unsupported in both modes.

## Protocol v1

`state` is required and contains JSON data. `questions` is an ordered object of
1–16 named fields. Each question has `type`, optional `instructions`, and
`criteria` as described below. Input order determines answer-symbol order, using
`A..Z a..z 0..9`; each symbol must encode to one token. Training `answer` and
`targets` are ignored. Reserved `<decision>` markers in prompt content are rejected.

| Type | Criteria | Returned value |
| --- | --- | --- |
| `choice` | Object of 1–62 business values to descriptions | `choice`: highest-probability business value |
| `score` | Object of finite numeric keys, or list of 1–62 descriptions | `score`: probability-weighted expectation; list values are `0..N-1`; `legend` preserves descriptions |
| `noul` | Optional yes/no descriptions | `noul`: probability of yes, with candidate order no then yes |

Each answer contains `type`, `probabilities`, `confidence` (largest candidate
probability), `decision` (discrete best business value), and `source='local'`.
Ties use lexicographic business-value order. Probabilities use FP32 softmax over
legal candidates only, at temperature 1; calibration is unsupported. In
particular, score `decision` differs from its expectation and noul `confidence`
can differ from its probability of yes.

The response contains `request_id`, `model`, `answers`, `usage`, `timing`, and
`thinking`. Model defaults to `intern-decision`, or the configured serving alias.
An explicitly supplied model must match. Thinking is disabled; `enabled=true`
is rejected. `confidence_threshold` defaults to 0.7 and is compatibility metadata
only; it must be a finite non-boolean number in [0,1].

Usage reports `input_tokens`, `output_tokens=question_count`,
`decision_count=question_count`, and `generated_tokens=0`. The compatibility
`output_tokens` field counts decisions, not generated tokens, and does not
increment generation metrics. Timing uses the same boundaries in Python/HTTP:
`queue_ms` runs from core admission entry to acquisition of the serial execution
slot; `inference_ms` covers CPU preparation, prefill, selection and confirmed
session cleanup; `server_ms` ends after output processing. HTTP decoding and
network transfer are outside these timings.

Final expanded input length must not exceed 8192, configured session length,
or the engine's actual cache-derived capacity. There is no truncation or padding.
Up to eight static images are supported. HTTP limits are 12 MiB per encoded
upload, 32 MiB raw total, 32 MiB normalized total, and 16 million pixels per image.
JPEG, PNG, WebP and static GIF are accepted; animation, MIME mismatch, invalid
base64 and decompression bombs are rejected. Uploads are EXIF-transposed, converted
to RGB and stripped of metadata. Trusted Python paths follow the historical RGB
conversion without EXIF transposition. These transport-specific semantics are
intentional. Video and arbitrary image/text interleaving are unsupported.

## Admission, errors and shutdown

The task accepts at most 32 pending requests, including one running request.
Configure Python initialization with
`decision_config={'max_pending_requests': 8, 'queue_timeout': 120,
'inference_timeout': 120, 'cleanup_timeout': 10}` as needed. Queue wait and execution have separate timeout bounds; `inference_timeout`
covers CPU preparation and prefill together, followed by bounded cleanup. Shared processor access is serialized, and
CPU work runs outside the event loop. Cancelling a request waits for bounded
session cleanup; images remain owned by the CPU worker until it exits.

| HTTP status | Meaning |
| --- | --- |
| 400 | Invalid encoded input, image upload, or final token length |
| 401 | Missing/invalid configured API key |
| 404 | Unknown model |
| 422 | Request schema validation failure |
| 429 | Admission full or queue wait timeout |
| 499 | Client disconnected (when a response is still possible) |
| 500 | Execution, output validation or cleanup failure |
| 503 | Task disabled, closing, or engine quarantined after cleanup failure |
| 504 | CPU preparation or inference timeout |

Disconnect and caller cancellation are forwarded to the shared operation. END_SESSION
must be acknowledged before returning a handle. Cleanup failure quarantines the
handle and makes subsequent decisions unavailable; restart that engine. Pipeline
and server shutdown cancel active decisions before stopping the engine. No
additional model, inference backend thread or event loop is created for decisions.

## Validation and limitations

CPU tests are in `tests/test_lmdeploy/test_decision*.py`. Set
`DECISION_TEST_CHECKPOINT` for tokenizer/processor golden and MRoPE checks. The
separate real-weight test additionally requires `DECISION_TEST_GPU=1` and an
explicit authorized GPU selection. Fixture expectations were captured with
checkpoint revision `85a0cc5a99d67ea8d56dfe98115689212867171d`.

See [the validation report](intern_decision_validation.md) for measured results.
Interface migration and cross-backend numerical acceptance are separate checks.
Do not treat this task as having passed HF numerical parity or general performance
acceptance. Sparse LM-head projection remains deferred; positions are selected
after full `L x V` projection. Generic PyTorch full-logits scoring also bypasses
shared prefix matching and allocation, because cached KV state cannot reproduce omitted prefix
logits. Ordinary generation retains its configured prefix-cache policy.


Process proxies are excluded from the initial decision task. The existing proxy
marks session readiness before the remote inference coroutine has submitted its
session. Combined with cancellation of its RPC stream, an early END_SESSION
response cannot prove that no later session allocation remains. Decision HTTP
therefore uses the existing in-process PyTorch engine and its owning server loop.
The explicit token-capacity API also remains available to ordinary proxy engines.
