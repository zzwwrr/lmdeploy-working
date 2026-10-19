# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) 通过**打分**而不是生成来回答结构化问题。一个请求包含
`state`、最多 16 个 `questions` 和最多 8 张图片。模型读取一个以 JSON 骨架结尾的提示，骨架中每个问题对应一个 `<decision>`
标记；取每个标记前一位置的 logits，并限定在该问题的答案符号（`A`、`B`……）上，即得到答案分布。

LMDeploy 在 PyTorch 引擎上以 `intern-decision` 任务提供该模型。checkpoint 本身是普通的 Qwen3.5 视觉语言模型，任务只是在其上增加协议层。
决策协议与预处理改编自 Apache-2.0 许可的 [Intern-Decision 项目](https://github.com/InternLM/Intern-Decision)，改编文件保留了署名。

## Python

```python
from lmdeploy import pipeline

pipe = pipeline('/path/to/Intern-Decision-0.8B', task='intern-decision')
result = pipe.decide({
    'state': {'signal': 'red', 'pedestrians': True},
    'images': ['/trusted/local/crossing.jpg'],          # 可选：本地路径或上传对象
    'questions': {
        'action': {'type': 'choice', 'instructions': 'Select an action.',
                   'criteria': {'stop': 'Stop at the light.', 'go': 'Drive through.'}},
        'red': {'type': 'noul', 'instructions': 'Is the signal red?'},
    },
})
print(result['answers'])
pipe.close()
```

## HTTP

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task intern-decision
```

`--task intern-decision` 会选择 PyTorch 后端，其余 PyTorch 引擎参数（批大小、TP、前缀缓存等）照常可用，服务同时提供常规的 OpenAI 接口。

`POST /v1/decisions`（别名 `POST /v1/jev`）接收相同的 JSON。通过 HTTP 时图片必须以上传对象给出：
`{"type": "image/png", "data": "<base64 或 data URL>"}`。服务器端路径和远程 URL 会被拒绝。支持 JPEG、PNG、WebP 与静态 GIF，
单张不超过 12 MB、合计不超过 32 MB、每张不超过 1600 万像素；与 checkpoint 的 `inference.py` 一致，像素按存储方向使用，**不会**应用 EXIF 旋转，请先自行旋转照片；Python 本地路径使用同一加载逻辑。

错误使用 OpenAI 错误格式：输入无效返回 400，字段不符合协议返回 422，待处理请求过多返回 429，未启用任务或引擎没有视觉编码器却收到图片时返回 503。

## 响应与校准

每个答案包含 `probabilities`、`confidence`、`decision`，以及按类型给出的 `choice`、`score`（按概率加权的数值期望，附 `legend`）或
`noul`（`yes` 的概率）。概率使用 checkpoint 中 `inference.py` 自带的温度（2.747760550703）校准，与官方 `DecisionEngine` 一致，响应中的
`calibration` 字段注明了该温度。`usage.generated_tokens` 恒为 0。

## 运行方式

一次决策就是共享引擎上的一次“只打分”预填充：按 checkpoint 的对话模板渲染请求（图片在内存中，由引擎的 Qwen3.5 处理器展开）；
`GenerationConfig(max_new_tokens=0, logprob_start_len=...)` 让引擎只在答案骨架上计算 LM head，`logprob_token_ids` 返回这些行上
62 个答案符号的 logits；目标 token 为 `<decision>` 的行即决策行。决策请求与其他请求一样由调度器批处理；前缀缓存是安全的，因为引擎不会复用超过
`logprob_start_len` 的缓存前缀。

## 限制

- 仅支持 PyTorch 后端；不支持投机解码与 PD 分离角色。
- 不支持 `thinking.enabled=true`。
- 图片展开后超过 8192 token 或超过引擎会话长度的输入会被拒绝，不会截断。
- `model` 字段可省略，或填写服务模型名（`--model-name`，默认为模型路径）或 checkpoint 目录名（如 `Intern-Decision-0.8B`）。
- `--dp > 1` 时请直接访问各 rank 的服务：DP 代理不转发 `/v1/decisions`。
