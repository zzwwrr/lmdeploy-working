# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) 通过**打分**而不是生成来回答结构化问题。一个请求包含
`state`、最多 16 个 `questions` 和可选图片。模型读取一个以 JSON 骨架结尾的提示，骨架中每个问题对应一个 `<decision>`
标记；取每个标记前一位置的 logits，并限定在该问题的答案符号（`A`、`B`……）上，即得到答案分布。

LMDeploy 在 PyTorch 引擎上以 `intern-decision` 任务提供该模型。checkpoint 本身是普通的 Qwen3.5 视觉语言模型，任务只是在其上增加协议层。
决策协议与预处理改编自 Apache-2.0 许可的 [Intern-Decision 项目](https://github.com/InternLM/Intern-Decision)，改编文件保留了署名。

当前仅通过 HTTP 接收 Decision 请求，暂不支持 Python Pipeline 的 Decision 入口。

## HTTP

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task Intern-Decision-0.8B
```

`--task` 取 checkpoint 名称，同时用于选择该模型的校准温度和 PyTorch 后端，其余 PyTorch 引擎参数（批大小、TP、前缀缓存等）照常可用。决策请求只做输入评分，不生成 token。

`POST /v1/decisions`（别名 `POST /v1/jev`）接收包含 `state`、`questions` 和可选 `images` 的 JSON。
`images` 复用 `/generate.image_data` 的 `ImageDataFormat`：接受字符串、`{"url": "..."}` 参数字典，或二者混合的列表。
例如 `"images": ["data:image/png;base64,...", {"url": "https://example.com/image.png"}]`；单图也可以直接传字符串或字典。
字符串支持 HTTP(S) URL、完整 base64 data URL 或本地路径。省略 `images` 或传 `[]` 表示纯文本输入。
字典内部字段交给公共多模态处理器检查，额外参数原样传递，其效果以已有处理器为准。
**迁移说明：** 传图片引用，不传完整 `{"type": "image_url", "image_url": {...}}` 消息块；compiler 会统一包装一次。
上游 `{"type": "image/png", "data": "<裸 base64>"}` 上传格式不做兼容转换，上传内容请先编码为完整 data URL。
本地路径指运行 LMDeploy 的机器上的文件。Decision HTTP 请求复用 chat 的多模态加载器，包括 URL 检查与 `allowed_media_domains` 配置。
图片格式、base64 解码和动画处理遵循框架已有行为，不额外设置 Decision 专属的图片张数、上传字节数或原图像素数限制。
checkpoint processor 负责缩放和视觉 token 展开；加载器按存储方向读取像素，不应用 EXIF 旋转。

错误使用 OpenAI 错误格式：输入无效或字段不符合协议返回 400，未启用任务或引擎没有视觉编码器却收到图片时返回 503。

另见 [HTTP 客户端示例](../../../examples/decision/decision.py)。

## 响应与校准

每个答案包含 `probabilities`、`confidence`、`decision`，以及按类型给出的 `choice`、`score`（按概率加权的数值期望，附 `legend`）或
`noul`（`yes` 的概率）。候选 softmax 与温度校准沿用 checkpoint 的 `inference.py`，Intern-Decision-0.8B 的默认温度为
2.747760550703，响应中的 `calibration` 字段注明了该温度。推理内核与浮点精度不同可能带来小幅数值误差，因此不保证与官方
`DecisionEngine` 的输出逐位一致。`usage.output_tokens` 为兼容协议而表示回答的问题数，`usage.generated_tokens` 恒为 0。

## 运行方式

决策端点直接驱动引擎并持有本次请求的 session。`DecisionParser` 只负责决策协议与引擎输入/输出之间的转换，不执行引擎调度。
不另设 `DecideRunner`、不依赖公共 runner 的扩展钩子，也没有决策专属视觉流程或自行加载资源的 `DecisionEncoder`：

1. `compile_row` 将 `state`、`questions` 编译成含训练时 assistant 骨架的 messages，沿用 `/generate` 做法，
   将图片引用包装为 `image_url` 块并放在 user 文本前；
   `DecisionParser` 直接把完整消息交给公共预处理。
   启动时只使用引擎已有 tokenizer 校验标记及候选 token，保存少量共享元数据，不再加载第二套 tokenizer/processor。
2. 公共预处理按 checkpoint 模板渲染完整输入，设置 `enable_thinking=False`、`add_vision_id=True`，不添加生成提示，
   并保留 assistant 结束标记。纯文本复用引擎 tokenizer，不额外添加 BOS 或特殊 token；有图时复用已经加载的 Qwen3.5
   processor 完成 tokenization 和图片 token 展开。
3. 预处理完成后，parser 在最终 input IDs 中定位 `<decision>`，把 `logprob_start_len` 设置为首个标记前一位置，
   并保存各问题的评分行索引和预期评分行数，无须提前计算图片展开后的 token 数。
4. `GenerationConfig(max_new_tokens=0, logprobs=0, logprob_extra_token_ids=...)` 请求输入评分。引擎对所选后缀计算 LM head，
   返回指定的 62 个候选分数，候选不受 top-k 排名限制。这里缩减的是返回的分数，不是 LM head 投影的词表维度。
5. parser 校验返回评分行数，按保存的行索引取各问题的分数，按各问题的候选集合归一化、应用温度校准并构造响应。
   引擎返回 raw logits 或 raw log probabilities 都支持这一步候选归一化。

整个请求只执行 prefill，必要时分块预填充，不进入自回归 decode。调度、session 释放与取消复用
`ServerContext.create_session` 和公共的请求清理函数；前缀缓存复用在需要评分的位置之前停止，保证这些分数重新计算。

Decision 图片引用包装成消息块后进入公共多模态预处理，包装过程不解码图片。加载器创建的图片在视觉预处理完成后关闭；
部分失败或请求取消时，先等待仍在运行的 worker，再释放图片。

## 限制

- 仅支持 PyTorch 后端；不支持投机解码与 PD 分离角色。
- 图片展开后必须小于引擎会话长度（即模型自身的上下文长度，除非用 `--session-len` 覆盖）；
  超限输入会被拒绝，不会截断。
- 每个请求包含 1–16 个问题；每个 `choice` 或 `score` 问题有 1–62 个选项，按请求顺序映射到答案符号，
  响应中恢复原始选项名称。
- `model` 字段可省略，或填写服务模型名（`--model-name`，默认为模型路径）或 checkpoint 目录名（如 `Intern-Decision-0.8B`）。
- `--dp > 1` 时请直接访问各 rank 的服务：DP 代理不转发 `/v1/decisions`。
