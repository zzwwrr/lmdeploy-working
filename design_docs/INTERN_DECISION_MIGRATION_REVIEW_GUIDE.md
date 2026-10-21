# Intern-Decision v2 协议包迁移快速 Review 指南

本轮将 `serve/processors/intern_decision/` 拆回模型编码、服务协议、Runner 和通用图片加载层。Review 的重点是：职责迁移后，训练时的输入输出语义、评分位置、请求隔离和资源释放是否仍然成立。

对应分支：`feat/intern-decision-v2`。比较范围：`0d446ee3..1ed07ba6`，共 5 个 commit。`BaseRunner` 和 `ChatRunner` 的初步复用已在基线 `0d446ee3` 中完成；本轮继续调整生命周期接口，并将 `DecisionRuntime` 并入共享 `DecideRunner`。本轮没有修改 GPU 模型实现或增加新的候选 logits 引擎能力。

时间有限时，先看 `DecideRunner.run()`、`_score()`、`BaseRunner.close()` 和 `ImageMediaIO._load_image()`，再回看编码与协议。

## 1 按 commit 看改动

| 顺序 | Commit | 主要改动 | Review 要回答的问题 |
|---|---|---|---|
| 1 | `38a21aad` | 合并 schema、encoding、output 到独立编码模块 | 是否保留 prompt、选项顺序、模板参数和校准计算？是否消除了对 serve 层的依赖？ |
| 2 | `ff0c7e75` | 迁移请求响应协议，使用公共服务错误 | 数据契约是否保留？429、503、500 是否仍能正确区分？ |
| 3 | `ee377221` | `ImageMediaIO` 增加可选校验与上传限制 | Decision 策略能否复用加载逻辑？普通图片调用的默认行为是否兼容？ |
| 4 | `1a09dfe0` | Runtime 并入共享 Runner，删除旧包 | 共享对象是否串请求？取消后是否过早关闭图片或释放准入名额？ |
| 5 | `1ed07ba6` | 更新架构文档 | 文档描述的是共享 Runner 组合请求级 BaseRunner，还是已过时的继承关系？ |

前几个 commit 的中间状态仍保留部分旧包文件，最终文件布局以 `1ed07ba6` 为准。

## 2 先认清文件和对象的归属

| 原位置或职责 | 当前入口 | 生命周期与边界 |
|---|---|---|
| `schema.py`、`encoding.py`、`output.py` | [intern_decision_encoding.py](../lmdeploy/intern_decision_encoding.py) | 编译 prompt、编码 token、解释候选分数；不导入 `lmdeploy.serve` |
| `protocol.py` 的请求响应模型 | [decision_protocol.py](../lmdeploy/serve/core/decision_protocol.py) | Python 和 HTTP 共用的数据契约；不包含引擎生命周期 |
| `runtime.py`、决策编排、图片输入策略 | [decide_runner.py](../lmdeploy/serve/core/decide_runner.py) | 每个启用决策服务的引擎持有一个共享实例 |
| session、preprocess、generate、清理 | [base_runner.py](../lmdeploy/serve/core/base_runner.py) | 每次 `DecideRunner.run()` 创建一个实例 |
| 图片解码和 RGB 转换 | [vl/media/image.py](../lmdeploy/vl/media/image.py) | 提供通用机制；Decision 通过参数和 validator 启用自己的策略 |
| 服务错误 | [exceptions.py](../lmdeploy/serve/core/exceptions.py) | 使用 `RequestError` 和 `ErrorCode`；编码模块仍保留纯模型错误 `DecisionOutputError` |

`ServerContext.decision_runner` 和 `Pipeline.decision_runner` 持有共享 Runner。`ServerContext` 继续管理服务上下文，`DecideRunner` 负责决策请求编排，`BaseRunner` 负责单次引擎执行。

当前 `ChatRunner` **继承** `BaseRunner`，`DecideRunner` **组合** `BaseRunner`。两者复用相同的执行机制，但共享 Decision 实例不能把 session、generator、images、encoded 存成“当前请求”字段。

## 3 沿一条请求看复用发生在哪里

```text
HTTP /v1/decisions、/v1/jev          Pipeline.decide()
  allow_local_paths=False            allow_local_paths=True
                 \                    /
                  DecideRunner.run()
                    校验请求、检查模型与视觉能力、占用 pending
                    |
                    _prepare() 在线程中执行
                      _load_images() -> ImageMediaIO + validator
                      DecisionEncoder.encode() -> token IDs、tail、候选 IDs
                    |
                    _score() 使用本请求的 BaseRunner
                      prepare_engine()
                        创建 session 并记录 engine epoch
                        AsyncEngine.preprocess()
                          有图时经过 MultimodalProcessor / ImageEncoder
                        after_preprocess 回调设置评分起点
                        AsyncEngine.generate()
                      iter_outputs() -> 原始 GenOut
                      取目标 token 为 <decision> 的行及候选 logits
                    |
                    DecisionOutputProcessor -> DecisionResponse
                    finally 清理执行资源、关闭图片、释放 pending
```

相关入口：[HTTP endpoint](../lmdeploy/serve/openai/endpoints/decisions.py)、[Pipeline](../lmdeploy/pipeline.py)、[API server](../lmdeploy/serve/openai/api_server.py)。

这里有三种不同的“预处理”：`ImageMediaIO` 把输入解码为 PIL 图片；`DecisionEncoder` 构造训练协议对应的 token 序列；原有多模态路径完成模型所需的视觉预处理和图片占位符展开。

[VLAsyncEngine](../lmdeploy/serve/core/vl_async_engine.py) 继承 `AsyncEngine`，初始化 `ImageEncoder` 并安装 [MultimodalProcessor](../lmdeploy/serve/processors/multimodal.py)。`preprocess()`、`generate()` 的公共入口仍来自 [AsyncEngine](../lmdeploy/serve/core/async_engine.py)。本轮没有把视觉模型预处理搬进 `DecideRunner`。

## 4 四处需要重点检查

### 请求隔离和清理顺序

读 `DecideRunner.run()` 与 `BaseRunner.prepare_engine()`、`close()`：

- `encoded`、图片、评分结果和 `BaseRunner` 都是请求局部变量；实例共享 encoder、校准器、配置、pending 和清理任务集合。
- pending 上限检查与加一之间没有 `await`。这一点依赖所有请求都在所属引擎的同一个事件循环上执行；它不是跨线程计数器。
- 输入准备使用 `to_thread()`，引擎视觉预处理也可能使用工作线程。取消等待者不会终止线程，因此两段准备任务都需要保留并等待，之后才能关闭图片。
- `BaseRunner.close()` 只创建一次清理任务；Decision 外层清理等待它，再关闭图片，最后减 pending。二次取消不能取消这个清理任务，`_cleanup_tasks` 为它保留引用并取回异常。
- 活跃 session 冲突不能重写其他请求的 epoch 或删除其 session；Runner 提前关闭后，预处理返回也不能再创建 generator。

需要接受的取舍：如果工作线程迟迟不结束，清理和 pending 释放也会等待。当前没有新增工作线程超时或强制终止机制。

### 图片展开后的评分位置

读 `DecisionEncoder.encode()` 和 `DecideRunner._score()`，重点核对：

```text
tail = 原始 token 数 - (首个 decision marker 位置 - 1)
logprob_start_len = 图片展开后的 token 数 - tail
```

例如原长 100、首个 marker 位于索引 80，则 tail 为 21。图片展开增加 40 个 token 后，评分起点为 `140 - 21 = 119`，对应新 marker 索引 120 的前一个位置。

这个关系依赖图片占位符均在决策骨架之前。编码器检查 marker 和图片占位符数量、位置；同步 `after_preprocess` 回调在图片展开后、generator 创建前计算起点并检查长度。超过限制会拒绝，不能截断骨架。

同时确认 `max_new_tokens=0`、`logprobs=0` 和 `logprob_token_ids` 的候选集合仍被传入。返回行按**该行预测的目标 token 是否为 marker**筛选，再按每个字段的候选 ID 取值；不能把行号直接当成问题编号。

### 通用图片加载与 Decision 策略

并排看 `ImageMediaIO` 和 `decide_runner.py` 的 `_decode_upload()`、`_validate_image()`、`_load_images()`：

| 检查项 | 当前行为 |
|---|---|
| validator 时机 | `Image.open()` 后、RGB 转换前，能读取原格式和帧数 |
| validator 状态 | 每张图独立绑定声明格式，避免并发或多图之间串值 |
| 通用默认参数 | `validator=None`、`clear_metadata=False`、`strict_base64=False`、`max_bytes=None` |
| Decision 上传策略 | 最多 8 张，单张上传 12 MiB，累计上传 32 MiB；严格 base64、MIME 与内容一致 |
| 图片内容策略 | 每张最多 1600 万像素、仅单帧；上传格式为 JPEG、PNG、WebP、GIF |
| 来源边界 | HTTP 只收上传对象；Python 可用本地路径；Decision 不抓取远程 URL |
| 本地文件 | 不应用上传字节限制；仍检查像素数和帧数 |
| 像素与 metadata | 转为 RGB，不应用 EXIF 旋转；Decision 清空 metadata；加载失败时关闭此前已得到的图片 |

8 张、12/32 MiB 等属于服务策略。训练 prompt、候选映射和评分位置则直接影响模型语义。Review 时应分别判断这两类约束。

`ImageMediaIO.max_bytes` 当前只约束 base64 解码，不约束 `load_bytes()` 和 `load_file()`。Decision 上传先调用受限的 `decode_base64()`，然后再加载图片。通用层原有 `LOAD_TRUNCATED_IMAGES=True` 仍保留，不能据此声称所有截断图片都会被拒绝。

### 编码语义和错误边界

独立编码模块重点看 `compile_row()`、`question_options()`、`DecisionEncoder.encode()`、`DecisionOutputProcessor.__call__()`：

- 字段和选项保持输入顺序；编译与输出处理共用 `question_options()`，不能排序后改变字母对应关系。
- 模板参数保持 `add_generation_prompt=False`、`enable_thinking=False`、`add_vision_id=True`；marker 和候选符号必须满足单 token 要求。
- 输出仍先对候选 logits 做 float32 softmax，再应用默认温度 `2.747760550703` 的校准；`choice`、`score`、`noul` 的解释不能随文件合并改变。
- 训练 metadata 不进入推理 prompt；请求仍限制问题数和选项数，并拒绝启用 thinking。

| 失败场景 | 当前处理 |
|---|---|
| Runner 内输入或图片校验产生 `ValueError` | endpoint 转 HTTP 400 |
| pending 达到上限 | `RequestError(OVERLOADED)`，HTTP 429 |
| 引擎 abort、请求图片但引擎无视觉能力 | `RequestError(ENGINE_UNAVAILABLE)`，HTTP 503 |
| 缺失评分行、候选分数或输出不一致 | `RequestError(INTERNAL_ERROR)`，HTTP 500 |
| `DecisionOutputError` | 在 Runner 边界转换为 `RequestError`，编码模块不反向依赖 serve |

还需扫一眼 [OpenAI 错误映射](../lmdeploy/serve/openai/errors.py)、[Anthropic 错误映射](../lmdeploy/serve/anthropic/errors.py) 和 [Responses 请求处理](../lmdeploy/serve/openai/responses/request.py)：本轮增加了公共 `OVERLOADED` 的 `rate_limit_error` 映射。

## 5 兼容性和验证范围

HTTP 路由及 `Pipeline.decide()` 的使用入口保留。内部导入路径已经迁移，旧协议包没有兼容转发层；直接导入旧包或访问 `decision_runtime` 的调用方需要更新。

已有用例可以作为阅读索引：

| 文件 | 优先阅读的场景 |
|---|---|
| [test_decision.py](../tests/test_lmdeploy/test_decision.py) | 图片展开后评分、共享实例重复及并发调用、准备阶段二次取消、图片释放、HTTP/Pipeline 入口 |
| [test_base_runner.py](../tests/test_lmdeploy/serve/core/test_base_runner.py) | session 冲突、预处理及回调失败、断连、生成阶段取消、重复等待清理 |
| [test_decision_golden.py](../tests/test_lmdeploy/test_decision_golden.py) | 编码和 checkpoint 对齐入口；旧 v1 fixture 只能作为历史参照，不能单独证明当前实现与官方模型一致 |

本轮按约定未运行测试，仅完成代码审阅、修改文件的 Python 静态语法解析和 `git diff --check`。真实 checkpoint、GPU 数值及吞吐不在已验证范围内。尤其是**引擎视觉预处理期间的取消**和**普通 Chat 图片默认加载行为**，仍值得在后续运行验证中单独确认。

以下命令只用于查看本轮提交，执行目录为仓库根目录：

```powershell
git log --reverse --oneline 0d446ee3..1ed07ba6
git diff --stat 0d446ee3 1ed07ba6
git show 1a09dfe0 -- lmdeploy/serve/core/base_runner.py lmdeploy/serve/core/decide_runner.py
git show ee377221 -- lmdeploy/vl/media/image.py
```
