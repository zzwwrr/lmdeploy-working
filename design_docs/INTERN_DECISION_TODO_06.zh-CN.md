# Intern-Decision TODO [06]：会话清理改动与解耦方案

日期：2026-10-19。

本文记录原生 Intern-Decision 接入的 [06] 改动，并整理讨论中的接口解耦方案。**本次仅补充文档，不修改运行时代码；下文“建议方案”尚未实现。**

## 1. 版本与范围

- 执行依据：任务目录下 `NATIVE_LMDEPLOY_ADAPTATION_PLAN.zh-CN.md` 第 6.2、6.3 节及 TODO [06]。
- [06] 原提交：`d83dbf2f14d01c98c8c9dcbc5b38f2d3d5940a51`，`fix(decision): [06] confirm session cleanup and quarantine uncertain handles`。
- 原提交规模：7 个文件，新增 97 行，删除 21 行。
- 当前代码说明基于 `35c1b8793aa91c1a93baa79e9c0dd2a44fec4a77`。该版本包含 [09] 对取消传播、关闭失败和异常上下文的后续补强，不能把所有现状都归入 [06] 原提交。

## 2. 为什么需要 [06]

决策请求通过零生成 prefill 取得 logits。即使没有生成 token，底层仍可能创建调度会话、序列和缓存，`Session.step == 0` 不代表引擎中没有资源。

必须区分三种对象：

| 对象 | 含义 | 清理要求 |
| --- | --- | --- |
| 上层 `Session` | 保存会话 ID、历史、step 和当前 handle 等状态 | 根据所有权保留或删除本地记录 |
| 请求 handle，即 `EngineInstance` | 池中的轻量推理实例，持有请求发送器，复用共享引擎 | 状态可靠时才能归还池 |
| 底层引擎会话 | 由 `session_id` 标识的调度状态及相关缓存 | 按结束协议处理，确认结果 |

之前的 `end()` 只发送 `END_SESSION(response=False)`，发送后立即返回。若把这个返回当成“结束已完成”，上层就无法确认后端是否已经处理该请求。

另一个问题是持有 handle 时调用 `Session.async_close()`：后者会等待当前 handle 释放，导致评分清理等待自己。因而需要在持有 handle 的作用域内直接完成清理，之后再归还实例。

目标顺序为：

```text
取得 handle → 提交 prefill → 读取并校验 logits
                             ↓ 成功 / 异常 / 取消 / 超时
                 必要时停止尚未完成的请求
                 → 关闭已停止迭代的结果生成器
                 → 需要结束时等待 END_SESSION 回执
                 → 归还确认可复用的 handle
                 → 按所有权处理上层 Session 记录
```

## 3. [06] 原提交逐文件说明

| 文件 | 原提交中的实际改动 | 目的与影响 |
| --- | --- | --- |
| `lmdeploy/pytorch/engine/engine.py` | 新增 `get_max_session_len()` | 公开缓存配置之后的实际会话容量，供输入长度校验使用 |
| `lmdeploy/pytorch/engine/engine_instance.py` | `end()` 从只发送改为同步等待回执；`async_end()` 改为异步等待回执；异步取消接受会话不存在并返回状态，其他失败抛异常 | 让上层能够确认结束、识别取消失败；同时改变了公共结束接口的既有语义 |
| `lmdeploy/pytorch/engine/mp_engine/base.py` | 初始化时通过 RPC 查询实际长度并提供 getter；结束后的状态删除改为 `pop(session_id, None)` | 透传容量能力，避免本地状态已删除时再次删除报错 |
| `lmdeploy/pytorch/engine/mp_engine/base_worker.py` | 增加容量查询转发 | 让代理通过 worker 获取实际容量 |
| `lmdeploy/serve/core/async_engine.py` | 删除完成后的 [06] TODO 注释 | 评分实现此前已在 [05] 接入，实际清理逻辑落在共享评分和 Session 中 |
| `lmdeploy/serve/core/scoring.py` | 新增 `await_bounded_cleanup()`；评分前检查引擎清理故障状态 | 调用方取消后仍允许清理有界完成；清理故障后拒绝继续评分 |
| `lmdeploy/serve/managers/session_manager.py` | 增加严格异常模式、handle 可复用状态、引擎会话存在状态和 `finish_score()`；调整关闭的提前返回条件 | 防止异常被吞、清理未确认即归还实例，以及 step 为 0 时遗漏评分会话 |

`request.py` 本来就具备请求和回执机制，[06] 没有修改该文件，也没有新增底层请求类型。

## 4. 当前清理行为（包含 [09] 补强）

### 4.1 评分入口与所有权

```text
AsyncEngine.async_decide()
  → decision_runtime.decide()
  → AsyncEngine.async_get_logits()
  → score_logits()
  → Session.request_handle(strict=True)
  → handle.async_stream_infer(max_new_tokens=0)
  → finally: Session.finish_score()
```

普通 PyTorch `async_get_logits()` 也进入 `score_logits()`，因此该机制覆盖共享评分路径，并非决策 HTTP 路由独有。

- 未传入 sessions：评分层创建并拥有上层会话；当前要求 `sequence_end=True`，离开评分作用域时移除本地记录。
- 调用方传入 sessions：上层记录归调用方所有。成功且 `sequence_end=False` 时保留底层会话，后续由调用方关闭。
- 评分出现异常时，当前实现会尝试结束已提交的会话，即使调用方原先要求保留。
- 批量评分中任一子任务失败，会取消并等待其余子任务，再退出本次调用。

### 4.2 清理确认与失败处理

`finish_score()` 在 handle 仍被持有时执行：未完成的请求先取消，结果生成器随后关闭，需要结束时等待 `async_end()`。

结束状态接受 `SUCCESS` 与 `SESSION_NOT_EXIST`，后者表示无需再结束的幂等结果。其他状态视为失败。回执表示引擎已处理结束协议；不能据此推断 CUDA 分配器已把全部显存归还给系统。引擎对 `preserve_cache` 也有自身的处理分支，评分配置则明确使用 `with_cache=False`、`preserve_cache=False`。

清理默认最多等待 10 秒；决策路径可通过 `cleanup_timeout` 配置。`await_bounded_cleanup()` 使用有界等待并屏蔽外层取消对内部清理的直接打断。清理完成后仍向调用方传播取消；当前版本用 `cleanup_completed` 标记区分“取消已传播但资源清理完成”和真正的清理失败。

清理失败或超时会使 `_handle_reusable=False`，该 handle 不再归还池；评分层还会设置 `_scoring_failure`，决策 runtime 存在时同步设置其故障状态。后续评分拒绝继续，要求重启引擎。已有业务异常优先保留，附带清理失败上下文。

当前 `Session.async_close()` 也采用严格异常处理和有界结束等待。这是相较原有公共关闭路径的行为变化，应纳入后续兼容性解耦范围。

### 4.3 容量查询与 8K 限制

`get_max_session_len()` 暴露已有的实际容量，不是新增一个 8K 模型上限。决策输入校验当前使用：

```python
limit = min(8192, engine.session_len, engine.engine.get_max_session_len())
```

其中 8192 是首版接入的任务限制；实际缓存容量还可能更小。长度计算包含图片展开后的 token。模型配置的上下文长度与该首版服务限制是两个概念，仅调整 `session_len` 不能解除上述 8192 限制。

多进程相关容量 RPC 不代表决策模式已支持多进程。当前首版明确拒绝 `enable_mp_engine=True`；已有代理的会话就绪与取消所有权问题，不因这些转发改动而得到完整解决。

## 5. 原来的 send_async 有什么问题

它本身没有协议错误；问题在于它只保证提交请求，满足不了评分清理所需的完成确认。

| 调用方式 | 等待回执 | 对调用线程的影响 |
| --- | --- | --- |
| `send_async(..., response=False)` | 否 | 提交后返回 |
| `send(..., response=True)` | 是 | 同步等待 |
| `await async_send(..., response=True)` | 是 | 协程挂起，事件循环可运行其他任务 |

`send_async` 是普通发送方法，`async_send` 是发送后等待回执的协程，名称相近但含义不同。确认结束不等于必须阻塞线程，异步评分应使用异步等待。

当前实现直接修改公共 `end()` / `async_end()`，让原调用方也承担新的等待、返回值和异常语义。这正是提出新增独立接口的原因。

## 6. 建议的解耦方案（尚未实现）

### 6.1 拆分结束接口

| 接口 | 建议行为 | 使用场景 |
| --- | --- | --- |
| `end()` / `async_end()` | 恢复 [06] 之前只发送结束请求的行为，使用 `response=False`，返回 `None` | 保持原有推理路径的调用语义 |
| `end_confirmed()` | `send(..., response=True)`，同步等待并校验结束回执 | 确实需要同步确认的调用方 |
| `async_end_confirmed()` | `await async_send(..., response=True)`，异步等待并校验结束回执 | 评分清理与保留评分会话的关闭 |

新接口接受 `SUCCESS`、`SESSION_NOT_EXIST`，其他结果抛异常。外层评分清理继续提供超时、取消保护和失败隔离。同步确认接口不能直接用于需要继续运行的同一事件循环，否则可能阻塞处理回执所需的执行。

### 6.2 上层必须对应修改

只新增底层接口并恢复旧实现还不够，否则 `finish_score()` 会把旧 `async_end()` 的 `None` 当成未确认状态。必须同步迁移调用点：

1. `Session.finish_score()` 直接调用 `async_end_confirmed()`，继续等待确认后再归还 handle。
2. `Session.async_close()` 根据会话清理要求选择旧接口或确认式接口。
3. 多进程实例、worker 实例池和 worker RPC 补充新异步接口的转发，保留旧 RPC 路径；这些转发仍不等于开放决策多进程支持。
4. 按需要补充基类接口声明，明确不具备确认能力时的行为，不能悄悄退回只发送请求并当作已确认。

HTTP 路由、APIClient 和用户请求格式不需要新增清理参数。

### 6.3 上层如何知道是不是决策请求

建议记录“本会话是否要求确认式结束”，而不是根据模型名称、schema 或 HTTP 路由判断请求身份。

原因有三点：清理对象由 `session_id` 确定；普通 logits 评分也需要同样的生命周期保证；保留会话以后调用公共 `async_close()` 时，原始请求上下文已经不在该入口。

可以在 Session 增加独立状态，暂名 `_requires_confirmed_end`：

```python
# 以下仅为设计示意，不是当前实现。
# 应在成功取得 handle 后、可能向引擎提交评分请求前设置。
session._requires_confirmed_end = True

# 后续公共关闭入口，在等待当前工作结束并取得清理 handle 后选择接口。
if session._requires_confirmed_end:
    await handle.async_end_confirmed(session.session_id)
else:
    await handle.async_end(session.session_id)
```

状态生命周期要求：

- 初始化为 `False`；只有清理责任实际进入评分路径后才设为 `True`，不在 HTTP 接纳阶段提前设置。
- 成功评分但 `sequence_end=False`：保留为 `True`，以后关闭仍需等待确认。
- 确认结束后：清除确认要求及底层会话活动状态；`reset()` 也应重置新增状态。
- 请求未真正提交时：结合原有状态处理，不能把已有待确认要求误清除。
- 结束失败或超时：不得假装清理完成、清除责任或复用状态不明的 handle。
- 公共关闭需在当前工作结束后读取状态，防止与正在执行的评分竞争。

现有 `_engine_session_active` 表达“可能还有底层会话”，不等同于清理策略，不宜让同一个布尔变量承担两种含义。真正提交状态仍应由评分流程及引擎协议确认，策略标记本身不能证明后端会话已经存在。

此方案允许一个会话按实际使用记录清理要求；不能仅因为引擎启用了决策任务，就把所有普通生成请求都改成新的关闭语义。

### 6.4 兼容性边界

- 旧结束接口和普通生成的关闭路径应一并审查，恢复其原有等待与异常语义；仅恢复方法名对应的实现不足以完成隔离。
- 评分与其后续关闭维持确认式结束、超时、取消保护、handle 隔离等保证。
- 本轮建议聚焦结束接口；`async_cancel()` 在 [06] 中也有语义调整，如需恢复其兼容性，应单独审查，不把它隐含混入本次方案。
- 旧外置后端固定校验 `engine_instance.py`、`request.py`、`session_manager.py` 的 SHA256。[06] 修改其中两个文件会触发旧后端校验失败。保持隔离基线，不删除校验或直接更新哈希来宣称兼容。

## 7. 验证依据与后续验收

现有 `tests/test_lmdeploy/test_decision.py` 已包含相关测试：

- `test_prefill_config_and_confirmed_cleanup`：回执未到时任务未完成、handle 未归还。
- `test_running_cancellation_and_queued_cancellation`、`test_cancellation_during_end_ack_wait`：运行与等待回执期间的取消。
- `test_cleanup_failure_quarantines_handle`：结束失败或超时后隔离 handle。
- `test_step_zero_close_confirmation_failure_is_visible`：step 为 0 的保留评分会话仍需清理，失败可见。
- `test_end_session_uses_request_receipt`：当前 `async_end()` 使用 `response=True`。
- `test_multimodal_forward_and_actual_capacity`：多模态透传及实际容量约束。

这些是当前实现的测试依据，不代表待议的解耦方案已被验证。本次仅写文档，没有重新运行 CPU 或 GPU 测试。

若后续批准实施解耦，应增加或调整以下验收：

1. 旧 `end()` / `async_end()` 只发送、不等待回执，保持原返回值。
2. 新同步与异步确认接口正确等待、接受幂等结果，并传播失败。
3. 评分清理明确走新接口；确认前 handle 不归还。
4. 普通生成关闭仍走原路径；保留评分会话即使 step 为 0，也走确认路径。
5. 清理标记在成功、保留、失败、取消和重置时正确转换，批量失败仍等待其他任务收尾。
6. 代理新旧接口正确转发；不扩大决策多进程支持范围。
7. 在允许且空闲的 GPU 6 上回归真实决策推理、取消恢复和普通生成，记录资源回收结果。

解耦实现应另行提交，和本次说明文档提交分开。
