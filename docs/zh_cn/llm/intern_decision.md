# Intern-Decision 原生决策

通过显式 `task='intern-decision'` 使用 Intern-Decision-0.8B。输入为
`state / questions / images`；支持 `choice`、`score`、`noul`，复用 LMDeploy
现有 Qwen3.5 网络和同一个引擎实例，不需要安装原始 Intern-Decision 仓库。

```python
from lmdeploy import pipeline

pipe = pipeline('/path/to/Intern-Decision-0.8B', task='intern-decision')
try:
    result = pipe.decide({
        'state': {'signal': 'red'},
        'questions': {'action': {
            'type': 'choice', 'instructions': '选择行动',
            'criteria': {'stop': '停车', 'go': '通行'},
        }},
    })
    print(result)
finally:
    pipe.close()
```

默认配置为 PyTorch、BF16、TP/DP/EP=1、batch=1、eager 模式、8192 prefill、
session_len=8192、关闭 prefix caching、cache fraction=0.05。决策 HTTP 复用服务进程内
已有的 PyTorch Engine；首版明确拒绝 enable_mp_engine=True。自定义配置不满足首版
限制时会在初始化拒绝。最终输入还受底层实际缓存容量限制，超长不截断。

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B \
  --backend pytorch --task intern-decision --dtype bfloat16 \
  --eager-mode --max-batch-size 1 --max-prefill-token-num 8192 \
  --session-len 8192 --cache-max-entry-count 0.05 \
  --server-name 127.0.0.1 --server-port 23333
```

`POST /v1/decisions` 和 `/v1/jev` 是同一个自定义决策处理器，共用现有服务鉴权。
HTTP 图片只能上传 base64 图片对象，不能指定服务器文件路径或 URL。Python 可传可信
本地路径；APIClient 会在客户端读取路径并上传字节。HTTP 会执行 EXIF 转正，Python
本地路径只转 RGB，保留历史语义。最多 8 张静态图；每张原始 12 MiB、原始及规范化
总量分别 32 MiB、每图 1600 万像素。

候选仅在合法集合内执行 FP32 softmax；score 返回期望值，noul 返回 P(yes)。
每题的 confidence 是最大候选概率，decision 是离散最佳业务值。usage 的旧字段
output_tokens 表示题数，实际 generated_tokens 固定为 0。thinking.enabled=true
及 calibration 不支持。默认最多接纳 32 个请求，串行评分，取消后等待有界清理和
END_SESSION 回执；清理失败会隔离 handle 并要求重启引擎。

完整合同、错误码、超时配置、字段说明及边界见
[英文接口文档](../../en/llm/intern_decision.md)；实测状态见
[验收报告](../../en/llm/intern_decision_validation.md)。接口迁入成功不等于 HF 数值门槛
通过；尚未验证的模型规格、TP>1、视频、量化和性能不得视为受支持。[10] 的 LM-head
前稀疏投影按计划延期。
