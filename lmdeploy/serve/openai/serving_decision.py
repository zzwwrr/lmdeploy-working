# Copyright (c) OpenMMLab. All rights reserved.

# TODO(intern-decision): [07] 本文件预留原生决策 HTTP 处理器，当前只有学习用注释。
# 拟承接 /v1/decisions 与 /v1/jev 两个路由，二者共用处理逻辑；属于 LMDeploy 自定义 API。

# TODO(intern-decision): [07] 校验共享 DecisionRequest 和引擎任务能力，保留原协议
# state/questions/images 及 choice/score/noul；model 可省略，提供时校验已加载的模型名。
# HTTP 图片接受 data/base64 内容，不开放任意服务器本地路径；输入模块负责解码和资源清理。

# TODO(intern-decision): [07] 通过 server_context 使用现有 async_engine，调用拟新增的
# async_decide(request)：输入编译与编码 -> 多模态 prefill 评分 -> 决策输出处理。
# 返回统一 DecisionResponse，HTTP 层不重复 softmax、不另建模型、线程或调度器。

# TODO(intern-decision): [07] 将断连、取消和超时传递到评分协程，等待其结束 session；
# 按现有 HTTP 错误结构区分非法输入、任务不可用和执行失败，避免清理遗漏或隐藏故障。
